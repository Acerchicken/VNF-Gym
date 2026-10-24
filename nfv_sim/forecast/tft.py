"""Temporal Fusion Transformer (Lim et al. 2021, "Temporal Fusion Transformers for interpretable multi-horizon
time series forecasting") — bản tự viết gọn, cùng giao diện với LSTM (vòng train/loss ở neural.py).

Ba loại đầu vào (đúng như bài báo):
    tĩnh (static)        : ID chuỗi, ID dataset                      -> embedding
    biết trước (known)   : 5 feature lịch (data.N_CALENDAR), có cho cả L bước quá khứ và H bước tương lai
    chỉ biết quá khứ     : tải (target)

Luồng tính (mỗi khối là một class bên dưới):
    1. Mỗi biến -> vector d chiều (Linear cho số thực, Embedding cho ID).
    2. Chọn biến (_VariableSelection): học trọng số softmax cho từng biến ở mỗi bước -> biết biến nào quan trọng.
    3. Bộ mã hoá biến tĩnh: 4 vector ngữ cảnh c_s (cho chọn biến), c_e (làm giàu), c_h, c_c (trạng thái đầu LSTM).
    4. LSTM encoder trên L bước quá khứ, LSTM decoder trên H bước tương lai (TFT dùng lại LSTM bên trong).
    5. Làm giàu bằng biến tĩnh (GRN có ngữ cảnh c_e).
    6. Attention có thể diễn giải (_InterpretableAttention): mỗi bước tương lai nhìn lại mọi bước trước nó;
       trọng số attention cho biết model dựa vào bước quá khứ nào.
    7. GRN theo từng vị trí + lớp ra Q quantile, dạng phần dư so với giá trị hiện tại (như LSTM).

Các "cổng" (GLU) và nối tắt (skip) cho phép model bỏ qua khối không có ích (vd khi dữ liệu không có giờ thật).
`explain()` trả về trọng số chọn biến và attention trung bình để vẽ hình (examples/explain_tft.py).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from ..data import N_CALENDAR
from .base import as_series_list, make_extra, make_windows, register_forecaster
from .neural import NeuralForecaster

CALENDAR_NAMES = ("giờ (sin)", "giờ (cos)", "thứ (sin)", "thứ (cos)", "có giờ thật")
STATIC_NAMES = ("ID chuỗi", "ID dataset")


class _GLU(nn.Module):
    """Gated Linear Unit: GLU(a) = (W1 a) * sigmoid(W2 a). Cổng sigmoid ~0 thì khối phía trước bị "tắt"."""

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_in, 2 * d_out)

    def forward(self, a):
        return nn.functional.glu(self.fc(self.drop(a)), dim=-1)


class _GRN(nn.Module):
    """Gated Residual Network: GRN(a, c) = LayerNorm(skip(a) + GLU(W2 ELU(W1 a + Wc c))).
    c là ngữ cảnh tĩnh (tuỳ chọn). Nhờ cổng GLU, GRN có thể chỉ trả về skip(a) nếu phần phi tuyến vô ích."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float, d_ctx: Optional[int] = None):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ctx = nn.Linear(d_ctx, d_hidden, bias=False) if d_ctx else None
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.glu = _GLU(d_hidden, d_out, dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, a, c=None):
        h = self.fc1(a)
        if self.ctx is not None and c is not None:
            while c.dim() < h.dim():           # (B, d) -> (B, 1, d) để cộng với chuỗi theo thời gian
                c = c.unsqueeze(-2)
            h = h + self.ctx(c)
        h = self.fc2(nn.functional.elu(h))
        return self.norm(self.skip(a) + self.glu(h))


class _VariableSelection(nn.Module):
    """Chọn biến: đầu vào (..., n_vars, d) -> (..., d) là tổng có trọng số các biến, trọng số (..., n_vars).
    Trọng số = softmax(GRN(mọi biến ghép lại, ngữ cảnh)); mỗi biến còn qua một GRN riêng trước khi cộng."""

    def __init__(self, n_vars: int, d: int, dropout: float, d_ctx: Optional[int] = None):
        super().__init__()
        self.weight_grn = _GRN(n_vars * d, d, n_vars, dropout, d_ctx)
        self.var_grns = nn.ModuleList([_GRN(d, d, d, dropout) for _ in range(n_vars)])

    def forward(self, x, c=None):
        w = torch.softmax(self.weight_grn(x.flatten(-2), c), dim=-1)
        h = torch.stack([g(x[..., i, :]) for i, g in enumerate(self.var_grns)], dim=-2)
        return (w.unsqueeze(-1) * h).sum(-2), w


class _InterpretableAttention(nn.Module):
    """Multi-head attention "có thể diễn giải": mỗi head có Q, K riêng nhưng dùng CHUNG V, rồi lấy trung bình
    các head -> một ma trận trọng số duy nhất (bước tương lai x bước quá khứ) đọc được trực tiếp."""

    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.dk = max(1, d // n_heads)
        self.q = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(n_heads)])
        self.k = nn.ModuleList([nn.Linear(d, self.dk) for _ in range(n_heads)])
        self.v = nn.Linear(d, self.dk)
        self.out = nn.Linear(self.dk, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, mask):
        v = self.v(k)
        heads, attns = [], []
        for wq, wk in zip(self.q, self.k):
            s = wq(q) @ wk(k).transpose(-1, -2) / self.dk ** 0.5
            a = torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1)
            heads.append(self.drop(a) @ v)
            attns.append(a)
        return self.out(torch.stack(heads).mean(0)), torch.stack(attns).mean(0)


class _TFTNet(nn.Module):
    def __init__(self, context_len: int, horizon: int, n_out: int, n_series: int, n_datasets: int,
                 d: int = 32, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.L, self.H, self.n_out = context_len, horizon, n_out
        K = N_CALENDAR
        # 1. embedding từng biến
        self.target_emb = nn.Linear(1, d)
        self.known_emb = nn.ModuleList([nn.Linear(1, d) for _ in range(K)])
        self.series_emb = nn.Embedding(n_series, d)
        self.dataset_emb = nn.Embedding(n_datasets, d)
        # 2-3. chọn biến + bộ mã hoá biến tĩnh
        self.static_vsn = _VariableSelection(2, d, dropout)
        self.c_s, self.c_e, self.c_h, self.c_c = (_GRN(d, d, d, dropout) for _ in range(4))
        self.past_vsn = _VariableSelection(1 + K, d, dropout, d_ctx=d)
        self.future_vsn = _VariableSelection(K, d, dropout, d_ctx=d)
        # 4. LSTM encoder-decoder + cổng nối tắt
        self.encoder = nn.LSTM(d, d, batch_first=True)
        self.decoder = nn.LSTM(d, d, batch_first=True)
        self.lstm_gate, self.lstm_norm = _GLU(d, d, dropout), nn.LayerNorm(d)
        # 5-6. làm giàu tĩnh + attention
        self.enrich = _GRN(d, d, d, dropout, d_ctx=d)
        self.attn = _InterpretableAttention(d, n_heads, dropout)
        self.attn_gate, self.attn_norm = _GLU(d, d, dropout), nn.LayerNorm(d)
        # 7. GRN theo vị trí + lớp ra
        self.pos_grn = _GRN(d, d, d, dropout)
        self.out_gate, self.out_norm = _GLU(d, d), nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)
        T = context_len + horizon
        # bước i chỉ được nhìn bước j <= i (không nhìn trước tương lai của chính nó)
        self.register_buffer("mask", torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)[context_len:])
        self.last_weights: Dict[str, torch.Tensor] = {}

    def forward(self, x, cal, sid, did):           # x (B, L), cal (B, L+H, K) -> (B, H, Q)
        L = self.L
        # biến tĩnh -> 4 vector ngữ cảnh
        static, w_static = self.static_vsn(torch.stack([self.series_emb(sid), self.dataset_emb(did)], dim=1))
        cs, ce, ch, cc = self.c_s(static), self.c_e(static), self.c_h(static), self.c_c(static)
        # biến theo thời gian -> chọn biến
        known = torch.stack([emb(cal[..., i:i + 1]) for i, emb in enumerate(self.known_emb)], dim=-2)  # (B,T,K,d)
        past_in = torch.cat([self.target_emb(x.unsqueeze(-1)).unsqueeze(-2), known[:, :L]], dim=-2)
        past, w_past = self.past_vsn(past_in, cs)                    # (B, L, d)
        future, w_future = self.future_vsn(known[:, L:], cs)         # (B, H, d)
        # LSTM encoder-decoder, trạng thái đầu lấy từ biến tĩnh
        enc, state = self.encoder(past, (ch.unsqueeze(0), cc.unsqueeze(0)))
        dec, _ = self.decoder(future, state)
        inp = torch.cat([past, future], dim=1)
        phi = self.lstm_norm(inp + self.lstm_gate(torch.cat([enc, dec], dim=1)))   # (B, L+H, d)
        theta = self.enrich(phi, ce)
        # attention: chỉ H bước tương lai làm query, nhìn lại toàn bộ chuỗi (có mask)
        att, a = self.attn(theta[:, L:], theta, self.mask)
        delta = self.attn_norm(theta[:, L:] + self.attn_gate(att))
        out = self.out_norm(phi[:, L:] + self.out_gate(self.pos_grn(delta)))
        self.last_weights = dict(static=w_static, past=w_past, future=w_future, attention=a)
        return x[:, -1:, None] + self.head(out)                       # phần dư so với x[t]


@register_forecaster("tft")
class TFTForecaster(NeuralForecaster):

    def __init__(self, d_model: int = 32, n_heads: int = 4, dropout: float = 0.1, **kw):
        super().__init__(**kw)
        self.d_model, self.n_heads, self.dropout = d_model, n_heads, dropout

    def _build_net(self):
        return _TFTNet(self.context_len, self.horizon, self.n_out, self.n_series, self.n_datasets,
                       self.d_model, self.n_heads, self.dropout)

    @torch.no_grad()
    def explain(self, data, max_windows: int = 20000, seed: int = 0) -> Dict[str, np.ndarray]:
        """Trọng số trung bình trên các cửa sổ của `data` (TimeSeries hoặc list):
            static    (2,)        : ID chuỗi, ID dataset
            past      (1 + K,)    : tải + K feature lịch, ở các bước quá khứ
            future    (K,)        : K feature lịch ở các bước tương lai
            attention (H, L + H)  : bước tương lai h nhìn vào bước nào (cột L-1 = hiện tại t)
        Tên biến: STATIC_NAMES, ("tải",) + CALENDAR_NAMES."""
        if self.net is None:
            raise RuntimeError("TFT chưa fit")
        rng = np.random.default_rng(seed)
        acc: Dict[str, list] = {k: [] for k in ("static", "past", "future", "attention")}
        for ts in as_series_list(data):
            t = np.arange(self.context_len - 1, len(ts) - self.horizon)
            if len(t) == 0:
                continue
            t = np.sort(rng.choice(t, min(len(t), max_windows), replace=False))
            windows = make_windows(ts.values, t, self.context_len).astype(np.float32)
            D = self._windows_to_tensors(windows, make_extra(ts.ctx, t, self.context_len, self.horizon))
            self.net.eval()
            for i in range(0, len(D["x"]), 4096):
                self.net(*self._inputs(D, slice(i, i + 4096)))
                w = self.net.last_weights
                acc["static"].append(w["static"].sum(0).numpy())
                acc["past"].append(w["past"].sum((0, 1)).numpy() / w["past"].shape[1])
                acc["future"].append(w["future"].sum((0, 1)).numpy() / w["future"].shape[1])
                acc["attention"].append(w["attention"].sum(0).numpy())
                acc.setdefault("n", []).append(len(w["static"]))
        n = float(sum(acc.pop("n")))
        return {k: np.sum(v, axis=0) / n for k, v in acc.items()}
