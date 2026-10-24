"""Phần dùng chung cho các forecaster mạng nơ-ron (LSTM, TFT): dữ liệu, vòng train, loss, dự báo.

Lớp con chỉ cần cài `_build_net()` trả về một nn.Module với
    forward(x, cal, sid, did) -> (B, H, Q)
    x   : (B, L) lịch sử tải
    cal : (B, L + H, N_CALENDAR) feature lịch của quá khứ và tương lai (giờ của tương lai đã biết trước)
    sid, did : (B,) ID số của chuỗi và dataset (0 = chưa gặp khi train), dùng cho embedding
    Q = số quantile (hoặc 1 khi dự báo một giá trị). Net không dùng thông tin phụ thì bỏ qua cal/sid/did.

Embedding = bảng vector số model tự học, mỗi ID một vector (như "thẻ tên" để phân biệt WEB với máy Google).
Khi train, `id_dropout` (10%) mẫu bị thay ID bằng 0 để model cũng học cách dự báo cho chuỗi chưa từng gặp.

Hai chế độ:
* quantiles=() : dự báo một giá trị, loss MSE (có under_weight phạt dự báo thiếu).
* quantiles=(0.1, 0.5, 0.9): dự báo khoảng, loss quantile (pinball). Với P90, dự báo THẤP hơn thực tế bị phạt
  0,9 * sai số, dự báo CAO hơn chỉ bị phạt 0,1 * sai số -> model tự đẩy P90 lên cao, sao cho ~90% giá trị thật
  nằm dưới P90.
"""
from __future__ import annotations

import copy
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from ..config import FORECAST
from ..data import N_CALENDAR
from .base import Forecaster, as_series_list, make_training_set


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Loss quantile trung bình. pred (B, H, Q), y (B, H), q (Q,) các mức quantile.
    Sai số e = y - pred; loss = q*e nếu e >= 0 (dự báo thấp), (q-1)*e nếu e < 0 (dự báo cao).
    Ví dụ q = 0.9, y = 1.0: pred = 0.8 -> loss 0.18; pred = 1.2 -> loss 0.02."""
    e = y.unsqueeze(-1) - pred
    return torch.maximum(q * e, (q - 1.0) * e).mean()


class NeuralForecaster(Forecaster):
    """Lớp cơ sở cho forecaster PyTorch. Tham số chung: xem FORECASTERS["lstm"] trong config.py."""

    def __init__(self, quantiles: Sequence[float] = FORECAST["quantiles"], epochs: int = 30, batch_size: int = 256,
                 lr: float = 1e-3, patience: int = 5, under_weight: float = 1.0, stride: int = 1,
                 id_dropout: float = 0.1, seed: int = 0, verbose: bool = True, **kw):
        super().__init__(**kw)
        self.quantiles = tuple(float(q) for q in quantiles) if quantiles else ()
        if self.quantiles and 0.5 not in self.quantiles:
            raise ValueError("quantiles phải có 0.5 (P50 dùng làm dự báo chính)")
        self.epochs, self.batch_size, self.lr, self.patience = epochs, batch_size, lr, patience
        self.under_weight, self.stride, self.seed, self.verbose = under_weight, stride, seed, verbose
        self.id_dropout = id_dropout
        self.vocab = dict(series={}, dataset={})   # tên -> ID (1, 2, ...); 0 = chưa gặp
        self.net: Optional[nn.Module] = None
        self.history: list = []                 # (epoch, train_loss, val_loss)

    @property
    def n_out(self) -> int:
        return len(self.quantiles) or 1

    # ---- lớp con cài đặt ----
    def _build_net(self) -> nn.Module:
        raise NotImplementedError

    # ---- dữ liệu ----
    @property
    def n_series(self) -> int:          # số ID chuỗi, gồm ID 0
        return len(self.vocab["series"]) + 1

    @property
    def n_datasets(self) -> int:
        return len(self.vocab["dataset"]) + 1

    def _ids(self, names, kind: str) -> np.ndarray:
        table = self.vocab[kind]
        return np.array([table.get(n, 0) for n in names], np.int64)

    def _tensors(self, series) -> dict:
        """list[TimeSeries] -> dict tensor x (M, L), y (M, H), cal (M, L+H, K), sid, did (M,)."""
        d = make_training_set(series, self.context_len, self.horizon, self.stride)
        return dict(x=torch.as_tensor(d["x"]), y=torch.as_tensor(d["y"]), cal=torch.as_tensor(d["cal"]),
                    sid=torch.as_tensor(self._ids(d["name"], "series")),
                    did=torch.as_tensor(self._ids(d["dataset"], "dataset")))

    @staticmethod
    def _inputs(D: dict, idx) -> tuple:
        """Các tensor đưa vào net.forward cho các mẫu idx."""
        return D["x"][idx], D["cal"][idx], D["sid"][idx], D["did"][idx]

    def _loss(self, pred, y):
        if self.quantiles:
            return pinball_loss(pred, y, self._q)
        p = pred[..., 0]
        w = torch.where(p < y, self.under_weight, 1.0)       # dự báo thiếu -> phạt nặng hơn
        return (w * (p - y) ** 2).mean()

    # ---- train ----
    def fit(self, train, val=None):
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        series = as_series_list(train)
        val = [v for v in as_series_list(val) if len(v) > self.context_len + self.horizon]
        self._setup(series)
        self.net = self._build_net()
        self._q = torch.tensor(self.quantiles or (0.5,), dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        D = self._tensors(series)
        Dv = self._tensors(val) if val else None
        n = len(D["y"])
        tag = self.name

        best, best_state, bad = float("inf"), None, 0
        self.history = []
        for ep in range(1, self.epochs + 1):
            self.net.train()
            perm = torch.randperm(n, generator=g)
            tot = 0.0
            for i in range(0, n, self.batch_size):
                b = perm[i:i + self.batch_size]
                x, cal, sid, did = self._inputs(D, b)
                if self.id_dropout > 0:            # một phần mẫu coi như "chuỗi lạ" (ID 0)
                    drop = torch.rand(len(b), generator=g) < self.id_dropout
                    sid, did = sid.masked_fill(drop, 0), did.masked_fill(drop, 0)
                loss = self._loss(self.net(x, cal, sid, did), D["y"][b])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(b)
            tr = tot / n
            va = self._eval_loss(Dv) if Dv is not None else tr
            self.history.append((ep, tr, va))
            if self.verbose:
                print(f"  [{tag}] epoch {ep:3d}  train={tr:.5f}  val={va:.5f}")
            if va < best - 1e-7:
                best, best_state, bad = va, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(f"  [{tag}] early stopping (best val={best:.5f})")
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def _setup(self, series) -> None:
        """Lập bảng ID chuỗi/dataset từ dữ liệu train (trước khi tạo net, vì embedding cần biết số ID)."""
        names = sorted({n for ts in series for n in ts.ctx.names})
        dsets = sorted({d for ts in series for d in ts.ctx.datasets})
        self.vocab = dict(series={n: i + 1 for i, n in enumerate(names)},
                          dataset={d: i + 1 for i, d in enumerate(dsets)})

    @torch.no_grad()
    def _eval_loss(self, D):
        self.net.eval()
        n, tot = len(D["y"]), 0.0
        for i in range(0, n, 4096):
            idx = slice(i, i + 4096)
            tot += self._loss(self.net(*self._inputs(D, idx)), D["y"][idx]).item() * len(D["y"][idx])
        return tot / n

    # ---- dự báo ----
    def _windows_to_tensors(self, windows, extra=None) -> dict:
        """(N, L, n) + thông tin phụ -> dict tensor, mỗi (thời điểm i, cột j) là một mẫu, thứ tự i*n + j.
        Không có extra: feature lịch = 0, ID = 0 (chuỗi lạ)."""
        N, L, n = windows.shape
        x = np.ascontiguousarray(windows.transpose(0, 2, 1)).reshape(-1, L)
        if extra is None:
            cal = np.zeros((N * n, L + self.horizon, N_CALENDAR), np.float32)
            sid = did = np.zeros(N * n, np.int64)
        else:
            cal = np.repeat(extra["cal"], n, axis=0)                 # cùng giờ cho mọi cột
            sid = np.tile(self._ids(extra["names"], "series"), N)
            did = np.tile(self._ids(extra["datasets"], "dataset"), N)
        return dict(x=torch.as_tensor(x, dtype=torch.float32), cal=torch.as_tensor(cal),
                    sid=torch.as_tensor(sid), did=torch.as_tensor(did))

    @torch.no_grad()
    def _predict_raw(self, windows, extra=None) -> np.ndarray:
        """(N, L, n) -> (N, H, n, Q) đầu ra của net (quantile đã sắp tăng dần)."""
        if self.net is None:
            raise RuntimeError(f"{self.name} chưa fit")
        self.net.eval()
        N, L, n = windows.shape
        D = self._windows_to_tensors(windows, extra)
        M = N * n
        y = torch.cat([self.net(*self._inputs(D, slice(i, i + 4096))) for i in range(0, M, 4096)]).numpy()
        y = np.sort(y, axis=-1)          # tránh P10 > P50 > P90 bị "chéo nhau"
        return y.reshape(N, n, self.horizon, self.n_out).transpose(0, 2, 1, 3)

    def predict_dist_batch(self, windows, extra=None):
        if not self.quantiles:
            return super().predict_dist_batch(windows, extra)
        return self._predict_raw(windows, extra)

    def predict_batch(self, windows, extra=None):
        out = self._predict_raw(windows, extra)
        return out[..., self.quantile_index(0.5)] if self.quantiles else out[..., 0]
