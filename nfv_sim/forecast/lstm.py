"""Forecaster LSTM (PyTorch).

Thiết kế channel-independent: một mạng dùng chung cho mọi VNF, mỗi VNF là một chuỗi đơn biến
(tải đã chuẩn hoá nên cùng thang). Lợi ích: gấp n_vnf lần dữ liệu train, chạy với số VNF bất kỳ.
Mạng dự báo trực tiếp H bước, dạng phần dư so với giá trị hiện tại: ŷ[t+h] = x[t] + Δ_h.
Dự báo khoảng (quantiles): mỗi bước h ra Q giá trị (P10/P50/P90), cùng dạng phần dư.

use_context=True (Bước 3): mỗi bước đầu vào thêm feature lịch + embedding chuỗi/dataset;
feature lịch của H bước tương lai được ghép vào trạng thái cuối trước lớp ra.
Vòng train, loss và dự báo nằm ở neural.py (dùng chung với TFT).
"""
from __future__ import annotations

import torch
from torch import nn

from ..data import N_CALENDAR
from .base import register_forecaster
from .neural import NeuralForecaster


class _LSTMNet(nn.Module):
    def __init__(self, horizon: int, n_out: int, hidden_size: int, num_layers: int, dropout: float,
                 use_context: bool = False, n_series: int = 1, n_datasets: int = 1, emb_dim: int = 8):
        super().__init__()
        self.horizon, self.n_out, self.use_context = horizon, n_out, use_context
        in_size, head_in = 1, hidden_size
        if use_context:
            self.series_emb = nn.Embedding(n_series, emb_dim)
            self.dataset_emb = nn.Embedding(n_datasets, emb_dim)
            in_size += N_CALENDAR + 2 * emb_dim
            head_in += horizon * N_CALENDAR
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(head_in, horizon * n_out)

    def forward(self, x, cal, sid, did):       # x: (B, L) -> (B, H, Q)
        B, L = x.shape
        inp = x.unsqueeze(-1)
        if self.use_context:
            static = torch.cat([self.series_emb(sid), self.dataset_emb(did)], -1)      # (B, 2*emb)
            inp = torch.cat([inp, cal[:, :L], static.unsqueeze(1).expand(B, L, -1)], -1)
        out, _ = self.lstm(inp)
        h = out[:, -1]
        if self.use_context:
            h = torch.cat([h, cal[:, L:].flatten(1)], -1)        # giờ của H bước tương lai
        d = self.head(h).view(B, self.horizon, self.n_out)
        return x[:, -1:, None] + d


@register_forecaster("lstm")
class LSTMForecaster(NeuralForecaster):

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1,
                 use_context: bool = True, emb_dim: int = 8, **kw):
        super().__init__(**kw)
        self.hidden_size, self.num_layers, self.dropout = hidden_size, num_layers, dropout
        self.use_context, self.emb_dim = use_context, emb_dim

    def _build_net(self):
        return _LSTMNet(self.horizon, self.n_out, self.hidden_size, self.num_layers, self.dropout,
                        self.use_context, self.n_series, self.n_datasets, self.emb_dim)
