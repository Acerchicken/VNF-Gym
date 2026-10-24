"""Forecaster LSTM (PyTorch).

Thiết kế channel-independent: một mạng dùng chung cho mọi VNF, mỗi VNF là một chuỗi đơn biến
(tải đã chuẩn hoá nên cùng thang). Lợi ích: gấp n_vnf lần dữ liệu train, chạy với số VNF bất kỳ.
Mạng dự báo trực tiếp H bước, dạng phần dư so với giá trị hiện tại: ŷ[t+h] = x[t] + Δ_h.
Dự báo khoảng (quantiles): mỗi bước h ra Q giá trị (P10/P50/P90), cùng dạng phần dư.
Vòng train, loss và dự báo nằm ở neural.py (dùng chung với TFT).
"""
from __future__ import annotations

from torch import nn

from .base import register_forecaster
from .neural import NeuralForecaster


class _LSTMNet(nn.Module):
    def __init__(self, horizon: int, n_out: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.horizon, self.n_out = horizon, n_out
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, horizon * n_out)

    def forward(self, x):                      # x: (B, L) -> (B, H, Q)
        out, _ = self.lstm(x.unsqueeze(-1))
        d = self.head(out[:, -1]).view(-1, self.horizon, self.n_out)
        return x[:, -1:, None] + d


@register_forecaster("lstm")
class LSTMForecaster(NeuralForecaster):

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1, **kw):
        super().__init__(**kw)
        self.hidden_size, self.num_layers, self.dropout = hidden_size, num_layers, dropout

    def _build_net(self):
        return _LSTMNet(self.horizon, self.n_out, self.hidden_size, self.num_layers, self.dropout)
