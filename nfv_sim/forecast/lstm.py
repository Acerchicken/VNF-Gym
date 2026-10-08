"""Forecaster LSTM (PyTorch).

Thiết kế channel-independent: một mạng dùng chung cho mọi VNF, mỗi VNF là một chuỗi đơn biến
(tải đã chuẩn hoá nên cùng thang). Lợi ích: gấp n_vnf lần dữ liệu train, chạy với số VNF bất kỳ.
Mạng dự báo trực tiếp H bước, dạng phần dư so với giá trị hiện tại: ŷ[t+h] = x[t] + Δ_h.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
from torch import nn

from .base import Forecaster, make_targets, make_windows, register_forecaster


class _LSTMNet(nn.Module):
    def __init__(self, horizon: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):                      # x: (B, L) -> (B, H)
        out, _ = self.lstm(x.unsqueeze(-1))
        return x[:, -1:] + self.head(out[:, -1])


@register_forecaster("lstm")
class LSTMForecaster(Forecaster):

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1,
                 epochs: int = 30, batch_size: int = 256, lr: float = 1e-3, patience: int = 5,
                 under_weight: float = 1.0, seed: int = 0, verbose: bool = True, **kw):
        super().__init__(**kw)
        self.hidden_size, self.num_layers, self.dropout = hidden_size, num_layers, dropout
        self.epochs, self.batch_size, self.lr, self.patience = epochs, batch_size, lr, patience
        self.under_weight, self.seed, self.verbose = under_weight, seed, verbose
        self.net: Optional[_LSTMNet] = None
        self.history: list = []                 # (epoch, train_loss, val_loss)

    # ---- dữ liệu: (T, n) -> X (N*n, L), Y (N*n, H) ----
    def _xy(self, load):
        t = np.arange(self.context_len - 1, len(load) - self.horizon)
        X = make_windows(load, t, self.context_len).transpose(0, 2, 1).reshape(-1, self.context_len)
        Y = make_targets(load, t, self.horizon).transpose(0, 2, 1).reshape(-1, self.horizon)
        return torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)

    def _loss(self, pred, y):
        w = torch.where(pred < y, self.under_weight, 1.0)    # dự báo thiếu -> phạt nặng hơn
        return (w * (pred - y) ** 2).mean()

    def fit(self, train, val=None):
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        self.net = _LSTMNet(self.horizon, self.hidden_size, self.num_layers, self.dropout)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        X, Y = self._xy(train)
        Xv, Yv = self._xy(val) if val is not None and len(val) > self.context_len + self.horizon else (None, None)

        best, best_state, bad = float("inf"), None, 0
        self.history = []
        for ep in range(1, self.epochs + 1):
            self.net.train()
            perm = torch.randperm(len(X), generator=g)
            tot = 0.0
            for i in range(0, len(X), self.batch_size):
                b = perm[i:i + self.batch_size]
                loss = self._loss(self.net(X[b]), Y[b])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(b)
            tr = tot / len(X)
            va = self._eval_loss(Xv, Yv) if Xv is not None else tr
            self.history.append((ep, tr, va))
            if self.verbose:
                print(f"  [lstm] epoch {ep:3d}  train={tr:.5f}  val={va:.5f}")
            if va < best - 1e-7:
                best, best_state, bad = va, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(f"  [lstm] early stopping (best val={best:.5f})")
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    @torch.no_grad()
    def _eval_loss(self, X, Y):
        self.net.eval()
        tot = 0.0
        for i in range(0, len(X), 4096):
            tot += self._loss(self.net(X[i:i + 4096]), Y[i:i + 4096]).item() * len(X[i:i + 4096])
        return tot / len(X)

    @torch.no_grad()
    def predict_batch(self, windows):
        if self.net is None:
            raise RuntimeError("LSTMForecaster chưa fit")
        self.net.eval()
        N, L, n = windows.shape
        x = torch.as_tensor(np.ascontiguousarray(windows.transpose(0, 2, 1)).reshape(-1, L), dtype=torch.float32)
        y = torch.cat([self.net(x[i:i + 4096]) for i in range(0, len(x), 4096)]).numpy()
        return y.reshape(N, n, self.horizon).transpose(0, 2, 1)
