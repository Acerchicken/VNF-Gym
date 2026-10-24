"""Phần dùng chung cho các forecaster mạng nơ-ron (LSTM, TFT): dữ liệu, vòng train, loss, dự báo.

Lớp con chỉ cần cài `_build_net()` trả về một nn.Module với
    forward(x) -> (B, H, Q)
    x: (B, L) lịch sử tải; Q = số quantile (hoặc 1 khi dự báo một giá trị).

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
                 seed: int = 0, verbose: bool = True, **kw):
        super().__init__(**kw)
        self.quantiles = tuple(float(q) for q in quantiles) if quantiles else ()
        if self.quantiles and 0.5 not in self.quantiles:
            raise ValueError("quantiles phải có 0.5 (P50 dùng làm dự báo chính)")
        self.epochs, self.batch_size, self.lr, self.patience = epochs, batch_size, lr, patience
        self.under_weight, self.stride, self.seed, self.verbose = under_weight, stride, seed, verbose
        self.net: Optional[nn.Module] = None
        self.history: list = []                 # (epoch, train_loss, val_loss)

    @property
    def n_out(self) -> int:
        return len(self.quantiles) or 1

    # ---- lớp con cài đặt ----
    def _build_net(self) -> nn.Module:
        raise NotImplementedError

    # ---- dữ liệu ----
    def _tensors(self, series) -> dict:
        """list[TimeSeries] -> dict tensor x (M, L), y (M, H)."""
        d = make_training_set(series, self.context_len, self.horizon, self.stride)
        return dict(x=torch.as_tensor(d["x"]), y=torch.as_tensor(d["y"]))

    @staticmethod
    def _inputs(D: dict, idx) -> tuple:
        """Các tensor đưa vào net.forward cho các mẫu idx."""
        return (D["x"][idx],)

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
                loss = self._loss(self.net(*self._inputs(D, b)), D["y"][b])
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
        """Chuẩn bị trước khi tạo net (lớp con dùng để lập bảng ID chuỗi...)."""

    @torch.no_grad()
    def _eval_loss(self, D):
        self.net.eval()
        n, tot = len(D["y"]), 0.0
        for i in range(0, n, 4096):
            idx = slice(i, i + 4096)
            tot += self._loss(self.net(*self._inputs(D, idx)), D["y"][idx]).item() * len(D["y"][idx])
        return tot / n

    # ---- dự báo ----
    def _windows_to_tensors(self, windows) -> dict:
        """(N, L, n) -> dict tensor, mỗi (thời điểm, cột) là một mẫu, thứ tự i*n + j."""
        N, L, n = windows.shape
        x = np.ascontiguousarray(windows.transpose(0, 2, 1)).reshape(-1, L)
        return dict(x=torch.as_tensor(x, dtype=torch.float32))

    @torch.no_grad()
    def _predict_raw(self, windows) -> np.ndarray:
        """(N, L, n) -> (N, H, n, Q) đầu ra của net (quantile đã sắp tăng dần)."""
        if self.net is None:
            raise RuntimeError(f"{self.name} chưa fit")
        self.net.eval()
        N, L, n = windows.shape
        D = self._windows_to_tensors(windows)
        M = N * n
        y = torch.cat([self.net(*self._inputs(D, slice(i, i + 4096))) for i in range(0, M, 4096)]).numpy()
        y = np.sort(y, axis=-1)          # tránh P10 > P50 > P90 bị "chéo nhau"
        return y.reshape(N, n, self.horizon, self.n_out).transpose(0, 2, 1, 3)

    def predict_dist_batch(self, windows):
        if not self.quantiles:
            return super().predict_dist_batch(windows)
        return self._predict_raw(windows)

    def predict_batch(self, windows):
        out = self._predict_raw(windows)
        return out[..., self.quantile_index(0.5)] if self.quantiles else out[..., 0]
