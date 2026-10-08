"""Các forecaster đơn giản — làm baseline để so sánh với LSTM/foundation model."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Forecaster, make_targets, make_windows, register_forecaster


@register_forecaster("naive")
class NaiveForecaster(Forecaster):
    """Dự báo mọi bước tương lai = giá trị hiện tại (persistence)."""

    needs_fit = False

    def predict_batch(self, windows):
        return np.repeat(windows[:, -1:, :], self.horizon, axis=1)


@register_forecaster("moving_avg")
class MovingAverageForecaster(Forecaster):
    """Dự báo = trung bình `window` bước gần nhất."""

    needs_fit = False

    def __init__(self, window: int = 5, **kw):
        super().__init__(**kw)
        self.window = int(window)
        if self.window > self.context_len:
            raise ValueError("window phải <= context_len")

    def predict_batch(self, windows):
        m = windows[:, -self.window:, :].mean(axis=1, keepdims=True)
        return np.repeat(m, self.horizon, axis=1)


@register_forecaster("linear")
class LinearForecaster(Forecaster):
    """Hồi quy tuyến tính tự hồi quy (AR) có ridge, dự báo trực tiếp H bước:
        y[t+h] = w_h · x[t-L+1..t] + b_h
    Một bộ trọng số dùng chung cho mọi VNF (mỗi VNF là một mẫu) — nghiệm đóng, train vài giây."""

    def __init__(self, ridge: float = 1e-2, **kw):
        super().__init__(**kw)
        self.ridge = float(ridge)
        self.W: Optional[np.ndarray] = None       # (L + 1, H)

    @staticmethod
    def _flat(windows):                            # (N, L, n) -> (N*n, L)
        return windows.transpose(0, 2, 1).reshape(-1, windows.shape[1])

    def fit(self, train, val=None):
        t = np.arange(self.context_len - 1, len(train) - self.horizon)
        X = self._flat(make_windows(train, t, self.context_len)).astype(np.float64)
        Y = self._flat(make_targets(train, t, self.horizon)).astype(np.float64)
        X = np.hstack([X, np.ones((len(X), 1))])
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.W = np.linalg.solve(A, X.T @ Y)
        return self

    def predict_batch(self, windows):
        if self.W is None:
            raise RuntimeError("LinearForecaster chưa fit")
        N, L, n = windows.shape
        X = self._flat(windows)
        Y = np.hstack([X, np.ones((len(X), 1))]) @ self.W          # (N*n, H)
        return Y.reshape(N, n, self.horizon).transpose(0, 2, 1)


@register_forecaster("oracle")
class OracleForecaster(Forecaster):
    """Trả về tương lai THẬT của chuỗi — chỉ là cận trên, không báo cáo như kết quả thật."""

    needs_fit = False

    def forecast_series(self, load, t_idx):
        return make_targets(load, t_idx, self.horizon).astype(np.float32)

    def predict_batch(self, windows):
        raise RuntimeError("Oracle cần biết tương lai — dùng forecast_series/env, không dùng predict")
