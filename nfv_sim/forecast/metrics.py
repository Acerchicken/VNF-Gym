"""Đánh giá forecaster trên một chuỗi tải (chỉ số chung cho mọi thuật toán)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Forecaster, make_targets


def evaluate_forecaster(fc: Forecaster, load: np.ndarray, horizon: Optional[int] = None,
                        vnfs: Optional[list] = None) -> dict:
    """Dự báo tại mọi bước t (chỉ dùng dữ liệu <= t) và so với giá trị thật t+1..t+H.

    Trả về dict:
        mae, rmse         : sai số trung bình trên mọi bước dự báo và mọi VNF
        mae_h<k>          : MAE riêng cho bước thứ k (độ chính xác giảm dần theo k)
        under_rate        : tỉ lệ dự báo THẤP hơn thực tế > 5% -> nguy cơ cấp thiếu, vi phạm SLA
        peak_mae          : MAE ở 10% thời điểm tải cao nhất (thời điểm quan trọng cho SLA)
        mae_<VNF>         : MAE theo từng VNF
    """
    H = horizon or fc.horizon
    t = np.arange(fc.context_len - 1, len(load) - H)
    pred = fc.forecast_series(load, t)[:, :H]           # (N, H, n)
    true = make_targets(load, t, H)
    err = pred - true
    out = dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())))
    for k in range(H):
        out[f"mae_h{k + 1}"] = float(np.abs(err[:, k]).mean())
    out["under_rate"] = float((err < -0.05).mean())
    thr = np.quantile(true, 0.9)
    out["peak_mae"] = float(np.abs(err[true >= thr]).mean())
    for j, v in enumerate(vnfs or range(load.shape[1])):
        out[f"mae_{v}"] = float(np.abs(err[..., j]).mean())
    return out
