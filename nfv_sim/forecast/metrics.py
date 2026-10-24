"""Đánh giá forecaster trên một chuỗi tải (chỉ số chung cho mọi thuật toán)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional, Sequence

import numpy as np

from ..data import TimeSeries, as_timeseries
from .base import Forecaster, make_targets


def _predict(fc: Forecaster, ts: TimeSeries, H: int):
    """Dự báo tại mọi bước t (chỉ dùng dữ liệu <= t) -> (pred, true, dist).
    pred, true: (N, H, n); dist: (N, H, n, Q) theo fc.quantiles, hoặc None nếu model dự báo một giá trị."""
    t = np.arange(fc.context_len - 1, len(ts) - H)
    true = make_targets(ts.values, t, H)
    if fc.quantiles:
        dist = fc.forecast_series(ts.values, t, quantiles=fc.quantiles)[:, :H]
        return dist[..., fc.quantile_index(0.5)], true, dist
    return fc.forecast_series(ts.values, t)[:, :H], true, None


def _quantile_metrics(dist: np.ndarray, true: np.ndarray, quantiles) -> dict:
    """pinball : loss quantile trung bình (thấp hơn = tốt hơn, so được giữa các model dự báo khoảng)
    cov_p<q>   : tỉ lệ giá trị thật <= dự báo P<q> (lý tưởng = q, vd cov_p90 ~ 0.90)
    width      : độ rộng khoảng trung bình giữa quantile thấp nhất và cao nhất (P90 - P10)"""
    q = np.asarray(quantiles, np.float32)
    e = true[..., None] - dist
    out = dict(pinball=float(np.maximum(q * e, (q - 1) * e).mean()))
    for i, x in enumerate(quantiles):
        out[f"cov_p{int(round(x * 100))}"] = float((true <= dist[..., i]).mean())
    out["width"] = float((dist[..., -1] - dist[..., 0]).mean())
    return out


def _summary(pred: np.ndarray, true: np.ndarray, dist: Optional[np.ndarray] = None, quantiles=()) -> dict:
    err = pred - true
    H = err.shape[1]
    out = dict(mae=float(np.abs(err).mean()), rmse=float(np.sqrt((err ** 2).mean())))
    for k in range(H):
        out[f"mae_h{k + 1}"] = float(np.abs(err[:, k]).mean())
    out["under_rate"] = float((err < -0.05).mean())
    thr = np.quantile(true, 0.9)
    out["peak_mae"] = float(np.abs(err[true >= thr]).mean())
    if dist is not None:
        out.update(_quantile_metrics(dist, true, quantiles))
    return out


def evaluate_forecaster(fc: Forecaster, load, horizon: Optional[int] = None,
                        vnfs: Optional[list] = None) -> dict:
    """Dự báo tại mọi bước t (chỉ dùng dữ liệu <= t) và so với giá trị thật t+1..t+H.

    load: mảng (T, n) hoặc TimeSeries.
    Trả về dict:
        mae, rmse         : sai số trung bình trên mọi bước dự báo và mọi VNF
        mae_h<k>          : MAE riêng cho bước thứ k (độ chính xác giảm dần theo k)
        under_rate        : tỉ lệ dự báo THẤP hơn thực tế > 5% -> nguy cơ cấp thiếu, vi phạm SLA
        peak_mae          : MAE ở 10% thời điểm tải cao nhất (thời điểm quan trọng cho SLA)
        mae_<VNF>         : MAE theo từng VNF
        pinball, cov_p<q>, width : chỉ với model dự báo khoảng (xem _quantile_metrics); MAE tính trên P50
    """
    ts = load if isinstance(load, TimeSeries) else as_timeseries(load)
    H = horizon or fc.horizon
    pred, true, dist = _predict(fc, ts, H)
    out = _summary(pred, true, dist, fc.quantiles)
    for j, v in enumerate(vnfs or range(ts.values.shape[1])):
        out[f"mae_{v}"] = float(np.abs(pred[..., j] - true[..., j]).mean())
    return out


def evaluate_by_dataset(fc: Forecaster, series: Sequence[TimeSeries],
                        horizon: Optional[int] = None) -> Dict[str, dict]:
    """Đánh giá trên phần test của nhiều chuỗi, gộp kết quả theo dataset -> {dataset: dict chỉ số}.
    Chuỗi ngắn hơn context_len + horizon bị bỏ qua."""
    H = horizon or fc.horizon
    by: Dict[str, list] = defaultdict(list)
    for ts in series:
        if len(ts) <= fc.context_len + H:
            continue
        pred, true, dist = _predict(fc, ts, H)
        for j, ds in enumerate(ts.ctx.datasets):     # tách từng cột rồi gộp theo dataset
            by[ds].append((pred[..., j:j + 1], true[..., j:j + 1], None if dist is None else dist[:, :, j:j + 1]))
    out = {}
    for ds, v in by.items():
        dist = None if v[0][2] is None else np.concatenate([d for _, _, d in v])
        out[ds] = _summary(np.concatenate([p for p, _, _ in v]), np.concatenate([t for _, t, _ in v]),
                           dist, fc.quantiles)
    return out
