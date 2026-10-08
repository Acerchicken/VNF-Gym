"""
Đọc dữ liệu SNDZoo (DatasetSNDZoo) thành chuỗi thời gian đã căn chỉnh cho simulator.

Cấu trúc thư mục mong đợi (đúng như bộ dữ liệu gốc):

    DatasetSNDZoo/
        WEB/  IOT/  SEC/
            TrainTest/NoHoles/<metric>.csv     -> tập train (~14 ngày, 1 phút/mẫu)
            18-02/CSVmergedFIXED/<metric>.csv  -> tập test (1 ngày)

Mỗi CSV có cột: <timestamp>, value_3, item_id. Giá trị đã chuẩn hoá min-max về [0, 1]
(riêng CPU là tỉ lệ sử dụng so với quota 1 core, có thể > 1 một chút).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import MAX_GAP_MINUTES, METRIC_FILES, SPLIT_DIRS, VNF_NAMES


def _read_series(path: str) -> pd.Series:
    df = pd.read_csv(path)
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col])
    s = df.set_index(ts_col)["value_3"].astype(float)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


def _find_split_dir(root: str, vnf: str, split: str) -> str:
    for sub in SPLIT_DIRS[split]:
        d = os.path.join(root, vnf, sub)
        if os.path.isfile(os.path.join(d, METRIC_FILES["rx"])):
            return d
    raise FileNotFoundError(
        f"Không tìm thấy dữ liệu {split} cho {vnf} trong {root}. "
        f"Đã thử: {[os.path.join(root, vnf, s) for s in SPLIT_DIRS[split]]}"
    )


@dataclass
class TraceSet:
    """Chuỗi thời gian đã căn chỉnh theo phút cho nhiều VNF.

    values[metric] có shape (T, n_vnf). index là timestamp chung.
    """

    vnfs: List[str]
    index: pd.DatetimeIndex
    values: Dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.index)

    @property
    def minute_of_day(self) -> np.ndarray:
        return (self.index.hour * 60 + self.index.minute).to_numpy()

    def aggregate(self, step_minutes: int) -> "TraceSet":
        """Gộp k phút thành 1 bước quyết định (lấy trung bình cho mem, max cho tải
        để không 'giấu' đỉnh tải bên trong một bước)."""
        if step_minutes <= 1:
            return self
        n = len(self) // step_minutes
        out = {}
        for m, arr in self.values.items():
            a = arr[: n * step_minutes].reshape(n, step_minutes, -1)
            out[m] = a.mean(axis=1) if m == "mem" else a.max(axis=1)
        return TraceSet(self.vnfs, self.index[: n * step_minutes : step_minutes], out)


def load_sndzoo(
    root: str,
    split: str = "train",
    vnfs: Sequence[str] = VNF_NAMES,
    metrics: Sequence[str] = ("cpu", "mem", "rx", "tx"),
    max_gap_minutes: int = MAX_GAP_MINUTES,
) -> TraceSet:
    """Đọc và căn chỉnh dữ liệu.

    root: đường dẫn tới thư mục DatasetSNDZoo
    split: "train" hoặc "test"

    Các VNF được đo trong những thí nghiệm song song nên timestamp gần trùng nhau;
    ta lấy khoảng giao nhau, đưa về lưới 1 phút, nội suy các lỗ ngắn (<= max_gap_minutes)
    và điền 0 cho phần còn lại (thời điểm không có tải).
    """
    vnfs = [v.upper() for v in vnfs]
    raw: Dict[str, Dict[str, pd.Series]] = {m: {} for m in metrics}
    for v in vnfs:
        d = _find_split_dir(root, v, split)
        for m in metrics:
            raw[m][v] = _read_series(os.path.join(d, METRIC_FILES[m]))

    # khoảng thời gian chung
    starts = [s.index.min() for m in metrics for s in raw[m].values()]
    ends = [s.index.max() for m in metrics for s in raw[m].values()]
    start, end = max(starts).floor("min"), min(ends).floor("min")
    if end <= start:
        raise ValueError("Các chuỗi không có khoảng thời gian chung.")
    index = pd.date_range(start, end, freq="1min")

    values = {}
    for m in metrics:
        cols = []
        for v in vnfs:
            s = raw[m][v]
            s.index = s.index.floor("min")
            s = s[~s.index.duplicated(keep="first")]
            s = s.reindex(index).interpolate(limit=max_gap_minutes, limit_area="inside").fillna(0.0)
            cols.append(np.clip(s.to_numpy(), 0.0, None))
        values[m] = np.stack(cols, axis=1).astype(np.float32)
    return TraceSet(vnfs, index, values)


def augment_load(
    load: np.ndarray,
    rng: np.random.Generator,
    load_multiplier: float = 1.0,
    noise_std: float = 0.0,
    burst_prob: float = 0.0,
    burst_scale: tuple = (1.5, 3.0),
    burst_len: tuple = (3, 20),
) -> np.ndarray:
    """Tạo kịch bản quá tải từ trace thật: nhân hệ số tải, thêm nhiễu và các đợt burst.

    load: (T, n_vnf) tải đã chuẩn hoá. Trả về bản sao đã biến đổi.
    """
    out = load.astype(np.float32) * load_multiplier
    T, n = out.shape
    if noise_std > 0:
        out = out * (1.0 + rng.normal(0.0, noise_std, size=out.shape)).astype(np.float32)
    if burst_prob > 0:
        for j in range(n):
            t = 0
            while t < T:
                if rng.random() < burst_prob:
                    L = int(rng.integers(burst_len[0], burst_len[1] + 1))
                    k = float(rng.uniform(*burst_scale))
                    # burst dạng hình thang: tăng nhanh, giữ, giảm
                    ramp = np.ones(min(L, T - t), dtype=np.float32) * k
                    r = max(1, L // 4)
                    ramp[:r] = np.linspace(1.0, k, r, dtype=np.float32)[: len(ramp[:r])]
                    out[t : t + len(ramp), j] *= ramp
                    t += L
                else:
                    t += 1
    return np.clip(out, 0.0, None)
