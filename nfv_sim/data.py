"""
Đọc dữ liệu SNDZoo (DatasetSNDZoo) thành chuỗi thời gian đã căn chỉnh cho simulator.

Cấu trúc thư mục mong đợi (đúng như bộ dữ liệu gốc):

    DatasetSNDZoo/
        WEB/  IOT/  SEC/
            TrainTest/NoHoles/<metric>.csv     -> tập train (~14 ngày, 1 phút/mẫu)
            18-02/CSVmergedFIXED/<metric>.csv  -> tập test (1 ngày)

Mỗi CSV có cột: <timestamp>, value_3, item_id. Giá trị đã chuẩn hoá min-max về [0, 1]
(riêng CPU là tỉ lệ sử dụng so với quota 1 core, có thể > 1 một chút).

Ngoài SNDZoo, `load_catalog` đọc các dataset bổ sung (NAB, Google, Alibaba, Azure) khai báo trong
config.DATASETS thành danh sách `TimeSeries` để train forecaster trên nhiều loại tải.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import DATASETS, MAX_GAP_MINUTES, METRIC_FILES, SPLIT_DIRS, TRAIN_FRAC, VNF_NAMES


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


# số feature lịch: sin/cos giờ trong ngày, sin/cos thứ trong tuần, cờ "có giờ thật"
N_CALENDAR = 5


@dataclass
class SeriesContext:
    """Thông tin mô tả các cột của một mảng tải (T, n) — forecaster dùng làm "thông tin phụ".

    names    : tên duy nhất của từng cột, dạng "<dataset>/<chuỗi>" (vd "sndzoo/WEB")
    datasets : dataset của từng cột
    start    : thời điểm của bước 0; None = không biết giờ
    step_minutes : độ dài một bước (phút)
    calendar : False nếu mốc thời gian là giả định -> không được dùng giờ/thứ làm feature
    """

    names: List[str]
    datasets: List[str]
    start: Optional[pd.Timestamp] = None
    step_minutes: int = 1
    calendar: bool = True

    @property
    def n(self) -> int:
        return len(self.names)

    def calendar_features(self, pos) -> np.ndarray:
        """Feature lịch tại các bước `pos` (mảng số nguyên bất kỳ shape; pos >= len chuỗi = tương lai, vẫn tính
        được vì giờ của tương lai đã biết trước) -> shape pos.shape + (N_CALENDAR,).

        Giờ mã hoá bằng sin/cos để 23:59 và 0:00 gần nhau. Ví dụ 6:00 -> góc 90° -> (sin, cos) = (1, 0);
        18:00 -> (-1, 0). Không có giờ thật (calendar=False) -> toàn 0, kể cả cờ."""
        pos = np.asarray(pos, np.int64)
        out = np.zeros(pos.shape + (N_CALENDAR,), np.float32)
        if not self.calendar or self.start is None:
            return out
        minutes = self.start.hour * 60 + self.start.minute + pos * self.step_minutes   # phút kể từ 0h ngày đầu
        day = 2 * np.pi * (minutes % 1440) / 1440.0
        week = 2 * np.pi * ((self.start.dayofweek + minutes // 1440) % 7) / 7.0
        out[..., 0], out[..., 1] = np.sin(day), np.cos(day)
        out[..., 2], out[..., 3] = np.sin(week), np.cos(week)
        out[..., 4] = 1.0
        return out

    def shifted(self, steps: int) -> "SeriesContext":
        """Context của phần chuỗi bắt đầu từ bước `steps` (dùng khi cắt train/val)."""
        start = None if self.start is None else self.start + pd.Timedelta(minutes=steps * self.step_minutes)
        return SeriesContext(list(self.names), list(self.datasets), start, self.step_minutes, self.calendar)


@dataclass
class TimeSeries:
    """Một mảng tải (T, n) kèm context — đơn vị dữ liệu forecaster nhận khi train/đánh giá."""

    values: np.ndarray
    ctx: SeriesContext

    def __len__(self) -> int:
        return len(self.values)

    def split(self, cut: int):
        """Cắt theo thời gian tại bước `cut`: (đầu, cuối)."""
        return (TimeSeries(self.values[:cut], self.ctx),
                TimeSeries(self.values[cut:], self.ctx.shifted(cut)))


def as_timeseries(load: np.ndarray, ctx: Optional[SeriesContext] = None) -> TimeSeries:
    """Mảng (T, n) không kèm thông tin -> TimeSeries với tên "col<j>", không có giờ."""
    load = np.asarray(load, np.float32)
    if load.ndim == 1:
        load = load[:, None]
    if ctx is None:
        n = load.shape[1]
        ctx = SeriesContext([f"col{j}" for j in range(n)], ["unknown"] * n, None, 1, False)
    return TimeSeries(load, ctx)


@dataclass
class TraceSet:
    """Chuỗi thời gian đã căn chỉnh theo phút cho nhiều VNF.

    values[metric] có shape (T, n_vnf). index là timestamp chung.
    dataset/calendar: tên dataset (key của config.DATASETS) và giờ có phải giờ thật không.
    """

    vnfs: List[str]
    index: pd.DatetimeIndex
    values: Dict[str, np.ndarray] = field(default_factory=dict)
    dataset: str = "sndzoo"
    calendar: bool = True

    def __len__(self) -> int:
        return len(self.index)

    @property
    def step_minutes(self) -> int:
        if len(self.index) < 2:
            return 1
        return max(1, int(round((self.index[1] - self.index[0]).total_seconds() / 60)))

    def slice(self, sl: slice) -> "TraceSet":
        return TraceSet(self.vnfs, self.index[sl], {m: a[sl] for m, a in self.values.items()},
                        self.dataset, self.calendar)

    def subsample(self, k: int) -> "TraceSet":
        """Lấy 1 mẫu mỗi k bước (dữ liệu gốc k phút đã bị nội suy lên lưới 1 phút -> lấy lại đúng mẫu thật)."""
        return self if k <= 1 else self.slice(slice(None, None, k))

    def context(self) -> SeriesContext:
        return SeriesContext([f"{self.dataset}/{v}" for v in self.vnfs], [self.dataset] * len(self.vnfs),
                             self.index[0] if len(self.index) else None, self.step_minutes, self.calendar)

    def series(self, metric: str = "rx") -> TimeSeries:
        return TimeSeries(self.values[metric], self.context())

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
        return TraceSet(self.vnfs, self.index[: n * step_minutes : step_minutes], out, self.dataset, self.calendar)


def load_sndzoo(
    root: str,
    split: str = "train",
    vnfs: Sequence[str] = VNF_NAMES,
    metrics: Sequence[str] = ("cpu", "mem", "rx", "tx"),
    max_gap_minutes: int = MAX_GAP_MINUTES,
    train_frac: Optional[float] = TRAIN_FRAC,
) -> TraceSet:
    """Đọc tập train hoặc test theo cách chia trong config (TRAIN_FRAC).

    train_frac None: train = thư mục TrainTest, test = thư mục 18-02 (cách chia gốc của SNDZoo).
    train_frac trong (0, 1): đọc thư mục TrainTest rồi chia theo thời gian (không xáo trộn),
    train = train_frac đầu, test = phần còn lại.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split phải là 'train' hoặc 'test', nhận '{split}'")
    if train_frac is None:
        return _load_folder(root, split, vnfs, metrics, max_gap_minutes)
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"TRAIN_FRAC phải trong (0, 1) hoặc None, nhận {train_frac}")
    return _time_split(_load_folder(root, "train", vnfs, metrics, max_gap_minutes), split, train_frac)


def _time_split(full: TraceSet, split: str, train_frac: float) -> TraceSet:
    """train = train_frac đầu chuỗi, test = phần còn lại (không xáo trộn)."""
    cut = int(len(full) * train_frac)
    return full.slice(slice(0, cut) if split == "train" else slice(cut, None))


def load_dataset(name: str, split: str = "train", metrics: Sequence[str] = ("rx",),
                 train_frac: Optional[float] = TRAIN_FRAC) -> List[TraceSet]:
    """Đọc một dataset trong config.DATASETS -> danh sách TraceSet (đã về đúng bước gốc của dữ liệu).

    joint=True: một TraceSet chứa mọi chuỗi (cắt về khoảng thời gian chung, như env đọc SNDZoo).
    joint=False: mỗi chuỗi một TraceSet (các chuỗi NAB, Google... nằm ở các khoảng thời gian khác nhau).
    """
    if name not in DATASETS:
        raise KeyError(f"Không có dataset '{name}' trong config.DATASETS. Có: {sorted(DATASETS)}")
    cfg = DATASETS[name]
    groups = [list(cfg["series"])] if cfg["joint"] else [[v] for v in cfg["series"]]
    out = []
    for vnfs in groups:
        if name == "sndzoo":
            ts = load_sndzoo(cfg["root"], split, vnfs, metrics, train_frac=train_frac)
        else:
            if train_frac is None:
                raise ValueError(f"Dataset '{name}' chỉ chia được theo TRAIN_FRAC (không có thư mục test riêng)")
            # cho phép nội suy lỗ dài tới 3 mẫu gốc; lấy lại mẫu thật TRƯỚC khi cắt train/test
            full = _load_folder(cfg["root"], "train", vnfs, metrics, max(MAX_GAP_MINUTES, 3 * cfg["step"]))
            ts = _time_split(full.subsample(cfg["step"]), split, train_frac)
        ts.dataset, ts.calendar = name, cfg["calendar"]
        out.append(ts)
    return out


def load_catalog(names: Sequence[str], split: str = "train", metric: str = "rx",
                 train_frac: Optional[float] = TRAIN_FRAC) -> List[TimeSeries]:
    """Đọc nhiều dataset -> danh sách TimeSeries (tải `metric` + context) để train/đánh giá forecaster."""
    return [ts.series(metric) for name in names for ts in load_dataset(name, split, (metric,), train_frac)]


def _load_folder(
    root: str,
    split: str,
    vnfs: Sequence[str],
    metrics: Sequence[str],
    max_gap_minutes: int,
) -> TraceSet:
    """Đọc và căn chỉnh dữ liệu của một thư mục (SPLIT_DIRS[split]).

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
