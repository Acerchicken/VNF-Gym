"""
Khung chung cho mọi thuật toán dự báo tải (Bước 4).

Thêm một thuật toán mới (Lag-Llama, TimesFM, GRU, Prophet...) chỉ cần:

    from nfv_sim.forecast import Forecaster, register_forecaster

    @register_forecaster("my_model")
    class MyForecaster(Forecaster):
        def __init__(self, my_param=1, **kw):
            super().__init__(**kw)              # nhận horizon, context_len
            self.my_param = my_param

        def fit(self, train, val=None):         # tuỳ chọn — model zero-shot thì bỏ qua
            series = as_series_list(train)      # list[TimeSeries], mỗi cái values (T, n)
            ...
            return self

        def predict_batch(self, windows, extra=None):   # (N, L, n_vnf) -> (N, H, n_vnf)
            ...

rồi thêm `"my_model": dict(my_param=...)` vào FORECASTERS trong config.py.

`extra` (Bước 3) là thông tin phụ của các cửa sổ, None nếu không có (model đơn giản bỏ qua):
    extra["cal"]      : (N, L + H, N_CALENDAR) feature lịch của L bước quá khứ + H bước tương lai
    extra["names"]    : tên n cột (vd "sndzoo/WEB"), extra["datasets"]: dataset của n cột
Tạo bằng `make_extra(ctx, t_idx, L, H)`; forecast_series/env/metrics tự truyền khi biết SeriesContext.

Model dự báo khoảng (Bước 2) đặt `self.quantiles = (0.1, 0.5, 0.9)` và cài `predict_dist_batch`
((N, L, n) -> (N, H, n, Q)); `predict_batch` khi đó trả về trung vị P50. Ví dụ: nfv_sim/forecast/neural.py. Sau đó mọi
script (train_forecaster, run_baselines, train_ppo) dùng được qua `--forecaster my_model`.

Quy ước dữ liệu: tải đã chuẩn hoá (giống `env._load`), mảng (T, n_vnf), float32.
`fit` nhận một mảng (T, n), một `TimeSeries` hoặc danh sách `TimeSeries` (nhiều dataset, xem data.load_catalog);
dùng `as_series_list` để đưa về danh sách.

Một key trong FORECASTERS có thể là biến thể của model khác (`model="lstm"`, vd "lstm_snd"):
make_forecaster tạo đúng class đó, đặt tên instance = key để lưu thành <key>.pkl.
"""
from __future__ import annotations

import os
import pickle
from typing import Callable, Dict, List, Optional, Sequence, Type

import numpy as np

from ..config import FORECAST, FORECASTERS
from ..data import SeriesContext, TimeSeries, as_timeseries

# ----------------------------------------------------------------------------- registry
_REGISTRY: Dict[str, Type["Forecaster"]] = {}


def register_forecaster(name: str) -> Callable:
    """Decorator đăng ký một class Forecaster dưới tên `name`."""

    def deco(cls):
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"Forecaster '{name}' đã được đăng ký")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _model_of(name: str) -> str:
    """Tên class đã đăng ký mà key `name` dùng (biến thể trong FORECASTERS có model=...)."""
    return FORECASTERS.get(name, {}).get("model", name)


def available_forecasters() -> list:
    """Tên đã đăng ký + các biến thể trong FORECASTERS (bỏ biến thể của model chưa cài, vd thiếu PyTorch)."""
    return sorted(set(_REGISTRY) | {k for k in FORECASTERS if _model_of(k) in _REGISTRY})


def make_forecaster(name: str, **overrides) -> "Forecaster":
    """Tạo forecaster theo tên, tham số = FORECAST (chung) + FORECASTERS[name] + overrides."""
    model = _model_of(name)
    if model not in _REGISTRY:
        raise KeyError(f"Không có forecaster '{name}'. Có: {available_forecasters()}")
    kw = dict(horizon=FORECAST["horizon"], context_len=FORECAST["context_len"])
    kw.update({k: v for k, v in FORECASTERS.get(name, {}).items() if k != "model"})
    kw.update(overrides)
    fc = _REGISTRY[model](**kw)
    fc.name = name                       # biến thể lưu/nạp theo tên riêng
    return fc


def model_path(name: str, models_dir: Optional[str] = None) -> str:
    return os.path.join(models_dir or FORECAST["models_dir"], f"{name}.pkl")


def load_forecaster(name_or_path: str, models_dir: Optional[str] = None) -> "Forecaster":
    """Nạp forecaster đã lưu. Nhận tên (tìm <models_dir>/<tên>.pkl) hoặc đường dẫn file.
    Model không cần train (naive, oracle...) mà chưa có file thì tạo mới từ config."""
    path = name_or_path if os.path.isfile(name_or_path) else model_path(name_or_path, models_dir)
    if os.path.isfile(path):
        with open(path, "rb") as f:   # chỉ nạp file do chính dự án tạo ra (pickle)
            return pickle.load(f)
    if _model_of(name_or_path) in _REGISTRY and not _REGISTRY[_model_of(name_or_path)].needs_fit:
        return make_forecaster(name_or_path)
    raise FileNotFoundError(f"Chưa có model '{name_or_path}' ({path}). "
                            f"Chạy: python examples\\train_forecaster.py --models {name_or_path}")


# ----------------------------------------------------------------------------- tiện ích
def make_windows(load: np.ndarray, t_idx: np.ndarray, context_len: int) -> np.ndarray:
    """Cửa sổ lịch sử kết thúc tại mỗi t (gồm cả t): (N, L, n_vnf).
    Đầu chuỗi chưa đủ L bước thì lặp lại giá trị đầu tiên."""
    t_idx = np.asarray(t_idx, np.int64)
    idx = t_idx[:, None] + np.arange(-context_len + 1, 1)[None, :]
    return load[np.clip(idx, 0, len(load) - 1)]


def make_targets(load: np.ndarray, t_idx: np.ndarray, horizon: int) -> np.ndarray:
    """Giá trị thật t+1..t+H: (N, H, n_vnf). Cuối chuỗi lặp lại giá trị cuối."""
    t_idx = np.asarray(t_idx, np.int64)
    idx = t_idx[:, None] + np.arange(1, horizon + 1)[None, :]
    return load[np.clip(idx, 0, len(load) - 1)]


def make_extra(ctx: Optional[SeriesContext], t_idx, context_len: int, horizon: int) -> Optional[Dict]:
    """Thông tin phụ cho các cửa sổ kết thúc tại t_idx (xem docstring đầu file). ctx None -> None."""
    if ctx is None:
        return None
    t_idx = np.asarray(t_idx, np.int64)
    pos = t_idx[:, None] + np.arange(-context_len + 1, horizon + 1)[None, :]     # (N, L + H)
    return dict(cal=ctx.calendar_features(pos), names=list(ctx.names), datasets=list(ctx.datasets))


def as_series_list(data) -> List[TimeSeries]:
    """Mảng (T, n) | TimeSeries | list[TimeSeries] | None -> list[TimeSeries]."""
    if data is None:
        return []
    if isinstance(data, TimeSeries):
        return [data]
    if isinstance(data, (list, tuple)):
        return [d if isinstance(d, TimeSeries) else as_timeseries(d) for d in data]
    return [as_timeseries(data)]


def split_train_val(load, val_frac: float = FORECAST["val_frac"]):
    """Phần cuối mỗi chuỗi làm validation. Nhận mảng (T, n) (trả về 2 mảng như trước)
    hoặc TimeSeries / danh sách TimeSeries (trả về 2 danh sách, cắt trong từng chuỗi)."""
    if isinstance(load, np.ndarray):
        n_val = int(len(load) * val_frac)
        if n_val <= 0:
            return load, None
        return load[:-n_val], load[-n_val:]
    tr, va = [], []
    for ts in as_series_list(load):
        n_val = int(len(ts) * val_frac)
        a, b = ts.split(len(ts) - n_val)
        tr.append(a)
        if n_val > 0:
            va.append(b)
    return tr, va


def make_training_set(series: Sequence[TimeSeries], context_len: int, horizon: int, stride: int = 1) -> Dict:
    """Cắt mọi chuỗi (mọi cột) thành cặp (lịch sử L bước, tương lai H bước), mỗi cột là một mẫu đơn biến.

    Trả về dict:
        x    : (M, L)  lịch sử
        y    : (M, H)  giá trị thật t+1..t+H
        cal  : (M, L + H, N_CALENDAR) feature lịch (quá khứ + tương lai)
        name : (M,)    tên chuỗi của từng mẫu (vd "sndzoo/WEB")
        dataset : (M,) tên dataset
    Chuỗi ngắn hơn L + H bị bỏ qua.
    """
    xs, ys, cals, names, dsets = [], [], [], [], []
    for ts in series:
        t = np.arange(context_len - 1, len(ts) - horizon, max(1, int(stride)))
        if len(t) == 0:
            continue
        w = make_windows(ts.values, t, context_len)          # (N, L, n)
        y = make_targets(ts.values, t, horizon)              # (N, H, n)
        cal = make_extra(ts.ctx, t, context_len, horizon)["cal"]
        for j in range(ts.values.shape[1]):
            xs.append(w[:, :, j])
            ys.append(y[:, :, j])
            cals.append(cal)
            names += [ts.ctx.names[j]] * len(t)
            dsets += [ts.ctx.datasets[j]] * len(t)
    if not xs:
        raise ValueError(f"Không chuỗi nào dài hơn L + H = {context_len + horizon} bước")
    return dict(x=np.concatenate(xs).astype(np.float32), y=np.concatenate(ys).astype(np.float32),
                cal=np.concatenate(cals), name=np.array(names), dataset=np.array(dsets))


# ----------------------------------------------------------------------------- lớp cơ sở
class Forecaster:
    """Lớp cơ sở. Lớp con BẮT BUỘC cài `predict_batch`; `fit` nếu model cần train.

    Một forecaster vừa là model, vừa cắm thẳng được vào env:
        NFVScalingEnv(..., forecast_horizon=H, forecaster=fc)   # env gọi fc(env, t)
    và vào baseline dự báo:
        PredictiveThresholdPolicy(env, forecast_fn=fc.demand_fn())
    """

    name = "base"
    needs_fit = True          # False: dùng được ngay không cần train (naive, oracle...)
    quantiles: tuple = ()     # () = dự báo một giá trị; (0.1, 0.5, 0.9) = dự báo khoảng (P10/P50/P90)
    _CHUNK = 256              # số bước dự báo cùng lúc khi gọi trong env (tăng tốc)
    _MAX_SERIES = 32          # số chuỗi tải giữ cache cùng lúc (nhiều env dùng chung 1 forecaster)

    def __init__(self, horizon: int = FORECAST["horizon"], context_len: int = FORECAST["context_len"],
                 train_datasets: Sequence[str] = FORECAST["train_datasets"]):
        self.horizon = int(horizon)
        self.context_len = int(context_len)
        self.train_datasets = tuple(train_datasets)   # dataset script train dùng cho model này
        self._cache: Dict[int, tuple] = {}     # id(load) -> (load, {khối: dự báo})

    # ---- lớp con cài đặt ----
    def fit(self, train, val=None) -> "Forecaster":
        """train/val: mảng (T, n), TimeSeries hoặc list[TimeSeries]. Mặc định không làm gì."""
        return self

    def predict_batch(self, windows: np.ndarray, extra: Optional[Dict] = None) -> np.ndarray:
        """windows: (N, L, n_vnf) lịch sử -> (N, H, n_vnf) dự báo cho các bước tiếp theo.
        Model dự báo khoảng: mặc định trả về P50 của predict_dist_batch."""
        if self.quantiles:
            return self.predict_dist_batch(windows, extra)[..., self.quantile_index(0.5)]
        raise NotImplementedError

    def predict_dist_batch(self, windows: np.ndarray, extra: Optional[Dict] = None) -> np.ndarray:
        """windows: (N, L, n_vnf) -> (N, H, n_vnf, Q) theo thứ tự self.quantiles (chỉ model dự báo khoảng)."""
        raise NotImplementedError(f"{self.name} không dự báo khoảng")

    def quantile_index(self, q: float) -> int:
        for i, x in enumerate(self.quantiles):
            if abs(x - q) < 1e-6:
                return i
        raise ValueError(f"{self.name} không có quantile {q} (có: {self.quantiles})")

    # ---- dùng chung ----
    def predict(self, history: np.ndarray) -> np.ndarray:
        """history: (L, n_vnf) -> (H, n_vnf)."""
        return self.predict_batch(np.asarray(history, np.float32)[None])[0]

    def forecast_series(self, load: np.ndarray, t_idx, quantiles: Optional[Sequence[float]] = None,
                        ctx: Optional[SeriesContext] = None) -> np.ndarray:
        """Dự báo tại các thời điểm t_idx trên chuỗi load: (N, H, n_vnf).
        quantiles (vd (0.5, 0.9)): trả về (N, H, n_vnf, len(quantiles)); model dự báo một giá trị thì
        mọi quantile = giá trị đó. ctx: mô tả các cột của load (giờ, tên chuỗi) -> thông tin phụ cho model.
        Chỉ dùng dữ liệu <= t (oracle ghi đè hàm này để nhìn tương lai)."""
        windows = make_windows(load, t_idx, self.context_len).astype(np.float32)
        extra = make_extra(ctx, t_idx, self.context_len, self.horizon)
        if quantiles is None:
            out = self.predict_batch(windows, extra)
        elif self.quantiles:
            out = self.predict_dist_batch(windows, extra)[..., [self.quantile_index(q) for q in quantiles]]
        else:
            out = np.repeat(np.asarray(self.predict_batch(windows, extra))[..., None], len(quantiles), axis=-1)
        return np.clip(np.asarray(out, np.float32), 0.0, None)

    def __call__(self, env, t: int, horizon: Optional[int] = None,
                 quantiles: Optional[Sequence[float]] = None) -> np.ndarray:
        """Giao diện forecaster của NFVScalingEnv: f(env, t) -> (H, n_vnf), hoặc (H, n_vnf, Q) khi có quantiles.
        Dự báo theo khối _CHUNK bước rồi cache theo từng chuỗi env._load. Giờ và tên VNF lấy từ env.traces."""
        H = horizon or env.cfg.forecast_horizon or self.horizon
        if H > self.horizon:
            raise ValueError(f"{self.name}: model dự báo {self.horizon} bước, env yêu cầu {H}")
        q = tuple(quantiles) if quantiles else None
        load = env._load
        entry = self._cache.get(id(load))
        if entry is None or entry[0] is not load:
            if len(self._cache) >= self._MAX_SERIES:
                self._cache.pop(next(iter(self._cache)))      # bỏ chuỗi cũ nhất
            entry = self._cache[id(load)] = (load, {})
        chunks = entry[1]
        c = t // self._CHUNK
        if (c, q) not in chunks:
            lo = c * self._CHUNK
            chunks[(c, q)] = self.forecast_series(load, np.arange(lo, min(lo + self._CHUNK, len(load))), q,
                                                  env.traces.context())
        return chunks[(c, q)][t - c * self._CHUNK, :H]

    def demand_fn(self, lookahead: Optional[int] = None, quantile: Optional[float] = None) -> Callable:
        """Hàm cho PredictiveThresholdPolicy: env -> (n_vnf,) nhu cầu (vCPU) lớn nhất
        từ hiện tại tới `lookahead` bước sau (mặc định = scale_out_delay).
        quantile (vd 0.9): cấp theo P90 thay vì P50 -> dự phòng cho burst."""

        def fn(env):
            L = lookahead or env.cfg.scale_out_delay
            t = env.t0 + env.t
            if quantile is None:
                fc = self(env, t, horizon=min(L, self.horizon))
            else:
                fc = self(env, t, horizon=min(L, self.horizon), quantiles=(quantile,))[..., 0]
            return np.maximum(env._load[t], fc.max(axis=0)) * env.peak

        return fn

    def save(self, path: Optional[str] = None) -> str:
        path = path or model_path(self.name)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}                               # không lưu cache
        return state

    def __repr__(self):
        return f"{type(self).__name__}(H={self.horizon}, L={self.context_len})"
