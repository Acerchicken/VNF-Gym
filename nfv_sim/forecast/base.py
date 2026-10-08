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
            ...
            return self

        def predict_batch(self, windows):       # (N, L, n_vnf) -> (N, H, n_vnf)
            ...

rồi thêm `"my_model": dict(my_param=...)` vào FORECASTERS trong config.py. Sau đó mọi
script (train_forecaster, run_baselines, train_ppo) dùng được qua `--forecaster my_model`.

Quy ước dữ liệu: tải đã chuẩn hoá (giống `env._load`), mảng (T, n_vnf), float32.
"""
from __future__ import annotations

import os
import pickle
from typing import Callable, Dict, Optional, Type

import numpy as np

from ..config import FORECAST, FORECASTERS

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


def available_forecasters() -> list:
    return sorted(_REGISTRY)


def make_forecaster(name: str, **overrides) -> "Forecaster":
    """Tạo forecaster theo tên, tham số = FORECAST (chung) + FORECASTERS[name] + overrides."""
    if name not in _REGISTRY:
        raise KeyError(f"Không có forecaster '{name}'. Có: {available_forecasters()}")
    kw = dict(horizon=FORECAST["horizon"], context_len=FORECAST["context_len"])
    kw.update(FORECASTERS.get(name, {}))
    kw.update(overrides)
    return _REGISTRY[name](**kw)


def model_path(name: str, models_dir: Optional[str] = None) -> str:
    return os.path.join(models_dir or FORECAST["models_dir"], f"{name}.pkl")


def load_forecaster(name_or_path: str, models_dir: Optional[str] = None) -> "Forecaster":
    """Nạp forecaster đã lưu. Nhận tên (tìm <models_dir>/<tên>.pkl) hoặc đường dẫn file.
    Model không cần train (naive, oracle...) mà chưa có file thì tạo mới từ config."""
    path = name_or_path if os.path.isfile(name_or_path) else model_path(name_or_path, models_dir)
    if os.path.isfile(path):
        with open(path, "rb") as f:   # chỉ nạp file do chính dự án tạo ra (pickle)
            return pickle.load(f)
    if name_or_path in _REGISTRY and not _REGISTRY[name_or_path].needs_fit:
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


def split_train_val(load: np.ndarray, val_frac: float = FORECAST["val_frac"]):
    n_val = int(len(load) * val_frac)
    if n_val <= 0:
        return load, None
    return load[:-n_val], load[-n_val:]


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
    _CHUNK = 256              # số bước dự báo cùng lúc khi gọi trong env (tăng tốc)
    _MAX_SERIES = 32          # số chuỗi tải giữ cache cùng lúc (nhiều env dùng chung 1 forecaster)

    def __init__(self, horizon: int = FORECAST["horizon"], context_len: int = FORECAST["context_len"]):
        self.horizon = int(horizon)
        self.context_len = int(context_len)
        self._cache: Dict[int, tuple] = {}     # id(load) -> (load, {khối: dự báo})

    # ---- lớp con cài đặt ----
    def fit(self, train: np.ndarray, val: Optional[np.ndarray] = None) -> "Forecaster":
        """train/val: (T, n_vnf). Mặc định không làm gì (model không cần train)."""
        return self

    def predict_batch(self, windows: np.ndarray) -> np.ndarray:
        """windows: (N, L, n_vnf) lịch sử -> (N, H, n_vnf) dự báo cho các bước tiếp theo."""
        raise NotImplementedError

    # ---- dùng chung ----
    def predict(self, history: np.ndarray) -> np.ndarray:
        """history: (L, n_vnf) -> (H, n_vnf)."""
        return self.predict_batch(np.asarray(history, np.float32)[None])[0]

    def forecast_series(self, load: np.ndarray, t_idx) -> np.ndarray:
        """Dự báo tại các thời điểm t_idx trên chuỗi load: (N, H, n_vnf).
        Chỉ dùng dữ liệu <= t (oracle ghi đè hàm này để nhìn tương lai)."""
        windows = make_windows(load, t_idx, self.context_len).astype(np.float32)
        out = np.asarray(self.predict_batch(windows), np.float32)
        return np.clip(out, 0.0, None)

    def __call__(self, env, t: int, horizon: Optional[int] = None) -> np.ndarray:
        """Giao diện forecaster của NFVScalingEnv: f(env, t) -> (H, n_vnf).
        Dự báo theo khối _CHUNK bước rồi cache theo từng chuỗi env._load."""
        H = horizon or env.cfg.forecast_horizon or self.horizon
        if H > self.horizon:
            raise ValueError(f"{self.name}: model dự báo {self.horizon} bước, env yêu cầu {H}")
        load = env._load
        entry = self._cache.get(id(load))
        if entry is None or entry[0] is not load:
            if len(self._cache) >= self._MAX_SERIES:
                self._cache.pop(next(iter(self._cache)))      # bỏ chuỗi cũ nhất
            entry = self._cache[id(load)] = (load, {})
        chunks = entry[1]
        c = t // self._CHUNK
        if c not in chunks:
            lo = c * self._CHUNK
            chunks[c] = self.forecast_series(load, np.arange(lo, min(lo + self._CHUNK, len(load))))
        return chunks[c][t - c * self._CHUNK, :H]

    def demand_fn(self, lookahead: Optional[int] = None) -> Callable:
        """Hàm cho PredictiveThresholdPolicy: env -> (n_vnf,) nhu cầu (vCPU) lớn nhất
        từ hiện tại tới `lookahead` bước sau (mặc định = scale_out_delay)."""

        def fn(env):
            L = lookahead or env.cfg.scale_out_delay
            t = env.t0 + env.t
            fc = self(env, t, horizon=min(L, self.horizon))
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
