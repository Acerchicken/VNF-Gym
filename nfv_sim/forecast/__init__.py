"""
Dự báo tải VNF (Bước 4).

    from nfv_sim.forecast import make_forecaster, load_forecaster, evaluate_forecaster

    fc = make_forecaster("lstm").fit(train_load, val_load)   # tham số lấy từ config.FORECASTERS
    fc.save()                                                # -> results/forecasters/lstm.pkl
    fc = load_forecaster("lstm")
    env = NFVScalingEnv(..., forecast_horizon=fc.horizon, forecaster=fc)

Thêm thuật toán mới: xem docstring của base.py.
"""
from .base import (Forecaster, available_forecasters, load_forecaster, make_forecaster, make_targets,
                   make_windows, model_path, register_forecaster, split_train_val)
from .metrics import evaluate_forecaster
from . import simple  # noqa: F401  (đăng ký naive, moving_avg, linear, oracle)

try:
    from . import lstm  # noqa: F401  (cần PyTorch)
except ImportError:
    pass

__all__ = ["Forecaster", "register_forecaster", "make_forecaster", "load_forecaster", "available_forecasters",
           "evaluate_forecaster", "make_windows", "make_targets", "model_path", "split_train_val"]
