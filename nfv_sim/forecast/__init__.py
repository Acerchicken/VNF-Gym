"""
Dự báo tải VNF (Bước 4).

    from nfv_sim.forecast import make_forecaster, load_forecaster, evaluate_forecaster

    fc = make_forecaster("lstm").fit(train_load, val_load)   # tham số lấy từ config.FORECASTERS
    # hoặc train trên nhiều dataset: fit(split_train_val(load_catalog(["sndzoo", "nab_aws"]))...)
    fc.save()                                                # -> results/forecasters/lstm.pkl
    fc = load_forecaster("lstm")
    env = NFVScalingEnv(..., forecast_horizon=fc.horizon, forecaster=fc)

Thêm thuật toán mới: xem docstring của base.py.
"""
from .base import (Forecaster, as_series_list, available_forecasters, load_forecaster, make_forecaster,
                   make_targets, make_training_set, make_windows, model_path, register_forecaster, split_train_val)
from .metrics import evaluate_by_dataset, evaluate_forecaster
from . import simple  # noqa: F401  (đăng ký naive, moving_avg, linear, oracle)

try:
    from . import lstm, tft  # noqa: F401  (cần PyTorch)
except ImportError:
    pass

__all__ = ["Forecaster", "register_forecaster", "make_forecaster", "load_forecaster", "available_forecasters",
           "evaluate_forecaster", "evaluate_by_dataset", "make_windows", "make_targets", "make_training_set",
           "as_series_list", "model_path", "split_train_val"]
