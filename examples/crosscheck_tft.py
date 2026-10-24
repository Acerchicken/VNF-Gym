"""So chéo TFT tự viết (nfv_sim/forecast/tft.py) với TFT của thư viện pytorch-forecasting (Bước 4).

Chạy MỘT lần để kiểm tra bản tự viết không sai lệch lớn; thư viện không được dùng ở chỗ nào khác của dự án.

    pip install pytorch-forecasting            (kéo theo lightning)
    python examples\\crosscheck_tft.py          (cần results/forecasters/tft.pkl từ train_forecaster.py)

Hai model dùng cùng dữ liệu train (FORECASTERS["tft"]["train_datasets"]), cùng L, H, quantile, cùng thông tin phụ
(5 feature lịch biết trước, ID chuỗi và dataset là biến tĩnh), cùng kích thước ẩn và số head.
Khác nhau: thư viện tự chuẩn hoá theo từng cửa sổ (EncoderNormalizer) và có thêm vài chi tiết của bài báo gốc;
bản tự viết dự báo phần dư so với giá trị hiện tại.

Đầu ra: <out>/crosscheck_tft.csv (MAE, pinball, cov_p90... theo dataset, cho cả hai model).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):  # console Windows (cp1252) không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from nfv_sim.config import FORECAST, FORECASTERS
from nfv_sim.data import N_CALENDAR, load_catalog
from nfv_sim.forecast import evaluate_by_dataset, load_forecaster, split_train_val
from nfv_sim.forecast.metrics import _summary

try:
    import lightning.pytorch as pl
    import torch
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import EncoderNormalizer, NaNLabelEncoder
    from pytorch_forecasting.metrics import QuantileLoss
except ImportError:
    raise SystemExit("Cần cài: pip install pytorch-forecasting")

p = argparse.ArgumentParser()
p.add_argument("--model", default="tft", help="TFT tự viết đã train (mặc định: %(default)s)")
p.add_argument("--epochs", type=int, default=FORECASTERS["tft"]["epochs"])
p.add_argument("--out", default=FORECAST["models_dir"])
args = p.parse_args()

CAL = [f"cal{i}" for i in range(N_CALENDAR)]
cfg = FORECASTERS["tft"]
L, H, Q = FORECAST["context_len"], FORECAST["horizon"], list(cfg["quantiles"])
ours = load_forecaster(args.model, args.out)


def to_frame(series):
    """list[TimeSeries] -> bảng dài: mỗi dòng = (chuỗi, bước), cột target + feature lịch."""
    parts = []
    for ts in series:
        pos = np.arange(len(ts))
        cal = ts.ctx.calendar_features(pos)
        for j, (name, ds) in enumerate(zip(ts.ctx.names, ts.ctx.datasets)):
            df = pd.DataFrame(cal, columns=CAL)
            df["series"], df["dataset"], df["time_idx"] = name, ds, pos
            df["target"] = ts.values[:, j].astype(np.float32)
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


tr, va = split_train_val(load_catalog(ours.train_datasets, "train"))
train_ds = TimeSeriesDataSet(
    to_frame(tr), time_idx="time_idx", target="target", group_ids=["series"],
    max_encoder_length=L, min_encoder_length=L, max_prediction_length=H, min_prediction_length=H,
    static_categoricals=["series", "dataset"], time_varying_known_reals=CAL,
    time_varying_unknown_reals=["target"], target_normalizer=EncoderNormalizer(),
    categorical_encoders={"series": NaNLabelEncoder(add_nan=True), "dataset": NaNLabelEncoder(add_nan=True)},
    add_relative_time_idx=False, add_target_scales=False, add_encoder_length=False)
val_ds = TimeSeriesDataSet.from_dataset(train_ds, to_frame(va), stop_randomization=True)
bs = cfg["batch_size"]
pl.seed_everything(cfg["seed"])
model = TemporalFusionTransformer.from_dataset(
    train_ds, hidden_size=cfg["d_model"], attention_head_size=cfg["n_heads"], dropout=cfg["dropout"],
    hidden_continuous_size=cfg["d_model"], learning_rate=cfg["lr"], loss=QuantileLoss(Q), log_interval=-1)
print(f"pytorch-forecasting TFT: {sum(p.numel() for p in model.parameters())} tham số; "
      f"tự viết: {sum(p.numel() for p in ours.net.parameters())}")
trainer = pl.Trainer(max_epochs=args.epochs, gradient_clip_val=1.0, enable_checkpointing=False, logger=False,
                     callbacks=[pl.callbacks.EarlyStopping("val_loss", patience=cfg["patience"])],
                     enable_model_summary=False)
t0 = time.time()
trainer.fit(model, train_ds.to_dataloader(train=True, batch_size=bs, num_workers=0),
            val_ds.to_dataloader(train=False, batch_size=4 * bs, num_workers=0))
print(f"train xong sau {time.time() - t0:.0f} s, {trainer.current_epoch} epoch")

rows = {}
test = load_catalog(FORECAST["eval_datasets"], "test")
for ds, m in evaluate_by_dataset(ours, test).items():
    rows[(args.model + " (tự viết)", ds)] = m
by = {}
for ts in test:
    if len(ts) <= L + H:
        continue
    test_ds = TimeSeriesDataSet.from_dataset(train_ds, to_frame([ts]), stop_randomization=True, predict=False)
    out = model.predict(test_ds.to_dataloader(train=False, batch_size=4 * bs, num_workers=0),
                        mode="quantiles", return_y=True, trainer_kwargs=dict(logger=False))
    dist, true = out.output.cpu().numpy(), out.y[0].cpu().numpy()          # (N, H, Q), (N, H)
    by.setdefault(ts.ctx.datasets[0], []).append((np.clip(dist, 0, None), true))
for ds, v in by.items():
    dist = np.concatenate([d for d, _ in v])[:, :, None, :]                 # (N, H, 1, Q)
    true = np.concatenate([t for _, t in v])[:, :, None]
    rows[("pytorch-forecasting", ds)] = _summary(dist[..., Q.index(0.5)], true, dist, tuple(Q))

df = pd.DataFrame(rows).T
df.index.names = ["model", "dataset"]
cols = ["mae", "mae_h1", "mae_h5", "peak_mae", "pinball", "cov_p10", "cov_p90", "width"]
pd.set_option("display.width", 200)
print(df[cols].round(4).to_string())
df.to_csv(os.path.join(args.out, "crosscheck_tft.csv"))
