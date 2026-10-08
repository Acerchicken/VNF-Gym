"""Train và so sánh các thuật toán dự báo tải (Bước 4) trên ngày test 18-02.

    python examples\\train_forecaster.py                       (mọi forecaster đã đăng ký)
    python examples\\train_forecaster.py --models lstm linear  (chỉ một số model)

Tham số lấy từ nfv_sim/config.py (FORECAST, FORECASTERS). Model đã train lưu ở
results/forecasters/<tên>.pkl để train_ppo.py / run_baselines.py dùng qua --forecaster <tên>.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):  # console Windows (cp1252) không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from nfv_sim import SimConfig, load_sndzoo
from nfv_sim.config import DATA_ROOT, FORECAST
from nfv_sim.forecast import available_forecasters, evaluate_forecaster, make_forecaster, split_train_val

p = argparse.ArgumentParser()
p.add_argument("--data", default=DATA_ROOT, help="thư mục DatasetSNDZoo (mặc định lấy từ config.py)")
p.add_argument("--models", nargs="+", default=available_forecasters(), choices=available_forecasters())
p.add_argument("--out", default=FORECAST["models_dir"])
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

metric = SimConfig().load_metric          # cùng chuỗi tải mà simulator dùng
train = load_sndzoo(args.data, "train")
test = load_sndzoo(args.data, "test")
tr, va = split_train_val(train.values[metric])
te = test.values[metric]
print(f"metric={metric} | train {len(tr)} / val {0 if va is None else len(va)} / test {len(te)} bước | "
      f"H={FORECAST['horizon']} L={FORECAST['context_len']}")

rows = {}
for name in args.models:
    print(f"\n=== {name} ===")
    fc = make_forecaster(name)
    t0 = time.time()
    fc.fit(tr, va)
    fit_s = time.time() - t0
    if fc.needs_fit:
        print(f"  đã lưu {fc.save(os.path.join(args.out, f'{name}.pkl'))}")
    rows[name] = dict(evaluate_forecaster(fc, te, vnfs=test.vnfs), fit_s=round(fit_s, 1))

df = pd.DataFrame(rows).T.sort_values("mae")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\nKết quả trên ngày test (tải chuẩn hoá, oracle = cận trên):")
print(df.round(4).to_string())
df.to_csv(os.path.join(args.out, "forecast_metrics.csv"))

try:
    import matplotlib.pyplot as plt
    import numpy as np

    from nfv_sim.forecast import load_forecaster

    k = FORECAST["horizon"]                 # vẽ dự báo bước xa nhất (khó nhất)
    t = np.arange(FORECAST["context_len"] - 1, len(te) - k)
    fig, axes = plt.subplots(len(test.vnfs), 1, figsize=(12, 8), sharex=True)
    for j, v in enumerate(test.vnfs):
        axes[j].plot(t + k, te[t + k, j], color="black", lw=1, label="thực tế")
        for name in args.models:
            if name == "oracle":
                continue
            fc = load_forecaster(name, args.out)
            axes[j].plot(t + k, fc.forecast_series(te, t)[:, k - 1, j], lw=0.8, label=name)
        axes[j].set_ylabel(v)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("bước (phút)")
    fig.suptitle(f"Dự báo t+{k} trên ngày test")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "forecast_test.png"), dpi=120)
    print(f"Đã lưu hình vào {args.out}/")
except ImportError:
    print("(cài matplotlib để vẽ hình)")
