"""So sánh các baseline trên ngày test 18-02, ở kịch bản bình thường và quá tải.

    python examples/run_baselines.py                          (tham số lấy từ configs/config.py)
    python examples/run_baselines.py --forecasters lstm linear

Mỗi forecaster đã train (examples/train_forecaster.py) tạo thêm một policy "predictive(<tên>)".
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):  # console Windows (cp1252) không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from nfv_sim import (NFVScalingEnv, PredictiveThresholdPolicy, StaticPolicy, ThresholdPolicy,
                     load_sndzoo, run_episode)
from nfv_sim.config import DATA_ROOT, EVAL_SEED, FORECAST, RESULTS_DIR, SCENARIOS
from nfv_sim.forecast import load_forecaster

p = argparse.ArgumentParser()
p.add_argument("--data", default=DATA_ROOT, help="thư mục DatasetSNDZoo (mặc định lấy từ config.py)")
p.add_argument("--out", default=RESULTS_DIR)
p.add_argument("--forecasters", nargs="*", default=[FORECAST["default"]],
               help="forecaster đã train dùng cho PredictiveThresholdPolicy")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

test = load_sndzoo(args.data, "test")
POLICIES = {
    "static": lambda e: StaticPolicy(e),
    "threshold": lambda e: ThresholdPolicy(e),
    "threshold+consolidate": lambda e: ThresholdPolicy(e, consolidate=True),
    "vertical": lambda e: ThresholdPolicy(e, mode="vertical"),
    "hybrid": lambda e: ThresholdPolicy(e, mode="hybrid"),
    "predictive(oracle)": lambda e: PredictiveThresholdPolicy(e),
}
for fname in args.forecasters:
    try:
        fc = load_forecaster(fname)
    except FileNotFoundError as ex:
        print(f"Bỏ qua predictive({fname}): {ex}")
        continue
    POLICIES[f"predictive({fname})"] = lambda e, fc=fc: PredictiveThresholdPolicy(e, forecast_fn=fc.demand_fn())

rows, traces = {}, {}
for sname, kw in SCENARIOS.items():
    for pname, make in POLICIES.items():
        env = NFVScalingEnv(traces=test, episode_steps=None, random_start=False, **kw)
        summary, tr = run_episode(env, make(env), seed=EVAL_SEED, record=True)
        rows[(sname, pname)] = summary
        traces[(sname, pname)] = tr

df = pd.DataFrame(rows).T.drop(columns="steps")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(df.to_string())
df.to_csv(os.path.join(args.out, "baselines.csv"))

try:
    import matplotlib.pyplot as plt

    for sname in SCENARIOS:
        fig, axes = plt.subplots(len(test.vnfs), 1, figsize=(12, 8), sharex=True)
        for j, v in enumerate(test.vnfs):
            ax = axes[j]
            tr = traces[(sname, "threshold")]
            ax.plot([x["demand"][j] for x in tr], color="black", lw=1, label="demand")
            for pname, c in [("threshold", "tab:orange"), ("predictive(oracle)", "tab:blue")]:
                tr = traces[(sname, pname)]
                ax.step(range(len(tr)), [x["capacity"][j] for x in tr], color=c, lw=1, label=f"capacity {pname}")
            ax.set_ylabel(v)
        axes[0].legend(loc="upper left", fontsize=8)
        axes[-1].set_xlabel("bước (phút)")
        fig.suptitle(f"Demand vs capacity — {sname}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"capacity_{sname}.png"), dpi=120)
    print(f"Đã lưu hình vào {args.out}/")
except ImportError:
    print("(cài matplotlib để vẽ hình)")
