"""Gộp các file ppo_eval*.csv (do train_ppo.py ghi) thành một bảng so sánh: trung bình ± độ lệch chuẩn theo seed.

    python examples\\compare_ppo.py                       (đọc results/ppo_eval*.csv)
    python examples\\compare_ppo.py --scenario overload

Mỗi cấu hình state là một tag: "" (không dự báo), "_oracle5", "_lstm", "_lstm_q50-90", "_tft", ...
Các lần chạy với --seed 1, 2 có thêm đuôi "_s1", "_s2" và được gộp vào cùng cấu hình.
Đầu ra: in bảng ra màn hình và lưu <dir>/ppo_compare.csv.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):  # console Windows (cp1252) không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from nfv_sim.config import RESULTS_DIR

COLS = ["sla_violation_rate", "avg_vcpu", "utilization", "oscillations", "total_reward"]

p = argparse.ArgumentParser()
p.add_argument("--dir", default=RESULTS_DIR)
p.add_argument("--scenario", default="normal", help="normal hoặc overload (mặc định: %(default)s)")
args = p.parse_args()

rows = []
for f in sorted(glob.glob(os.path.join(args.dir, "ppo_eval*.csv"))):
    tag = os.path.basename(f)[len("ppo_eval"):-len(".csv")]
    m = re.search(r"_s(\d+)$", tag)
    seed = int(m.group(1)) if m else 0
    config = (tag[:m.start()] if m else tag).lstrip("_") or "none"
    df = pd.read_csv(f, index_col=[0, 1])
    for (scen, policy), r in df.iterrows():
        if scen == args.scenario and str(policy).startswith("ppo"):
            rows.append(dict(config=config, seed=seed, **{c: r[c] for c in COLS}))
if not rows:
    raise SystemExit(f"Không có ppo_eval*.csv trong {args.dir} (chạy train_ppo.py trước)")

df = pd.DataFrame(rows)
g = df.groupby("config")[COLS]
out = g.mean().round(4).astype(str) + " ± " + g.std(ddof=0).round(4).astype(str)
out.insert(0, "n_seeds", df.groupby("config").size())
pd.set_option("display.width", 200)
print(f"Kịch bản {args.scenario}, trung bình ± độ lệch chuẩn qua các seed:")
print(out.to_string())
out.to_csv(os.path.join(args.dir, "ppo_compare.csv"))
