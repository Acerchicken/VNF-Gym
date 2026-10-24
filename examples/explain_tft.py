"""Vẽ phần "giải thích" của TFT đã train (Bước 4): biến nào quan trọng và model nhìn lại bước quá khứ nào.

    python examples\\explain_tft.py                    (dùng results/forecasters/tft.pkl)
    python examples\\explain_tft.py --model tft --datasets sndzoo nab_aws

Đầu ra (trong --out):
    tft_importance.csv           trọng số chọn biến trung bình theo từng dataset (phần test)
    tft_importance.png           cột: biến tĩnh, biến quá khứ, biến tương lai
    tft_attention.png            trọng số attention trung bình theo độ trễ, cho từng bước dự báo t+1..t+H
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):  # console Windows (cp1252) không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from nfv_sim.config import DATASETS, FORECAST
from nfv_sim.data import load_catalog
from nfv_sim.forecast import load_forecaster
from nfv_sim.forecast.tft import CALENDAR_NAMES, STATIC_NAMES

p = argparse.ArgumentParser()
p.add_argument("--model", default="tft", help="tên model TFT đã train (mặc định: %(default)s)")
p.add_argument("--datasets", nargs="+", default=list(FORECAST["eval_datasets"]), choices=sorted(DATASETS))
p.add_argument("--out", default=FORECAST["models_dir"])
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

fc = load_forecaster(args.model, args.out)
names = ([f"tĩnh: {n}" for n in STATIC_NAMES] + [f"quá khứ: {n}" for n in ("tải",) + CALENDAR_NAMES]
         + [f"tương lai: {n}" for n in CALENDAR_NAMES])
rows, attn = {}, {}
for ds in args.datasets:
    e = fc.explain(load_catalog([ds], "test"))
    rows[ds] = np.concatenate([e["static"], e["past"], e["future"]])
    attn[ds] = e["attention"]
    print(f"{ds}: tải chiếm {e['past'][0]:.0%} trọng số quá khứ; ID dataset {e['static'][1]:.0%} trọng số tĩnh")
df = pd.DataFrame(rows, index=names)
print(df.round(3).to_string())
df.to_csv(os.path.join(args.out, f"{args.model}_importance.csv"))

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("trọng số chọn biến trung bình (mỗi nhóm cộng lại = 1)")
    ax.set_title(f"{args.model}: biến nào quan trọng")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{args.model}_importance.png"), dpi=150)
    plt.close(fig)

    L, H = fc.context_len, fc.horizon
    fig, axes = plt.subplots(1, len(attn), figsize=(4 * len(attn), 3.5), sharey=True, squeeze=False)
    for ax, (ds, a) in zip(axes[0], attn.items()):
        lag = np.arange(L + H) - (L - 1)           # 0 = thời điểm hiện tại t, âm = quá khứ
        for h in range(H):
            ax.plot(lag, a[h], lw=1, label=f"t+{h + 1}")
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_title(ds)
        ax.set_xlabel("bước so với hiện tại t")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("trọng số attention")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"{args.model}: model nhìn lại bước nào khi dự báo")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"{args.model}_attention.png"), dpi=150)
    plt.close(fig)
    print(f"Đã lưu hình vào {args.out}/")
except ImportError:
    print("(cài matplotlib để vẽ hình)")
