"""Train and compare load-forecasting algorithms (Step 4) on the 18-02 test day.

    python examples\\train_forecaster.py                       (all registered forecasters)
    python examples\\train_forecaster.py --models lstm linear  (only some models)
    python examples\\train_forecaster.py --plot naive lstm linear  (choose plotted lines, default FORECAST["plot"])
    python examples\\train_forecaster.py --models naive --plot naive lstm  (re-plot only, reuse the trained lstm.pkl)
    python examples\\train_forecaster.py --models naive --range 0 30  (plot only the first 30% of the test day)

Run with --help for the full list of options. Parameters come from configs/config.py (FORECAST, FORECASTERS).
Trained models are saved to results/forecasters/<name>.pkl so train_ppo.py / run_baselines.py can use them
via --forecaster <name>.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):  # Windows console (cp1252) cannot print non-ASCII text
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from nfv_sim import SimConfig, load_sndzoo
from nfv_sim.config import DATA_ROOT, FORECAST
from nfv_sim.forecast import available_forecasters, evaluate_forecaster, make_forecaster, split_train_val

ALGOS = {  # one-line description of each algorithm for --help
    "naive": "repeat the last observed value (simplest baseline)",
    "moving_avg": "mean of the most recent steps",
    "linear": "linear autoregression (AR) with ridge regularization",
    "lstm": "LSTM neural network (requires PyTorch)",
    "oracle": "sees the future = upper bound, for reference only, not a real result",
}
EPILOG = "Available algorithms:\n" + "\n".join(
    f"  {n:<11} {ALGOS.get(n, '')}" for n in available_forecasters()) + r"""

Examples:
  python examples\train_forecaster.py
      train + evaluate every algorithm, plot actual + naive + lstm over the whole test day
  python examples\train_forecaster.py --models naive --plot naive lstm
      re-plot only (does not retrain LSTM, reuses the existing lstm.pkl)
  python examples\train_forecaster.py --models naive --plot naive lstm linear --range 0 30
      add the linear line and plot only the first 30% of the test day so the time axis is easier to read

Outputs (in --out):
  <name>.pkl                      trained model
  forecast_metrics.csv            error metrics on the test day
  forecast_test_<VNF>.png         forecast plot per VNF (WEB, IOT, SEC)
  forecast_test_<VNF>_A-Bpct.png  plot when --range A B is used
"""

p = argparse.ArgumentParser(description="Train and compare load-forecasting algorithms on the test day, "
                                        "saving one plot per VNF.",
                            epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--data", default=DATA_ROOT, metavar="DIR",
               help="DatasetSNDZoo directory (default: %(default)s)")
p.add_argument("--models", nargs="+", default=available_forecasters(), choices=available_forecasters(),
               metavar="NAME", help="algorithms to train + evaluate, separated by spaces "
                                    "(default: all). Choices: %(choices)s")
p.add_argument("--out", default=FORECAST["models_dir"], metavar="DIR",
               help="where models, metrics and plots are saved (default: %(default)s)")
p.add_argument("--plot", nargs="+", default=FORECAST["plot"], choices=available_forecasters(), metavar="NAME",
               help="algorithms drawn on the plots; the actual line is always drawn (default: %(default)s). "
                    "Models not trained in this run are loaded from --out. Choices: %(choices)s")
p.add_argument("--range", nargs=2, type=float, default=FORECAST["plot_range"], metavar=("START", "END"),
               help="part of the test day to plot, in percent, 0 <= START < END <= 100 "
                    "(default: %(default)s = whole day; e.g. 0 30 = first 30%%, 40 60 = middle part)")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

metric = SimConfig().load_metric          # same load series the simulator uses
train = load_sndzoo(args.data, "train")
test = load_sndzoo(args.data, "test")
tr, va = split_train_val(train.values[metric])
te = test.values[metric]
print(f"metric={metric} | train {len(tr)} / val {0 if va is None else len(va)} / test {len(te)} steps | "
      f"H={FORECAST['horizon']} L={FORECAST['context_len']}")

rows = {}
for name in args.models:
    print(f"\n=== {name} ===")
    fc = make_forecaster(name)
    t0 = time.time()
    fc.fit(tr, va)
    fit_s = time.time() - t0
    if fc.needs_fit:
        print(f"  saved {fc.save(os.path.join(args.out, f'{name}.pkl'))}")
    rows[name] = dict(evaluate_forecaster(fc, te, vnfs=test.vnfs), fit_s=round(fit_s, 1))

df = pd.DataFrame(rows).T.sort_values("mae")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\nResults on the test day (normalized load, oracle = upper bound):")
print(df.round(4).to_string())
df.to_csv(os.path.join(args.out, "forecast_metrics.csv"))

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np

    from nfv_sim.forecast import load_forecaster

    k = FORECAST["horizon"]                 # plot the furthest (hardest) forecast step
    t = np.arange(FORECAST["context_len"] - 1, len(te) - k)
    preds = {}                              # forecast once, reuse for every plot
    for name in args.plot:
        try:
            preds[name] = load_forecaster(name, args.out).forecast_series(te, t)[:, k - 1]
        except FileNotFoundError as e:
            print(f"  skipping {name} in plots: {e}")
    lo, hi = (int(round(len(t) * r / 100)) for r in args.range)   # plot only the --range part so the time axis spreads out
    if not 0 <= lo < hi <= len(t):
        raise SystemExit(f"--range {args.range} is invalid (need 0 <= START < END <= 100)")
    t = t[lo:hi]
    preds = {name: y[lo:hi] for name, y in preds.items()}
    full = args.range[0] <= 0 and args.range[1] >= 100
    suffix = "" if full else f"_{args.range[0]:g}-{args.range[1]:g}pct"   # do not overwrite the full-day plot
    x = test.index[t + k]                   # x axis = real time of the forecast point (1 step = 1 minute)
    for j, v in enumerate(test.vnfs):       # one plot per VNF for readability
        fig, ax = plt.subplots(figsize=FORECAST["plot_figsize"])
        ax.plot(x, te[t + k, j], color="black", lw=1.2, label="actual")
        for name, y in preds.items():
            ax.plot(x, y[:, j], lw=1, alpha=0.85, label=name)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        ax.set_xlabel("time")
        ax.set_ylabel("load (normalized)")
        ax.set_title(f"{v}: {k}-minute-ahead forecast on test day")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = os.path.join(args.out, f"forecast_test_{v}{suffix}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")
except ImportError:
    print("(install matplotlib to generate plots)")
