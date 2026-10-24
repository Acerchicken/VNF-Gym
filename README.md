# nfv_sim — Gymnasium-compatible VNF auto-scaling simulator (SNDZoo data)

A lightweight simulator written in plain Python (numpy + pandas + gymnasium) for training and evaluating RL
agents on the **VNF auto-scaling** problem. It runs on real SNDZoo traces: Nginx (WEB),
Mosquitto (IOT) and Snort (SEC). It replaces CloudSimNFV and runs at roughly 5,000 steps/second on a single CPU core.

## 1. Installation and usage on Windows

Requires Python ≥ 3.10 (tested with 3.13). Run all commands from the project folder `D:\Git_repo\VNF Gym`.

**PowerShell / VS Code terminal**

```powershell
cd "D:\Git_repo\VNF Gym"
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
pip install stable-baselines3         # only needed to train PPO/DQN (pulls in PyTorch)
```

* **cmd**: same commands as above, but activate the venv with `.venv\Scripts\activate.bat`.
* **Git Bash**: activate the venv with `source .venv/Scripts/activate`.

> On Windows, avoid running `python -m pip install --upgrade pip` inside the venv unless necessary. pip can lock
> its own files and corrupt the venv; if that happens, delete the `.venv` folder and recreate it.

**Configure**: all parameters live in [`configs/config.py`](configs/config.py). Set `DATA_ROOT` there to your
**DatasetSNDZoo** folder once; the scripts then need no arguments (`--data`, `--steps`, … still override it).

**Run**:

```powershell
python examples\check_env.py
python examples\run_baselines.py
python examples\train_ppo.py
python examples\play.py          # step through the simulator by hand
```

Outputs (`baselines.csv`, `capacity_*.png` plots, the PPO model) are written to the `results/` folder.

**VS Code**:

1. Open the project folder.
2. `Ctrl+Shift+P` → *Python: Select Interpreter* → choose `.venv`.
3. Open a new terminal; the venv is activated automatically. Run the commands above.

## 2. How the data is used

The split is set by `TRAIN_FRAC` in `configs/config.py` (default 0.7): the `<VNF>/TrainTest/NoHoles/*.csv`
folder (~20,000 minutes, 27/01 → 10/02/2025) is cut in time, the first 70% is train and the rest is test.

| Split | `TRAIN_FRAC = 0.7` (default) | `TRAIN_FRAC = None` (original SNDZoo split) |
|---|---|---|
| train | first 70% of TrainTest (~14,000 min, 27/01 → 06/02) | `<VNF>/TrainTest/NoHoles/*.csv` (~20,000 min) |
| test  | last 30% of TrainTest (~6,000 min, 06/02 → 10/02) | `<VNF>/18-02/CSVmergedFIXED/*.csv` (~1,300 min) |

* **Incoming load** (`load_metric="rx"`, the default) is `container_network_receive_bytes_total`, normalised to [0,1].
  It can be switched to `"cpu"`.
* **Conversion to instance count:** `demand_t = load_t × peak_instances[VNF]`. Defaults are WEB=6, IOT=4, SEC=5,
  meaning that at the trace's peak load about 6 WEB instances are needed running at 100%. The data was min-max
  normalised, so the absolute scale is lost; this is therefore an **assumption that must be stated explicitly in the thesis**.
* **Memory working set** (`mem`) is used to compute **migration time and cost**.
  A VNF using more RAM takes longer and costs more to migrate (modelling pre-copy live migration).
* The tx/cpu columns are loaded as well, so you can use them as features or change `load_metric`.

### Additional datasets (used only to train the forecasters)

`DATASETS` in `configs/config.py` lists extra load traces stored in the SNDZoo folder layout under
`EXTERNAL_ROOT` (`E:/SNDZoo dataset/ExternalDatasets`, see the README there). `nfv_sim.data.load_catalog(names, split)`
reads them as a list of `TimeSeries` (load array + `SeriesContext`: series name, dataset, start time, step, whether
the clock is real). Every series is split in time with `TRAIN_FRAC`, like SNDZoo.

| Key | Series | Step | Real clock? | Content |
|---|---|---|---|---|
| `sndzoo` | 3 (WEB, IOT, SEC) | 1 min | yes | incoming traffic of the three VNFs |
| `nab_aws` | 12 | 5 min | yes | AWS CloudWatch CPU / network in / ELB requests (NAB) |
| `google_machines` | 6 | 5 min | yes | CPU of single Google Borg machines, May 2019 |
| `alibaba_cluster` | 1 | 1 min | no | Alibaba 2018 cluster-average CPU |
| `cluster_5min` | 2 | 5 min | no | Google 2019 and Azure 2019 cluster-wide CPU |

5-minute data is taken back to its real samples (the loader's 1-minute interpolation is dropped), so a 5-step
forecast means 25 minutes there. "Real clock = no" means the start date is a placeholder, so the calendar features
are switched off for those series. WorldCup98 is in the folder but deliberately not used (1998 data).
The simulator itself still runs on SNDZoo only (it needs `peak_instances`, SLA and latency for each VNF).

## 3. System model

```
          ┌─ node 0 [■■■□] ─┐      1 node = node_slots (4) vCPU; 1 instance = size vCPU (1..max_vcpu)
 load ──► │  node 1 [■■□□]  │ ───► a node is on if it hosts ≥1 instance; power P = P_idle + (P_max−P_idle)·util
          └─ node 5 [□□□□] ─┘
```

**Instance lifecycle**

| State | Serves | Incurs cost | Notes |
|---|---|---|---|
| BOOT | 0% | yes | lasts `scale_out_delay` steps (default 3 minutes) → **scale-out latency** |
| RUN  | size^α (α = `vertical_alpha` = 0.9) | yes | |
| MIG  | `migration_capacity` (50%) | yes, occupies a slot on both the source and destination node | lasts `migration_base_delay + ceil(mem × migration_mem_delay)` steps |

**Vertical scaling**

* +1 vCPU: applies to a running instance, only if its node still has a free slot. Takes effect after `vertical_delay`
  steps (default 1, faster than boot), but is limited by free space on the node and diminishing returns (size^α).
* −1 vCPU: takes effect immediately.

This is a trade-off worth analysing: vertical scaling is fast but bounded, horizontal scaling is slow (due to boot) but scalable.

**Performance:** each VNF has one queue. Load exceeding capacity is carried over to the next step (backlog, up to
`buffer_steps` × capacity); anything beyond that is dropped.

```
latency = base_latency / (1 − min(ρ, 0.98)) + backlog/capacity × step_length
SLA violation  ⇔  latency > sla_ms  or  there is a drop
```

Default values: base latency WEB/IOT/SEC = 20/5/10 ms, SLA = 100/30/60 ms. **These are assumed parameters,
not measured from the data**, and can be changed in `SimConfig`.

**The two main metrics of the thesis**

* **Utilization** = served load / provisioned capacity. Provisioned capacity includes booting instances and
  pending vCPUs, since both are billed.
  * Whole episode: `episode_summary()["utilization"]`.
  * Per step: `info["utilization"]`.
* **Oscillation**: a reversal occurs when a VNF changes its resources in the opposite direction to its most recent
  change and within `osc_window` steps of it (default 10). Example: scale-out followed by scale-in 4 minutes later.
  * `oscillations`: total number of reversals.
  * `oscillation_rate` = reversals / (steps × number of VNFs).
  * `scaling_actions`: total number of scaling operations.

## 4. State – Action – Reward

**Action**, by default `MultiDiscrete([6, 6, 6])`, one choice per VNF:

| idx | meaning |
|---|---|
| 0 | horizontal scale-in −1 (prefers cancelling a booting instance, then removes an instance from the emptiest node) |
| 1 | no-op |
| 2 | horizontal scale-out +1 (best-fit placement; powers on a new node if there is no room) |
| 3 | migrate 1 instance to consolidate nodes (from the emptiest node to the fullest node that has room) |
| 4 | vertical scale-up +1 vCPU |
| 5 | vertical scale-down −1 vCPU |

* `max_step_change=K` allows changing by ±K instances per step. In that case:
  * idx 0..2K are horizontal;
  * idx 2K+1 is migrate;
  * idx 2K+2 and 2K+3 are vertical.
* `enable_vertical=False` removes the two vertical actions (back to 4 actions).
* `discrete_action=True` flattens the action into `Discrete(6^3 = 216)`, for use with DQN.
* Invalid actions (exceeding min/max, no free slot, nothing to consolidate) are ignored and penalised by `w_invalid`.

**Observation**, 55 dimensions with the default configuration:

* Per VNF (17 dimensions):
  * the 6 most recent load values;
  * [H-step forecast, if enabled];
  * number of run/boot/mig instances and total provisioned vCPU;
  * current demand;
  * ρ (utilization);
  * latency/SLA;
  * backlog;
  * memory;
  * direction of the last change and the number of steps since it. These two values let the agent know whether it
    just scaled up or down, so it can avoid reversals, and keep the oscillation-penalty reward Markov.
* Global (4 dimensions): fraction of nodes on, fraction of slots used, sin/cos of the hour of day.

**Reward** (always ≤ 0):

```
r = −( w_sla·Σ SLA_violations + w_drop·Σ drop_ratio + w_latency·Σ latency/SLA        ← SLA constraint
       + w_resource·n_vCPU + w_util·Σ(1 − utilization) + w_energy·energy             ← utilization objective
       + w_osc·n_reversals + w_scale·n_scaling_actions                                ← oscillation objective
       + w_migration·Σ(fixed_cost + mem_cost·mem) + w_invalid·n_invalid_actions )
```

Each term is returned in `info["reward_terms"]`, so it is easy to plot analyses or tune the weights.

## 5. Overload scenarios

Overload scenarios can be generated directly from the real trace (`augment_load`):

```python
env = NFVScalingEnv(data_root=ROOT, split="test", episode_steps=None, random_start=False,
                    load_multiplier=1.5,      # load increased by 50%
                    burst_prob=0.005,         # probability of a burst starting each minute
                    burst_scale=(1.5, 3.0),   # burst ×1.5–3
                    burst_len=(3, 20),        # lasting 3–20 minutes
                    noise_std=0.05)
```

**Train** with augmentation so the agent gets used to overload, then **test** on the original 18-02 trace plus a
fixed overload scenario (same seed) for a fair comparison.

## 6. Baseline results on the test set

Test set = last 30% of SNDZoo (`TRAIN_FRAC = 0.7`), ~6,000 minutes. `python examples\run_baselines.py --forecasters lstm tft`.

| Scenario | Policy | Utilization | Oscillation | Scaling actions | SLA violations | Avg vCPU |
|---|---|---|---|---|---|---|
| normal | static (provisioned for peak) | 0.17 | 0 | 17 | 0.0% | 20.0 |
| normal | threshold (HPA 80%/40%) | 0.48 | 757 | 1674 | 11.0% | 7.0 |
| normal | vertical (VPA) | 0.49 | 695 | 1550 | 9.2% | 7.1 |
| normal | hybrid | 0.50 | 701 | 1547 | 11.1% | 7.3 |
| normal | predictive (oracle) | 0.48 | 966 | 2052 | 4.8% | 7.0 |
| normal | predictive (lstm, P50) | 0.48 | 1415 | 2527 | 8.4% | 7.0 |
| normal | predictive (lstm, P90) | 0.43 | 2026 | 2979 | 2.6% | 7.9 |
| normal | predictive (tft, P50) | 0.48 | 1538 | 2665 | 9.4% | 7.0 |
| normal | predictive (tft, P90) | 0.42 | 2103 | 3123 | 3.2% | 8.0 |
| overload | static | 0.27 | 0 | 17 | 5.2% | 20.0 |
| overload | threshold | 0.53 | 757 | 2005 | 13.3% | 10.1 |
| overload | vertical | 0.54 | 709 | 1859 | 11.5% | 10.2 |
| overload | hybrid | 0.56 | 733 | 1906 | 14.8% | 10.3 |
| overload | predictive (oracle) | 0.53 | 1118 | 2901 | 7.3% | 10.2 |
| overload | predictive (lstm, P50) | 0.55 | 1576 | 3467 | 14.8% | 9.8 |
| overload | predictive (lstm, P90) | 0.52 | 1737 | 3529 | 9.7% | 10.4 |
| overload | predictive (tft, P50) | 0.55 | 1666 | 3495 | 15.6% | 9.8 |
| overload | predictive (tft, P90) | 0.51 | 1875 | 3547 | 9.8% | 10.4 |

The table shows four things:

* Static has almost no SLA violations and no oscillation, but utilization is only 17–27%.
* Threshold and vertical raise utilization to about 50%, but reverse direction constantly and violate the SLA
  often because of boot latency.
* Predictive (perfect forecast) reduces SLA violations, but **oscillates even more** because it tracks every load peak.
* Provisioning on the **P90** of a range forecast instead of the median cuts SLA violations from 8.4% to 2.6%
  (normal) — below the oracle — for about 13% more vCPU, but oscillates the most.

An RL agent has to achieve high utilization, good SLA compliance and low oscillation at the same time. This is the
gap the thesis targets.

## 7. Step 4 — load forecasting (`nfv_sim/forecast/`)

```
python examples\train_forecaster.py                      # train + compare every registered forecaster (~1.5 h on CPU)
python examples\train_forecaster.py --models lstm tft    # only some models
python examples\run_baselines.py --forecasters lstm tft  # adds "predictive(lstm)" and "predictive(lstm,P90)" policies
python examples\train_ppo.py --forecaster lstm --quantiles 0.5 0.9   # P50 + P90 forecast in the RL state
python examples\explain_tft.py                           # TFT variable importance + attention plots
python examples\crosscheck_tft.py                        # one-off check against pytorch-forecasting
```

Every algorithm is a subclass of `Forecaster` (`nfv_sim/forecast/base.py`) registered by name.
Built-in: `naive`, `moving_avg`, `linear` (ridge AR), `lstm` and `tft` (PyTorch, shared training code in
`neural.py`), `oracle` (upper bound only).
Common parameters (`horizon`, `context_len`) live in `FORECAST`, per-model ones in `FORECASTERS` (config.py).
Trained models are pickled to `results/forecasters/<name>.pkl`.

A forecaster plugs into both consumers directly:

```python
from nfv_sim.forecast import load_forecaster
fc = load_forecaster("lstm")
env = NFVScalingEnv(traces=test, forecast_horizon=fc.horizon, forecaster=fc)     # RL state
policy = PredictiveThresholdPolicy(env, forecast_fn=fc.demand_fn())               # baseline
```

**Adding a new algorithm** (Lag-Llama, TimesFM, GRU, …): one class + one config line.

```python
@register_forecaster("timesfm")
class TimesFMForecaster(Forecaster):
    def fit(self, train, val=None):        # optional (zero-shot models skip it)
        return self
    def predict_batch(self, windows):      # (N, L, n_vnf) history -> (N, H, n_vnf) forecast
        ...
```

Import the module in `nfv_sim/forecast/__init__.py` and add `"timesfm": dict(...)` to `FORECASTERS`.
All scripts then accept it by name, and `evaluate_forecaster` reports the same metrics
(MAE/RMSE per horizon step, under-prediction rate, MAE at load peaks, MAE per VNF).

### What changed in Steps 1–5 (TFT integration)

| Step | What | Where |
|---|---|---|
| 1 | Train on several datasets (`train_datasets`, default all of `DATASETS`); evaluate per dataset | `data.load_catalog`, `base.make_training_set`, `metrics.evaluate_by_dataset` |
| 2 | Range forecast P10/P50/P90 with quantile (pinball) loss | `neural.py`, `Forecaster.quantiles`, `predict_dist_batch` |
| 3 | Extra inputs: calendar (hour/day of week as sin/cos) and series/dataset ID embeddings | `SeriesContext.calendar_features`, `base.make_extra`, `lstm.py` `use_context` |
| 4 | Temporal Fusion Transformer written from scratch + one-off cross-check with pytorch-forecasting | `forecast/tft.py`, `examples/explain_tft.py`, `examples/crosscheck_tft.py` |
| 5 | P50 + P90 in the PPO state, P90 provisioning in the predictive baseline | `SimConfig.forecast_quantiles`, `train_ppo.py --quantiles`, `demand_fn(quantile=0.9)` |

`predict_batch(windows, extra=None)` still returns one value per step (the median for range models), so older code
keeps working. `forecast_series(load, t, quantiles=(0.5, 0.9), ctx=...)` returns several quantiles; `ctx`
(a `SeriesContext`) carries the clock and series name, the env passes it automatically.

Variants in `FORECASTERS` (one key per row of the table below, `model="lstm"` reuses the LSTM class):
`lstm_snd` (Step 0: single value, SNDZoo only) → `lstm_point` (Step 1: all datasets) → `lstm_q` (Step 2: range)
→ `lstm` (Step 3: range + extra inputs) → `tft` (Step 4: same data and inputs as `lstm`).

### Results on the SNDZoo test set (H=5, L=30, normalised load)

| Forecaster | MAE | MAE t+1 | MAE t+5 | under-rate | peak MAE | pinball | P90 coverage | P90−P10 | train time |
|---|---|---|---|---|---|---|---|---|---|
| lstm (range + extra inputs) | **0.0389** | 0.0177 | **0.0593** | **11.5%** | **0.096** | **0.0123** | 87.6% | 0.121 | 10 min |
| lstm_q (range) | 0.0395 | **0.0173** | 0.0608 | 11.7% | 0.104 | 0.0127 | 90.7% | 0.120 | 7 min |
| tft | 0.0408 | 0.0206 | 0.0599 | 13.2% | 0.111 | 0.0131 | 91.5% | 0.131 | 63 min |
| lstm_snd (old LSTM) | 0.0421 | 0.0202 | 0.0630 | 12.8% | 0.111 | – | – | – | 4 min |
| lstm_point | 0.0484 | 0.0263 | 0.0678 | 13.3% | 0.109 | – | – | – | 5 min |
| linear | 0.0582 | 0.0343 | 0.0740 | 16.6% | 0.160 | – | – | – | 2 s |
| naive | 0.0645 | 0.0358 | 0.0832 | 18.0% | 0.169 | – | – | – | – |
| moving_avg | 0.0735 | 0.0568 | 0.0856 | 19.7% | 0.193 | – | – | – | – |

MAE of range models is measured on P50. Pinball = average quantile loss (lower is better). P90 coverage = share of
true values below the P90 forecast (ideal 90%). Train time on a 4-core CPU.

MAE on the test part of every dataset (`forecast_metrics_by_dataset.csv`):

| Forecaster | sndzoo | nab_aws | google_machines | alibaba_cluster | cluster_5min |
|---|---|---|---|---|---|
| lstm | **0.0389** | 0.0344 | 0.0415 | 0.0529 | **0.0477** |
| lstm_q | 0.0395 | **0.0293** | **0.0403** | 0.0547 | 0.0508 |
| tft | 0.0408 | 0.0316 | 0.0412 | 0.0529 | 0.0499 |
| lstm_snd | 0.0421 | 0.0696 | 0.0723 | 0.0697 | 0.0700 |
| lstm_point | 0.0484 | 0.0362 | 0.0408 | 0.0554 | 0.0516 |
| naive | 0.0645 | 0.0418 | 0.0452 | 0.0588 | 0.0533 |

Reading the two tables:

* Training a single-value LSTM on all datasets **without** telling it which series it sees (`lstm_point`) makes
  SNDZoo worse (0.0484 vs 0.0421); the 5-minute cluster traces behave differently. The range loss (`lstm_q`) and
  the series/dataset ID (`lstm`) bring it back and below the old LSTM, while one model now also covers every dataset.
* TFT does not beat the LSTM here (0.0408 vs 0.0389 on SNDZoo; close on the other datasets). With a single load
  input, 30 steps of history and 5 calendar features most of TFT's blocks have little to select from; it took
  6× longer to train (best validation loss at epoch 34 of a 60-epoch budget). Its value for the thesis is the explanation below.
* `tft_importance.png`: the past load gets 57–68% of the past-input weight; calendar features share the rest.
  `tft_attention.png`: attention grows towards the most recent steps (the oldest step of the window also gets
  extra weight); on SNDZoo and NAB the future steps also look at the known calendar of earlier future steps.

TFT cross-check against pytorch-forecasting (same data, L, H, quantiles, hidden size 32, 4 heads, 30 epochs):
| Dataset | MAE ours | MAE library | pinball ours | pinball library | P90 coverage ours | P90 coverage library |
|---|---|---|---|---|---|---|
| sndzoo | 0.0408 | 0.0403 | 0.0131 | 0.0130 | 91.5% | 88.5% |
| nab_aws | 0.0316 | 0.0295 | 0.0114 | 0.0107 | 89.5% | 86.5% |
| google_machines | 0.0412 | 0.0399 | 0.0135 | 0.0130 | 88.3% | 88.2% |
| alibaba_cluster | 0.0529 | 0.0534 | 0.0170 | 0.0172 | 88.5% | 88.2% |
| cluster_5min | 0.0499 | 0.0496 | 0.0165 | 0.0164 | 86.5% | 86.6% |

Both land within ~0.002 MAE of each other, so the self-written TFT has no large bug. The library model has
103,569 parameters, ours 131,660. Rerun: `pip install pytorch-forecasting`, then `python examples\crosscheck_tft.py`
(about 75 min on a 4-core CPU); the table is saved to `results/forecasters/crosscheck_tft.csv`.

### PPO with forecasts in the state (Step 5)

`python examples\train_ppo.py --forecaster lstm --quantiles 0.5 0.9` adds 5 P50 + 5 P90 values per VNF to the
state (obs 85 dims instead of 55 + 15).
How to run and compare (one configuration per line, repeat each with `--seed 0`, `--seed 1`, `--seed 2`):

```bat
python examples\train_ppo.py --seed 0                                         :: no forecast
python examples\train_ppo.py --forecast 5 --seed 0                            :: oracle (upper bound)
python examples\train_ppo.py --forecaster lstm --seed 0                       :: LSTM P50
python examples\train_ppo.py --forecaster lstm --quantiles 0.5 0.9 --seed 0   :: LSTM P50 + P90
python examples\train_ppo.py --forecaster tft --quantiles 0.5 0.9 --seed 0    :: TFT P50 + P90
python examples\compare_ppo.py                                                :: mean ± std over seeds
```

Each run (300k steps) takes about 5 min with LSTM and 7 min with TFT on a 4-core CPU and writes `results/ppo_eval<tag>.csv`; `compare_ppo.py` groups
the files by configuration (`_s1`, `_s2` are merged) and saves `results/ppo_compare.csv`. Compare first
`sla_violation_rate` (lower is better), then `avg_vcpu` (cost), then `oscillations`. Differences smaller than the
std over seeds are noise.

If `forecast_horizon>0` and no forecaster is passed, the env uses an **oracle** (the true future values).
The oracle is only for measuring an upper bound and must not be reported as a real result.

## 8. Commonly tuned parameters (`configs/config.py`)

Everything is in one file, grouped as:

* paths: `DATA_ROOT`, `RESULTS_DIR`;
* data: `VNF_NAMES`, `SPLIT_DIRS`, …;
* simulator: `SimConfig`;
* evaluation scenarios: `SCENARIOS`, `EVAL_SEED`;
* baselines: `BASELINE`;
* training: `TRAIN`, `TRAIN_AUGMENT`, `PPO`;
* forecasting: `FORECAST` (horizon, context_len, quantiles, train/eval datasets), `FORECASTERS` (models and variants);
* extra datasets: `EXTERNAL_ROOT`, `DATASETS` (`VNF_GYM_DATA_ROOT` / `VNF_GYM_EXTERNAL_ROOT` environment variables override the paths);
* `play.py`: `PLAY`.

To try a value without editing the file, override it when creating the env: `NFVScalingEnv(traces=..., w_osc=1.0)`.

Most-tuned `SimConfig` fields:

| Group | Parameters |
|---|---|
| Time | `step_minutes` (aggregate k minutes per step, taking the max load within the step), `episode_steps` |
| Forecast in state | `forecast_horizon`, `forecast_quantiles` (e.g. `(0.5, 0.9)`) |
| Infrastructure | `n_nodes`, `node_slots`, `min_instances`, `max_instances` |
| Delays | `scale_out_delay`, `vertical_delay`, `migration_base_delay`, `migration_mem_delay`, `migration_capacity` |
| Vertical | `enable_vertical`, `max_vcpu`, `vertical_alpha` |
| Oscillation | `osc_window`, `w_osc`, `w_scale` |
| Cost | `w_*`, `migration_fixed_cost`, `migration_mem_cost`, `p_idle`, `p_max` |
| SLA | `base_latency_ms`, `sla_ms`, `buffer_steps` |

## 9. Limitations to state in the thesis

* The absolute load scale is lost because the data is normalised. The peak instance counts (`peak_instances`) are assumptions.
* Vertical efficiency (size^α, α = 0.9) and vertical delay are assumptions, not measured from SNDZoo.
* The latency model is an M/M/1 queueing approximation plus backlog, not a packet-level simulation.
* The three VNFs in SNDZoo are three independent services, not an SFC chain. They only share infrastructure
  (nodes), so their interaction is limited to energy and migration.
* The data contains about 2 hours of near-zero load (08:00–10:00, due to an experiment restart), followed by a
  sudden surge at 10:00. This accidentally provides a good "sudden load spike" scenario for testing.
