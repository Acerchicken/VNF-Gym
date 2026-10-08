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

**Configure**: all parameters live in [`nfv_sim/config.py`](nfv_sim/config.py). Set `DATA_ROOT` there to your
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

| Split | Folder | Length |
|---|---|---|
| train | `<VNF>/TrainTest/NoHoles/*.csv` | ~20,000 minutes (27/01 → 10/02/2025) |
| test  | `<VNF>/18-02/CSVmergedFIXED/*.csv` | ~1,300 minutes (17/02 10:00 → 18/02 07:47) |

* **Incoming load** (`load_metric="rx"`, the default) is `container_network_receive_bytes_total`, normalised to [0,1].
  It can be switched to `"cpu"`.
* **Conversion to instance count:** `demand_t = load_t × peak_instances[VNF]`. Defaults are WEB=6, IOT=4, SEC=5,
  meaning that at the trace's peak load about 6 WEB instances are needed running at 100%. The data was min-max
  normalised, so the absolute scale is lost; this is therefore an **assumption that must be stated explicitly in the thesis**.
* **Memory working set** (`mem`) is used to compute **migration time and cost**.
  A VNF using more RAM takes longer and costs more to migrate (modelling pre-copy live migration).
* The tx/cpu columns are loaded as well, so you can use them as features or change `load_metric`.

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

## 6. Baseline results on the test day (trial run)

| Scenario | Policy | Utilization | Oscillation | Scaling actions | SLA violations | Avg vCPU |
|---|---|---|---|---|---|---|
| normal | static (provisioned for peak) | 0.20 | 0 | 17 | 0.2% | 20.0 |
| normal | threshold (HPA 80%/40%) | 0.51 | 166 | 370 | 10.3% | 8.0 |
| normal | vertical (VPA) | 0.51 | 160 | 348 | 8.8% | 8.0 |
| normal | hybrid | 0.53 | 182 | 389 | 13.2% | 8.0 |
| normal | predictive (oracle) | 0.51 | 238 | 496 | 5.6% | 7.8 |
| overload | static | 0.32 | 0 | 17 | 6.9% | 20.0 |
| overload | threshold | 0.55 | 192 | 490 | 15.1% | 11.6 |
| overload | vertical | 0.56 | 183 | 458 | 13.2% | 11.8 |
| overload | hybrid | 0.59 | 180 | 441 | 16.6% | 11.8 |
| overload | predictive (oracle) | 0.55 | 254 | 700 | 8.4% | 11.5 |

The table shows three things:

* Static has almost no SLA violations and no oscillation, but utilization is only 20–30%.
* Threshold and vertical raise utilization to about 50%, but reverse direction constantly (about once every
  8 minutes) and violate the SLA often because of boot latency.
* Predictive (perfect forecast) reduces SLA violations, but **oscillates even more** because it tracks every load peak.

An RL agent has to achieve high utilization, good SLA compliance and low oscillation at the same time. This is the
gap the thesis targets.

## 7. Step 4 — load forecasting (`nfv_sim/forecast/`)

```
python examples\train_forecaster.py                 # train + compare every registered forecaster
python examples\run_baselines.py --forecasters lstm  # adds a "predictive(lstm)" policy
python examples\train_ppo.py --forecaster lstm       # forecast goes into the RL state
```

Every algorithm is a subclass of `Forecaster` (`nfv_sim/forecast/base.py`) registered by name.
Built-in: `naive`, `moving_avg`, `linear` (ridge AR), `lstm` (PyTorch), `oracle` (upper bound only).
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

First results on the test day (H=5, L=30, normalised load):

| Forecaster | MAE | MAE t+1 | MAE t+5 | under-rate | peak MAE | predictive policy: oscillation / SLA viol. |
|---|---|---|---|---|---|---|
| lstm | 0.044 | 0.021 | 0.066 | 14.0% | 0.106 | 368 / 9.3% |
| linear | 0.057 | 0.029 | 0.077 | 17.4% | 0.139 | 422 / 12.7% |
| naive | 0.071 | 0.041 | 0.090 | 19.9% | 0.160 | 395 / 20.0% |
| moving_avg | 0.080 | 0.062 | 0.093 | 21.7% | 0.183 | – |
| oracle | 0 | 0 | 0 | 0 | 0 | 238 / 5.6% |

If `forecast_horizon>0` and no forecaster is passed, the env uses an **oracle** (the true future values).
The oracle is only for measuring an upper bound and must not be reported as a real result.

## 8. Commonly tuned parameters (`nfv_sim/config.py`)

Everything is in one file, grouped as:

* paths: `DATA_ROOT`, `RESULTS_DIR`;
* data: `VNF_NAMES`, `SPLIT_DIRS`, …;
* simulator: `SimConfig`;
* evaluation scenarios: `SCENARIOS`, `EVAL_SEED`;
* baselines: `BASELINE`;
* training: `TRAIN`, `TRAIN_AUGMENT`, `PPO`;
* forecasting: `FORECAST`, `FORECASTERS`;
* `play.py`: `PLAY`.

To try a value without editing the file, override it when creating the env: `NFVScalingEnv(traces=..., w_osc=1.0)`.

Most-tuned `SimConfig` fields:

| Group | Parameters |
|---|---|
| Time | `step_minutes` (aggregate k minutes per step, taking the max load within the step), `episode_steps` |
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
