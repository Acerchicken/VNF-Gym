"""Huấn luyện PPO (Stable-Baselines3) trên tập train, đánh giá trên ngày test 18-02.

    pip install stable-baselines3
    python examples/train_ppo.py            (tham số lấy từ configs/config.py: TRAIN, TRAIN_AUGMENT, PPO)

Mẹo:
* Train với augmentation (burst, nhân tải) để agent học xử lý quá tải; test trên trace gốc.
* --forecaster lstm thêm dự báo của model đã train (examples/train_forecaster.py) vào state,
  horizon lấy từ FORECAST["horizon"]. Dùng tên bất kỳ đã đăng ký trong nfv_sim/forecast.
* --forecast 5 không kèm --forecaster: dự báo oracle 5 bước (chỉ là cận trên).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):  # console Windows (cp1252) không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from nfv_sim import NFVScalingEnv, ThresholdPolicy, load_sndzoo, run_episode
from nfv_sim.config import (DATA_ROOT, EVAL_SEED, FORECAST, PPO as PPO_KW, RESULTS_DIR, SCENARIOS, TRAIN,
                            TRAIN_AUGMENT)
from nfv_sim.forecast import load_forecaster

p = argparse.ArgumentParser()
p.add_argument("--data", default=DATA_ROOT, help="thư mục DatasetSNDZoo (mặc định lấy từ config.py)")
p.add_argument("--steps", type=int, default=TRAIN["total_steps"])
p.add_argument("--forecast", type=int, default=TRAIN["forecast_horizon"])
p.add_argument("--forecaster", default=TRAIN["forecaster"], help="tên forecaster đã train (vd: lstm, linear)")
p.add_argument("--n-envs", type=int, default=TRAIN["n_envs"])
p.add_argument("--out", default=RESULTS_DIR)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

fc = None
if args.forecaster:
    fc = load_forecaster(args.forecaster)
    args.forecast = args.forecast or FORECAST["horizon"]
    print(f"Forecaster: {fc} -> state gồm dự báo {args.forecast} bước")
tag = f"_{args.forecaster}" if args.forecaster else ""

train = load_sndzoo(args.data, "train")
test = load_sndzoo(args.data, "test")

train_kw = dict(traces=train, episode_steps=TRAIN["episode_steps"], random_start=True,
                forecast_horizon=args.forecast, forecaster=fc, **TRAIN_AUGMENT)
venv = make_vec_env(lambda: NFVScalingEnv(**train_kw), n_envs=args.n_envs, seed=TRAIN["seed"])
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=PPO_KW["gamma"])

model = PPO("MlpPolicy", venv, verbose=1, seed=TRAIN["seed"], **PPO_KW)
model.learn(total_timesteps=args.steps)
model.save(os.path.join(args.out, f"ppo_nfv{tag}"))
venv.save(os.path.join(args.out, f"vecnormalize{tag}.pkl"))

# ---------------- đánh giá trên ngày test ----------------
venv.training = False
venv.norm_reward = False


def ppo_policy(obs):
    a, _ = model.predict(venv.normalize_obs(obs), deterministic=True)
    return a


for sname, kw in SCENARIOS.items():
    env = NFVScalingEnv(traces=test, episode_steps=None, random_start=False,
                        forecast_horizon=args.forecast, forecaster=fc, **kw)
    s_ppo, _ = run_episode(env, ppo_policy, seed=EVAL_SEED)
    env2 = NFVScalingEnv(traces=test, episode_steps=None, random_start=False,
                         forecast_horizon=args.forecast, **kw)
    s_thr, _ = run_episode(env2, ThresholdPolicy(env2), seed=EVAL_SEED)
    print(f"\n=== {sname} ===")
    for k in s_ppo:
        print(f"{k:24s} PPO={s_ppo[k]!s:>10}   threshold={s_thr[k]!s:>10}")
