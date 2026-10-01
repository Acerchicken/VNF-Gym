"""Huấn luyện PPO (Stable-Baselines3) trên tập train, đánh giá trên ngày test 18-02.

    pip install stable-baselines3
    python examples/train_ppo.py --data "E:/SNDZoo dataset/SNDZoo dataset/DatasetSNDZoo" --steps 300000

Mẹo:
* Train với augmentation (burst, nhân tải) để agent học xử lý quá tải; test trên trace gốc.
* --forecast 5 thêm dự báo 5 bước vào state (mặc định oracle -> cận trên). Ở Bước 4 hãy
  truyền forecaster thật (Lag-Llama/TimesFM/LSTM) qua tham số `forecaster=` của env.
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

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--steps", type=int, default=300_000)
p.add_argument("--forecast", type=int, default=0)
p.add_argument("--n-envs", type=int, default=8)
p.add_argument("--out", default="results")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

train = load_sndzoo(args.data, "train")
test = load_sndzoo(args.data, "test")

train_kw = dict(traces=train, episode_steps=720, random_start=True, forecast_horizon=args.forecast,
                load_multiplier=1.2, noise_std=0.05, burst_prob=0.003)
venv = make_vec_env(lambda: NFVScalingEnv(**train_kw), n_envs=args.n_envs, seed=0)
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=0.99)

model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=256, gamma=0.99, gae_lambda=0.95,
            learning_rate=3e-4, ent_coef=0.01, verbose=1, seed=0)
model.learn(total_timesteps=args.steps)
model.save(os.path.join(args.out, "ppo_nfv"))
venv.save(os.path.join(args.out, "vecnormalize.pkl"))

# ---------------- đánh giá trên ngày test ----------------
venv.training = False
venv.norm_reward = False


def ppo_policy(obs):
    a, _ = model.predict(venv.normalize_obs(obs), deterministic=True)
    return a


for sname, kw in {"normal": {}, "overload": dict(load_multiplier=1.5, burst_prob=0.005)}.items():
    env = NFVScalingEnv(traces=test, episode_steps=None, random_start=False,
                        forecast_horizon=args.forecast, **kw)
    s_ppo, _ = run_episode(env, ppo_policy, seed=1)
    env2 = NFVScalingEnv(traces=test, episode_steps=None, random_start=False,
                         forecast_horizon=args.forecast, **kw)
    s_thr, _ = run_episode(env2, ThresholdPolicy(env2), seed=1)
    print(f"\n=== {sname} ===")
    for k in s_ppo:
        print(f"{k:24s} PPO={s_ppo[k]!s:>10}   threshold={s_thr[k]!s:>10}")
