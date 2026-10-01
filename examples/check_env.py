"""Kiểm tra env đúng chuẩn Gymnasium và chạy thử 1 episode ngẫu nhiên.

    python examples/check_env.py --data "E:/SNDZoo dataset/SNDZoo dataset/DatasetSNDZoo"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):  # console Windows (cp1252) không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")

from gymnasium.utils.env_checker import check_env

from nfv_sim import NFVScalingEnv, load_sndzoo

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="đường dẫn thư mục DatasetSNDZoo")
args = p.parse_args()

train = load_sndzoo(args.data, "train")
test = load_sndzoo(args.data, "test")
print(f"train: {len(train)} phút ({train.index[0]} -> {train.index[-1]})")
print(f"test : {len(test)} phút ({test.index[0]} -> {test.index[-1]})")

env = NFVScalingEnv(traces=train)
check_env(env, skip_render_check=True)
print("check_env: OK")
print("observation:", env.observation_space.shape, "| action:", env.action_space)

obs, info = env.reset(seed=0)
for k in range(5):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    print(f"step {k}: reward={r:.3f} terms={ {a: round(b, 3) for a, b in info['reward_terms'].items()} }")
print(env.render())
