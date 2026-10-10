"""Chạy simulator từng bước bằng tay để hiểu nó hoạt động thế nào.

    python examples/play.py            (tham số lấy từ configs/config.py)

Mỗi bước nhập 3 số (cho WEB IOT SEC), cách nhau bởi dấu cách:
    0 = scale-in   1 = giữ   2 = scale-out   3 = migrate   4 = vertical +1 vCPU   5 = vertical -1 vCPU
Enter trống = giữ nguyên cả 3.  q = thoát.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):  # console Windows (cp1252) không in được tiếng Việt
    sys.stdout.reconfigure(encoding="utf-8")

from nfv_sim import NFVScalingEnv, load_sndzoo
from nfv_sim.config import DATA_ROOT, PLAY

p = argparse.ArgumentParser()
p.add_argument("--data", default=DATA_ROOT, help="thư mục DatasetSNDZoo (mặc định lấy từ config.py)")
p.add_argument("--start", type=int, default=PLAY["start"], help="phút bắt đầu trong ngày test")
args = p.parse_args()

env = NFVScalingEnv(traces=load_sndzoo(args.data, "test"), episode_steps=None, random_start=False)
obs, info = env.reset(seed=PLAY["seed"], options={"start": args.start})
print(env.render())

while True:
    s = input("\naction [WEB IOT SEC] (vd: 2 1 4) > ").strip()
    if s.lower() == "q":
        break
    acts = [int(x) for x in s.split()] if s else [1, 1, 1]
    if len(acts) != env.n_vnf or not all(0 <= a < env.n_act_per_vnf for a in acts):
        print(f"Cần {env.n_vnf} số trong khoảng 0..{env.n_act_per_vnf - 1}")
        continue
    obs, r, term, trunc, info = env.step(acts)
    print(env.render())
    print(f"  utilization={[round(float(u), 2) for u in info['utilization']]}  "
          f"vi phạm SLA={info['sla_violation'].astype(int).tolist()}  "
          f"đảo chiều={info['oscillation'].astype(int).tolist()}")
    print(f"  reward={r:.3f}  " + "  ".join(f"{k}={v:.2f}" for k, v in info["reward_terms"].items() if v))
    if term or trunc:
        break

print("\nTổng kết:", env.episode_summary())
