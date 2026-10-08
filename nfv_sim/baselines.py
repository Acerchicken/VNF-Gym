"""Các policy so sánh (baseline) để đối chiếu với agent RL."""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np

from .config import BASELINE
from .env import NFVScalingEnv


class StaticPolicy:
    """Giữ nguyên số instance (cấp phát tĩnh theo đỉnh) — không scale."""

    def __init__(self, env: NFVScalingEnv, target: Optional[Dict[str, int]] = None):
        self.env = env
        self.target = target or {v: int(math.ceil(env.peak[j] / BASELINE["static_target_util"])) for j, v in enumerate(env.vnfs)}

    def __call__(self, obs):
        deltas = []
        for st in self.env.vnf_status():
            have = st["run"] + st["boot"] + st["mig"]
            deltas.append(int(np.sign(self.target[st["name"]] - have)))
        return self.env.delta_to_action(deltas)


class ThresholdPolicy:
    """Autoscaler kiểu Kubernetes HPA: scale up khi utilization > upper, scale down khi < lower,
    có cooldown. Tuỳ chọn tự gom node (migrate) khi phân mảnh.

    mode="horizontal": thêm/bớt instance (HPA).
    mode="vertical"  : thêm/bớt vCPU (VPA); hết chỗ trên node thì mới thêm instance.
    mode="hybrid"    : tăng thì ưu tiên vertical (nhanh, không phải boot), giảm thì ưu tiên
                       gỡ cả instance (giải phóng slot, dễ tắt node), chỉ bớt vCPU khi đã ở
                       min_instances."""

    def __init__(self, env: NFVScalingEnv, upper: float = BASELINE["upper"],
                 lower: float = BASELINE["lower"], cooldown: int = BASELINE["cooldown"],
                 consolidate: bool = False, mode: str = "horizontal"):
        self.env, self.upper, self.lower, self.cooldown = env, upper, lower, cooldown
        self.consolidate, self.mode = consolidate, mode
        self.last = [-10**9] * env.n_vnf

    def __call__(self, obs):
        env = self.env
        deltas, mig, vert = [], [], []
        used = env.node_usage()
        on = used[used > 0]
        fragmented = len(on) > 1 and on.sum() <= (len(on) - 1) * env.cfg.node_slots
        for j, st in enumerate(env.vnf_status()):
            cap = st["alloc"]                      # tính cả instance đang boot / vCPU đang chờ
            util = st["demand"] / max(cap, 1e-6)
            d, m, v = 0, False, 0
            if env.t - self.last[j] >= self.cooldown:
                if util > self.upper or st["backlog"] > 0:
                    if self.mode in ("vertical", "hybrid") and st["can_vup"]:
                        v = 1
                    else:
                        d = 1
                elif cap > 1 and st["demand"] / (cap - 1) < self.lower * 1.5 and util < self.lower:
                    n_inst = st["run"] + st["boot"] + st["mig"]
                    prefer_v = self.mode == "vertical" or (
                        self.mode == "hybrid" and n_inst <= env.cfg.min_instances)
                    if prefer_v and st["can_vdown"]:
                        v = -1
                    else:
                        d = -1
            if d != 0 or v != 0:
                self.last[j] = env.t
            elif self.consolidate and fragmented and st["mig"] == 0 and env.t % BASELINE["consolidate_every"] == j:
                m = True
            deltas.append(d); mig.append(m); vert.append(v)
        return env.delta_to_action(deltas, mig, vert)


class PredictiveThresholdPolicy(ThresholdPolicy):
    """Giống ThresholdPolicy nhưng ra quyết định theo tải dự báo `lookahead` bước sau
    (bù cho độ trễ boot). forecast_fn(env) -> mảng (n_vnf,) demand dự báo (đơn vị instance).
    Mặc định dùng oracle từ trace (cận trên của một policy dự báo hoàn hảo)."""

    def __init__(self, env: NFVScalingEnv, forecast_fn: Optional[Callable] = None, **kw):
        super().__init__(env, **kw)
        self.forecast_fn = forecast_fn or self._oracle

    def _oracle(self, env):
        L = env.cfg.scale_out_delay
        t = min(env.t0 + env.t + L, len(env._load) - 1)
        lo = env.t0 + env.t
        return env._load[lo:t + 1].max(axis=0) * env.peak

    def __call__(self, obs):
        env = self.env
        fc = self.forecast_fn(env)
        deltas = []
        for j, st in enumerate(env.vnf_status()):
            have = st["vcpu"]
            need = int(math.ceil(fc[j] / self.upper)) if fc[j] > 0 else env.cfg.min_instances
            need = int(np.clip(need, env.cfg.min_instances, env.cfg.max_instances))
            deltas.append(int(np.sign(need - have)) if (need > have or env.t - self.last[j] >= self.cooldown) else 0)
            if deltas[-1] != 0:
                self.last[j] = env.t
        return env.delta_to_action(deltas)


def run_episode(env: NFVScalingEnv, policy, seed: Optional[int] = None,
                options: Optional[dict] = None, record: bool = False):
    obs, info = env.reset(seed=seed, options=options)
    if hasattr(policy, "last"):
        policy.last = [-10**9] * env.n_vnf
    trace = []
    done = False
    while not done:
        a = policy(obs)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        if record:
            trace.append(dict(t=info["t"], reward=r, demand=info["demand"].copy(),
                              capacity=info["capacity"].copy(), latency=info["latency_ms"].copy(),
                              n_instances=info["n_instances"], active_nodes=info["active_nodes"]))
    return env.episode_summary(), trace
