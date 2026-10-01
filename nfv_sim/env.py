"""
NFVScalingEnv — simulator nhẹ (chuẩn Gymnasium) cho bài toán auto-scaling VNF,
lái bằng trace thật của bộ SNDZoo (WEB/Nginx, IOT/Mosquitto, SEC/Snort).

Mô hình
-------
* Hạ tầng: `n_nodes` máy chủ, mỗi máy có `node_slots` vCPU (slot).
  Node bật khi có ít nhất 1 instance (đang boot/chạy/migrate) và tắt khi trống.
* Mỗi VNF có một nhóm instance; mỗi instance có kích thước `size` vCPU (1..max_vcpu).
  Tải đến lambda_t (đơn vị "vCPU-tương-đương") = trace_chuẩn_hoá_t * peak_instances[v].
  Một instance `size` vCPU đang chạy xử lý được size**vertical_alpha đơn vị
  (alpha<1: scale-up có hiệu suất giảm dần, giả định cần ghi rõ).
* Horizontal scale-out CÓ ĐỘ TRỄ: instance mới (1 vCPU) ở trạng thái BOOT `scale_out_delay`
  bước rồi mới phục vụ, nhưng tốn chi phí ngay từ lúc tạo.
* Horizontal scale-in: gỡ ngay (ưu tiên huỷ instance đang boot, sau đó lấy instance trên
  node vắng nhất để dễ gom node).
* Vertical scale-up: thêm 1 vCPU cho 1 instance đang chạy, cần node của nó còn slot trống;
  có hiệu lực sau `vertical_delay` bước (nhanh hơn boot) và slot bị giữ ngay.
  Vertical scale-down: bớt 1 vCPU, có hiệu lực ngay.
* Migration (gom node / consolidation): chuyển 1 instance từ node vắng nhất sang node đầy nhất
  còn chỗ. Trong lúc migrate (thời gian phụ thuộc memory working set của VNF) instance chiếm
  slot ở CẢ HAI node và chỉ phục vụ `migration_capacity` phần công suất; có chi phí cố định
  + chi phí theo memory. Lợi ích: node nguồn trống -> tắt -> giảm năng lượng.
* Hiệu năng: hàng đợi có backlog. Phần tải vượt công suất bị dồn sang bước sau (tối đa
  `buffer_steps` lần công suất), phần còn lại bị drop. Độ trễ xấp xỉ
  latency = base / (1 - min(rho, rho_cap)) + thời gian chờ do backlog.
  Vi phạm SLA khi latency > sla_ms hoặc có drop.
* Utilization (mỗi VNF) = tải được phục vụ / công suất đã cấp (tính cả instance đang boot
  và vCPU đang chờ vertical, vì đều đang bị tính tiền).
* Oscillation: một "lần đảo chiều" của VNF j xảy ra khi hướng thay đổi tài nguyên ở bước này
  (tăng/giảm, horizontal hoặc vertical) ngược với hướng thay đổi gần nhất và cách nó
  không quá `osc_window` bước (ví dụ scale-out rồi scale-in sau 4 phút).

Action (mặc định MultiDiscrete([2K+4] * n_vnf), K = max_step_change):
  idx 0..2K -> horizontal: thay đổi số instance delta = idx - K  (K=1: 0=-1, 1=giữ, 2=+1)
  idx 2K+1  -> migrate 1 instance của VNF đó để gom node
  idx 2K+2  -> vertical scale-up (+1 vCPU)     (chỉ khi enable_vertical=True)
  idx 2K+3  -> vertical scale-down (-1 vCPU)   (chỉ khi enable_vertical=True)
  `discrete_action=True` -> làm phẳng thành Discrete((2K+4)^n_vnf) cho DQN.

Reward = -( w_sla*Σ vi_phạm + w_drop*Σ tỉ_lệ_drop + w_latency*Σ latency/sla
            + w_resource*số_vCPU_đã_cấp + w_util*Σ(1 - utilization)
            + w_energy*năng_lượng_chuẩn_hoá + w_migration*chi_phí_migration
            + w_scale*số_thao_tác_scale + w_osc*số_lần_đảo_chiều
            + w_invalid*số_action_không_hợp_lệ )
Mọi thành phần nằm trong info["reward_terms"] để phân tích.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .data import TraceSet, augment_load, load_sndzoo, VNF_NAMES

BOOT, RUN, MIG = 0, 1, 2


@dataclass
class SimConfig:
    # ---- dữ liệu ----
    data_root: Optional[str] = None          # đường dẫn DatasetSNDZoo (nếu không truyền traces)
    split: str = "train"                     # "train" | "test"
    vnfs: Sequence[str] = VNF_NAMES
    load_metric: str = "rx"                  # "rx" (traffic vào) hoặc "cpu"
    step_minutes: int = 1                    # 1 bước quyết định = bao nhiêu phút dữ liệu
    episode_steps: Optional[int] = 720       # None = chạy hết chuỗi
    random_start: bool = True                # train: chọn điểm bắt đầu ngẫu nhiên

    # ---- quy đổi tải ----
    peak_instances: Dict[str, float] = field(
        default_factory=lambda: {"WEB": 6.0, "IOT": 4.0, "SEC": 5.0})
    # ---- kịch bản quá tải (augmentation) ----
    load_multiplier: float = 1.0
    noise_std: float = 0.0
    burst_prob: float = 0.0
    burst_scale: tuple = (1.5, 3.0)
    burst_len: tuple = (3, 20)

    # ---- hạ tầng ----
    n_nodes: int = 6
    node_slots: int = 4                      # vCPU mỗi node
    min_instances: int = 1
    max_instances: int = 10
    initial_instances: Optional[Dict[str, int]] = None   # None -> min_instances

    # ---- động học scale / migration ----
    scale_out_delay: int = 3                 # số bước boot trước khi phục vụ
    max_step_change: int = 1                 # K: mỗi bước thay đổi tối đa ±K instance
    enable_vertical: bool = True             # thêm 2 action vertical up/down
    max_vcpu: int = 4                        # vCPU tối đa của 1 instance (<= node_slots)
    vertical_delay: int = 1                  # số bước trước khi vCPU thêm có hiệu lực
    vertical_alpha: float = 0.9              # công suất instance = size**alpha
    migration_base_delay: int = 1            # bước
    migration_mem_delay: int = 3             # + ceil(mem_norm * giá trị này) bước
    migration_capacity: float = 0.5          # công suất phục vụ khi đang migrate
    migration_fixed_cost: float = 0.3
    migration_mem_cost: float = 0.7          # * mem_norm của VNF tại thời điểm migrate

    # ---- mô hình hiệu năng ----
    base_latency_ms: Dict[str, float] = field(
        default_factory=lambda: {"WEB": 20.0, "IOT": 5.0, "SEC": 10.0})
    sla_ms: Dict[str, float] = field(
        default_factory=lambda: {"WEB": 100.0, "IOT": 30.0, "SEC": 60.0})
    rho_cap: float = 0.98
    buffer_steps: float = 2.0                # backlog tối đa = buffer_steps * công suất

    # ---- năng lượng ----
    p_idle: float = 100.0                    # W mỗi node bật
    p_max: float = 200.0

    # ---- oscillation ----
    osc_window: int = 10                     # đảo chiều trong vòng N bước bị tính là dao động

    # ---- trọng số reward ----
    w_sla: float = 1.0
    w_drop: float = 2.0
    w_latency: float = 0.05
    w_resource: float = 0.03                 # mỗi vCPU đã cấp
    w_util: float = 0.2                      # * Σ_j (1 - utilization_j)
    w_energy: float = 0.5
    w_migration: float = 1.0
    w_scale: float = 0.02                    # mỗi thao tác scale thành công
    w_osc: float = 0.5                       # mỗi lần đảo chiều (oscillation)
    w_invalid: float = 0.1

    # ---- quan sát ----
    history_len: int = 6
    forecast_horizon: int = 0                # >0: thêm dự báo vào observation
    discrete_action: bool = False


class _Inst:
    __slots__ = ("vnf", "node", "state", "timer", "dest", "size", "grow", "vtimer")

    def __init__(self, vnf: int, node: int, state: int, timer: int = 0, dest: int = -1, size: int = 1):
        self.vnf, self.node, self.state, self.timer, self.dest = vnf, node, state, timer, dest
        self.size = size      # vCPU đang hoạt động
        self.grow = 0         # vCPU đang chờ vertical scale-up (đã giữ slot)
        self.vtimer = 0


class NFVScalingEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        config: Optional[SimConfig] = None,
        traces: Optional[TraceSet] = None,
        forecaster: Optional[Callable[["NFVScalingEnv", int], np.ndarray]] = None,
        render_mode: Optional[str] = None,
        **overrides,
    ):
        """
        config     : SimConfig (có thể ghi đè từng trường qua **overrides)
        traces     : TraceSet đã nạp sẵn (tiết kiệm thời gian khi tạo nhiều env)
        forecaster : f(env, t_idx) -> mảng (forecast_horizon, n_vnf) tải chuẩn hoá dự báo
                     cho các bước t_idx+1..t_idx+H. Mặc định (nếu horizon>0) dùng
                     dự báo "oracle" từ chính trace — chỉ để thử nghiệm cận trên.
                     Ở Bước 4, thay bằng Lag-Llama/TimesFM/LSTM.
        """
        super().__init__()
        cfg = config or SimConfig()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise AttributeError(f"SimConfig không có trường '{k}'")
            setattr(cfg, k, v)
        if cfg.max_vcpu > cfg.node_slots:
            raise ValueError("max_vcpu phải <= node_slots")
        self.cfg = cfg
        self.render_mode = render_mode
        self.forecaster = forecaster

        if traces is None:
            if cfg.data_root is None:
                raise ValueError("Cần truyền data_root (thư mục DatasetSNDZoo) hoặc traces.")
            traces = load_sndzoo(cfg.data_root, cfg.split, cfg.vnfs)
        self.traces = traces.aggregate(cfg.step_minutes)
        self.vnfs = list(self.traces.vnfs)
        self.n_vnf = len(self.vnfs)
        if cfg.load_metric not in self.traces.values:
            raise ValueError(f"load_metric '{cfg.load_metric}' không có trong traces")

        self._base_load = self.traces.values[cfg.load_metric]
        self._mem = self.traces.values.get("mem", np.zeros_like(self._base_load))
        self._mod = self.traces.minute_of_day
        self.peak = np.array([cfg.peak_instances.get(v, 5.0) for v in self.vnfs], np.float32)
        self.base_lat = np.array([cfg.base_latency_ms.get(v, 10.0) for v in self.vnfs], np.float32)
        self.sla = np.array([cfg.sla_ms.get(v, 50.0) for v in self.vnfs], np.float32)

        T = len(self._base_load)
        self.ep_len = T - 1 if cfg.episode_steps is None else min(cfg.episode_steps, T - 1)
        if self.ep_len < 1:
            raise ValueError("Chuỗi dữ liệu quá ngắn so với episode_steps.")

        # ----- action space -----
        K = cfg.max_step_change
        self.migrate_idx = 2 * K + 1
        self.vup_idx = 2 * K + 2 if cfg.enable_vertical else -1
        self.vdown_idx = 2 * K + 3 if cfg.enable_vertical else -1
        self.n_act_per_vnf = 2 * K + (4 if cfg.enable_vertical else 2)
        if cfg.discrete_action:
            self.action_space = spaces.Discrete(self.n_act_per_vnf ** self.n_vnf)
        else:
            self.action_space = spaces.MultiDiscrete([self.n_act_per_vnf] * self.n_vnf)

        # ----- observation space -----
        self.per_vnf_dim = cfg.history_len + cfg.forecast_horizon + 11
        self.global_dim = 4
        dim = self.n_vnf * self.per_vnf_dim + self.global_dim
        self.observation_space = spaces.Box(0.0, np.inf, shape=(dim,), dtype=np.float32)
        # (dùng [0, inf) vì tải có thể vượt 1 khi có burst; các trường còn lại đã chuẩn hoá ~[0,1])

        self._load = self._base_load
        self.insts: List[_Inst] = []

    # ------------------------------------------------------------------ helpers
    def encode_action(self, per_vnf: Sequence[int]):
        if not self.cfg.discrete_action:
            return np.asarray(per_vnf, dtype=np.int64)
        a = 0
        for x in reversed(per_vnf):
            a = a * self.n_act_per_vnf + int(x)
        return a

    def decode_action(self, action) -> np.ndarray:
        if self.cfg.discrete_action:
            a, out = int(action), []
            for _ in range(self.n_vnf):
                out.append(a % self.n_act_per_vnf)
                a //= self.n_act_per_vnf
            return np.array(out)
        return np.asarray(action, dtype=np.int64).reshape(self.n_vnf)

    def delta_to_action(self, deltas: Sequence[int], migrate: Sequence[bool] = None,
                        vertical: Sequence[int] = None):
        """Tiện ích cho policy heuristic: chuyển delta instance, cờ migrate và hướng vertical
        (+1 = up, -1 = down, 0 = không) thành action. Thứ tự ưu tiên: migrate > vertical > horizontal."""
        K = self.cfg.max_step_change
        idx = []
        for j, d in enumerate(deltas):
            if migrate is not None and migrate[j]:
                idx.append(self.migrate_idx)
            elif vertical is not None and vertical[j] != 0 and self.cfg.enable_vertical:
                idx.append(self.vup_idx if vertical[j] > 0 else self.vdown_idx)
            else:
                idx.append(int(np.clip(d, -K, K)) + K)
        return self.encode_action(idx)

    def _eff(self, size: int) -> float:
        return float(size) ** self.cfg.vertical_alpha

    def _node_used(self) -> np.ndarray:
        used = np.zeros(self.cfg.n_nodes, np.int64)
        for i in self.insts:
            used[i.node] += i.size + i.grow
            if i.state == MIG:
                used[i.dest] += i.size
        return used

    def _count(self, j: int, state: int) -> int:
        return sum(1 for i in self.insts if i.vnf == j and i.state == state)

    def _n_inst(self, j: int) -> int:
        return sum(1 for i in self.insts if i.vnf == j)

    def _vcpu(self, j: int) -> int:
        """Tổng vCPU đã cấp (đang bị tính tiền) cho VNF j."""
        return sum(i.size + i.grow for i in self.insts if i.vnf == j)

    def _alloc_capacity(self, j: int) -> float:
        """Công suất đã cấp khi mọi instance/vCPU đang chờ đã sẵn sàng (để đo utilization)."""
        return sum(self._eff(i.size + i.grow) for i in self.insts if i.vnf == j)

    def _place(self, used: np.ndarray, size: int = 1) -> int:
        """Best-fit: node đang bật có ít chỗ trống nhất mà vẫn đủ chỗ; hết thì bật node mới."""
        S = self.cfg.node_slots
        on = [n for n in range(self.cfg.n_nodes) if 0 < used[n] and used[n] + size <= S]
        if on:
            return max(on, key=lambda n: used[n])
        off = [n for n in range(self.cfg.n_nodes) if used[n] == 0]
        return off[0] if off else -1

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}
        cfg = self.cfg
        T = len(self._base_load)

        if "start" in options:
            self.t0 = int(options["start"])
        elif cfg.random_start and T - 1 > self.ep_len:
            self.t0 = int(self.np_random.integers(0, T - 1 - self.ep_len))
        else:
            self.t0 = 0
        self.t0 = int(np.clip(self.t0, 0, T - 1 - self.ep_len))

        if cfg.load_multiplier != 1.0 or cfg.noise_std > 0 or cfg.burst_prob > 0:
            self._load = augment_load(self._base_load, self.np_random, cfg.load_multiplier,
                                      cfg.noise_std, cfg.burst_prob, cfg.burst_scale, cfg.burst_len)
        else:
            self._load = self._base_load

        self.t = 0
        self.insts = []
        self.backlog = np.zeros(self.n_vnf, np.float32)
        self.last_latency = self.base_lat.copy()
        self.last_rho = np.zeros(self.n_vnf, np.float32)
        self.last_drop = np.zeros(self.n_vnf, np.float32)
        self.last_dir = np.zeros(self.n_vnf, np.int64)          # hướng thay đổi gần nhất (-1/+1)
        self.last_change_t = np.full(self.n_vnf, -10**9, np.int64)
        init = cfg.initial_instances or {}
        for j, v in enumerate(self.vnfs):
            for _ in range(int(init.get(v, cfg.min_instances))):
                n = self._place(self._node_used())
                if n < 0:
                    raise ValueError("Không đủ slot cho số instance khởi tạo.")
                self.insts.append(_Inst(j, n, RUN))
        self.stats = dict(steps=0, reward=0.0, sla_violations=0, dropped=0.0, demand=0.0,
                          served=0.0, alloc_capacity=0.0, vcpu_steps=0.0, instance_steps=0.0,
                          energy_kwh=0.0, migrations=0, scale_outs=0, scale_ins=0,
                          vertical_ups=0, vertical_downs=0, oscillations=0, invalid=0,
                          latency_sum=np.zeros(self.n_vnf))
        return self._obs(), self._info({})

    def step(self, action):
        cfg = self.cfg
        acts = self.decode_action(action)
        K = cfg.max_step_change
        tidx = self.t0 + self.t + 1          # chỉ số dữ liệu của bước sẽ được phục vụ
        invalid, mig_cost, n_mig, n_out, n_in, n_vup, n_vdown = 0, 0.0, 0, 0, 0, 0, 0
        change = np.zeros(self.n_vnf, np.int64)   # thay đổi vCPU ròng của từng VNF

        # ---------- 1. áp dụng action ----------
        for j in range(self.n_vnf):
            a = int(acts[j])
            if a == self.migrate_idx:
                c = self._migrate(j, tidx)
                if c is None:
                    invalid += 1
                else:
                    mig_cost += c
                    n_mig += 1
                continue
            if a == self.vup_idx:
                if self._vertical_up(j):
                    n_vup += 1; change[j] += 1
                else:
                    invalid += 1
                continue
            if a == self.vdown_idx:
                if self._vertical_down(j):
                    n_vdown += 1; change[j] -= 1
                else:
                    invalid += 1
                continue
            delta = a - K
            for _ in range(abs(delta)):
                if delta > 0:
                    if self._n_inst(j) >= cfg.max_instances:
                        invalid += 1; break
                    n = self._place(self._node_used())
                    if n < 0:
                        invalid += 1; break
                    self.insts.append(_Inst(j, n, BOOT, timer=cfg.scale_out_delay))
                    n_out += 1; change[j] += 1
                else:
                    if not self._scale_in(j):
                        invalid += 1; break
                    n_in += 1; change[j] -= 1

        # ---------- 1b. oscillation ----------
        direction = np.sign(change)
        osc = np.zeros(self.n_vnf, np.float32)
        for j in range(self.n_vnf):
            if direction[j] == 0:
                continue
            if self.last_dir[j] == -direction[j] and self.t - self.last_change_t[j] <= cfg.osc_window:
                osc[j] = 1.0
            self.last_dir[j] = direction[j]
            self.last_change_t[j] = self.t

        # ---------- 2. phục vụ tải ----------
        lam = self._load[tidx] * self.peak                    # (n_vnf,)
        cap = np.zeros(self.n_vnf, np.float32)
        for i in self.insts:
            if i.state == RUN:
                cap[i.vnf] += self._eff(i.size)
            elif i.state == MIG:
                cap[i.vnf] += cfg.migration_capacity * self._eff(i.size)

        work = lam + self.backlog
        served = np.minimum(work, cap)
        excess = work - served
        max_backlog = cfg.buffer_steps * np.maximum(cap, 1e-6)
        new_backlog = np.minimum(excess, max_backlog)
        dropped = excess - new_backlog
        rho = np.where(cap > 0, lam / np.maximum(cap, 1e-6), np.where(lam > 0, 10.0, 0.0))
        rho_eff = np.minimum(rho, cfg.rho_cap)
        step_ms = cfg.step_minutes * 60_000.0
        queue_wait = np.where(cap > 0, new_backlog / np.maximum(cap, 1e-6), cfg.buffer_steps) * step_ms
        latency = self.base_lat / (1.0 - rho_eff) + queue_wait
        drop_frac = np.where(lam > 1e-9, dropped / np.maximum(lam, 1e-9), 0.0)
        violation = ((latency > self.sla) | (dropped > 1e-6)).astype(np.float32)
        self.backlog = new_backlog.astype(np.float32)

        alloc = np.array([self._alloc_capacity(j) for j in range(self.n_vnf)], np.float32)
        util = np.where(alloc > 0, np.minimum(served / np.maximum(alloc, 1e-6), 1.0), 0.0)
        vcpu = np.array([self._vcpu(j) for j in range(self.n_vnf)], np.float32)

        # ---------- 3. năng lượng ----------
        used = self._node_used()
        busy_slots = np.zeros(cfg.n_nodes, np.float32)
        for i in self.insts:   # phần slot thực sự bận (theo tải) trên mỗi node
            if i.state in (RUN, MIG) and cap[i.vnf] > 0:
                c_i = self._eff(i.size) * (1.0 if i.state == RUN else cfg.migration_capacity)
                busy_slots[i.node] += served[i.vnf] * c_i / cap[i.vnf]
            elif i.state == BOOT:
                busy_slots[i.node] += 0.3            # boot cũng tiêu tốn CPU
        on = used > 0
        node_util = np.clip(busy_slots / cfg.node_slots, 0, 1)
        power = np.where(on, cfg.p_idle + (cfg.p_max - cfg.p_idle) * node_util, 0.0)
        energy_norm = float(power.sum() / (cfg.n_nodes * cfg.p_max))
        energy_kwh = float(power.sum()) * cfg.step_minutes / 60.0 / 1000.0
        n_inst_total = len(self.insts)

        # ---------- 4. reward ----------
        n_scale = n_out + n_in + n_vup + n_vdown
        terms = {
            "sla": cfg.w_sla * float(violation.sum()),
            "drop": cfg.w_drop * float(np.clip(drop_frac, 0, 1).sum()),
            "latency": cfg.w_latency * float(np.clip(latency / self.sla, 0, 5).sum()),
            "resource": cfg.w_resource * float(vcpu.sum()),
            "util": cfg.w_util * float((1.0 - util).sum()),
            "energy": cfg.w_energy * energy_norm,
            "migration": cfg.w_migration * mig_cost,
            "scale": cfg.w_scale * n_scale,
            "oscillation": cfg.w_osc * float(osc.sum()),
            "invalid": cfg.w_invalid * invalid,
        }
        reward = -float(sum(terms.values()))

        # ---------- 5. tiến thời gian: boot / migration / vertical hoàn tất ----------
        for i in self.insts:
            if i.state in (BOOT, MIG):
                i.timer -= 1
                if i.timer <= 0:
                    if i.state == MIG:
                        i.node, i.dest = i.dest, -1
                    i.state = RUN
            if i.grow > 0:
                i.vtimer -= 1
                if i.vtimer <= 0:
                    i.size += i.grow
                    i.grow = 0

        self.last_latency, self.last_rho, self.last_drop = latency, rho, drop_frac
        self.t += 1

        s = self.stats
        s["steps"] += 1; s["reward"] += reward
        s["sla_violations"] += int(violation.sum()); s["dropped"] += float(dropped.sum())
        s["demand"] += float(lam.sum()); s["served"] += float(served.sum())
        s["alloc_capacity"] += float(alloc.sum()); s["vcpu_steps"] += float(vcpu.sum())
        s["instance_steps"] += n_inst_total
        s["energy_kwh"] += energy_kwh; s["migrations"] += n_mig
        s["scale_outs"] += n_out; s["scale_ins"] += n_in
        s["vertical_ups"] += n_vup; s["vertical_downs"] += n_vdown
        s["oscillations"] += int(osc.sum()); s["invalid"] += invalid
        s["latency_sum"] = s["latency_sum"] + latency

        terminated = False
        truncated = self.t >= self.ep_len
        info = self._info(dict(
            demand=lam, capacity=cap, alloc_capacity=alloc, vcpu=vcpu, utilization=util,
            latency_ms=latency, rho=rho, dropped=dropped, sla_violation=violation,
            oscillation=osc, active_nodes=int(on.sum()), energy_kwh=energy_kwh,
            n_instances=n_inst_total, reward_terms=terms, invalid_actions=invalid,
            migrations=n_mig))
        if self.render_mode == "human":
            print(self.render())
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ actions
    def _scale_in(self, j: int) -> bool:
        cfg = self.cfg
        mine = [i for i in self.insts if i.vnf == j]
        if len(mine) <= cfg.min_instances:
            return False
        boots = [i for i in mine if i.state == BOOT]
        if boots:                                     # huỷ instance đang boot trước
            self.insts.remove(boots[-1]); return True
        runs = [i for i in mine if i.state == RUN]
        if not runs:
            return False
        used = self._node_used()
        victim = min(runs, key=lambda i: (used[i.node], i.size))  # node vắng nhất -> dễ tắt node
        self.insts.remove(victim)
        return True

    def _vertical_up(self, j: int) -> bool:
        """+1 vCPU cho instance RUN nhỏ nhất của VNF j mà node còn slot trống."""
        cfg = self.cfg
        used = self._node_used()
        cands = [i for i in self.insts if i.vnf == j and i.state == RUN and i.grow == 0
                 and i.size < cfg.max_vcpu and used[i.node] < cfg.node_slots]
        if not cands:
            return False
        inst = min(cands, key=lambda i: (i.size, -used[i.node]))
        if cfg.vertical_delay <= 0:
            inst.size += 1
        else:
            inst.grow, inst.vtimer = 1, cfg.vertical_delay
        return True

    def _vertical_down(self, j: int) -> bool:
        """-1 vCPU cho instance RUN lớn nhất của VNF j (có hiệu lực ngay)."""
        cands = [i for i in self.insts if i.vnf == j and i.state == RUN and i.grow == 0 and i.size > 1]
        if not cands:
            return False
        max(cands, key=lambda i: i.size).size -= 1
        return True

    def _migrate(self, j: int, tidx: int) -> Optional[float]:
        """Gom node: chuyển 1 instance RUN của VNF j từ node vắng nhất sang node đầy nhất
        còn chỗ. Trả về chi phí, hoặc None nếu không hợp lệ/không có lợi."""
        cfg = self.cfg
        used = self._node_used()
        cands = [i for i in self.insts if i.vnf == j and i.state == RUN and i.grow == 0]
        if not cands:
            return None
        src_inst = min(cands, key=lambda i: used[i.node])
        src = src_inst.node
        dests = [n for n in range(cfg.n_nodes)
                 if n != src and 0 < used[n] and used[n] + src_inst.size <= cfg.node_slots
                 and used[n] >= used[src]]
        if not dests:
            return None
        dest = max(dests, key=lambda n: used[n])
        mem = float(self._mem[min(tidx, len(self._mem) - 1), j])
        src_inst.state = MIG
        src_inst.dest = dest
        src_inst.timer = cfg.migration_base_delay + int(math.ceil(mem * cfg.migration_mem_delay))
        return cfg.migration_fixed_cost + cfg.migration_mem_cost * mem

    # ------------------------------------------------------------------ obs/info
    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        tcur = self.t0 + self.t
        H = cfg.history_len
        lo = max(0, tcur - H + 1)
        hist = self._load[lo:tcur + 1]
        if len(hist) < H:
            hist = np.vstack([np.repeat(hist[:1], H - len(hist), 0), hist])
        parts = []
        fc = None
        if cfg.forecast_horizon > 0:
            if self.forecaster is not None:
                fc = np.asarray(self.forecaster(self, tcur), np.float32).reshape(cfg.forecast_horizon, self.n_vnf)
            else:  # oracle (cận trên) — chỉ dùng để thử nghiệm
                idx = np.clip(np.arange(tcur + 1, tcur + 1 + cfg.forecast_horizon), 0, len(self._load) - 1)
                fc = self._load[idx]
        Mx = float(cfg.max_instances)
        Mv = float(cfg.max_instances * cfg.max_vcpu)
        W = float(max(cfg.osc_window, 1))
        for j in range(self.n_vnf):
            feats = list(hist[:, j])
            if fc is not None:
                feats += list(fc[:, j])
            n_run = self._count(j, RUN); n_boot = self._count(j, BOOT); n_mig = self._count(j, MIG)
            cap = sum(self._eff(i.size) * (1.0 if i.state == RUN else cfg.migration_capacity)
                      for i in self.insts if i.vnf == j and i.state != BOOT)
            since = min(self.t - self.last_change_t[j], W)
            feats += [
                n_run / Mx, n_boot / Mx, n_mig / Mx,
                self._vcpu(j) / Mv,
                # nhu cầu (theo tải hiện tại) tính bằng vCPU
                float(self._load[tcur, j] * self.peak[j]) / Mx,
                min(float(self.last_rho[j]), 3.0) / 3.0,
                min(float(self.last_latency[j] / self.sla[j]), 5.0) / 5.0,
                min(float(self.backlog[j] / max(cap, 1.0)), cfg.buffer_steps) / max(cfg.buffer_steps, 1e-6),
                float(self._mem[tcur, j]),
                # để agent biết mình vừa tăng/giảm gì -> tránh đảo chiều (reward phạt oscillation)
                (float(self.last_dir[j]) + 1.0) / 2.0,
                since / W,
            ]
            parts += feats
        used = self._node_used()
        mod = self._mod[tcur]
        parts += [
            float((used > 0).sum()) / cfg.n_nodes,
            float(used.sum()) / (cfg.n_nodes * cfg.node_slots),
            0.5 + 0.5 * math.sin(2 * math.pi * mod / 1440.0),
            0.5 + 0.5 * math.cos(2 * math.pi * mod / 1440.0),
        ]
        return np.asarray(parts, dtype=np.float32)

    def _info(self, extra: dict) -> dict:
        info = {
            "t": self.t,
            "data_index": self.t0 + self.t,
            "timestamp": str(self.traces.index[min(self.t0 + self.t, len(self.traces.index) - 1)]),
            "instances": {v: {"run": self._count(j, RUN), "boot": self._count(j, BOOT),
                              "mig": self._count(j, MIG), "vcpu": self._vcpu(j)}
                          for j, v in enumerate(self.vnfs)},
        }
        info.update(extra)
        return info

    def vnf_status(self) -> List[dict]:
        """Trạng thái thô cho policy heuristic (không dùng cho agent RL)."""
        cfg = self.cfg
        tcur = self.t0 + self.t
        used = self._node_used()
        out = []
        for j, v in enumerate(self.vnfs):
            mine = [i for i in self.insts if i.vnf == j]
            out.append(dict(
                name=v, run=self._count(j, RUN), boot=self._count(j, BOOT), mig=self._count(j, MIG),
                vcpu=self._vcpu(j), alloc=self._alloc_capacity(j),
                can_vup=any(i.state == RUN and i.grow == 0 and i.size < cfg.max_vcpu
                            and used[i.node] < cfg.node_slots for i in mine),
                can_vdown=any(i.state == RUN and i.grow == 0 and i.size > 1 for i in mine),
                demand=float(self._load[tcur, j] * self.peak[j]),
                rho=float(self.last_rho[j]), latency_ms=float(self.last_latency[j]),
                sla_ms=float(self.sla[j]), backlog=float(self.backlog[j]),
            ))
        return out

    def node_usage(self) -> np.ndarray:
        return self._node_used()

    def episode_summary(self) -> dict:
        s = dict(self.stats)
        n = max(s["steps"], 1)
        lat = s.pop("latency_sum") / n
        n_scale = s["scale_outs"] + s["scale_ins"] + s["vertical_ups"] + s["vertical_downs"]
        return {
            "steps": s["steps"],
            "total_reward": round(s["reward"], 2),
            # ---- utilization (mục tiêu 1) ----
            "utilization": round(s["served"] / max(s["alloc_capacity"], 1e-9), 4),
            "avg_vcpu": round(s["vcpu_steps"] / n, 2),
            "avg_instances": round(s["instance_steps"] / n, 2),
            # ---- oscillation (mục tiêu 2) ----
            "oscillations": s["oscillations"],
            "oscillation_rate": round(s["oscillations"] / (n * self.n_vnf), 4),
            "scaling_actions": n_scale,
            # ---- SLA (ràng buộc) ----
            "sla_violation_rate": round(s["sla_violations"] / (n * self.n_vnf), 4),
            "drop_rate": round(s["dropped"] / max(s["demand"], 1e-9), 4),
            **{f"avg_latency_ms_{v}": round(float(lat[j]), 1) for j, v in enumerate(self.vnfs)},
            # ---- khác ----
            "energy_kwh": round(s["energy_kwh"], 3),
            "scale_outs": s["scale_outs"],
            "scale_ins": s["scale_ins"],
            "vertical_ups": s["vertical_ups"],
            "vertical_downs": s["vertical_downs"],
            "migrations": s["migrations"],
            "invalid_actions": s["invalid"],
        }

    def render(self):
        used = self._node_used()
        lines = [f"t={self.t:4d} {self._info({})['timestamp']}  nodes={''.join(str(u) for u in used)}"]
        for st in self.vnf_status():
            lines.append(
                f"  {st['name']:4s} run={st['run']:2d} boot={st['boot']} mig={st['mig']} vcpu={st['vcpu']:2d} "
                f"demand={st['demand']:5.2f} rho={st['rho']:4.2f} lat={st['latency_ms']:7.1f}/{st['sla_ms']:.0f}ms "
                f"backlog={st['backlog']:.2f}")
        return "\n".join(lines)

    def get_config(self) -> dict:
        return asdict(self.cfg)
