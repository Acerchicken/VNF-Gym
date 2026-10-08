"""
TẤT CẢ THAM SỐ CỦA DỰ ÁN — mở file này để chỉnh.

Nhóm tham số:
    1. Đường dẫn               DATA_ROOT, RESULTS_DIR
    2. Dữ liệu SNDZoo          VNF_NAMES, METRIC_FILES, SPLIT_DIRS, MAX_GAP_MINUTES
    3. Simulator               SimConfig  (giá trị mặc định của NFVScalingEnv)
    4. Kịch bản đánh giá       SCENARIOS
    5. Baseline                BASELINE
    6. Train RL                TRAIN, TRAIN_AUGMENT, PPO
    7. Script play.py          PLAY
    8. Dự báo tải              FORECAST, FORECASTERS

Mọi module khác import từ đây; không đặt "số ma thuật" ở chỗ khác.
Muốn thử một giá trị khác mà không sửa file, vẫn có thể ghi đè khi tạo env:
    NFVScalingEnv(traces=..., w_osc=1.0, scale_out_delay=5)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

# =============================================================================
# 1. ĐƯỜNG DẪN
# =============================================================================
DATA_ROOT = "E:/SNDZoo dataset/SNDZoo dataset/DatasetSNDZoo"   # thư mục DatasetSNDZoo
RESULTS_DIR = "results"                                       # nơi lưu csv, hình, model

# =============================================================================
# 2. DỮ LIỆU SNDZoo
# =============================================================================
VNF_NAMES = ("WEB", "IOT", "SEC")  # Nginx, Mosquitto, Snort

METRIC_FILES = {
    "cpu": "container_cpu_usage_rate_normalized.csv",
    "mem": "container_memory_working_set_bytes.csv",
    "rx": "container_network_receive_bytes_total.csv",
    "tx": "container_network_transmit_bytes_total.csv",
}

# thư mục train/test bên trong <DATA_ROOT>/<VNF>/ (thử lần lượt, cái nào có dữ liệu thì dùng)
SPLIT_DIRS = {
    "train": [os.path.join("TrainTest", "NoHoles"), "TrainTest"],
    "test": [os.path.join("18-02", "CSVmergedFIXED"), os.path.join("18-02", "CSVmerged")],
}

MAX_GAP_MINUTES = 5          # lỗ dữ liệu <= N phút thì nội suy, dài hơn thì điền 0


# =============================================================================
# 3. SIMULATOR (NFVScalingEnv)
# =============================================================================
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
    # lúc tải đỉnh của trace cần bao nhiêu vCPU (GIẢ ĐỊNH — dữ liệu đã chuẩn hoá nên mất thang tuyệt đối)
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
    boot_cpu_load: float = 0.3               # phần vCPU bận của 1 instance đang boot

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


# =============================================================================
# 4. KỊCH BẢN ĐÁNH GIÁ (ghi đè SimConfig khi chạy trên ngày test)
# =============================================================================
SCENARIOS = {
    "normal": {},
    "overload": dict(load_multiplier=1.5, burst_prob=0.005),
}
EVAL_SEED = 1                # seed cố định -> mọi policy gặp cùng một kịch bản quá tải

# =============================================================================
# 5. BASELINE (nfv_sim/baselines.py)
# =============================================================================
BASELINE = dict(
    upper=0.8,               # utilization > upper -> tăng tài nguyên
    lower=0.4,               # utilization < lower -> giảm tài nguyên
    cooldown=3,              # số bước tối thiểu giữa hai lần thay đổi
    static_target_util=0.8,  # StaticPolicy cấp ceil(peak / giá trị này) instance
    consolidate_every=10,    # ThresholdPolicy(consolidate=True): thử gom node mỗi N bước
)

# =============================================================================
# 6. TRAIN RL (examples/train_ppo.py)
# =============================================================================
TRAIN = dict(
    total_steps=300_000,     # số bước tương tác khi train
    n_envs=8,                # số env song song
    episode_steps=720,       # độ dài 1 episode train (phút)
    forecast_horizon=0,      # >0: thêm dự báo vào state (mặc định oracle)
    forecaster=None,         # tên forecaster đã train (vd "lstm") -> dùng thay oracle, horizon = FORECAST["horizon"]
    seed=0,
)
# augmentation khi train để agent quen với quá tải (test vẫn dùng trace gốc)
TRAIN_AUGMENT = dict(load_multiplier=1.2, noise_std=0.05, burst_prob=0.003)

PPO = dict(
    n_steps=1024,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=3e-4,
    ent_coef=0.01,
)

# =============================================================================
# 7. SCRIPT play.py
# =============================================================================
PLAY = dict(
    start=600,               # phút bắt đầu trong ngày test
    seed=0,
)

# =============================================================================
# 8. DỰ BÁO TẢI (nfv_sim/forecast, examples/train_forecaster.py)
# =============================================================================
# Tham số chung cho mọi forecaster (từng model có thể ghi đè trong FORECASTERS).
FORECAST = dict(
    horizon=5,               # H: số bước dự báo (>= scale_out_delay để bù thời gian boot)
    context_len=30,          # L: số bước lịch sử đưa vào model
    val_frac=0.1,            # phần cuối tập train dùng làm validation (early stopping)
    models_dir=os.path.join(RESULTS_DIR, "forecasters"),   # nơi lưu model đã train (<tên>.pkl)
    default="lstm",          # forecaster dùng khi script không chỉ định
)

# Tham số riêng của từng thuật toán — key là tên đã đăng ký bằng @register_forecaster.
# Thêm thuật toán mới: viết class trong nfv_sim/forecast/, đăng ký tên, rồi thêm một dòng ở đây.
FORECASTERS = {
    "naive": dict(),                         # giữ nguyên giá trị cuối
    "moving_avg": dict(window=5),            # trung bình window bước gần nhất
    "linear": dict(ridge=1e-2),              # hồi quy tuyến tính (AR) có ridge, dự báo trực tiếp H bước
    "lstm": dict(
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        epochs=30,
        batch_size=256,
        lr=1e-3,
        patience=5,          # early stopping: dừng sau N epoch val không cải thiện
        under_weight=1.0,    # >1: phạt dự báo THẤP hơn thực tế nặng hơn (thiếu tài nguyên -> vi phạm SLA)
        seed=0,
    ),
    "oracle": dict(),                        # tương lai thật — chỉ là cận trên, không báo cáo như kết quả thật
}
