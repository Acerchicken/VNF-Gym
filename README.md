# nfv_sim — Simulator auto-scaling VNF chuẩn Gymnasium (dữ liệu SNDZoo)

Simulator nhẹ, viết bằng Python thuần (numpy + pandas + gymnasium) để huấn luyện và đánh giá agent RL
cho bài toán **tự động scale VNF**. Simulator chạy trên trace thật của SNDZoo: Nginx (WEB),
Mosquitto (IOT), Snort (SEC). Nó thay cho CloudSimNFV và chạy khoảng 5.000 bước/giây trên một core CPU.

## 1. Cài đặt và chạy trên Windows

Yêu cầu Python ≥ 3.10 (đã thử với 3.13). Mọi lệnh chạy trong thư mục dự án `D:\Git_repo\VNF Gym`.

**PowerShell / terminal trong VS Code**

```powershell
cd "D:\Git_repo\VNF Gym"
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # nếu bị chặn: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
pip install stable-baselines3         # chỉ cần khi train PPO/DQN (kéo theo PyTorch)
```

* **cmd**: các lệnh giống trên, nhưng kích hoạt venv bằng `.venv\Scripts\activate.bat`.
* **Git Bash**: kích hoạt venv bằng `source .venv/Scripts/activate`.

> Trên Windows, đừng chạy `python -m pip install --upgrade pip` bên trong venv nếu không cần. pip có thể tự
> khoá file của chính nó và làm hỏng venv; khi đó xoá thư mục `.venv` rồi tạo lại.

**Chạy** (tham số `--data` trỏ tới thư mục **DatasetSNDZoo**):

```powershell
python examples\check_env.py     --data "E:/SNDZoo dataset/SNDZoo dataset/DatasetSNDZoo"
python examples\run_baselines.py --data "E:/SNDZoo dataset/SNDZoo dataset/DatasetSNDZoo"
python examples\train_ppo.py     --data "E:/SNDZoo dataset/SNDZoo dataset/DatasetSNDZoo" --steps 300000
```

Kết quả (`baselines.csv`, hình `capacity_*.png`, model PPO) được ghi vào thư mục `results/`.

**VS Code**:

1. Mở thư mục dự án.
2. `Ctrl+Shift+P` → *Python: Select Interpreter* → chọn `.venv`.
3. Mở terminal mới; venv được kích hoạt tự động. Chạy các lệnh như trên.

## 2. Dữ liệu được dùng thế nào

| Tập | Thư mục | Độ dài |
|---|---|---|
| train | `<VNF>/TrainTest/NoHoles/*.csv` | ~20.000 phút (27/01 → 10/02/2025) |
| test  | `<VNF>/18-02/CSVmergedFIXED/*.csv` | ~1.300 phút (17/02 10:00 → 18/02 07:47) |

* **Tải đến** (`load_metric="rx"`, mặc định) là `container_network_receive_bytes_total`, đã chuẩn hoá về [0,1].
  Có thể đổi sang `"cpu"`.
* **Quy đổi ra số instance:** `demand_t = load_t × peak_instances[VNF]`. Mặc định WEB=6, IOT=4, SEC=5,
  nghĩa là lúc tải đỉnh của trace cần ~6 instance WEB chạy 100%. Dữ liệu đã bị chuẩn hoá min-max nên mất
  thang đo tuyệt đối, vì vậy đây là **giả định cần ghi rõ trong luận văn**.
* **Memory working set** (`mem`) được dùng để tính **thời gian và chi phí migration**.
  VNF đang dùng nhiều RAM thì migrate lâu hơn và tốn hơn (mô phỏng pre-copy live migration).
* Các cột tx/cpu được nạp sẵn để bạn dùng làm feature hoặc đổi `load_metric`.

## 3. Mô hình hệ thống

```
          ┌─ node 0 [■■■□] ─┐      1 node = node_slots (4) vCPU; 1 instance = size vCPU (1..max_vcpu)
 tải ───► │  node 1 [■■□□]  │ ───► node bật nếu có ≥1 instance; công suất P = P_idle + (P_max−P_idle)·util
          └─ node 5 [□□□□] ─┘
```

**Vòng đời instance**

| Trạng thái | Phục vụ | Tốn chi phí | Ghi chú |
|---|---|---|---|
| BOOT | 0% | có | kéo dài `scale_out_delay` bước (mặc định 3 phút) → **độ trễ khi scale-out** |
| RUN  | size^α (α = `vertical_alpha` = 0,9) | có | |
| MIG  | `migration_capacity` (50%) | có, chiếm slot ở cả node nguồn và node đích | kéo dài `migration_base_delay + ceil(mem × migration_mem_delay)` bước |

**Vertical scaling**

* +1 vCPU: áp dụng cho một instance đang chạy, chỉ khi node của nó còn slot. Có hiệu lực sau `vertical_delay`
  bước (mặc định 1, nhanh hơn boot), nhưng bị giới hạn bởi chỗ trống trên node và hiệu suất giảm dần (size^α).
* −1 vCPU: có hiệu lực ngay.

Đây là trade-off cần phân tích: vertical nhanh nhưng bị giới hạn, horizontal chậm (do boot) nhưng mở rộng được.

**Hiệu năng:** mỗi VNF có một hàng đợi. Phần tải vượt công suất được dồn sang bước sau (backlog, tối đa
`buffer_steps` × công suất); phần còn thừa nữa bị drop.

```
latency = base_latency / (1 − min(ρ, 0.98)) + backlog/capacity × độ dài bước
vi phạm SLA  ⇔  latency > sla_ms  hoặc  có drop
```

Giá trị mặc định: base latency WEB/IOT/SEC = 20/5/10 ms, SLA = 100/30/60 ms. **Đây là tham số giả định,
không đo từ dữ liệu**, và bạn chỉnh được trong `SimConfig`.

**Hai chỉ tiêu chính của đề tài**

* **Utilization** = tải được phục vụ / công suất đã cấp. Công suất đã cấp tính cả instance đang boot và vCPU
  đang chờ, vì đều bị tính tiền.
  * Cả episode: `episode_summary()["utilization"]`.
  * Từng bước: `info["utilization"]`.
* **Oscillation**: một lần đảo chiều xảy ra khi VNF thay đổi tài nguyên ngược hướng với lần thay đổi gần nhất
  và cách nó ≤ `osc_window` bước (mặc định 10). Ví dụ: scale-out rồi scale-in sau 4 phút.
  * `oscillations`: tổng số lần đảo chiều.
  * `oscillation_rate` = số lần / (số bước × số VNF).
  * `scaling_actions`: tổng số thao tác scale.

## 4. State – Action – Reward

**Action**, mặc định là `MultiDiscrete([6, 6, 6])`, mỗi VNF có một lựa chọn:

| idx | ý nghĩa |
|---|---|
| 0 | horizontal scale-in −1 (ưu tiên huỷ instance đang boot, sau đó gỡ instance trên node vắng nhất) |
| 1 | giữ nguyên |
| 2 | horizontal scale-out +1 (đặt theo best-fit; bật node mới nếu hết chỗ) |
| 3 | migrate 1 instance để gom node (từ node vắng nhất sang node đầy nhất còn chỗ) |
| 4 | vertical scale-up +1 vCPU |
| 5 | vertical scale-down −1 vCPU |

* `max_step_change=K` cho phép thay đổi ±K instance mỗi bước. Khi đó:
  * idx 0..2K là horizontal;
  * idx 2K+1 là migrate;
  * idx 2K+2 và 2K+3 là vertical.
* `enable_vertical=False` bỏ hai action vertical (quay về 4 action).
* `discrete_action=True` làm phẳng action thành `Discrete(6^3 = 216)`, dùng cho DQN.
* Action không hợp lệ (vượt min/max, hết slot, không có gì để gom) bị bỏ qua và bị phạt `w_invalid`.

**Observation**, gồm 55 chiều với cấu hình mặc định:

* Cho mỗi VNF (17 chiều):
  * 6 giá trị tải gần nhất;
  * [dự báo H bước, nếu bật];
  * số instance run/boot/mig và tổng vCPU đã cấp;
  * nhu cầu hiện tại;
  * ρ (utilization);
  * latency/SLA;
  * backlog;
  * memory;
  * hướng thay đổi gần nhất và số bước kể từ lần đó. Hai giá trị này giúp agent biết mình vừa tăng hay giảm,
    để tránh đảo chiều, và giữ cho reward phạt oscillation vẫn Markov.
* Toàn cục (4 chiều): tỉ lệ node đang bật, tỉ lệ slot đã dùng, sin/cos của giờ trong ngày.

**Reward** (luôn ≤ 0):

```
r = −( w_sla·Σ vi_phạm_SLA + w_drop·Σ tỉ_lệ_drop + w_latency·Σ latency/SLA        ← ràng buộc SLA
       + w_resource·số_vCPU + w_util·Σ(1 − utilization) + w_energy·năng_lượng       ← mục tiêu utilization
       + w_osc·số_lần_đảo_chiều + w_scale·số_thao_tác_scale                         ← mục tiêu oscillation
       + w_migration·Σ(chi_phí_cố_định + chi_phí_mem·mem) + w_invalid·số_action_sai )
```

Từng thành phần được trả về trong `info["reward_terms"]`, nên bạn dễ vẽ hình phân tích hoặc chỉnh trọng số.

## 5. Kịch bản quá tải

Có thể tạo kịch bản quá tải ngay từ trace thật (`augment_load`):

```python
env = NFVScalingEnv(data_root=ROOT, split="test", episode_steps=None, random_start=False,
                    load_multiplier=1.5,      # tải tăng 50%
                    burst_prob=0.005,         # xác suất bắt đầu burst mỗi phút
                    burst_scale=(1.5, 3.0),   # burst ×1.5–3
                    burst_len=(3, 20),        # dài 3–20 phút
                    noise_std=0.05)
```

Nên **train** với augmentation để agent quen với quá tải, rồi **test** trên trace gốc 18-02 cộng với một
kịch bản quá tải cố định (cùng seed) để so sánh công bằng.

## 6. Kết quả baseline trên ngày test (chạy thử)

| Kịch bản | Policy | Utilization | Oscillation | Thao tác scale | Vi phạm SLA | vCPU TB |
|---|---|---|---|---|---|---|
| normal | static (cấp theo đỉnh) | 0,20 | 0 | 17 | 0,2% | 20,0 |
| normal | threshold (HPA 80%/40%) | 0,51 | 166 | 370 | 10,3% | 8,0 |
| normal | vertical (VPA) | 0,51 | 160 | 348 | 8,8% | 8,0 |
| normal | hybrid | 0,53 | 182 | 389 | 13,2% | 8,0 |
| normal | predictive (oracle) | 0,51 | 238 | 496 | 5,6% | 7,8 |
| overload | static | 0,32 | 0 | 17 | 6,9% | 20,0 |
| overload | threshold | 0,55 | 192 | 490 | 15,1% | 11,6 |
| overload | vertical | 0,56 | 183 | 458 | 13,2% | 11,8 |
| overload | hybrid | 0,59 | 180 | 441 | 16,6% | 11,8 |
| overload | predictive (oracle) | 0,55 | 254 | 700 | 8,4% | 11,5 |

Bảng cho thấy ba điều:

* Static gần như không vi phạm SLA và không dao động, nhưng utilization chỉ 20–30%.
* Threshold và vertical đưa utilization lên khoảng 50%, nhưng đảo chiều liên tục (khoảng 1 lần mỗi 8 phút)
  và vi phạm SLA nhiều do độ trễ boot.
* Predictive (dự báo hoàn hảo) giảm vi phạm SLA, nhưng **dao động còn nhiều hơn** vì nó bám sát từng đỉnh tải.

Agent RL cần đồng thời đạt utilization cao, SLA tốt và oscillation thấp. Đây chính là khoảng trống mà đề tài
nhắm tới.

## 7. Nối với Bước 4 (dự báo)

Truyền forecaster của bạn vào env. Forecaster nhận `(env, t_idx)` và trả về mảng `(H, n_vnf)` chứa tải
chuẩn hoá dự báo:

```python
def my_forecaster(env, t):
    hist = env._load[max(0, t-64):t+1]          # lịch sử tải (đã chuẩn hoá)
    return model.predict(hist)                  # (H, n_vnf)  — Lag-Llama / TimesFM / LSTM ...

env = NFVScalingEnv(data_root=ROOT, forecast_horizon=5, forecaster=my_forecaster)
```

Nếu `forecast_horizon>0` mà không truyền forecaster, env dùng **oracle** (giá trị thật trong tương lai).
Oracle chỉ để đo cận trên, không được báo cáo như kết quả thật.

## 8. Các tham số hay chỉnh (`SimConfig`)

| Nhóm | Tham số |
|---|---|
| Thời gian | `step_minutes` (gộp k phút/bước, lấy max tải trong bước), `episode_steps` |
| Hạ tầng | `n_nodes`, `node_slots`, `min_instances`, `max_instances` |
| Độ trễ | `scale_out_delay`, `vertical_delay`, `migration_base_delay`, `migration_mem_delay`, `migration_capacity` |
| Vertical | `enable_vertical`, `max_vcpu`, `vertical_alpha` |
| Oscillation | `osc_window`, `w_osc`, `w_scale` |
| Chi phí | `w_*`, `migration_fixed_cost`, `migration_mem_cost`, `p_idle`, `p_max` |
| SLA | `base_latency_ms`, `sla_ms`, `buffer_steps` |

## 9. Giới hạn cần nêu trong luận văn

* Thang tải tuyệt đối bị mất do dữ liệu đã chuẩn hoá. Số instance ở đỉnh (`peak_instances`) là giả định.
* Hiệu suất vertical (size^α, α = 0,9) và độ trễ vertical là giả định, không đo từ SNDZoo.
* Mô hình độ trễ là xấp xỉ hàng đợi M/M/1 cộng backlog, không mô phỏng ở mức gói tin.
* Ba VNF trong SNDZoo là ba dịch vụ độc lập, không phải một chuỗi SFC. Chúng chỉ chia sẻ hạ tầng
  (node), nên tác động qua lại nằm ở phần năng lượng và migration.
* Trong dữ liệu có khoảng ~2 giờ tải ≈ 0 (08:00–10:00, do thí nghiệm khởi động lại), sau đó tải tăng vọt
  lúc 10:00. Đoạn này vô tình tạo ra một kịch bản "tải tăng đột ngột" tốt để kiểm thử.
