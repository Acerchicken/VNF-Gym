# VNF Gym — bối cảnh dự án

Đồ án tốt nghiệp: auto-scaling VNF bằng RL. Tại mỗi bước t, dự đoán tài nguyên VNF cần và chọn hành động
(giữ nguyên / vertical / horizontal / migration) sao cho:
1. Utilization cao (không cấp thừa).
2. Oscillation thấp (không tăng rồi giảm liên tục).
3. Ràng buộc: không vi phạm SLA (quá tải, độ trễ).

Lộ trình: (1–2) dữ liệu SNDZoo → **(3) simulator Gymnasium — đã xong** → **(4) dự báo tải — đã có khung + LSTM** (Lag-Llama/TimesFM thêm sau)
→ (5) train agent RL (PPO/DQN) và so sánh với baseline.

## Môi trường
- Windows 11, Python 3.13, venv ở `.venv` (đã cài numpy, pandas, gymnasium, matplotlib, stable-baselines3).
- Dữ liệu: `E:/SNDZoo dataset/SNDZoo dataset/DatasetSNDZoo` (WEB=Nginx, IOT=Mosquitto, SEC=Snort).
  - train: `<VNF>/TrainTest/NoHoles/*.csv` (27/01 → 10/02/2025, ~20.000 phút)
  - test: `<VNF>/18-02/CSVmergedFIXED/*.csv` (17/02 10:00 → 18/02 07:47, ~1.300 phút)
  - Giá trị đã chuẩn hoá min-max → mất thang tuyệt đối.
- Không chạy `pip install --upgrade pip` trong venv (pip tự khoá file trên Windows, hỏng venv).

## Lệnh
```
python examples\check_env.py
python examples\run_baselines.py
python examples\train_ppo.py
python examples\play.py          # chạy tay từng bước để hiểu simulator
python examples\train_forecaster.py   # train + so sánh forecaster, lưu results/forecasters/<tên>.pkl
```
Đường dẫn dữ liệu và mọi tham số lấy từ `configs/config.py`; tham số dòng lệnh (`--data`, `--steps`…) chỉ để ghi đè.

## Cấu trúc
- `configs/config.py` — **MỌI tham số** (`nfv_sim/config.py` chỉ chuyển tiếp tới đây): đường dẫn, dữ liệu, `SimConfig`, `SCENARIOS`, `BASELINE`, `TRAIN`/`PPO`, `PLAY`,
  `FORECAST` (chung) / `FORECASTERS` (riêng từng model).
  Không đặt hằng số/tham số ở file khác; thêm tham số mới vào đây rồi import.
- `nfv_sim/data.py` — đọc/căn chỉnh SNDZoo, `augment_load` tạo kịch bản quá tải.
- `nfv_sim/env.py` — `NFVScalingEnv` (đọc tham số từ `SimConfig` trong config.py).
- `nfv_sim/baselines.py` — Static, Threshold (mode horizontal/vertical/hybrid, consolidate), PredictiveThreshold (oracle).
- `nfv_sim/forecast/` — dự báo tải. `base.py`: lớp `Forecaster` (cài `predict_batch(N,L,n)->(N,H,n)`, tuỳ chọn `fit`)
  + registry `@register_forecaster("tên")`; `simple.py` (naive, moving_avg, linear, oracle), `lstm.py`, `metrics.py`.
  Forecaster cắm thẳng vào env (`forecaster=fc`) và `PredictiveThresholdPolicy(forecast_fn=fc.demand_fn())`.
  Thêm thuật toán: 1 class + import trong `forecast/__init__.py` + 1 dòng trong `FORECASTERS`.
- `examples/` — check_env, train_forecaster, run_baselines, train_ppo, play (nhập action bằng tay). Các script gọi `sys.stdout.reconfigure(encoding="utf-8")`
  vì console Windows cp1252 không in được tiếng Việt — giữ dòng này khi viết script mới.
- `README.md` — mô tả chi tiết mô hình, state/action/reward, bảng baseline, giới hạn.

## Mô hình simulator (tóm tắt)
- 6 node × 4 vCPU. Instance có `size` vCPU (1..4), công suất = size^0.9.
- Action mỗi VNF (MultiDiscrete [6,6,6]): 0 scale-in, 1 giữ, 2 scale-out, 3 migrate (gom node),
  4 vertical +1 vCPU, 5 vertical −1 vCPU. `discrete_action=True` → Discrete(216) cho DQN.
- Scale-out: BOOT 3 bước mới phục vụ. Vertical +1: hiệu lực sau 1 bước, cần node còn slot.
  Migration: thời gian & chi phí theo memory, phục vụ 50% trong lúc migrate.
- Hàng đợi có backlog; SLA vi phạm khi latency > sla_ms hoặc có drop.
- Utilization = served / công suất đã cấp (tính cả instance boot).
- Oscillation = đổi hướng tăng/giảm tài nguyên trong vòng `osc_window`=10 bước.
- Reward ≤ 0 gồm: sla, drop, latency, resource, util, energy, migration, scale, oscillation, invalid
  (xem `info["reward_terms"]`). Observation 55 chiều (có hướng thay đổi gần nhất để phạt oscillation vẫn Markov).
- Forecaster cho Bước 4: truyền `forecaster=f(env, t) -> (H, n_vnf)` (mọi `Forecaster` đều callable như vậy)
  và `forecast_horizon=H`; không truyền thì dùng oracle (chỉ là cận trên, không báo cáo như kết quả thật).

## Kết quả baseline hiện có (ngày test, kịch bản normal)
| Policy | Utilization | Oscillation | Vi phạm SLA |
|---|---|---|---|
| static | 0,20 | 0 | 0,2% |
| threshold (HPA) | 0,51 | 166 | 10,3% |
| vertical | 0,51 | 160 | 8,8% |
| hybrid | 0,53 | 182 | 13,2% |
| predictive (oracle) | 0,51 | 238 | 5,6% |

| predictive (lstm) | 0,51 | 368 | 9,3% |

Dự báo (H=5, L=30, MAE trên ngày test): lstm 0,044 · linear 0,057 · naive 0,071 · moving_avg 0,080.

Điểm đáng nêu: dự báo hoàn hảo giảm vi phạm SLA nhưng dao động nhiều nhất → động lực cho RL có phạt oscillation.

## Việc tiếp theo
- PPO mới chạy thử 8k bước (chỉ kiểm tra pipeline). Cần train ≥300k bước, tinh chỉnh `w_osc`, `w_invalid`.
- Bước 4: LSTM val loss còn giảm ở epoch 30 → thử tăng `epochs`; thêm Lag-Llama/TimesFM; thử `under_weight>1`.
- Giả định cần ghi trong luận văn: `peak_instances`, `vertical_alpha`, `vertical_delay`, ngưỡng SLA, base latency.

## Quy ước
- Trả lời và viết comment/docstring bằng tiếng Việt, giữ phong cách code hiện có.
