# VNF Gym — bối cảnh dự án

Đồ án tốt nghiệp: auto-scaling VNF bằng RL. Tại mỗi bước t, dự đoán tài nguyên VNF cần và chọn hành động
(giữ nguyên / vertical / horizontal / migration) sao cho:
1. Utilization cao (không cấp thừa).
2. Oscillation thấp (không tăng rồi giảm liên tục).
3. Ràng buộc: không vi phạm SLA (quá tải, độ trễ).

Lộ trình: (1–2) dữ liệu SNDZoo → **(3) simulator Gymnasium — đã xong** → **(4) dự báo tải — đã có khung, LSTM và TFT**
(dự báo khoảng P10/P50/P90, train trên nhiều dataset, thông tin phụ giờ/ID chuỗi) → (5) train agent RL (PPO/DQN) và so sánh với baseline.

## Môi trường
- Windows 11, Python 3.13, venv ở `.venv` (đã cài numpy, pandas, gymnasium, matplotlib, stable-baselines3).
- Dữ liệu: `E:/SNDZoo dataset/SNDZoo dataset/DatasetSNDZoo` (WEB=Nginx, IOT=Mosquitto, SEC=Snort).
  - train: `<VNF>/TrainTest/NoHoles/*.csv` (27/01 → 10/02/2025, ~20.000 phút)
  - test: `<VNF>/18-02/CSVmergedFIXED/*.csv` (17/02 10:00 → 18/02 07:47, ~1.300 phút)
  - Giá trị đã chuẩn hoá min-max → mất thang tuyệt đối.
  - Mặc định `TRAIN_FRAC = 0.7`: train = 70% đầu thư mục TrainTest, test = 30% cuối (~6.000 phút). `None` = test ngày 18-02.
- Dataset bổ sung (chỉ để train forecaster): `E:/SNDZoo dataset/ExternalDatasets` (`EXTERNAL_ROOT`, danh mục `DATASETS`):
  NAB AWS (12 chuỗi), Google từng máy (6), Alibaba/Google/Azure gộp cluster (3, giờ giả định). WorldCup98 bỏ (quá cũ).
- Biến môi trường `VNF_GYM_DATA_ROOT`, `VNF_GYM_EXTERNAL_ROOT` ghi đè đường dẫn (chạy trên máy khác).
- Không chạy `pip install --upgrade pip` trong venv (pip tự khoá file trên Windows, hỏng venv).

## Lệnh
```
python examples\check_env.py
python examples\run_baselines.py
python examples\train_ppo.py
python examples\play.py          # chạy tay từng bước để hiểu simulator
python examples\train_forecaster.py   # train + so sánh forecaster, lưu results/forecasters/<tên>.pkl (~1,5 giờ CPU)
python examples\train_ppo.py --forecaster lstm --quantiles 0.5 0.9 --seed 0   # PPO với P50+P90 trong state
python examples\explain_tft.py        # hình độ quan trọng của biến + attention của TFT
python examples\crosscheck_tft.py     # so chéo một lần với pytorch-forecasting (pip install pytorch-forecasting)
python examples\compare_ppo.py        # gộp results/ppo_eval*.csv thành bảng trung bình ± std theo seed
```
Đường dẫn dữ liệu và mọi tham số lấy từ `configs/config.py`; tham số dòng lệnh (`--data`, `--steps`…) chỉ để ghi đè.

## Cấu trúc
- `configs/config.py` — **MỌI tham số** (`nfv_sim/config.py` chỉ chuyển tiếp tới đây): đường dẫn, dữ liệu, `SimConfig`, `SCENARIOS`, `BASELINE`, `TRAIN`/`PPO`, `PLAY`,
  `FORECAST` (chung) / `FORECASTERS` (riêng từng model).
  Không đặt hằng số/tham số ở file khác; thêm tham số mới vào đây rồi import.
- `nfv_sim/data.py` — đọc/căn chỉnh SNDZoo, `augment_load` tạo kịch bản quá tải; `SeriesContext` (tên chuỗi, dataset,
  giờ bắt đầu, bước, giờ thật hay không, `calendar_features`), `TimeSeries` (mảng + context), `load_dataset`/`load_catalog`
  đọc dataset trong `DATASETS` (dữ liệu 5 phút lấy lại đúng mẫu thật).
- `nfv_sim/env.py` — `NFVScalingEnv` (đọc tham số từ `SimConfig` trong config.py).
- `nfv_sim/baselines.py` — Static, Threshold (mode horizontal/vertical/hybrid, consolidate), PredictiveThreshold (oracle).
- `nfv_sim/forecast/` — dự báo tải. `base.py`: lớp `Forecaster` (cài `predict_batch(windows, extra=None)` (N,L,n)->(N,H,n),
  tuỳ chọn `fit`, model dự báo khoảng có `quantiles` + `predict_dist_batch` -> (N,H,n,Q)), registry `@register_forecaster("tên")`,
  biến thể trong `FORECASTERS` qua `model="lstm"`, `make_training_set`, `make_extra` (thông tin phụ: feature lịch, tên chuỗi);
  `simple.py` (naive, moving_avg, linear, oracle); `neural.py` (`NeuralForecaster`: vòng train, pinball loss, bảng ID +
  embedding, dùng chung cho LSTM/TFT); `lstm.py`; `tft.py` (Temporal Fusion Transformer tự viết, `explain()`);
  `metrics.py` (`evaluate_forecaster`, `evaluate_by_dataset`, pinball/độ phủ P10/P90).
  Forecaster cắm thẳng vào env (`forecaster=fc`) và `PredictiveThresholdPolicy(forecast_fn=fc.demand_fn())`.
  Thêm thuật toán: 1 class + import trong `forecast/__init__.py` + 1 dòng trong `FORECASTERS`.
- `examples/` — check_env, train_forecaster, explain_tft, crosscheck_tft, run_baselines, train_ppo, compare_ppo, play (nhập action bằng tay). Các script gọi `sys.stdout.reconfigure(encoding="utf-8")`
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
  `forecast_quantiles=(0.5, 0.9)` → state có cả P50 và P90 (env gọi `f(env, t, quantiles=q)`).

## Kết quả hiện có (tập test = 30% cuối SNDZoo, kịch bản normal) — chi tiết ở README mục 6–7
| Policy | Utilization | Oscillation | Vi phạm SLA |
|---|---|---|---|
| static | 0,17 | 0 | 0,0% |
| threshold (HPA) | 0,48 | 757 | 11,0% |
| vertical | 0,49 | 695 | 9,2% |
| predictive (oracle) | 0,48 | 966 | 4,8% |
| predictive (lstm, P50) | 0,48 | 1415 | 8,4% |
| predictive (lstm, P90) | 0,43 | 2026 | 2,6% |

Dự báo (H=5, L=30, MAE trên SNDZoo test): lstm (khoảng + thông tin phụ, train mọi dataset) 0,0389 · lstm_q 0,0395 ·
tft 0,0408 · lstm_snd (LSTM cũ) 0,0421 · lstm_point 0,0484 · linear 0,0582 · naive 0,0645.
TFT không thắng LSTM (chỉ 1 biến tải, L=30), nhưng cho hình giải thích; train lâu gấp 6.

Điểm đáng nêu: dự báo hoàn hảo giảm vi phạm SLA nhưng dao động nhiều nhất → động lực cho RL có phạt oscillation.
Cấp theo P90 giảm vi phạm SLA còn 2,6% nhưng tốn thêm ~13% vCPU và dao động nhiều hơn nữa.

## Việc tiếp theo
- Dự báo: thử TFT với L dài hơn (60–120) và thêm biến quan sát (cpu/mem) khi không dùng augment; thêm Lag-Llama/TimesFM.
- PPO với P50+P90 trong state: Duong tự chạy theo README mục 7 (3 seed mỗi cấu hình, rồi `compare_ppo.py`); tinh chỉnh `w_osc`, `w_invalid`.
- Giả định cần ghi trong luận văn: `peak_instances`, `vertical_alpha`, `vertical_delay`, ngưỡng SLA, base latency.

## Quy ước
- Trả lời và viết comment/docstring bằng tiếng Việt, giữ phong cách code hiện có.
