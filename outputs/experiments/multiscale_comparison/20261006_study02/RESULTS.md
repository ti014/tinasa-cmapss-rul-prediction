# Kết quả study HGB multi-scale

**Tác giả:** Phan Nguyễn Mai Phương — **MSHV 25002711**  
**Ngày:** 2026-10-06  
**Run:** `20261006_study02`

## Kết luận ngắn

Study đầy đủ đã chạy 72 fits trên Windows với 5 outer folds và 3 inner folds
theo engine. `hgb_multiscale10_30` giảm trajectory RMSE điểm trên outer
predictions từ **29,8068** xuống **29,4802**, nhưng khoảng tin cậy 95% là
`[-0,6886; 0,0596]` và các kiểm soát endpoint/on-time không đạt đồng thời.
Vì vậy promotion gate **không đạt** và `selected_pipeline` outer vẫn là
`hgb_baseline169` ở cả 5 folds.

Final inner fit trên toàn bộ train chọn `hgb_multiscale10_30`; lựa chọn này được
lưu để tái lập và dùng cho test reporting. Test là reporting-only vì đã được
kiểm tra trong lịch sử, không phải bằng chứng độc lập để thay đổi gate.

## So sánh outer

| Arm | Trajectory RMSE | Endpoint MAE | Endpoint NASA | On-time | Early burden |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hgb_baseline169` | 29,8068 | 9,8913 | 6,8838 | 0,94 | 0,05216 |
| `hgb_multiscale10_30` | 29,4802 | 9,9876 | 7,5664 | 0,91 | 0,05273 |
| `hgb_initial_profile42` | 33,1280 | 13,1723 | 41,0964 | 0,95 | 0,07244 |

Paired bootstrap 10.000 lần cho multi-scale so với baseline:

- Trajectory RMSE: delta **−0,3267**, CI `[-0,6886; 0,0596]`.
- Endpoint MAE: delta **+0,0963**, CI `[-0,2417; 0,4360]`.
- Endpoint NASA: delta **+0,6827**, CI `[-0,7503; 3,0749]`.
- On-time rate: delta **−0,0300**, CI `[-0,0800; 0,0100]`.
- Early-alarm burden: delta **+0,00057**, CI `[-0,00369; 0,00478]`.

## Test reporting

Trên FD001 test đã được xem trong lịch sử, multi-scale có RMSE **19,9166**,
endpoint MAE **13,5042** và NASA **2321,63**; baseline tương ứng là
**20,9334**, **13,7534** và **3744,35**. Các số này chỉ mô tả kết quả reporting,
không dùng để chọn arm hoặc mở promotion gate.

## Kiểm tra độc lập

`verification.json` báo **passed** với **41/41 checks**; gồm parity 23 baseline
CV vectors và final model gốc, causality của 337 union features, tách engine
fit/validation, cache checksum, policy selection, pooled RMSE bootstrap và thứ
tự đọc test labels. Có **69 CV cache records + 3 final models = 72 fits**.

Artifacts chính:

- `aggregate_metrics.csv`, `paired_comparison.csv`, `region_metrics.csv`.
- `outer_predictions.parquet`, `final_predictions.parquet`.
- `decision.json`, `selection.json`, `baseline_replay.json`, `verification.json`.

Kết luận sử dụng cho bài: giữ `hgb_baseline169` làm pipeline được promote; ghi
`hgb_multiscale10_30` như hướng có tín hiệu cải thiện RMSE nhưng chưa chứng minh
được cải thiện an toàn trên outer folds. Không promote initial-profile42.
