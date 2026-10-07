**Phan Nguyễn Mai Phương — MSHV 25002711**

# Kết quả so sánh HGB và LightGBM

## Kết luận

Trong protocol này, **không chọn LightGBM thay HGB**. LightGBM uncapped có trajectory RMSE thấp hơn HGB uncapped một lượng nhỏ (29,653 so với 29,807), nhưng paired 95% CI của chênh lệch là `[-0,479; 0,186]`, nên chưa chứng minh được cải thiện. Endpoint mean-engine NASA của LightGBM uncapped là **20,988**, xấu hơn HGB uncapped **6,884**. Promotion gate không đạt. Procedure được chọn trong cả năm outer folds và final inner selection đều là `hgb_uncapped`.

Kết quả này loại LightGBM khỏi lựa chọn của protocol hiện tại. Nó chưa phải bằng chứng rằng LightGBM luôn kém trên mọi feature, cap nhãn, seed hoặc cách calibration khác.

## Thiết lập

- Full nested study: 5 outer folds, 3 inner folds, seed 42, engine-grouped split.
- Sáu candidates: hai HGB references và bốn LightGBM candidates (uncapped/cap125 × `reg_lambda` nominal 0/10).
- Cùng frozen **169 causal features**, raw-RUL evaluation và engine-total-one weights.
- LightGBM: 200 trees, learning rate 0,05, 31 leaves, `min_child_samples=20`, `max_bin=255`, CPU, deterministic/force-col-wise, 4 threads.
- Tổng số fits: **132**; thời gian chạy: **473,43 giây**.
- Tác giả, source identity, dependency lock và input hashes nằm trong `manifest.json`.

## Outer readout

| Procedure | Trajectory RMSE | Endpoint MAE | Endpoint mean-engine NASA | On-time | Early-alarm burden |
|---|---:|---:|---:|---:|---:|
| HGB uncapped | 29,806846 | 9,891336 | 6,883768 | 0,94 | 0,052159 |
| LightGBM uncapped | **29,653172** | 9,943892 | **20,988312** | 0,98 | 0,055359 |
| HGB cap125 | 36,208014 | **9,321925** | **3,683556** | 0,94 | 0,055003 |
| LightGBM cap125 | 36,098799 | 9,333136 | 3,859042 | 0,93 | 0,056454 |

LightGBM uncapped so với HGB uncapped có delta trajectory RMSE `-0,153674`, CI `[-0,479373; 0,185828]`; delta endpoint MAE `+0,052555`, CI `[-0,379974; 0,515986]`; delta endpoint NASA `+14,104544`, CI `[-1,406930; 44,210609]`. Với cap125, LightGBM giảm RMSE `0,109215` (CI `[-0,188614; -0,025010]`) nhưng endpoint MAE và NASA không tốt hơn HGB cap125 trong controls đã đóng băng. Endpoint panel dùng các fractions 50%, 70% và 90%; endpoint NASA trong bảng là trung bình theo engine của các điểm này, nên khác NASA tổng trong test reporting.

Các CI là paired bootstrap 10.000 lần trên engine, conditional trên outer predictions cố định. Chúng không bao gồm biến thiên khi huấn luyện lại.

## Final selection và test reporting

Inner selection trên toàn bộ train chọn:

- `hgb_uncapped` cho arm uncapped;
- `hgb_cap125` cho arm cap125;
- LightGBM uncapped `reg_lambda=10` là cấu hình được chọn trong tập ứng viên đạt numeric family controls;
- LightGBM cap125 `reg_lambda=0` là diagnostic winner nhưng không đạt numeric family controls.

Final models được lưu và reload với prediction replay exact trước khi đọc nhãn test. Test chỉ dùng để reporting:

| Procedure | MAE | RMSE | NASA tổng |
|---|---:|---:|---:|
| HGB uncapped | 13,753450 | 20,933433 | 3744,351401 |
| LightGBM uncapped | **13,037866** | **19,716541** | **3554,088413** |
| HGB cap125 | 9,375761 | **12,549297** | **212,123270** |
| LightGBM cap125 | **9,334126** | 12,817272 | 214,655286 |

Các số test này không được dùng để chọn model, policy, procedure hoặc thay đổi gate. Test labels đã được đọc sau khi selection, models và final predictions được đóng băng.

## Giới hạn diễn giải

- Folds và test đã từng được xem trong các nghiên cứu trước; đây là follow-up có historical inspection, không phải xác nhận độc lập hoàn toàn.
- Bootstrap là conditional trên predictions cố định và không đo seed variability hoặc retraining variability.
- Kết luận chỉ áp dụng cho frozen 169 features, split, weights, target caps và LightGBM grid của protocol này.
- Không thay published model hoặc registry. Artifact chính: [verification.json](verification.json), [aggregate_metrics.csv](aggregate_metrics.csv), [paired_comparison.csv](paired_comparison.csv), [decision.json](decision.json), [selection.json](selection.json), [test_metrics.csv](test_metrics.csv), [test_evaluation.json](test_evaluation.json) và [final_predictions_receipt.json](final_predictions_receipt.json).

Đã đạt [22 focused tests](tests_delivery.xml) và [read-only verification](verification.json): 108 inner fits, 20 outer fits, 4 final fits; dự đoán từ model đã lưu trùng chính xác. Verification được chạy trong worktree thực nghiệm với source và lock đã đóng băng. Bản bàn giao về project chính giữ nguyên byte của artifacts; dùng run ID mới nếu thử nghiệm tiếp với source hoặc lock khác.

Protocol chi tiết: [LIGHTGBM_COMPARISON_PROTOCOL.md](../../../../experiments/LIGHTGBM_COMPARISON_PROTOCOL.md).
