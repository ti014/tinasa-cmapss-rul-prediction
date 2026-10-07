# Đối chiếu HGB giữa Linux và Windows

**Học viên:** Phan Nguyễn Mai Phương  
**MSHV:** 25002711  
**Ngày:** 05/10/2026

Windows chạy lại xác nhận cùng thứ tự kết quả với Linux: HGB history không cắt nhãn có RMSE toàn quỹ đạo thấp hơn; HGB cap125 + history có endpoint MAE thấp hơn. Metrics gần nhau, nhưng prediction vectors của lượt tính lại features chưa trùng tuyệt đối.

## Kết quả chạy mới

| Candidate | RMSE quỹ đạo Linux | RMSE quỹ đạo Windows | Delta Windows − Linux | Endpoint MAE Linux | Endpoint MAE Windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| hgb_baseline | 33.693474 | 33.695182 | +0.001708 | 15.189438 | 15.198737 |
| hgb_history | 28.833687 | 28.835035 | +0.001348 | 10.565528 | 10.550971 |
| hgb_cap125_history | 36.537842 | 36.531259 | −0.006583 | 9.763407 | 9.758693 |

Đơn vị: cycles. Đây là final-inner CV trên 100 training engines, với 17.731 prediction rows mỗi candidate và 300 retrospective endpoint rows ở 50%, 70%, 90% lifetime. Evaluation truth luôn là raw RUL.

Ví dụ: history không cắt nhãn đạt RMSE khoảng 28,835 trên Windows, thấp hơn khoảng 7,696 cycles so với cap125 + history. Endpoint MAE lại giảm từ 10,551 xuống 9,759 khi dùng cap125. Vì vậy, lựa chọn phụ thuộc mục tiêu đã chốt trước thí nghiệm.

## Phạm vi và điều kiện đối chiếu

- Source commit: `919450a2e773f030f1e943dbe952814b0c337f30`.
- Linux reference: `outputs/experiments/audit_remediation/20261005_full0750`.
- Windows replay: `outputs/experiments/audit_remediation/20261005_windows_inner_replay`.
- Chạy 9 fits mới: baseline, history, cap125 + history × 3 GroupKFold. Không dùng Linux fit cache để tạo predictions Windows.
- Window30, 126 baseline features; history thêm cycle age và 42 sensor features so với first30 mean, tổng169 features.
- HGB200 iterations, learning_rate0.05, max_leaf_nodes31, min_samples_leaf20, l2=0, early_stopping=False, seed42, bốn fit threads; mỗi engine có tổng sample weight bằng1.
- Python3.12.14, scikit-learn1.9.1, NumPy/pandas2.3.3, joblib1.6.0 trùng Linux.
- SHA256 khớp cho12 source files, hai canonical inputs, năm baseline inputs và dependency lock sau khi chuẩn hóa LF trong bản worktree riêng. Workspace gốc được dùng để lưu bản sao kết quả.
- Mỗi fold có engine IDs, candidate spec, feature names/thứ tự, prediction keys và raw truth khớp chính xác với cache Linux.
- Chỉ chạy lại final-inner subset. Chưa chạy lại outer CV, final test hoặc full study. Không đọc test labels trong lượt replay; kết quả này không tạo promotion decision.

Lượt9 fits hoàn tất trong80,17 giây trên Windows. Dùng interpreter Windows hiện có và source trong worktree riêng. Một virtual environment mới do uv tạo gặp Windows mount error448 khi chạy interpreter trampoline; môi trường đang hoạt động có đúng versions đã kiểm chứng được sử dụng.

## Sai khác số học và kiểm tra có kiểm soát

So với Linux, số predictions khác nhau là5.882/17.731 ở baseline,5.877/17.731 ở history và5.976/17.731 ở cap125 + history. Max absolute prediction delta tương ứng1,796952;1,967917;1,187427 cycles. Không nới tolerance để gọi các vector này là exact match.

Windows tính lại126 rolling features trên27.927 rows có197.348 giá trị khác Linux, với max absolute feature delta **7,105427357601002 × 10⁻¹⁵**.

Để kiểm tra nguyên nhân, chạy thêm một fit baseline ở inner fold1 bằng chính rolling feature values Linux đã lưu, giữ nguyên model, labels, engine IDs, sample weights, seed và threads. Kết quả tái tạo **5.878/5.878 predictions Linux trùng tuyệt đối**, max delta0. Trước khi thay features, cùng fold này lệch tối đa1,796952 cycles.

Kiểm tra này xác định nguồn sai khác của fold baseline đã kiểm tra nằm ở feature arithmetic. Chưa thực hiện kiểm tra tương tự cho mọi fold và cả hai history candidates.

## Bằng chứng

- [Kiểm tra source, inputs, versions](preflight.json).
- [Bảng metrics đầy đủ](metrics_comparison.csv).
- [Đối chiếu cache và prediction vectors](comparison.json).
- [Sai khác features](feature_comparison.json).
- [Kiểm tra bằng features Linux đã lưu](linux_feature_control.json).
- [Prediction vectors Linux và Windows](prediction_comparison.parquet).
- Raw Windows fits, metadata, frozen source và manifest nằm trong thư mục `audit_remediation/20261005_windows_inner_replay` đã nêu.

Nhận xét trước khi fetch remote dựa trên checkout ngày29/09. Commit05/10 có đánh giá endpoint trong validation và dùng history169 features; báo cáo này sử dụng đúng phiên bản05/10.

