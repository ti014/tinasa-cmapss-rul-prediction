# Kết quả cải thiện RUL — 27/09/2026

Đã chạy nested GroupKFold 5×3, window 10/20/30, 24 cấu hình HGB và ba phương
án trọng số. Giữ baseline theo tiêu chí đã chốt. Run mất 78,61 phút; 511 fits
riêng được cache và một fit cuối. Cache không đưa vào Git.

| Outer evaluation trên 100 train engines | Baseline | Quy trình mới |
| --- | ---: | ---: |
| MAE toàn bộ (cycles) | 20,2143 | 20,2459 |
| MAE RUL ≤30 (cycles) | 3,6676 | 3,6730 |
| Cảnh báo trễ/bỏ lỡ | 49/100 | 53/100 |
| Tỷ lệ thời gian báo sớm | 1,2933% | 1,2580% |

Cấu hình cuối trên toàn train trùng baseline; MAE test vẫn 15,3490 cycles.
Ví dụ `hgb_08` giảm low-RUL MAE 9,75% trong inner CV nhưng tăng MAE toàn bộ
0,32 cycles. Đây là kết quả lựa chọn, chưa phải đánh giá ngoài độc lập.

- [Cách chạy, protocol và giới hạn](../../../../experiments/README.md).
- [Manifest trước fit](search_manifest.json), [thông tin run](run_info.json).
- [So sánh bốn stage](ablation_metrics.csv), [chênh lệch và CI](paired_comparison.csv).
- [Lựa chọn cuối](final_selection_metrics.csv), [so sánh test](test_comparison.csv).
- [Validation mô hình](validation.json), [AppTest](dashboard_validation.json).
- `baseline_snapshot/` lưu mô hình gốc; `baseline_documents/` giữ tài liệu trước sửa.
- `source_snapshot/` lưu mã dùng để chạy. `fit_cache/` chỉ có trên máy chạy;
  clone mới sẽ fit lại các phần thiếu, sau khi kiểm tra manifest.

Test FD001 đã được xem trong phân tích trước đợt này. Không tuning theo kết
quả test mới. Khoảng tin cậy paired bootstrap chỉ phản ánh biến thiên trên
predictions đã fit, không bao gồm refit. Chưa đánh giá chi phí/lịch bảo trì thực tế.
