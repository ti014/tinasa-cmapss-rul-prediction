**Phan Nguyễn Mai Phương — MSHV 25002711**

# TCN pilot — FD001, 05/10/2026

Đây là pilot sàng lọc kiến trúc, không phải kết quả promotion. Pilot dùng 21
raw sensors, sequence dài 30 cycles, 5 outer folds theo engine và 8 epochs cố
định. Không đọc test labels và không thay published model.

## Kết quả outer pilot

| Arm | RMSE trung bình theo fold | MAE trung bình theo fold |
| --- | ---: | ---: |
| TCN uncapped | 36,60 | 24,63 |
| TCN cap125 | 41,83 | 27,34 |

Chi tiết từng fold nằm trong [outer_metrics.csv](outer_metrics.csv). TCN
uncapped tốt hơn TCN cap125 trong cả năm folds, nhưng vẫn kém HGB uncapped
outer trajectory RMSE **29,81** của nghiên cứu trước. Vì ngân sách 8 epochs
có thể còn underfit, pilot không được dùng để kết luận TCN kém ở mọi cấu hình.

## Protocol và giới hạn

- Cửa sổ tại cycle `t` dùng đúng cycles `t-29..t` của cùng engine.
- Standardization fit riêng trên fitting engines của từng fold.
- Target cap125 chỉ áp dụng khi fit; metric dùng raw RUL.
- Không có inner selection, paired gate, warning policy hoặc test evaluation.
- Kiến trúc chi tiết và lệnh full nested CV nằm trong
  [TCN_COMPARISON_PROTOCOL.md](../../../../experiments/TCN_COMPARISON_PROTOCOL.md).

Lượt nested 50 epochs được bắt đầu rồi dừng trước khi có kết quả đầy đủ, vì
runner chưa có endpoint panel và warning/provenance safeguards đã chốt. TCN
chưa có bằng chứng xác nhận; không loại kiến trúc từ pilot 8 epochs. Trước khi
chạy tiếp cần hoàn thiện protocol và dùng run ID mới.
