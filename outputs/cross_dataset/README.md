# Evaluation FD001--FD004

Artifact này ghi kết quả chạy cùng pipeline trên bốn subset NASA C-MAPSS và ba
phép transfer từ FD001. Kết quả được tạo ngày 2026-09-25 từ raw files trong
`data/raw`.

## Protocol

- Model candidates và GroupKFold giữ nguyên từ pipeline modeling hiện tại.
- Mỗi subset có một evaluation nội bộ: train và test thuộc cùng subset.
- Transfer dùng model đã fit trên toàn bộ train FD001, sau đó dự đoán test của
  FD002, FD003 hoặc FD004; không fit lại trên target test.
- Matrix dùng trailing window `10 cycles` vì FD004 có test engine ngắn nhất chỉ
  19 cycles. Kết quả này không thay thế kết quả FD001 chính thức dùng window 30.
- Test labels chỉ được đọc sau khi final predictions đã được ghi.
- Bootstrap dùng 10.000 engine resamples, seed 42.

## Results

| Source | Target | Type | Mean engine MAE | RMSE | NASA score |
|---|---|---|---:|---:|---:|
| FD001 | FD001 | within-subset | 21.9258 | 29.5936 | 8,103.0148 |
| FD002 | FD002 | within-subset | 22.3556 | 31.3077 | 34,004.9782 |
| FD003 | FD003 | within-subset | 28.3255 | 41.7606 | 4,109,107.7036 |
| FD004 | FD004 | within-subset | 27.2218 | 36.9311 | 492,703.0709 |
| FD001 | FD002 | transfer | 60.3215 | 70.4603 | 14,193,434.0842 |
| FD001 | FD003 | transfer | 47.3949 | 59.2140 | 59,380,624.2646 |
| FD001 | FD004 | transfer | 58.5731 | 69.8148 | 14,691,764.3175 |

Transfer từ FD001 làm MAE tăng rõ rệt so với evaluation nội bộ trên target
subset. Model fit riêng trên FD001 không nên được dùng trực tiếp cho FD002--FD004
nếu chưa có retraining hoặc domain adaptation.

## Files

- `matrix.json`: bảng tổng hợp toàn bộ evaluation.
- `FD001/`--`FD004/`: ETL và rolling features theo subset.
- `evaluations/FD*_to_FD*/`: predictions, metrics, bootstrap CI và context.
