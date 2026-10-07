# Protocol pilot TCN trên C-MAPSS FD001

**Tác giả:** Phan Nguyễn Mai Phương — MSHV 25002711

Thí nghiệm này là nhánh nghiên cứu độc lập, không thay published model hoặc
registry. TCN nhận trực tiếp 21 raw sensors theo cửa sổ nhân quả dài 30 chu kỳ.
Mỗi cửa sổ kết thúc tại chu kỳ `t` dùng đúng các quan sát `t-29..t` của cùng
engine. Kiến trúc gồm projection `21 -> 32`, bốn residual blocks với hai causal
`Conv1d` trong mỗi block, dilations `1, 2, 4, 8`, kernel size 3, ReLU và
dropout 0.10; head là `Linear(32, 1)` đọc timestep cuối.

Hai arms được khai báo trước: `tcn_uncapped` (nhãn fitting nguyên bản) và
`tcn_cap125` (chỉ cắt nhãn fitting ở 125). Mọi metric dùng raw RUL. Sensor
standardization được fit trên các engine thuộc fitting fold. GroupKFold theo
`engine_id` dùng 5 outer folds và 3 inner folds; chọn theo trajectory RMSE.
`test_labels` chỉ được đọc sau khi hai final model và prediction đã lưu.

Runner hiện là scaffold thăm dò, chưa đủ protocol chọn mô hình: endpoint của
runner là cycle cuối ở RUL=0, chưa phải panel tại 50/70/90% lifetime; chưa có
warning calibration, paired promotion gate, source snapshot hoặc model replay
receipt. Vì vậy `scientific=False` cho mọi cấu hình và không đưa kết quả vào
registry. TCN pilot chỉ sàng lọc pipeline; khác thông tin đầu vào với HGB169
có age/history nên không quy kết riêng hiệu quả kiến trúc.

Chạy smoke:

```powershell
$env:PYTHONPATH = "$PWD/src"
& .venv-xgb/Scripts/python.exe scripts/run_tcn_comparison.py `
  --output-dir outputs/experiments/tcn_comparison/20261005_smoke `
  --dry-run --skip-test-evaluation
```

Full run cần PyTorch tương thích host. Linux ARM phải cài build PyTorch phù hợp
trước khi chạy; runner không tự thêm dependency vào published environment.

Lượt 50 epochs được bắt đầu rồi dừng khi rà soát phát hiện các thiếu sót trên.
Không có kết quả đầy đủ hay quyết định loại TCN từ lượt đó. Pilot 8 epochs có
thể underfit; chưa đủ bằng chứng kết luận TCN không phù hợp.
