# Kết quả kiểm tra dữ liệu FD001

Ngày 2026-09-22, package data contract đã qua **63 tests** và validation trên
FD001 thật. Đây là bằng chứng cho bước chuẩn bị dữ liệu, không phải kết quả
huấn luyện hoặc đánh giá mô hình.

| Artifact | Nội dung | Kết quả |
| --- | --- | --- |
| [fd001.json](fd001.json) | Observations train/test và train targets; không đọc test labels | Passed |
| [fd001-integrity.json](fd001-integrity.json) | Như trên, thêm kiểm tra riêng tính toàn vẹn test labels | Passed |

Đã xác nhận 26 cột nguồn, 20.631 dòng train, 13.096 dòng test, 100 engines trong
mỗi split và 100 test labels. Train targets có đủ 20.631 dòng trước/sau bước
tạo nhãn. Ba checksum SHA-256 sau khi chạy trùng với checksum trước triển khai.
Giá trị nhãn test không được ghi trong báo cáo và không được join observations.

Ví dụ positive proof: train engine có cycle `1, 2, 3, 4` nhận RUL `3, 2, 1, 0`.
Negative proof phát hiện sai schema, giá trị không hợp lệ, trùng khóa, đứt
cycle, ánh xạ nhãn sai, RUL sai hoặc truyền test vào hàm tạo train targets.

Lệnh kiểm thử đã chạy:

```powershell
uv run --frozen pytest -q --basetemp=tmp/pytest-contract-final --tb=short
```

Kết quả: `63 passed in 5.58s`. Thư mục Temp mặc định trên Windows bị từ chối
quyền truy cập; thư mục riêng trong `tmp/` đã chạy thành công. Xem
[README](../../README.vi.md#bắt-đầu) để tạo môi trường và chạy lại validation.
Timestamp trong JSON dùng UTC, tương ứng ngày 2026-09-22 theo Asia/Saigon.

Local validation đã chạy; chưa có optional hook hoặc CI invocation cho các
kiểm tra này. Branch protection chưa được xác minh. Rolling features,
predictions và leakage tại pipeline mô hình chưa có bằng chứng runtime.

Sau refactor reader, exception và hướng dẫn sử dụng trong cùng ngày, toàn bộ
63 tests tiếp tục đạt (`5.20s`, thư mục test `tmp/pytest-readability-1`). Lệnh
chính `uv run cmapss-validate` đã chạy thành công và cập nhật `fd001.json`.
