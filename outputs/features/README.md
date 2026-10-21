# Rolling features FD001

Pipeline tạo Gold rolling features từ Silver observations. Artifact
này chuyển chuỗi cảm biến thành bảng đặc trưng deterministic cho các bước mô
hình sau, nhưng chưa fit scaler, chưa chọn feature, chưa huấn luyện mô hình và
chưa đọc nhãn test `RUL_FD001.txt`.

## Cách chạy

Từ thư mục gốc, sau khi đã có `outputs/etl/silver/observations.parquet`:

```powershell
.venv\Scripts\python -m cmapss.features `
  --silver-observations outputs/etl/silver/observations.parquet `
  --output-dir outputs/features
```

Lệnh mặc định dùng chính hai đường dẫn trên. Nếu Silver observations chưa có,
chạy ETL trước bằng `python -m cmapss.etl`.

## Schema và artifact

| Lớp | Tệp | Nội dung | Số dòng hiện tại |
| --- | --- | --- | ---: |
| Gold | `gold/rolling_features.parquet` | Khóa `(split, engine_id, cycle)` và 126 feature columns | 27.927 |
| Metadata | `run_info.json` | Cấu hình window, checksum input, số dòng và contract feature | 1 |

Feature artifact giữ đủ 21 sensors. Mỗi sensor có sáu cột theo mẫu
`<sensor>__<stat>`:

- `last`: giá trị tại cycle cuối window.
- `mean`: trung bình của 30 observations trong window.
- `std`: sample standard deviation với `ddof=1`.
- `min` và `max`: giá trị nhỏ nhất và lớn nhất trong window.
- `slope`: hệ số hồi quy tuyến tính theo vị trí 1--30 trong window.

Số dòng hiện tại:

| Split | Engines | Rolling windows |
| --- | ---: | ---: |
| Train | 100 | 17.731 |
| Test | 100 | 10.196 |
| Tổng | 200 | 27.927 |

Ví dụ: nếu `sensor_1` bằng đúng số cycle trong một engine, row tại cycle 30 dùng
window 1--30 và có `sensor_1__last=30`, `sensor_1__mean=15,5`,
`sensor_1__min=1`, `sensor_1__max=30`, `sensor_1__slope=1`. Row tại cycle 31
chỉ dùng window 2--31; thay đổi giá trị ở cycle 31 không làm đổi feature của
row cycle 30.

## Quy tắc chống leakage

- Window không vượt qua ranh giới engine hoặc split.
- Mỗi output row chỉ dùng 30 cycle kết thúc tại chính key của row đó.
- Engine dưới 30 cycle bị từ chối; pipeline không padding, không impute và
  không rút ngắn window.
- `RUL_FD001.txt` không được đọc hoặc join ở bước này.
- Artifact giữ đủ 21 sensors để chưa thực hiện feature selection trên toàn bộ
  train. Nếu bước mô hình muốn loại sensor hằng, quyết định đó phải nằm trong
  training fold.

## Kiểm tra

- `tests/test_features.py`: kiểm tra giá trị tính tay cho sensor bằng cycle,
  sample standard deviation, slope, sensor hằng, không dùng tương lai, không
  nối qua engine và negative proof cho engine dưới 30 cycle.
- Lần chạy FD001 tạo 27.927 dòng, 129 cột gồm 3 khóa và 126 feature columns.
- Parquet được đọc lại trong test round-trip.
- `run_info.json` ghi `joins_test_labels=false` và `modeling.status=not_started`.

Giới hạn: artifact này chưa có fold assignment, scaler, PCA, model predictions
hoặc metric. Các bước đó thuộc workflow modeling riêng.
