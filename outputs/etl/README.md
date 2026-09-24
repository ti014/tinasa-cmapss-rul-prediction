# ETL Silver/Gold FD001

Mốc M3.1 đã hiện thực hóa pipeline Pandas cho Bronze → Silver → Gold và chạy
PySpark parity trên hai file observations FD001. Pipeline đọc raw qua
reader/validator hiện có, giữ checksum input, ghi observations typed vào Silver
Parquet, tạo train RUL trong Silver và tổng hợp một dòng cho mỗi
`(split, engine_id)` vào Gold Parquet và SQLite.

Rolling features window 30 được ghi riêng trong [`outputs/features`](../features/README.md); predictions và metrics chưa thuộc artifact ETL này.
Contract rolling vẫn được giữ riêng trong [`data/CONTRACTS.md`](../../data/CONTRACTS.md)
để không biến engine summary thành kết quả mô hình.

## Cách chạy

Từ thư mục gốc, sau khi cài `requirements-dev.txt`:

```powershell
.venv\Scripts\python -m cmapss.etl `
  --raw-dir data/raw `
  --output-dir outputs/etl `
  --benchmark-scales 1,10,100 `
  --benchmark-repetitions 2
```

PySpark là optional extra và cần Java runtime:

```powershell
uv sync --frozen --extra etl
.venv\Scripts\python -m cmapss.etl `
  --with-spark `
  --raw-dir data/raw `
  --output-dir outputs/etl
```

`--with-spark` chạy parity theo schema, khóa, số dòng và giá trị số giữa Pandas
và PySpark. Lần chạy bàn giao ngày 22/09/2026 dùng JRE portable trong `tmp/`
để không cài vào hệ thống; artifact public vẫn chỉ là kết quả parity tại
`outputs/etl/parity.json`.

## Schema và artifact

| Lớp | Tệp | Nội dung | Số dòng hiện tại |
| --- | --- | --- | ---: |
| Silver | `silver/observations.parquet` | `split`, khóa observation, 3 settings và 21 sensors | 33.727 |
| Silver | `silver/train_targets.parquet` | Khóa train và `rul = max_cycle - cycle` | 20.631 |
| Gold | `gold/engine_summary.parquet` | `n_cycles`, `first_cycle`, `last_cycle` theo engine | 200 |
| Gold | `gold/etl.sqlite` | Hai bảng trên và index theo khóa | 200 engine rows |

`run_info.json` ghi schema, row counts, số engine, checksum raw và trạng thái
rolling features. `benchmark.json` ghi runtime Python/Pandas, nền tảng, số lần
lặp cold/warm và kết quả ở 1×, 10×, 100×.

Ví dụ: engine train 1 có toàn bộ lịch sử đến cycle cuối thì target tại cycle
cuối bằng `0`; engine summary của cùng engine ghi tổng số cycle và hai ranh giới
cycle, không làm thay đổi raw observations.

## Kết quả Pandas scalability benchmark

Benchmark chỉ nhân bản observations trong bộ nhớ và remap engine ID; bản sao
không được ghi vào Silver/Gold khoa học hoặc dùng huấn luyện. Lần chạy hiện tại
dùng Python 3.12.14, Pandas 2.3.3 trên Windows 11:

| Scale | Input rows | Output engine rows | Cold (s) | Warm (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1× | 33.727 | 200 | 0,0195 | 0,0196 |
| 10× | 337.270 | 2.000 | 0,1589 | 0,1474 |
| 100× | 3.372.700 | 20.000 | 1,6752 | 1,6732 |

Các số đo này chỉ đo transform `build_engine_summary` trên bảng đã ở trong
bộ nhớ; không gồm replication, đọc raw hay ghi Parquet/SQLite. Nhãn cold/warm
trong metadata chỉ là lần 1/lần 2 liên tiếp, không phải cache được kiểm soát.
Pandas materialize kết quả ngay; chưa đo runtime PySpark. Hai lần đo mỗi scale
chưa đủ đánh giá biến thiên runtime; không kết luận tốc độ cluster.

## Kiểm tra

- `tests/test_etl.py`: 5 tests đạt, gồm schema/row count, parity positive và
  negative proof, remap engine ID, benchmark isolation và round-trip
  Parquet/SQLite.
- `outputs/etl/run_info.json`: raw SHA-256 khớp các checksum đã ghi nhận.
- `outputs/etl/parity.json`: Pandas/PySpark parity đạt cho Silver observations
  33.727 dòng, Silver train targets 20.631 dòng và Gold engine summary 200
  dòng; `max_abs_diff=0.0` ở cả ba bảng.
- Raw files không bị ghi đè; output path được chặn nếu nằm trong `data/raw`.
- PySpark trên Windows vẫn phát cảnh báo thiếu `winutils.exe` và native Hadoop;
  parity local trả mã 0 và không ghi HDFS hoặc cluster artifact.
