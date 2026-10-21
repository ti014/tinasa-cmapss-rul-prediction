# ETL theo subset FD001–FD004

**Học viên:** Phan Nguyễn Mai Phương · **MSHV:** 25002711.

Thư mục này lưu kết quả ETL Pandas của từng subset NASA C-MAPSS, mỗi subset một
thư mục riêng. FD002–FD004 chỉ đi qua bước ETL; mô hình chỉ được huấn luyện và
đánh giá trên FD001. Bốn subset không được union thành một bảng chung vì khóa
`(split, engine_id, cycle)` chưa có `dataset_id`.

## Số bản ghi

Số liệu lấy từ `FD00x/etl/run_info.json` (lần chạy ngày 25/09/2026, giờ Việt Nam).

| Subset | Observations (train + test) | Train targets | Động cơ train | Động cơ test |
|---|---:|---:|---:|---:|
| FD001 | 33.727 | 20.631 | 100 | 100 |
| FD002 | 87.750 | 53.759 | 260 | 259 |
| FD003 | 41.316 | 24.720 | 100 | 100 |
| FD004 | 102.463 | 61.249 | 249 | 248 |
| Tổng | 265.256 | 160.359 | 709 | 707 |

Ví dụ: FD002 có 53.759 dòng train và 33.991 dòng test, nên Silver observations có
87.750 dòng; `engine_summary` có 519 dòng, mỗi dòng một động cơ (260 train, 259 test).

## File

Mỗi `FD00x/etl/` gồm:

- `silver/observations.parquet`: observations train và test, giữ cột `split`.
- `silver/train_targets.parquet`: RUL train theo `(split, engine_id, cycle)`.
- `gold/engine_summary.parquet` và `gold/etl.sqlite`: bảng tổng hợp theo động cơ.
- `run_info.json`: checksum SHA-256 của hai file raw, số dòng, số động cơ và schema.

Gói nộp bài chỉ giữ README này và bốn file `run_info.json`; các file Parquet và
SQLite tạo lại được bằng lệnh bên dưới.

## Tạo lại

Chạy ETL CLI cho từng subset từ thư mục gốc repository, ví dụ FD002:

```powershell
uv run --frozen python -m cmapss.etl --dataset-id FD002 --output-dir outputs/cross_dataset/FD002/etl
```

Đổi `FD002` thành `FD001`, `FD003` hoặc `FD004` cho các subset còn lại. Lệnh đọc
`data/raw/train_FD00x.txt` và `data/raw/test_FD00x.txt`, kiểm tra schema và số dòng
theo profile của subset, rồi ghi Silver/Gold cùng `run_info.json`. CLI cũng ghi
`benchmark.json`; thêm `--benchmark-scales 1 --benchmark-repetitions 1` nếu chỉ cần
ETL.
