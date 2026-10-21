# Benchmark quy mô Pandas/Spark ngày 21/10/2026

**Học viên:** Phan Nguyễn Mai Phương — **MSHV:** 25002711

Câu hỏi: khi dữ liệu FD001 được nhân bản 1×–200×, một máy với Pandas còn xử lý
được đến đâu, và Spark local thay đổi điều gì?

| Mức | Pandas tính toán | Spark tính toán | RAM đỉnh Pandas | RAM đỉnh Spark |
|---|---|---|---|---|
| 1× | 11,5 giây | 12,7 giây | 0,3 GiB | 1,6 GiB |
| 10× | 88,6 giây | 31,4 giây | 1,7 GiB | 1,6 GiB |
| 50× | 583,7 giây | 105,7 giây | 7,3 GiB | 4,6 GiB |
| 100× | 1.075,4 giây | 196,3 giây | 12,9 GiB | 8,4 GiB |
| 200× | quá 30 phút, dừng | 392,2 giây | — | 11,4 GiB |

"Tính toán" gồm đọc Parquet, tổng hợp theo động cơ và tạo 126 đặc trưng window 30
cycle; không gồm ghi file. Mỗi mức một lượt. Máy: 22 luồng CPU, 23,4 GiB RAM; Spark
`local[4]`, heap driver 12 GiB.

## Hai lượt chạy và cách ghép

- `measurements.csv`, `run_info.json`: lượt đầy đủ cả hai engine, có bước ghi file.
  Spark thất bại từ 50× ở bước ghi: trên Windows thiếu `winutils` và Python worker
  của Spark bị crash, nên bảng đặc trưng phải gom về driver rồi ghi bằng PyArrow;
  bước gom này vượt `spark.driver.maxResultSize` ở 50× và hết heap ở 100×–200×.
- `spark_compute_only/`: lượt Spark với `--no-write`, đủ năm mức.
- `compute_only.csv`: do `slides/extract_slide_data.py` ghép; Pandas lấy
  `pipeline_seconds − write_seconds` từ lượt đầy đủ, Spark lấy lượt `--no-write`.

Lệnh tái lập (cần `JAVA_HOME`):

```powershell
uv run --frozen python -m cmapss.etl.scalability --scales 1 10 50 100 200 --repetitions 1 --timeout-seconds 1800 --spark-driver-memory 12g --output outputs/benchmarks/20261021 --work-dir tmp/scalability-20261021
uv run --frozen python -m cmapss.etl.scalability --engines spark --no-write --scales 1 10 50 100 200 --repetitions 1 --timeout-seconds 1800 --spark-driver-memory 12g --output outputs/benchmarks/20261021/spark_compute_only --work-dir tmp/scalability-20261021-spark
```

## Giới hạn

- Một máy, không phải cluster; Spark nhanh hơn ở đây nhờ chạy song song 4 luồng và
  xử lý theo partition, chưa đo lợi ích khi thêm máy.
- Pandas ở 200× dừng vì quá thời gian 30 phút. RAM Pandas tăng gần tuyến tính, nên
  200× ước tính cần khoảng 26 GiB, vượt RAM máy; đây là suy luận, chưa đo trực tiếp.
- Mỗi mức chỉ một lượt; lượt 20260929 (1×, 10×, ba lượt) cho kết quả cùng chiều.
