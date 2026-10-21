# Mạch trình bày của dự án

**Học viên:** Phan Nguyễn Mai Phương — **MSHV:** 25002711

**Giảng viên hướng dẫn:** TS. Võ Thị Thanh Vân

File này là mạch trình bày chung cho slide, lời nói (`handoff/SPEECH_SCRIPT_VI.tex`)
và dashboard. Mỗi con số phải lấy từ file nguồn ghi ở bảng dưới.

Report không dùng file này làm khung: report viết theo văn phong học thuật và khung
chương chuẩn, chỉ dùng chung số liệu và thuật ngữ (xem `AGENTS.md`, mục "Mạch trình bày").

## Giọng văn

Nghiêm túc, nói như một học viên trình bày trước hội đồng. Không dùng nhân vật, ẩn dụ
hay câu đùa ("đèn vàng/đèn đỏ", "vào xưởng", "câu hỏi buổi sáng"...). Hai mức cảnh
báo gọi là **cảnh báo mức một** (Health Index báo bất thường) và **cảnh báo mức hai**
(RUL dự đoán ≤ 30 cycle).

## Bài toán

Động cơ turbofan gửi 21 số đo cảm biến sau mỗi cycle. Bảo trì quá sớm gây lãng phí,
quá muộn có thể dẫn đến hỏng hóc. Bài xây dựng pipeline dữ liệu để phát hiện bất
thường, dự đoán RUL và lập lịch bảo trì cho 100 động cơ test. Động cơ train 1 minh
họa phần phát hiện và cảnh báo; động cơ test 90 minh họa một dòng của lịch bảo trì.

## Số liệu chính theo nội dung

| Nội dung | Số chính | Hình | Nguồn |
|---|---|---|---|
| Nguồn và cấu trúc dữ liệu | 4 tập con, 265.256 bản ghi; FD001 33.727 bản ghi, 26 trường | Bảng bốn tập con, bốn dòng đầu của `train_FD001.txt` | `outputs/acquisition/manifest.json`, `outputs/cross_dataset/FD00x/etl/run_info.json` |
| Chất lượng dữ liệu | 0 giá trị thiếu, 0 khóa trùng, 7 cột không đổi; tuổi thọ 128–362 cycle | Phân phối tuổi thọ (notebook 01) | `outputs/etl/silver/observations.parquet`, `outputs/eda/tables/` |
| Lưu trữ Bronze–Silver–Gold | observations 33.727 × 27; engine_summary 200 × 5; đặc trưng 27.927 × 129 | Sơ đồ luồng dữ liệu của report | `outputs/etl/`, `outputs/features/gold/` |
| HDFS, MapReduce, YARN | 3 map + 1 reduce task; 33.727 bản ghi vào, 200 dòng ra; fsck HEALTHY; khớp 200/200 | Ảnh chụp YARN, log `hdfs dfs -ls` | `outputs/hdfs/` |
| PySpark và Pandas | 3 bảng ETL lệch 0; đặc trưng 27.927 dòng khớp (sai số < 10⁻⁷) | Bảng đối chiếu, đoạn code Window | `outputs/etl/parity.json`, `outputs/benchmarks/20260929/run_info.json` |
| Truy vấn | SQL trên SQLite và Spark SQL window function | Câu truy vấn và kết quả chạy thật | `outputs/etl/gold/etl.sqlite` |
| Biến đổi | 6 bước, ra 17.731 × 126 mẫu train | Bảng sáu bước | `outputs/etl/run_info.json`, `outputs/features/run_info.json` |
| Mở rộng quy mô | 100×: Spark nhanh hơn Pandas 5,5 lần; 200×: Pandas quá 30 phút, Spark 392 giây | Đường cong thời gian và RAM | `outputs/benchmarks/20261021/` |
| Làm mượt EWMA | giảm khoảng 86% dao động; không làm mượt thì cảnh báo mức một còn 72,5 cycle thay vì 121 | Sensor 11 và 4 của động cơ train 1 | `outputs/degradation/` |
| Xu hướng suy giảm | 12 sensor đổi cùng chiều ở 100/100 động cơ; 15/21 có ý nghĩa | Sensor 11 tăng, sensor 12 giảm (notebook 01) | `outputs/statistics/tables/sensor_inference.csv` |
| Cảnh báo mức một | 100/100 động cơ train, trung vị 121 cycle trước khi hỏng | Health Index của động cơ train 1 | `outputs/degradation/tables/anomaly_events.csv` |
| Dự đoán RUL | Test: MAE 15,35 cycle; 4,38 khi RUL ≤ 30; cross-validation 20,21 | RUL thật so với dự đoán | `outputs/modeling/tables/final_metrics.json` |
| Cảnh báo mức hai | Train (OOF): 49/100 muộn ở ngưỡng 30; ngưỡng 40: kịp thời 51% lên 94% | Quỹ đạo cảnh báo động cơ train 1 | `outputs/course_alignment/20261007/warning_fixed_policies.csv` |
| Lịch bảo trì | Test: 21/21 đúng; 4 ca sót (dự đoán 31–38); ngưỡng 40 bắt 25/25 | Trạng thái 100 động cơ test | `outputs/maintenance/` |

## Hai tập đánh giá, luôn ghi rõ

- **Train (OOF):** dự đoán chéo theo fold, theo dõi suốt đời 100 động cơ train.
  Dùng cho câu hỏi cảnh báo đến sớm hay muộn.
- **Test:** 100 động cơ, mỗi động cơ một dự đoán tại cycle cuối được quan sát.
  Dùng cho câu hỏi dự đoán lệch bao nhiêu và lịch có đúng không.

## Kịch bản slide

Hai mươi slide, khoảng 15–16 phút kể cả 2 phút demo dashboard sau slide 18. Môn
Phân tích dữ liệu lớn nên phần dữ liệu và hạ tầng chiếm 9 slide.

| Slide | Phần | Nội dung |
|---|---|---|
| 1 | Giới thiệu | Bìa: đề tài, giảng viên hướng dẫn, học viên |
| 2 | Giới thiệu | Bài toán, định nghĩa RUL, mục tiêu |
| 3 | Giới thiệu | Quy trình năm bước |
| 4 | Dữ liệu và hạ tầng | Nguồn dữ liệu: bốn tập con C-MAPSS, cách tải và kiểm tra |
| 5 | Dữ liệu và hạ tầng | Cấu trúc bản ghi và ba file của một tập con |
| 6 | Dữ liệu và hạ tầng | Chất lượng dữ liệu và phân phối tuổi thọ |
| 7 | Dữ liệu và hạ tầng | Lưu trữ Bronze–Silver–Gold |
| 8 | Dữ liệu và hạ tầng | HDFS, MapReduce, YARN |
| 9 | Dữ liệu và hạ tầng | PySpark đối chiếu với Pandas |
| 10 | Dữ liệu và hạ tầng | Truy vấn SQL và Spark SQL |
| 11 | Dữ liệu và hạ tầng | Sáu bước biến đổi |
| 12 | Dữ liệu và hạ tầng | Mở rộng quy mô 1×–200× |
| 13 | Phân tích và dự đoán | Làm mượt EWMA |
| 14 | Phân tích và dự đoán | Xu hướng suy giảm |
| 15 | Phân tích và dự đoán | Health Index và cảnh báo mức một |
| 16 | Phân tích và dự đoán | Mô hình RUL |
| 17 | Phân tích và dự đoán | Cảnh báo mức hai và ngưỡng 40 |
| 18 | Phân tích và dự đoán | Lịch bảo trì; sau đó demo dashboard |
| 19 | Kết luận | Kết luận, hạn chế, hướng phát triển |
| 20 | Kết luận | Cảm ơn, mời đặt câu hỏi |

Nội dung chỉ có trong report: từ điển dữ liệu, quan hệ giữa các bảng, độ nhạy của
quy tắc phát hiện bất thường, sai số OOF theo RUL. Khi được hỏi thì trả lời theo
report. FD002–FD004 chỉ đi qua bước ETL; mô hình chỉ được huấn luyện và đánh giá
trên FD001.

## Quy tắc slide

- Mỗi slide một hình thật (file do module, notebook sinh ra, sơ đồ của report hoặc
  ảnh chụp hệ thống) hoặc một bảng số liệu thật, kèm vài con số chính.
- Tiêu đề nói điều slide chứng minh, bằng câu bình thường.
- Chi tiết, công thức, giới hạn và nguồn số liệu để trong ghi chú người nói.
- Không đưa con số nào lên slide nếu chưa có trong cột Nguồn.
