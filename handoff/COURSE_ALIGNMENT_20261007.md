# Đối chiếu đề cương và bằng chứng bàn giao

**Học viên:** Phan Nguyễn Mai Phương  
**MSHV:** 25002711  
**Ngày rà soát:** 07/10/2026

Tài liệu này đối chiếu bài làm hiện tại với hai nguồn yêu cầu: đề cương môn
Phân tích dữ liệu lớn và brief **Đề tài 8 — Phân tích dữ liệu IoT & bảo trì dự
đoán**. Đây là ma trận kiểm tra trước khi chốt bản nộp; trạng thái **Chưa chốt**
hoặc **Một phần** không được trình bày như đã hoàn thành. Số liệu mô hình trong
tài liệu này phải được cập nhật theo quyết định protocol/model cuối cùng trong
artifact alignment, không tự suy ra từ một run thí điểm.

## Kết luận ngắn

Pipeline đã có bằng chứng khá đầy đủ cho lifecycle dữ liệu: đọc nhiều file,
kiểm tra contract, tạo nhãn, tích hợp train/test, lưu Parquet/SQLite, đối chiếu
Pandas--PySpark và tạo các artifact có provenance. Phần cần giữ khi bàn giao là
vai trò của model/protocol đã chọn, cách trình bày cảnh báo như một đầu ra hỗ trợ
bảo trì, và ranh giới của các claim về xử lý nhiễu, Spark và khả năng áp dụng.

Brief không yêu cầu hệ thống thời gian thực, chi phí bảo trì hoặc lịch bảo trì tối
ưu. Vì vậy các nội dung đó chỉ được ghi là giới hạn, không biến thành tiêu chí
mới làm lệch đề tài.

## Quyết định alignment về predictor và warning

Artifact [`outputs/course_alignment/20261007/summary.json`](../outputs/course_alignment/20261007/summary.json)
đã được tạo ở chế độ **read-only replay**: không fit model mới và không đọc nhãn
test để chọn. Hai prediction panel có cùng 17.731 khóa train, 100 engine, cùng
raw truth và cùng grouped fold; khác biệt nằm ở biểu diễn đầu vào:

- **HGB126**: 126 rolling features, là baseline lịch sử đang được dashboard cũ
  đọc.
- **HGB169**: 126 rolling features + `cycle` + 42 history features, là predictor
  tham chiếu được khuyến nghị cho phân tích whole-trajectory và endpoint.

Khuyến nghị này có điều kiện: HGB169 tốt hơn trên các panel chung theo summary,
nhưng bootstrap có điều kiện trên prediction đã lưu, không bao gồm bất định do
fit/seed/selection và không phải blind confirmation độc lập. Vì vậy văn bản dùng
cụm **predictor tham chiếu/ứng viên chính**, không nói đã chứng minh cải thiện
triển khai.

Warning là một quyết định riêng. Policy `threshold=30` và `threshold=40` với
persistence/lead requirement đã được đọc như các sensitivity policy cố định;
policy lịch sử calibrated169 chọn ngưỡng 40 ở các outer fold (một fold persistence
3), nhưng đây là historical nested-policy readout. Summary ghi rõ không được
claim unconditional whole-process improvement: dưới threshold 30 HGB169 kém
timely hơn, còn threshold 40 chỉ nhỉnh hơn về on-time cùng burden tăng nhẹ. Báo
cáo và slide phải giữ prediction model, policy cảnh báo và metric test thành ba
phần tách biệt.

## Trọng số chấm đồ án trong đề cương

| Tiêu chí | Trọng số | Bằng chứng cần nhìn thấy trong bản nộp |
|---|---:|---|
| Quy trình quản lý dữ liệu: thu thập → làm sạch → chuẩn hóa → tích hợp → lưu trữ | 40% | Input/output, schema, khóa, kiểm tra chất lượng, transform, storage và provenance nối được thành một pipeline |
| Trực quan hóa và truyền đạt kết quả | 25% | Xu hướng suy giảm, metric RUL, cảnh báo và dashboard/figure có thông điệp rõ |
| Code, cấu trúc pipeline và tài liệu kỹ thuật | 20% | Module, notebook, lệnh chạy, artifact, parity, test và giới hạn được ghi đúng |
| Trình bày, thuyết minh và phản biện | 15% | Deck tối đa 20 slide, phần nói 18–20 phút gồm demo khoảng 3 phút, trả lời được câu hỏi về dữ liệu và giới hạn |

## Ma trận yêu cầu → bằng chứng → trạng thái

### A. Yêu cầu trực tiếp của Brief Đề tài 8

| Yêu cầu | Bằng chứng hiện có | Trạng thái cần ghi khi trình bày |
|---|---|---|
| Phân tích chuỗi thời gian từ cảm biến | `data/raw/train_FD00x.txt`, `src/cmapss/data/readers.py`, `report/sections/problem-data.tex`; mỗi observation có `engine_id`, `cycle`, 3 settings và 21 sensor | **Đạt trong phạm vi dữ liệu batch mô phỏng** |
| Xử lý nhiều tín hiệu | `src/cmapss/features/rolling.py`; 21 sensor × 6 thống kê = 126 feature columns; EDA và kiểm định đầu/cuối trong `report/sections/current-results.tex` | **Đạt**; không gọi đây là feature selection nếu chưa có bước chọn biến |
| Xử lý nhiễu cảm biến | Rolling mean/std/min/max/slope tóm tắt dao động; phần Health Index dùng chuẩn hóa + PCA | **Một phần**: chưa có bộ lọc tín hiệu riêng, imputation hoặc benchmark khử nhiễu; phải nói rõ giới hạn |
| Chuẩn hóa dữ liệu | Reader/ETL kiểm tra schema và kiểu dữ liệu; `StandardScaler` được fit trong training fold cho Ridge và nhánh Health Index | **Đạt có điều kiện**: HGB dùng feature ở thang gốc; không nói toàn pipeline đã scaling chung |
| Tổng hợp theo time window | Window nhân quả 30 cycles trong `rolling_features`; 29 cycle đầu mỗi engine không padding; có parity rolling Pandas--PySpark ở scale 1× | **Đạt**; window 10 chỉ dùng ở ma trận cross-subset và không được so trực tiếp với protocol window 30 |
| Trực quan hóa xu hướng suy giảm | Hình quỹ đạo sensor, Health Index theo cycle, residual/RUL và cảnh báo trong `report/main.pdf`; dashboard Streamlit đọc artifact | **Đạt** cho mô tả xu hướng; không suy ra quan hệ nhân quả từ hình |
| Dự đoán thời điểm cần bảo trì | RUL prediction trên test engines, `outputs/modeling/tables/final_metrics.json`, notebook modeling và figure so sánh thật--dự đoán | **Đạt trong phạm vi FD001 và điểm cắt test**; model/protocol cuối phải lấy từ artifact alignment |
| Cảnh báo theo cycle | `warning_summary.json`, `warning_trajectory.png`, quy tắc `RUL dự đoán ≤ 30 cycles`, lead time và timing counts | **Một phần**: đây là cảnh báo nghiên cứu trên OOF, chưa phải policy bảo trì tối ưu; phải giữ số cảnh báo trễ trong báo cáo |

### B. Đề cương môn học và CLO liên quan

| CLO / yêu cầu | Bằng chứng hiện có | Trạng thái kiểm tra |
|---|---|---|
| CLO1: Big Data, 5V và Data Science | `report/sections/theory.tex`, `report/sections/discussion.tex` có phần 5V và phạm vi pipeline | **Đạt ở mức khái niệm**, cần giữ giới hạn volume/velocity của dữ liệu một máy |
| CLO2: Hadoop/HDFS/MapReduce ở mức khái niệm | Không đưa HDFS/MapReduce vào phạm vi triển khai; báo cáo tập trung Spark, Parquet và pipeline batch đã có artifact | **Chưa đánh giá thực hành HDFS/MapReduce**; không tạo claim khi chưa có artifact |
| CLO3: thu thập, làm sạch, chuẩn hóa, tích hợp, lưu trữ | `src/cmapss/data/readers.py`, `etl/pandas_pipeline.py`, `etl/spark_pipeline.py`, `report/sections/data-processing.tex`, Parquet/SQLite artifacts | **Đạt phần lớn**; thu thập là file dataset có sẵn, làm sạch chủ yếu là contract validation |
| CLO4/CLO7: visualization và storytelling | `report/main.pdf`, `slides/cmapss_fd001_presentation.pptx`, dashboard Streamlit, figure/metric artifacts | **Đạt nếu deck và speech dùng cùng số liệu và cùng protocol** |
| CLO5: thao tác Spark đơn giản | PySpark ETL/rolling, parity artifacts, benchmark local có I/O tại `outputs/benchmarks/20260929` | **Một phần**: Spark chạy local, ghi qua driver/Arrow trên Windows; chưa có cluster benchmark |
| CLO6: Pandas/PySpark cleaning và integration | Pandas concat, PySpark `unionByName`, schema/key checks và parity ở `outputs/etl/parity.json` | **Đạt trong phạm vi batch local** |
| CLO8: pipeline end-to-end | Raw → Silver → Gold → features → model → warning → dashboard; lệnh tái lập trong `README.vi.md` và `handoff/FINAL_HANDOFF.md` | **Đạt trong phạm vi artifact đã chốt**; model/protocol và tài liệu demo đã được đồng bộ |
| CLO9: làm việc nhóm | Không có yêu cầu nhóm được ghi trong brief cá nhân hiện tại | **Không áp dụng / không suy diễn** |
| CLO10: chất lượng dữ liệu, riêng tư, đạo đức | Contract, hash, giới hạn dữ liệu mô phỏng và khai báo sử dụng AI/LLM trong `report/sections/problem-data.tex` | **Đạt phần chất lượng và khai báo**; dữ liệu công khai, không có claim về dữ liệu riêng tư |

### C. Bằng chứng Big Data cần nói chính xác

| Nội dung | Bằng chứng | Cách diễn đạt được phép |
|---|---|---|
| Nhiều file và nhiều tầng dữ liệu | Raw text → Silver observations/targets → Gold summary/features; Parquet và SQLite | “Batch ETL nhiều file và lưu trữ phân lớp” |
| Hai implementation | Pandas và PySpark đối chiếu schema, khóa, số dòng và giá trị; rolling scale 1× khớp tolerance đã ghi trong artifact | “Parity trên cùng input”, không nói hai hệ thống đã được chứng minh hiệu năng tương đương |
| Benchmark scale | Benchmark local có stage đọc, tổng hợp, rolling, ghi; run đã lưu đo scale 1×/10× | Nêu đúng scale và stage; dữ liệu nhân bản không dùng để train và không đại diện cluster |
| Hạ tầng | Spark `local[4]`; trên Windows ghi qua Arrow/PyArrow về driver; chưa có HDFS/cluster/distributed writer | Đây là **bằng chứng pipeline Big Data ở môi trường local**, không phải benchmark production cluster |
| Provenance và reliability | `run_info.json`, manifest/hash, staging/rollback, raw boundary, test suite và handoff reliability | Dùng để chứng minh khả năng truy vết/tái lập; không gọi đây là transaction nhiều file chống mất điện |

## Các điểm phải khóa trước khi tạo bản nộp

1. Giữ một protocol/model trung tâm từ artifact alignment. HGB169 là predictor
   candidate có điều kiện; HGB126 là baseline lịch sử và dashboard legacy. Báo cáo,
   slide, dashboard và speech phải gọi đúng vai trò của từng bản; không được trộn
   metric hoặc warning policy giữa chúng.
2. Dùng một bảng cảnh báo duy nhất trong deck và speech. Ghi rõ dữ liệu OOF,
   ngưỡng nghiên cứu, số cảnh báo trước/đúng/sau ngưỡng và giới hạn false alarm.
3. Kiểm tra lại mọi claim “xử lý nhiễu”, “Spark rolling”, “benchmark” và “tái
   lập” với artifact tương ứng. Nếu chưa có bằng chứng, chuyển claim thành giới
   hạn hoặc công việc tiếp theo.
4. Giữ phần trình bày trong 18–20 phút gồm demo khoảng 3 phút. Nếu rút ngắn,
   rút phần phụ (ARIMA hoặc chi tiết follow-up), không bỏ pipeline, cảnh báo và
   giới hạn vì đó là phần trả lời trực tiếp brief/rubric.

## Nguồn kiểm tra

- `D:/IUH/BigData/Resources/De Cuong PTDLL.docx` — đề cương và rubric gốc.
- `references/context/assignment-brief.png` — Brief Đề tài 8.
- `report/main.pdf` và các section trong `report/sections/` — nội dung báo cáo.
- `README.vi.md`, `handoff/DATA_WALKTHROUGH.md`, `handoff/DEMO_SCRIPT.md` — đường chạy và cách demo.
- `outputs/etl/`, `outputs/features/`, `outputs/modeling/`, `outputs/experiments/` — artifact số liệu; không thay số bằng số đẹp hơn.
