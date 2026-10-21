# Đối chiếu đề cương và bằng chứng bàn giao

**Học viên:** Phan Nguyễn Mai Phương  
**MSHV:** 25002711  
**Ngày đồng bộ:** 12/10/2026 (đối chiếu code và artifact hiện có)

Tài liệu này đối chiếu bài làm với hai nguồn yêu cầu: đề cương môn Phân tích dữ liệu
lớn và brief **Đề tài 8 — Phân tích dữ liệu IoT & bảo trì dự đoán**. Trạng thái
**Một phần** hoặc **Chưa đánh giá** không được trình bày như đã hoàn thành.

## Kết luận ngắn

Bài làm đáp ứng đủ các yêu cầu kỹ thuật trọng tâm của Đề tài 8 và có sản phẩm đặc
thù là lịch dự đoán thời điểm cần bảo trì kèm biểu đồ xu hướng cảnh báo sớm. So với
bản 07/10, ba chỗ còn yếu đã được bổ sung: xử lý nhiễu (EWMA), phát hiện dấu hiệu
bất thường (Health Index) và mục khó khăn, bài học trong báo cáo. Phần còn ở mức
một phần là các giới hạn của dữ liệu mô phỏng và hạ tầng một máy.

Brief không yêu cầu hệ thống thời gian thực, chi phí bảo trì hay lịch bảo trì tối ưu.
Các nội dung đó chỉ được nêu là giới hạn, không biến thành tiêu chí mới.

## Quyết định về mô hình chính

Mô hình chính là HistGradientBoosting với 126 đặc trưng window 30 cycle, chọn bằng
GroupKFold 5 fold theo động cơ; đây là mô hình duy nhất có đánh giá test đầy đủ (MAE
15,35 cycle). FD002–FD004 chỉ đi qua bước ETL; mô hình chỉ được huấn luyện và đánh
giá trên FD001. Báo cáo, slide, dashboard và kịch bản dùng cùng một mô hình, một
bảng cảnh báo.

## Trọng số chấm đồ án trong đề cương

| Tiêu chí | Trọng số | Bằng chứng trong bản nộp |
|---|---:|---|
| Quy trình quản lý dữ liệu: thu thập → làm sạch → chuẩn hóa → tích hợp → lưu trữ | 40% | Mục 4 và 6.1 báo cáo; slide 3–12 (slide 3 gắn năm bước với quy trình); `src/cmapss/{data,etl,features}`; parity và benchmark |
| Trực quan hóa và truyền đạt kết quả | 25% | Hình khử nhiễu, Health Index, cảnh báo, lịch bảo trì; dashboard tám trang; slide 13–18 |
| Code, cấu trúc pipeline và tài liệu kỹ thuật | 20% | Package theo bước, notebook 00, 01, 02, 03 và 06, README song ngữ, bộ test tự động |
| Trình bày, thuyết minh và phản biện | 15% | Deck 20 slide, kịch bản nói khoảng 15–16 phút gồm 2 phút demo, sổ tay `SPEECH_SCRIPT_VI.pdf` có lời đọc từng slide, demo và câu hỏi phản biện |

## Ma trận yêu cầu → bằng chứng → trạng thái

### A. Yêu cầu trực tiếp của Brief Đề tài 8

| Yêu cầu | Bằng chứng | Trạng thái |
|---|---|---|
| Phân tích chuỗi thời gian từ cảm biến | `src/cmapss/data/readers.py`; mỗi quan sát có `engine_id`, `cycle`, 3 settings, 21 sensor; Mục 2 báo cáo | **Đạt** trong phạm vi dữ liệu batch mô phỏng |
| Xử lý nhiễu tín hiệu | `src/cmapss/degradation/denoise.py`: EWMA span 10, chỉ dùng quá khứ; giảm trung vị 86,3% dao động; hình trước/sau; Mục 5.2 và 6.2 | **Đạt**; giới hạn: không có tín hiệu sạch để đối chiếu |
| Chuẩn hóa dữ liệu | Kiểm tra schema/kiểu trong ETL; StandardScaler cho Ridge (trong fold) và trước PCA (Health Index) | **Đạt**; HGB dùng thang gốc vì cây không phụ thuộc thang đo |
| Tổng hợp theo time window | Window 30 cycle, chỉ dùng dữ liệu hiện tại và quá khứ, 126 đặc trưng; parity Pandas–PySpark | **Đạt** |
| Phát hiện dấu hiệu bất thường | `src/cmapss/degradation/anomaly.py`: Health Index giảm quá 3σ so với 30 cycle đầu, 5 cycle liên tiếp; 100/100 động cơ train, trung vị 121 cycle trước khi hỏng; phân tích độ nhạy 9 cấu hình (2/3/4σ × 3/5/10 cycle) đều phát hiện 100/100, lead time nhỏ nhất ≥ 50 cycle (Bảng 18); Mục 5.3 và 6.3 | **Đạt**; giới hạn: chưa đo trực tiếp được tỷ lệ báo động giả |
| Trực quan hóa xu hướng suy giảm | Quỹ đạo sensor, kiểm định đầu/cuối, Health Index, quỹ đạo cảnh báo | **Đạt** |
| Dự đoán thời điểm cần bảo trì | `src/cmapss/maintenance/`: lịch cho 100 động cơ test; 21/21 báo cần bảo trì là đúng; Mục 6.6 | **Đạt** trong phạm vi FD001 và điểm cắt test; giới hạn: 12 động cơ còn xa hỏng (RUL thật ≥ 77) có lịch muộn hơn lúc hỏng 4–27 cycle |
| Báo cáo dự đoán kèm biểu đồ xu hướng cảnh báo sớm | `outputs/maintenance/tables/maintenance_schedule.csv`, `fleet_status.png`, `maintenance_planning.png`, `warning_trajectory.png`; slide 17–18; trang dashboard Lịch bảo trì | **Đạt** |

### B. Yêu cầu nộp bài

| Yêu cầu | Bằng chứng | Trạng thái |
|---|---|---|
| Báo cáo: bài toán, quy trình xử lý, kết quả, khó khăn và bài học | `report/main.pdf`; Mục 8 "Khó khăn và bài học rút ra" | **Đạt** |
| Mã nguồn tổ chức theo từng bước pipeline | `src/cmapss/` theo package; notebook 00 (dữ liệu), 01–03 (EDA, thống kê, mô hình RUL) và 06 (khử nhiễu, bảo trì) | **Đạt** |
| Slide thuyết trình | `slides/cmapss_fd001_presentation.pptx` | **Đạt** |
| Dữ liệu đã xử lý hoặc script tái tạo | `outputs/` và lệnh tái lập trong `FINAL_HANDOFF.md` | **Đạt** |
| Một file zip theo cú pháp `Nhom<X>_TenDeTai_BaoCaoCuoiKy` | `dist/Nhom4_PhanTichDuLieuIoTBaoTriDuDoan_BaoCaoCuoiKy.zip`, tạo bằng `scripts/package_submission.py` | **Đạt**; nhóm 4, bài làm cá nhân (cần đăng ký với giảng viên) |

### C. Đề cương môn học và CLO liên quan

| CLO | Bằng chứng | Trạng thái |
|---|---|---|
| CLO1: Big Data, 5V và Data Science | Bảng 5V trong Mục 7.3 báo cáo | **Đạt ở mức khái niệm**; nêu rõ giới hạn volume/velocity |
| CLO2: Hadoop/HDFS/MapReduce | `infra/hadoop/` (Hadoop 3.4.3 một node trong Docker), `src/cmapss/hdfs/`: raw FD001 trên HDFS, job MapReduce (Hadoop Streaming) trên YARN, `fsck` HEALTHY; Mục 3.7 và 4.4 báo cáo; slide 7–8 | **Đạt ở mức sandbox** đúng phạm vi đề cương (không cần cluster thật); chưa có cụm nhiều máy |
| CLO3: thu thập, làm sạch, chuẩn hóa, tích hợp, lưu trữ | `cmapss.data.acquire` tải từ NASA, kiểm tra MD5/SHA-256, ghi manifest; reader, contract, ETL Pandas/PySpark, Parquet/SQLite, HDFS | **Đạt**; nguồn là dataset công khai, thu thập bằng script chứ không qua API/streaming |
| CLO4/CLO7: trực quan hóa, dashboard, storytelling | Báo cáo, slide, dashboard tám trang theo thứ tự pipeline | **Đạt** |
| CLO5: thao tác HDFS và chạy job MapReduce/Spark | Thao tác HDFS (`put`, `ls`, `fsck`), job MapReduce trên YARN, Spark đọc/ghi HDFS; PySpark ETL và rolling, parity, benchmark 1×/10× | **Đạt** ở mức một node; chưa đo hiệu năng trên cụm nhiều máy |
| CLO6: Pandas/PySpark cleaning và integration | concat/`unionByName`, kiểm tra schema và khóa, parity | **Đạt** trong phạm vi batch local |
| CLO8: pipeline end-to-end | Raw → Silver observations → rolling features → RUL; nhánh riêng EWMA → HI → bất thường; lịch bảo trì kết hợp RUL/HI → dashboard; engine_summary tổng hợp riêng từ Silver | **Đạt** |
| CLO9: làm việc nhóm | Brief cá nhân | **Không áp dụng** |
| CLO10: chất lượng dữ liệu, đạo đức | Contract, hash, ranh giới nhãn test, khai báo sử dụng AI/LLM trong Mục 1.3 | **Đạt** |

## Bằng chứng Big Data cần nói chính xác

| Nội dung | Cách diễn đạt được phép |
|---|---|
| Nhiều file, nhiều tầng | "Batch ETL nhiều file và lưu trữ phân lớp Bronze/Silver/Gold" |
| Hai implementation | "Pandas và PySpark khớp tuyệt đối trên cùng input" |
| Benchmark | "Trên một máy, Spark `local[4]` nhanh hơn Pandas khoảng 3,4 lần ở scale 10×"; không nói là cluster |
| Hạ tầng | "Hadoop/HDFS/YARN một node Docker; MapReduce và Spark trên HDFS đối chiếu engine_summary khớp 200/200"; chưa có cụm nhiều máy hay streaming |

## Các điểm phải khóa trước khi nộp

1. Kịch bản nói và kịch bản demo dùng đúng thứ tự 20 slide và tám trang dashboard.
2. Chốt tên file zip và danh sách thư mục đưa vào gói theo `scripts/package_submission.py` (không gồm `tmp/`, `.venv/`).
3. Chạy lại toàn bộ test và build báo cáo, slide ngay trước khi đóng gói.

## Nguồn kiểm tra

- `D:/IUH/BigData/Resources/De Cuong PTDLL.docx` và `Danh Sach De Tai.docx` — đề cương, rubric và yêu cầu nộp.
- `references/context/assignment-brief.png` — Brief Đề tài 8.
- `report/main.pdf`, `slides/cmapss_fd001_presentation.pdf`, `handoff/SPEECH_SCRIPT_VI.pdf`.
- `outputs/{etl,degradation,features,modeling,maintenance}/` — artifact số liệu.
