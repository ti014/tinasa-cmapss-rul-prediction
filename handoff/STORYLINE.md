# Câu chuyện dữ liệu của dự án

**Học viên:** Phan Nguyễn Mai Phương — **MSHV:** 25002711

File này là nguồn gốc chung cho slide, lời nói, report, dashboard và notebook.
Mọi tài liệu kể theo đúng thứ tự cảnh dưới đây. Nội dung không phục vụ câu hỏi
chính được chuyển ra phụ lục hoặc slide dự phòng. Mỗi con số phải lấy từ file
nguồn được ghi ở cảnh tương ứng.

## Nhân vật và câu hỏi

**Nhân vật:** trưởng bộ phận bảo trì của một đội 100 động cơ turbofan.

**Câu hỏi mỗi sáng:** "Hôm nay đưa động cơ nào vào xưởng?"

Đưa vào quá sớm thì tốn tiền và thời gian. Đưa vào quá muộn thì động cơ có thể
hỏng giữa chừng. Dữ liệu duy nhất có trong tay là các số đo cảm biến gửi về sau
mỗi cycle.

**Câu kết:** "Từ 21 con số thô mỗi cycle, hệ thống trả lời được câu hỏi mỗi
sáng: hôm nay đưa động cơ nào vào xưởng."

**Hai động cơ xuyên suốt:** động cơ train 1 minh họa phần học và cảnh báo; động
cơ test 90 minh họa quyết định bảo trì.

## Tám cảnh

| # | Cảnh | Một câu nói | Số chính | Hình chính | Nguồn |
|---|---|---|---|---|---|
| 1 | Có gì trong tay | Mỗi động cơ gửi 21 số đo sau mỗi cycle; 100 động cơ đã chạy đến hỏng để học. | Tuổi thọ 128–362 cycle, trung vị 199 | Phân phối tuổi thọ 100 động cơ | `outputs/eda/tables/lifetime_summary.csv` |
| 2 | Dữ liệu ồn | Số đo thô lên xuống liên tục; lọc nhiễu EWMA làm xu hướng hiện ra. | Giảm khoảng 86% dao động ngắn hạn | Sensor 11 của động cơ train 1, trước và sau lọc | `outputs/degradation/summary.json`, `outputs/degradation/figures/denoise_before_after.png` |
| 3 | Động cơ có triệu chứng | So 30 cycle cuối đời với 30 cycle đầu đời: hỏng hóc để lại dấu vết đo được. | 12 sensor đổi cùng chiều ở 100/100 động cơ | Sensor 4 đi lên, sensor 7 đi xuống | `outputs/statistics/tables/sensor_inference.csv` |
| 4 | Đèn vàng | Gộp 21 sensor thành Health Index; tụt khỏi mức bình thường của chính động cơ thì bật đèn vàng. | 100/100 động cơ, trung vị 121 cycle trước khi hỏng; động cơ train 1: cycle 84, hỏng ở 192 | Health Index của động cơ train 1 | `outputs/degradation/tables/anomaly_events.csv`, `outputs/degradation/figures/anomaly_health_index.png` |
| 5 | Đèn đỏ | Mô hình RUL ước lượng còn bao lâu; đèn đỏ khi dự đoán còn 30 cycle hoặc ít hơn. | Test, mỗi động cơ một dự đoán ở cycle cuối: MAE 15,35 cycle, gần hỏng (RUL ≤ 30) lệch 4,38 cycle. Ví dụ train (OOF): động cơ train 1 đỏ ở cycle 162 | Quỹ đạo cảnh báo động cơ train 1 | `outputs/modeling/tables/final_metrics.json`, `outputs/modeling/figures/warning_trajectory.png` |
| 6 | Mô hình có điểm yếu | Mô hình hay lạc quan nên khoảng một nửa cảnh báo bị trễ; nâng ngưỡng lên 40 sửa được phần lớn. | Test: 74/100 động cơ đoán cao hơn thực tế, động cơ 23 thật 113, đoán 170. Train (OOF, theo dõi cả đời 100 động cơ): cảnh báo 44/7/49 (sớm/đúng/trễ); ngưỡng 40: kịp thời 51% lên 94% | Sai số theo RUL thật | `outputs/modeling/tables/warning_summary.json`, `outputs/course_alignment/20261007/warning_fixed_policies.csv`, `slides/slide_data.json` |
| 7 | Câu trả lời buổi sáng | Lịch cho 100 động cơ test: ai vào xưởng ngay, ai theo dõi, ai bình thường. | Test: 21 cần bảo trì ngay và cả 21 đều đúng; 25 động cơ thật sự cần, 4 ca sót đều sát ngưỡng (dự đoán 31–38), hạn sau 1–8 cycle; 59 theo dõi, 20 bình thường. Ngưỡng 40 (chọn từ train): bắt 25/25, báo sớm 3. Động cơ 90 là một ca sót: cycle 146, dự đoán 31,2, thật 28, lịch ở cycle 147, trước lúc hỏng 27 cycle | Lịch bảo trì và chi tiết động cơ 90 | `outputs/maintenance/tables/maintenance_schedule.csv`, `outputs/maintenance/figures/maintenance_planning.png` |
| 8 | Vì sao cần big data | Khi dữ liệu lớn lên, một máy với Pandas không theo kịp; pipeline đã dựng trên HDFS và Spark. | Tính toán (không ghi file): 100× Spark nhanh hơn Pandas 5,5 lần; 200× Pandas không xong trong giới hạn 30 phút, Spark 392 giây. Với 100 động cơ FD001, bỏ Spark thì lịch không đổi | Đường cong thời gian và RAM theo quy mô | `outputs/benchmarks/20261021/` |

## Hai tập đánh giá, luôn ghi rõ

- **Train (OOF):** dự đoán chéo theo fold, theo dõi suốt đời 100 động cơ train.
  Dùng cho câu hỏi "cảnh báo đến sớm hay trễ" (cảnh 4, 6).
- **Test:** 100 động cơ, mỗi động cơ một dự đoán tại cycle cuối được quan sát.
  Dùng cho câu hỏi "dự đoán lệch bao nhiêu" và "lịch có đúng không" (cảnh 5, 7).

Mỗi con số trên slide, trong lời nói và report phải nói rõ thuộc tập nào.

## Kịch bản slide

Mười bốn slide chính, khoảng 15 phút kể cả 2,5 phút demo. Mỗi slide một ý, một
số chính, một hình.

| Slide | Nội dung | Cảnh |
|---|---|---|
| 1 | Bìa: tên đề tài và câu hỏi "Hôm nay đưa động cơ nào vào xưởng?" | — |
| 2 | Nhân vật, vấn đề sớm/muộn, và kết quả trong một câu: lịch test báo đúng 21/21 động cơ cần bảo trì ngay | mở đầu |
| 3 | Bản đồ câu chuyện: cảm biến → làm sạch → hai nhánh (đèn vàng, đèn đỏ) → lịch; tám cảnh trên một dải | mở đầu |
| 4 | Dữ liệu trong tay | 1 |
| 5 | Lọc nhiễu | 2 |
| 6 | Triệu chứng đo được | 3 |
| 7 | Đèn vàng | 4 |
| 8 | Đèn đỏ | 5 |
| 9 | Điểm yếu và cách sửa | 6 |
| 10 | Câu trả lời buổi sáng, động cơ 90 | 7 |
| 11 | Vì sao cần big data: đường cong benchmark và pipeline HDFS/Spark | 8 |
| 12 | Trả lời năm câu hỏi của đề tài, mỗi câu trỏ về một cảnh; ba giới hạn chính | kết |
| 13 | Demo dashboard | — |
| 14 | Cảm ơn | — |

Slide dự phòng (chỉ mở khi được hỏi): kiến trúc Bronze–Silver–Gold và parity
Pandas/PySpark/MapReduce; chi tiết HDFS và YARN; kiểm định Wilcoxon và
Benjamini–Hochberg; vì sao MAE cross-validation (20,21) khác MAE test (15,35);
thử nghiệm mở rộng (nested CV, 169 đặc trưng, FD002–FD004, ARIMA).

## Phần đưa ra phụ lục hoặc slide dự phòng

Nested CV và thử nghiệm cải thiện, LightGBM, TCN, so sánh cây, chuyển sang
FD002–FD004, dự báo Health Index bằng ARIMA, chi tiết kiểm định thống kê
(Wilcoxon, Benjamini–Hochberg), chi tiết hạ tầng Docker. Các phần này giữ nguyên
số liệu, chỉ đổi vị trí; khi giám khảo hỏi thì mở ra.

## Quy tắc viết theo câu chuyện

- Mỗi slide và mỗi mục report mở đầu bằng câu "phần này trả lời câu hỏi gì".
- Mỗi cảnh có đúng một con số chính và một hình chính; số phụ đưa vào lời nói thêm.
- Thuật ngữ mới được giải thích bằng một ví dụ ngay lần đầu xuất hiện.
- Không đưa con số nào vào slide hoặc report nếu chưa có trong cột Nguồn.
