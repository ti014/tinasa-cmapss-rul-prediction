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
| 5 | Đèn đỏ | Mô hình RUL ước lượng còn bao lâu; đèn đỏ khi dự đoán còn 30 cycle hoặc ít hơn. | MAE 15,35 cycle; gần hỏng (RUL ≤ 30) lệch 4,38 cycle; động cơ train 1 đỏ ở cycle 162 | Quỹ đạo cảnh báo động cơ train 1 | `outputs/modeling/tables/final_metrics.json`, `outputs/modeling/figures/warning_trajectory.png` |
| 6 | Mô hình có điểm yếu | Mô hình hay lạc quan nên khoảng một nửa cảnh báo bị trễ; nâng ngưỡng lên 40 sửa được phần lớn. | 74/100 động cơ test đoán cao hơn thực tế; cảnh báo 44/7/49 (sớm/đúng/trễ); ngưỡng 40: kịp thời 51% lên 94%; động cơ 23 thật 113, đoán 170 | Sai số theo RUL thật | `outputs/modeling/tables/warning_summary.json`, `outputs/course_alignment/20261007/warning_fixed_policies.csv`, `slides/slide_data.json` |
| 7 | Câu trả lời buổi sáng | Lịch cho 100 động cơ test: ai vào xưởng ngay, ai theo dõi, ai bình thường. | 21 cần bảo trì ngay (đúng 21/21), 59 theo dõi, 20 bình thường; động cơ 90: cycle 146, còn 31 cycle, lịch ở cycle 147 | Lịch bảo trì và chi tiết động cơ 90 | `outputs/maintenance/tables/maintenance_schedule.csv`, `outputs/maintenance/figures/maintenance_planning.png` |
| 8 | Vì sao cần big data | Khi dữ liệu lớn lên, một máy với Pandas không theo kịp; pipeline đã dựng trên HDFS và Spark. | Benchmark 1×–200× (đang đo, cập nhật khi xong) | Đường cong thời gian và RAM theo quy mô | `outputs/benchmarks/20261021/` |

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
