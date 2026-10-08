# Kịch bản demo dashboard — Đề tài 8

**Học viên:** Phan Nguyễn Mai Phương · **MSHV:** 25002711.
**Cập nhật:** 08/10/2026, theo dashboard chín trang và deck 20 slide trên template IUH.

Demo dài khoảng 3 phút, đặt sau slide 19 (khó khăn và bài học), trước slide cảm
ơn. Mục tiêu: cho người nghe thấy sản phẩm đặc thù của đề tài (lịch dự đoán bảo
trì) chạy trên dữ liệu thật, rồi lần ngược về khử nhiễu và bất thường.

## Chuẩn bị trước buổi báo cáo

1. Chạy dashboard từ thư mục gốc repository và để sẵn trình duyệt ở trang Tổng quan:

   ```powershell
   uv run --extra dashboard streamlit run app.py
   ```

2. Kiểm tra trang `http://localhost:8501/?page=Lịch bảo trì` hiện đủ bảng 100 động cơ.
3. Phóng to trình duyệt (Ctrl + dấu cộng) để chữ đọc được từ cuối phòng.

Dashboard chỉ đọc artifact trong `outputs/`, không huấn luyện lại mô hình và không
mở file nhãn test gốc, nên không cần mạng và không có bước chờ dài.

## Trình tự ba phút

### 1. Lịch bảo trì — 60 giây

- Mở trang **Lịch bảo trì**. Chỉ vào bốn ô số: 21 động cơ cần bảo trì ngay, 59
  theo dõi tăng cường, 20 bình thường, 14 đến hạn trong 25 cycles tới.
- Lọc trạng thái chỉ còn "Cần bảo trì": "Đây là danh sách cần can thiệp ngay, sắp
  theo mức ưu tiên."
- Ở ô "Xem chi tiết động cơ test", chọn **Engine 090**: đang ở cycle 146, RUL dự
  đoán 31,2, cần bảo trì ở cycle 147; Health Index đã báo bất thường từ cycle 67.
- Kéo xuống bảng kiểm tra: "21/21 động cơ báo cần bảo trì là đúng; 4 động cơ còn
  lại đến hạn sau 1–8 cycles và vẫn ở mức theo dõi."

### 2. Khử nhiễu — 35 giây

- Mở trang **Khử nhiễu**, giữ Engine 001 và `sensor_11`.
- "Đường xám là tín hiệu gốc, đường xanh là sau EWMA. Dao động giữa hai cycle giảm
  khoảng 86%, nhưng xu hướng tăng sau cycle 150 vẫn giữ nguyên."
- Nếu được hỏi: đổi sang `sensor_4` để thấy cùng hiện tượng.

### 3. Xu hướng và bất thường — 45 giây

- Mở trang **Xu hướng & bất thường**, Engine 001.
- "Vùng xám là 30 cycles tham chiếu lúc động cơ còn khỏe. Đường đứt là ngưỡng 3σ.
  Health Index cắt ngưỡng và giữ 5 cycles liên tiếp ở cycle 84, tức 108 cycles trước
  khi hỏng ở cycle 192."
- Chỉ vào histogram: "Trên 100 động cơ train, trung vị báo trước 121 cycles."

### 4. Cảnh báo RUL — 30 giây

- Mở trang **Cảnh báo sớm**, chọn Engine 001.
- "RUL dự đoán chạm ngưỡng 30 ở cycle 162, đúng lúc RUL thật chạm 30. Ghép với
  trang trước: cycle 84 là mức 1, cycle 162 là mức 2, cycle 192 là hỏng."

### 5. Kết — 10 giây

- Quay về **Tổng quan**: "Toàn bộ số liệu trên dashboard đọc từ cùng artifact với
  báo cáo và slide."

## Phương án dự phòng khi dashboard lỗi

Chuyển sang slide 12 (bất thường), slide 15 (hai mức cảnh báo) và slide 16 (lịch
bảo trì). Không mở terminal hoặc chạy lại huấn luyện trước lớp. Nếu cần bảng đầy
đủ, mở `outputs/maintenance/tables/maintenance_schedule.csv` hoặc `report/main.pdf`
(Mục 6.6).

## Câu hỏi về dữ liệu và Big Data

- **Nhiều file gom thế nào?** Trong từng subset, reader đọc train/test cùng schema,
  thêm cột `split`, rồi concat (Pandas) hoặc `unionByName` (PySpark). FD001–FD004
  được xử lý riêng vì khóa chưa có `dataset_id`.
- **Truy vấn ở đâu?** SQLite truy vấn `engine_summary` và `train_targets`; dữ liệu
  sensor chi tiết đọc từ Silver Parquet.
- **Transform là gì?** Đổi text thành bảng có kiểu, tạo nhãn RUL train, engine
  summary, khử nhiễu EWMA, rồi rolling features 30 cycles.
- **Spark có nhanh hơn Pandas không?** Ở scale 1×, Pandas nhanh hơn vì Spark tốn chi
  phí khởi động. Ở scale 10×, Spark `local[4]` nhanh hơn khoảng 3,4 lần trên toàn
  pipeline có I/O. Đây là đo trên một máy, chưa phải cluster.
- **Có many-to-many join không?** Không. Quan hệ chính là động cơ–quan sát 1:N;
  đặc trưng–nhãn và dự đoán–nhãn test là 1:1 theo khóa đầy đủ.

## Câu hỏi về phương pháp và giới hạn

- **Xử lý nhiễu bằng gì, vì sao chọn EWMA?** EWMA chỉ dùng quá khứ nên không rò rỉ
  tương lai, có một tham số (span) và phản ứng nhanh hơn trung bình trượt thường
  khi xu hướng đổi. Span 10 được cố định trước. Giới hạn: không có tín hiệu sạch để
  chứng minh đã loại đúng nhiễu đo.
- **Phát hiện bất thường không có nhãn thì kiểm chứng thế nào?** So mỗi động cơ với
  chính nó lúc khỏe. Đo được thời gian báo trước khi hỏng, nhưng chưa đo được tỷ lệ
  báo động giả vì mọi động cơ train đều đi đến hỏng.
- **Vì sao GroupKFold theo động cơ?** Hai window liền kề của cùng động cơ trùng 29/30
  cycles; chia ngẫu nhiên theo dòng làm metric đẹp giả tạo.
- **Vì sao 49% cảnh báo trễ?** Động cơ suy giảm nhanh có RUL thật thấp hơn điều mô
  hình kỳ vọng ở cùng mức sensor. Ngưỡng 40 giảm trễ còn 6% nhưng báo sớm nhiều hơn
  khoảng bốn lần; chọn ngưỡng thật cần chi phí bảo trì.
- **Mức theo dõi tăng cường có quá rộng không?** Có: 41/59 động cơ ở mức này có RUL
  thật trên 60. Mức này dùng để tăng tần suất kiểm tra, không dùng để lên lịch can
  thiệp.
- **Đã thử mô hình mạnh hơn chưa?** Đã thử nested CV, thêm 42 đặc trưng lịch sử,
  similarity-based RUL, LightGBM, TCN. Không thử nghiệm nào cải thiện đồng thời sai số
  và cảnh báo, nên giữ mô hình chính. Chi tiết ở slide 17 và Mục 6.7 báo cáo.
- **Có áp dụng cho động cơ thật không?** Chưa thể kết luận. C-MAPSS là dữ liệu mô
  phỏng; chưa có telemetry thật, lịch sử can thiệp, downtime hay chi phí bảo trì.
