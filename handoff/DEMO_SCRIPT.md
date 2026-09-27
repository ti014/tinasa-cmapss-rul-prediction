# Kịch bản demo dashboard FD001

Mục tiêu là trình bày trong khoảng bốn phút bằng các artifact đã có. Khởi động
theo lệnh trong `README.vi.md` và mở `http://localhost:8501`. Không chạy lại
training trong lúc thuyết trình.

1. **Tổng quan (45 giây).** Nêu phạm vi 100 test engines, model được chọn là
   `hist_gradient_boosting`, mean engine MAE 15.3490 và RMSE 20.5413. Chỉ vào
   bảng so sánh để giải thích lựa chọn bằng 5-fold GroupKFold trên train:
   HistGradientBoosting 20.2143, Ridge 23.9922, median baseline 46.9543.
2. **Đánh giá RUL (45 giây).** Chọn test engine 23 để chỉ ra một trường hợp
   sai số lớn: RUL thật 113, dự đoán 170.4768, absolute error 57.4768 cycles.
   Dùng scatter plot để đặt điểm này trong phân bố của 100 engines. Đây là ví
   dụ giới hạn, không loại khỏi metric cuối.
3. **Cảnh báo sớm (30 giây).** Mở trang cảnh báo. Ngưỡng là 30 cycles; giải
   thích phân phối 44 trước ngưỡng, 7 đúng ngưỡng, 49 sau ngưỡng. Median lead
   time là 30 cycles. Nhấn mạnh đây là OOF nội bộ, chưa phải policy bảo trì.
4. **Health Index / ARIMA (45 giây).** Chọn validation engine 70. Forecast MAE
   là 1.0372 theo thang Health Index và prediction interval coverage 95% chỉ
   21.43%. Chỉ ra rằng coverage thấp được giữ lại trong kết quả; Health Index
   forecast không phải RUL forecast.
5. **Provenance (30 giây).** Nêu 100 evaluation rows, 704 forecast rows và
   `test_label_join.status=completed_after_final_predictions` trong
   `modeling/run_info.json`. Dashboard chỉ đọc artifact.
6. **Kết (15 giây).** Nhắc FD001 chỉ mô phỏng một điều kiện vận hành và một
   dạng lỗi; chưa kiểm chứng tổng quát hóa sang động cơ thực tế.

Nếu browser hoặc mạng trình chiếu gặp sự cố, dùng
`report/main.pdf`; slide cũ cần đối chiếu lại trước khi dùng. Lệnh
`/_stcore/health` chỉ chứng minh server sẵn sàng; cần mở các view cần trình bày để xác
nhận flow demo.

## Câu hỏi về dữ liệu và Big Data

Nếu cô hỏi trước khi xem dashboard, trả lời theo luồng đã triển khai:

- **Nhiều file gom thế nào?** Trong từng subset, reader đọc train/test cùng
  schema, thêm `split` rồi concat/`unionByName`. FD001--FD004 được xử lý riêng;
  không gộp toàn bộ vì khóa hiện tại chưa có `dataset_id`.
- **Truy vấn ở đâu?** SQLite truy vấn `engine_summary` và `train_targets`.
  Sensor observations được lọc từ Silver Parquet bằng Pandas. Project chưa có
  query engine SQL trên Parquet hoặc partition pruning.
- **Transform là gì?** Đổi text thành typed observations, tạo RUL train,
  engine summary, rồi rolling features 30 cycles. Đây là data transformation,
  không phải Transformer model.
- **Lấy feature nào?** Model dùng 126 cột từ 21 sensors, mỗi sensor có
  `last`, `mean`, `std`, `min`, `max`, `slope`. Ba settings không nằm trong X
  hiện tại. Đây là feature engineering, chưa phải feature selection.
- **Có quan hệ nhiều-nhiều không?** Không có many-to-many join được triển khai.
  Quan hệ chính là engine--observations 1:N; feature--target và prediction--label
  là 1:1 theo khóa. Join thiếu `cycle` sẽ làm nhân dòng.
- **Big Data nằm ở đâu?** Ở batch ETL nhiều file, Parquet Silver/Gold và
  PySpark parity local. Benchmark 100× chỉ đo Pandas in-memory; không nói là
  cluster benchmark, Airflow hoặc Power BI.

## Phần bổ sung về cải thiện mô hình (60 giây)

Mở **Thử nghiệm cải thiện**. Đọc bảng vòng ngoài: low-RUL MAE 3,6676 → 3,6730
cycles, cảnh báo trễ 49 → 53/100. Chọn train engine 98 để xem trường hợp sai số
tăng. Sau đó đọc bảng đánh đổi: `hgb_08` giảm low-RUL MAE khoảng 9,75% trong
inner CV nhưng tăng MAE toàn bộ khoảng 0,32 cycles. Nhấn mạnh hai bảng dùng
hai cấp đánh giá khác nhau. Kết luận: đã thử cải thiện, chưa đạt điều kiện thay
mô hình; baseline vẫn dự đoán RUL với test MAE 15,3490 cycles.

Dùng slide 14–16 của deck hiện hành hoặc phần thực nghiệm trong báo cáo làm
phương án dự phòng. Có AppTest cho trang mới; vẫn nên mở trực tiếp trang này
trên máy trình chiếu trước buổi báo cáo.
