# Kịch bản demo dashboard FD001

Mục tiêu là trình bày trong khoảng ba phút bằng các artifact đã có. Khởi động
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
3. **Health Index / ARIMA (45 giây).** Chọn validation engine 70. Forecast MAE
   là 1.0396 theo thang Health Index và prediction interval coverage 95% chỉ
   21.43%. Chỉ ra rằng coverage thấp được giữ lại trong kết quả; Health Index
   forecast không phải RUL forecast.
4. **Provenance (30 giây).** Nêu 100 evaluation rows, 704 forecast rows và
   `test_label_join.status=completed_after_final_predictions` trong
   `modeling/run_info.json`. Dashboard chỉ đọc artifact.
5. **Kết (15 giây).** Nhắc FD001 chỉ mô phỏng một điều kiện vận hành và một
   dạng lỗi; chưa kiểm chứng tổng quát hóa sang động cơ thực tế.

Nếu browser hoặc mạng trình chiếu gặp sự cố, dùng
`slides/cmapss_fd001_presentation.pptx` và `report/main.pdf`. Lệnh
`/_stcore/health` chỉ chứng minh server sẵn sàng; cần mở đủ bốn view để xác
nhận flow demo.
