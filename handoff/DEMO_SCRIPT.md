# Kịch bản demo dashboard FD001

**Học viên:** Phan Nguyễn Mai Phương · **MSHV:** 25002711
**Thời lượng mục tiêu:** khoảng 3 phút, nằm trong tổng thời gian trình bày 18–20 phút.

Demo dùng các artifact đã sinh sẵn. Không chạy training trong lúc trình bày.
Khởi động dashboard theo `README.vi.md`, mở `http://localhost:8501` và kiểm tra
trước buổi báo cáo rằng các trang cần dùng đã tải được.

## Quy tắc đồng bộ model

Model/protocol được trình bày ở slide và báo cáo lấy từ
`outputs/course_alignment/20261007/summary.json`. Artifact này khuyến nghị
HGB169 làm predictor tham chiếu cho whole-trajectory/endpoint; HGB126 là
baseline lịch sử. Dashboard hiện tại đọc `outputs/modeling/` của run HGB126 nên
được dùng như **baseline audit/legacy view**, không dùng số của nó để thay cho
metric HGB169. Warning policy phải được nói riêng với prediction model; không
trộn timing của HGB126 với metric của HGB169.

## Trình tự ba phút

### 1. Mở đầu và phạm vi — 25 giây

Nói một câu: “Đây là dashboard đọc artifact cho pipeline NASA C-MAPSS FD001.
Pipeline đi từ raw sensor đến RUL, cảnh báo theo cycle và provenance; các con số
model cuối được đọc từ slide/bảng alignment.”

Chỉ vào phạm vi 100 train engines, 100 test engines, 21 sensor mỗi cycle và
window 30 ở protocol chính. Nhắc rằng FD001 là dữ liệu mô phỏng một điều kiện
vận hành và một fault mode.

### 2. Provenance và dữ liệu — 35 giây

Mở trang **Provenance**. Chỉ vào:

- 100 evaluation rows và 704 forecast rows;
- trạng thái `test_label_join.status=completed_after_final_predictions`;
- schema/run information và checksum nếu trang hiển thị.

Nói rõ dashboard chỉ đọc artifact. Việc ghép test label diễn ra sau khi ghi
prediction cuối; nhãn test không đi vào EDA, feature engineering hoặc chọn model.

Nếu được hỏi về Big Data, trả lời ngắn: “Pipeline dùng batch ETL nhiều file,
Parquet Silver/Gold, PySpark `local[4]` để kiểm tra parity và benchmark có đọc,
rolling, ghi ở scale 1×/10×. Spark ghi qua driver trên Windows nên chưa có claim
hiệu năng cluster.”

### 3. RUL và một trường hợp sai — 40 giây

Mở trang **Đánh giá RUL**. Chọn Engine 023 để chỉ ra trường hợp sai lớn đã lưu:
RUL thật 113, dự đoán 170,4768, absolute error 57,4768 cycles. Không loại điểm
này khỏi metric; dùng nó để giải thích vì sao phải xem residual và phân bố lỗi,
không chỉ đọc một MAE.

Đọc metric HGB169 trên slide/bảng alignment. Nói rõ “view dashboard này là
baseline audit HGB126; metric predictor tham chiếu HGB169 nằm ở bảng alignment”
rồi chuyển sang trang cảnh báo. Không đọc 15,3490 như metric HGB169.

### 4. Cảnh báo theo cycle — 45 giây

Mở trang **Cảnh báo sớm** và chọn trajectory có vạch cảnh báo. Nói ba điểm:

1. Quy tắc cảnh báo là một ngưỡng RUL theo cycle, không phải lịch bảo trì tối ưu.
2. Policy và model được đánh giá riêng; số timing lịch sử của baseline cố định
   `RUL ≤ 30` là 44 trước ngưỡng, 7 đúng ngưỡng và 49 sau ngưỡng trên 100 train
   engines. Summary cũng có sensitivity threshold 40 và calibrated169 historical
   readout; chưa có deployment policy được xác nhận.
3. Đây là OOF nội bộ và mọi train engine đều đi đến failure, nên false-alarm rate
   cấp engine không được suy ra là 0%.

Nếu slide trình bày sensitivity threshold 40, đọc đúng bảng đó và giữ các số
44/7/49 như baseline tham chiếu có nhãn, không thay thế lẫn nhau. Không gọi
calibrated169 historical readout là policy triển khai độc lập.

### 5. Kết — 35 giây

Mở trang **Cross-dataset** hoặc quay lại slide kết luận. Nêu rằng giữ model FD001
để chạy trực tiếp trên FD002–FD004 làm MAE tăng rõ trong matrix window 10; đây là
bằng chứng cần retrain hoặc domain adaptation, không phải claim model dùng được
ngoài miền.

Kết thúc: “Pipeline đã nối được dữ liệu, feature, model, cảnh báo và provenance.
Giới hạn còn lại là dữ liệu mô phỏng, Spark local và thiếu chi phí bảo trì; vì vậy
đầu ra hiện là hỗ trợ phân tích, chưa phải lịch can thiệp tự động.”

## Phương án dự phòng khi dashboard lỗi

Chuyển sang slide 6–7 để trình bày pipeline/parity, slide 14–16 để trình bày RUL
và cảnh báo, rồi slide 20 để nêu giới hạn. Không mở terminal hoặc chạy lại
training trên sân khấu. Có thể dùng `report/main.pdf` nếu dashboard không khởi động được.

## Câu hỏi dữ liệu và Big Data

- **Nhiều file gom thế nào?** Trong từng subset, reader đọc train/test cùng
  schema, thêm `split`, rồi concat/`unionByName`. FD001–FD004 được xử lý riêng vì
  khóa hiện tại chưa có `dataset_id`.
- **Truy vấn ở đâu?** SQLite truy vấn `engine_summary` và `train_targets`; sensor
  detail được lọc từ Silver Parquet bằng Pandas. Chưa có query engine SQL trên
  Parquet hoặc partition pruning production.
- **Transform là gì?** Đổi text thành typed observations, tạo RUL train,
  engine summary, rồi rolling features 30 cycles. Đây là data transformation,
  không phải Transformer model.
- **Feature nào được dùng?** 21 sensor × 6 thống kê = 126 feature columns trong
  protocol window 30; ba setting không nằm trong X hiện tại. History features của
  đối chứng phải gọi đúng theo artifact alignment.
- **Có many-to-many join không?** Không. Quan hệ chính là engine–observations
  1:N; feature–target và prediction–label là 1:1 theo khóa. Bỏ `cycle` khi join
  sẽ nhân dòng.
- **Spark nhanh hơn Pandas không?** Không kết luận như vậy. Benchmark local có
  I/O ở scale 1×/10×, Spark `local[4]`, write qua driver và chưa đo cluster.

## Câu hỏi về phương pháp và giới hạn

- **Vì sao GroupKFold theo engine?** Các window cùng engine chồng lấn 29/30
  cycles; chia ngẫu nhiên theo row gây leakage thời gian và metric lạc quan.
- **Có xử lý nhiễu không?** Rolling statistics tóm tắt dao động; contract kiểm tra
  missing/key/cycle. Repository chưa có moving median, Kalman filter, imputation
  hoặc outlier removal được đánh giá riêng.
- **ARIMA có dự báo RUL không?** Không. ARIMA chỉ forecast Health Index và là phân
  tích chuỗi thời gian phụ.
- **Có áp dụng cho động cơ thật không?** Chưa thể kết luận. C-MAPSS là dữ liệu mô
  phỏng; chưa có telemetry, intervention records, downtime hoặc chi phí bảo trì.
- **Reliability mới kiểm tra gì?** Staging/rollback/manifest, raw boundary,
  dependency và worker save/load/resume đã có bằng chứng trong
  `handoff/CODE_RELIABILITY_20261007.json`; đây không phải kết quả training mới.
