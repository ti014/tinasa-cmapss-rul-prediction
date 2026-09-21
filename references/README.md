# Tài liệu tham khảo

Danh sách đọc phục vụ **Đề tài 8: Phân tích dữ liệu IoT và bảo trì dự đoán**,
tập trung vào **NASA C-MAPSS FD001** đang dùng trong báo cáo cuối kỳ. Nguồn được
kiểm chứng ngày **20/09/2026**.

Nên đọc **R1 → R2 → R3 → R4 → R5** trước. R6–R8 phục vụ mở rộng mô hình hoặc phần Big Data. Đây là tuyển chọn nền tảng theo mức độ phù hợp, không phải tổng quan toàn bộ nghiên cứu mới nhất.

## Cấu trúc

```text
references/
├── README.md          # Danh mục và thứ tự đọc
├── reading-guide.md   # Đọc phần nào, áp dụng ra sao, giới hạn
├── bibliography.bib   # 8 bài báo và 1 nguồn dữ liệu
└── papers/            # PDF để đọc offline và bảng nguồn tải
```

## Danh mục ưu tiên

| Mã | Bài báo, nguồn chính thức | Vai trò trong báo cáo |
| --- | --- | --- |
| R1 | Saxena et al. (2008), [Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation](https://doi.org/10.1109/PHM.2008.4711414) | Nguồn dữ liệu mô phỏng, quá trình suy giảm và định nghĩa RUL |
| R2 | Lei et al. (2018), [Machinery health prognostics: A systematic review from data acquisition to RUL prediction](https://doi.org/10.1016/j.ymssp.2017.11.016) | Cơ sở lý thuyết và cách tổ chức pipeline phân tích |
| R3 | Ramasso & Saxena (2014), [Review and Analysis of Algorithmic Approaches Developed for Prognostics on CMAPSS Dataset](https://papers.phmsociety.org/index.php/phmconf/article/view/2512) | Hiểu FD001, chuẩn bị dữ liệu và so sánh thực nghiệm |
| R4 | Li, Ding & Sun (2018), [Remaining useful life estimation in prognostics using deep convolution neural networks](https://doi.org/10.1016/j.ress.2017.11.021) | Window, chuẩn hóa, nhãn RUL và thiết kế thực nghiệm |
| R5 | Saxena et al. (2010), [Metrics for Offline Evaluation of Prognostic Performance](https://papers.phmsociety.org/index.php/ijphm/article/view/1336) | Đánh giá dự đoán và mức độ hữu ích của cảnh báo sớm |
| R6 | Zheng et al. (2017), [Long Short-Term Memory Network for Remaining Useful Life estimation](https://doi.org/10.1109/ICPHM.2017.7998311) | Đọc thêm khi cân nhắc LSTM |
| R7 | Akidau et al. (2015), [The Dataflow Model: A Practical Approach to Balancing Correctness, Latency, and Cost in Massive-Scale, Unbounded, Out-of-Order Data Processing](https://research.google/pubs/the-dataflow-model-a-practical-approach-to-balancing-correctness-latency-and-cost-in-massive-scale-unbounded-out-of-order-data-processing/) | Windowing, event time và dữ liệu đến trễ nếu mô phỏng streaming |
| R8 | Zaharia et al. (2012), [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](https://www.usenix.org/conference/nsdi12/technical-sessions/presentation/zaharia) | Nền tảng Spark nếu lựa chọn xử lý phân tán |

Xem [hướng dẫn đọc](reading-guide.md) để biết phần cần đọc, liên kết toàn văn và ví dụ áp dụng. Metadata trích dẫn nằm trong [bibliography.bib](bibliography.bib).

## Dữ liệu và phạm vi

- Nguồn dữ liệu: [DATA_SOURCE.md](../data/raw/DATA_SOURCE.md), [README gốc](../data/raw/readme.txt) và [NASA Open Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
- Kết quả đối chiếu file local: train có **20.631 dòng / 100 động cơ**, test có **13.096 dòng / 100 động cơ** và có **100 nhãn RUL**. Mỗi dòng gồm 26 cột: engine ID, cycle, 3 operational settings và 21 sensor readings.
- FD001 là dữ liệu mô phỏng theo **cycle**, không có timestamp lịch. Không chuyển RUL sang giờ/ngày nếu chưa có giả định quy đổi được giải thích.
- Bộ dữ liệu AI4I đã nằm trong `data/archive/ai4i/`; danh sách này tập trung vào FD001 đang dùng.
- Những ví dụ về window, mô hình và ngưỡng cảnh báo là **đề xuất thực nghiệm**, không phải yêu cầu bắt buộc của giảng viên hoặc kết quả đã quan sát.

Ví dụ: nếu một động cơ train kết thúc ở cycle 200 thì RUL tại cycle 170 bằng
30 cycles. Lịch sử cảm biến từ cycle 141 đến 170 có thể tạo thành một window
dài 30 cycles để dự đoán nhãn này. Đây là ví dụ về cách xây dựng dữ liệu, không
phải kết quả của mô hình.

PDF để đọc offline được tập hợp trong [papers/](papers/README.md). Bảng này ghi rõ file đã có, nguồn tải và bài chưa tải được. R1 được sao chép từ PDF gốc trong `data/raw/`; file dữ liệu gốc được giữ nguyên. `references/` không bị loại trừ bởi `.gitignore` hiện tại.
