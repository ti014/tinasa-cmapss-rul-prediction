# Hướng dẫn đọc và áp dụng vào báo cáo

Tài liệu này hỗ trợ các phần **nguồn dữ liệu → xử lý chuỗi cảm biến → window →
phân tích và dự đoán → trực quan hóa → báo cáo**. Các mô hình được nhắc đến là
gợi ý nghiên cứu; chúng không được xem là đã triển khai hoặc đã có kết quả.

## R1 — Hiểu nguồn gốc dữ liệu

**Saxena, Goebel, Simon & Eklund (2008). Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.** IEEE PHM 2008, 1–9. Key: `saxena2008damage`.

[DOI](https://doi.org/10.1109/PHM.2008.4711414) · [PDF local](papers/R1_Saxena_2008_Damage_Propagation.pdf).

- **Đọc trước:** Sections II, VI–VII về prognostics, competition data và performance evaluation; sau đó xem Sections III–V nếu cần giải thích mô phỏng vật lý.
- **Giá trị:** hiểu run-to-failure, sự suy giảm cảm biến và lý do việc dự đoán hỏng quá muộn cần được đánh giá riêng.
- **Áp dụng đề xuất:** viết phần mô tả dữ liệu, diễn giải nhãn RUL và giới hạn của dữ liệu mô phỏng. Ví dụ, RUL = 30 nghĩa là động cơ còn 30 cycles theo nhãn, không phải xác suất hỏng 30%.
- **Giới hạn:** bài gốc mô tả PHM’08 challenge; dùng R3 và README của bộ dữ liệu để phân biệt challenge với FD001.
- **Đã kiểm chứng:** DOI, metadata hội nghị, trang đầu và Sections VI–VII trong PDF local. Chưa tái lập thí nghiệm của bài báo.

## R2 — Xây dựng cơ sở lý thuyết cho toàn pipeline

**Lei, Li, Guo, Li, Yan & Lin (2018). Machinery health prognostics: A systematic review from data acquisition to RUL prediction.** Mechanical Systems and Signal Processing, 104, 799–834. Key: `lei2018machinery`.

[DOI](https://doi.org/10.1016/j.ymssp.2017.11.016) · [Trang nhà xuất bản](https://www.sciencedirect.com/science/article/pii/S0888327017305988).

- **Đọc trước:** Sections 2–5: data acquisition, health indicator construction, health stage division và RUL prediction.
- **Giá trị:** giải thích mối liên hệ giữa thu thập dữ liệu, biểu diễn sức khỏe thiết bị và dự đoán tuổi thọ.
- **Áp dụng đề xuất:** tổ chức phần cơ sở lý thuyết và giải thích vì sao phải xử lý tín hiệu cảm biến trước khi dự đoán. Ví dụ, xem xu hướng sensor theo cycle trước khi chọn rolling mean hoặc độ dốc làm đặc trưng.
- **Giới hạn:** bài tổng quan bao quát nhiều loại máy; không phải mọi kỹ thuật đều phù hợp với FD001.
- **Đã kiểm chứng:** metadata từ Crossref, phần giới thiệu và nội dung xem trước do nhà xuất bản cung cấp; chưa đọc toàn văn. Toàn văn có thể cần quyền truy cập thư viện.

## R3 — Sử dụng C-MAPSS và so sánh kết quả đúng phạm vi

**Ramasso & Saxena (2014). Review and Analysis of Algorithmic Approaches Developed for Prognostics on CMAPSS Dataset.** Annual Conference of the PHM Society, 6(1). Key: `ramasso2014review`.

[DOI](https://doi.org/10.36001/phmconf.2014.v6i1.2512) · [Toàn văn chính thức](https://papers.phmsociety.org/index.php/phmconf/article/download/2512/1473) · [PDF local](papers/R3_Ramasso_Saxena_2014_CMAPSS_Review.pdf).

- **Đọc trước:** Section 2 về các bộ dữ liệu và Section 5 về cách sử dụng C-MAPSS.
- **Giá trị:** phân biệt FD001–FD004 với PHM’08 challenge; hiểu tác động của noise, operational conditions và fault modes.
- **Áp dụng đề xuất:** mô tả rõ split, cách tạo tập validation, sensor và nhãn. Ví dụ, không thể dùng RMSE trên FD001 để kết luận một phương pháp tốt hơn nghiên cứu chạy trên FD004.
- **Giới hạn:** đây là bài conference năm 2014; không nhầm title hoặc DOI với bài journal cùng tác giả “Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets”.
- **Đã kiểm chứng:** metadata và nội dung liên quan trong toàn văn chính thức, đặc biệt là Sections 2 và 5.

## R4 — Tham khảo trực tiếp về window và RUL

**Li, Ding & Sun (2018). Remaining useful life estimation in prognostics using deep convolution neural networks.** Reliability Engineering & System Safety, 172, 1–11. Key: `li2018rul`.

[DOI](https://doi.org/10.1016/j.ress.2017.11.021) · [Author manuscript tại University of California](https://escholarship.org/content/qt5ns8r3fs/qt5ns8r3fs.pdf) · [PDF local](papers/R4_Li_Ding_Sun_2018_CNN_RUL.pdf).

- **Đọc trước:** Sections 3.1.1–3.1.5 và 3.2.3–3.2.4 về preprocessing, window và ảnh hưởng của nhãn.
- **Giá trị:** cung cấp một ví dụ cụ thể về dự đoán RUL từ chuỗi đa cảm biến. Bài báo dùng 14 sensor, window mặc định 30 cycles và RUL cap 125; đây là cấu hình của nghiên cứu, không phải mặc định của báo cáo này.
- **Áp dụng đề xuất:** biểu diễn một mẫu bằng `30 cycles × số sensor được chọn`, sau đó so sánh với baseline chỉ dùng giá trị tại cycle hiện tại.
- **Giới hạn quan trọng:** thí nghiệm với window dài hơn 31 cycles loại các test engine có chuỗi quá ngắn. Không so sánh kết quả trên tập rút gọn với kết quả trên toàn bộ 100 test engines. Việc dùng RUL cap khác nhau cũng làm thay đổi protocol so sánh.
- **Đã kiểm chứng:** metadata và các phần nêu trên trong author manuscript; chưa tái lập kết quả.

## R5 — Đánh giá dự đoán có đủ sớm để hành động không

**Saxena, Celaya, Saha, Saha & Goebel (2010). Metrics for Offline Evaluation of Prognostic Performance.** International Journal of Prognostics and Health Management, 1(1). Key: `saxena2010metrics`.

[DOI](https://doi.org/10.36001/ijphm.2010.v1i1.1336) · [Toàn văn chính thức](https://papers.phmsociety.org/index.php/ijphm/article/download/1336/324) · [PDF local](papers/R5_Saxena_2010_Prognostic_Metrics.pdf).

- **Đọc trước:** Section 6.2: Prognostic Horizon, alpha-lambda performance, Relative Accuracy và Convergence.
- **Giá trị:** xem chất lượng dự đoán thay đổi theo thời điểm, thay vì chỉ một sai số tổng hợp.
- **Áp dụng đề xuất:** ngoài RMSE và MAE, vẽ RUL dự đoán cùng RUL thật theo cycle trên các động cơ validation. Ví dụ, hai mô hình có RMSE gần nhau vẫn có thể đạt mức sai số chấp nhận được tại những thời điểm khác nhau.
- **Giới hạn:** đánh giá toàn bộ trajectory cần dự đoán tại nhiều thời điểm; một dự đoán ở cycle cuối của mỗi test engine không đủ cho phân tích này. Ngưỡng chấp nhận phải được định nghĩa và giải thích riêng.
- **Đã kiểm chứng:** metadata, abstract và Section 6.2. Dùng năm 2010 theo volume/issue và [NASA NTRS](https://ntrs.nasa.gov/citations/20100039169); trang publisher/Crossref hiện hiển thị thêm ngày 2021.

## R6 — Đọc thêm khi chọn mô hình sequence

**Zheng, Ristovski, Farahat & Gupta (2017). Long Short-Term Memory Network for Remaining Useful Life estimation.** IEEE ICPHM 2017, 88–95. Key: `zheng2017lstm`.

[DOI](https://doi.org/10.1109/ICPHM.2017.7998311) · [Danh mục nghiên cứu Hitachi](https://www.hitachi.com/en-us/about/rd/publications/).

- **Mục đích đọc:** tìm hiểu LSTM cho bài toán RUL sau khi đã xác định window và evaluation protocol.
- **Áp dụng đề xuất:** so sánh baseline với một mô hình khai thác lịch sử sensor trên cùng split, nhãn và evaluation protocol.
- **Giới hạn kiểm chứng:** đã xác minh title, tác giả, năm, venue, trang và DOI qua Crossref/Hitachi; chưa đọc được full text chính thống. Vì vậy tài liệu này không trích số lớp, hyperparameter hay điểm FD001 của bài.
- **Ưu tiên:** tài liệu bổ sung; chưa cần đọc để hoàn thành phần nguồn dữ liệu, EDA và trực quan hóa.

## R7 — Hiểu time window trong streaming

**Akidau et al. (2015). The Dataflow Model: A Practical Approach to Balancing Correctness, Latency, and Cost in Massive-Scale, Unbounded, Out-of-Order Data Processing.** PVLDB, 8(12), 1792–1803. Key: `akidau2015dataflow`.

[Google Research](https://research.google/pubs/the-dataflow-model-a-practical-approach-to-balancing-correctness-latency-and-cost-in-massive-scale-unbounded-out-of-order-data-processing/) · [Toàn văn VLDB](https://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf) · [PDF local](papers/R7_Akidau_2015_Dataflow_Model.pdf).

- **Mục đích đọc:** tìm hiểu event time, processing time, windowing và trade-off giữa độ đầy đủ, latency và chi phí.
- **Áp dụng đề xuất:** nếu replay dữ liệu để minh họa streaming, giải thích cách nhóm bản ghi và xử lý dữ liệu đến trễ. Ví dụ, bản ghi cycle 99 đến sau cycle 100 có thể làm thay đổi kết quả tổng hợp đã tạo.
- **Giới hạn:** FD001 không có timestamp thực. Window 30 cycles cho ML khác time window 30 giây của stream processor; mọi ánh xạ cycle sang thời gian replay phải được ghi là mô phỏng.
- **Đã kiểm chứng:** metadata và abstract Google Research; toàn văn VLDB truy cập được. Chưa đánh giá hay triển khai hệ thống streaming.

## R8 — Cơ sở cho phần Spark/Big Data nếu sử dụng

**Zaharia et al. (2012). Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing.** USENIX NSDI 2012, 15–28. Key: `zaharia2012rdd`.

[USENIX](https://www.usenix.org/conference/nsdi12/technical-sessions/presentation/zaharia) · [Toàn văn chính thức](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf) · [PDF local](papers/R8_Zaharia_2012_Resilient_Distributed_Datasets.pdf).

- **Đọc trước:** abstract, introduction và phần RDD abstraction: transformations, persistence, lineage.
- **Giá trị:** nền tảng để giải thích xử lý phân tán và phục hồi dữ liệu của Spark.
- **Áp dụng đề xuất:** nếu báo cáo dùng Spark, trình bày lợi ích của việc tái sử dụng dữ liệu đã xử lý khi tính nhiều thống kê sensor.
- **Giới hạn:** file FD001 hiện tại chỉ vài MB, nên kích thước chưa chứng minh nhu cầu cluster. Đây là bài nền tảng, không phải hướng dẫn Spark API hiện hành hay bằng chứng Spark sẽ nhanh hơn trên máy của bạn.
- **Đã kiểm chứng:** metadata, abstract và introduction; PDF USENIX truy cập được. Chưa chạy benchmark.

## Chuyển tài liệu đọc thành nội dung báo cáo

Các mục dưới đây được đề xuất từ yêu cầu đề tài và dữ liệu local; chúng chưa phải kết quả thực nghiệm.

| Phần cần làm | Tài liệu hỗ trợ | Sản phẩm có thể đưa vào báo cáo |
| --- | --- | --- |
| Nguồn và mô tả dữ liệu | R1, R3, README gốc | Mô tả FD001, schema 26 cột, train/test, RUL và giới hạn mô phỏng |
| Nhiễu và xu hướng suy giảm | R2, R3 | Biểu đồ sensor theo cycle, so sánh tín hiệu gốc với trailing smoothing |
| Chuẩn hóa và window | R3, R4 | Định nghĩa window, sensor, label và cấu hình preprocessing |
| Nghiệp vụ dự đoán bảo trì | R1, R4; R6 bổ sung | Baseline RUL, mô hình ứng viên và phân tích trường hợp dự đoán trễ |
| Trực quan hóa và cảnh báo sớm | R5 | Biểu đồ RUL thật–dự đoán, ngưỡng cảnh báo minh họa và sai số theo động cơ |
| Pipeline Big Data | R7, R8 nếu triển khai | Phân biệt batch/replay, cách windowing và bằng chứng vận hành pipeline |

## Những lựa chọn cần kiểm chứng khi thực nghiệm

1. **Split theo động cơ:** tách một nhóm engine trong train để validation trước khi tạo window. Không chia ngẫu nhiên các window chồng lấn của cùng engine sang train và validation; giữ test chính thức cho lần đánh giá cuối.
2. **Preprocessing chỉ dùng dữ liệu được phép biết:** fit scaler và chọn đặc trưng từ train; window hoặc smoothing tại cycle `t` chỉ dùng dữ liệu đến `t`. Centered smoothing có thể làm lộ thông tin tương lai.
3. **Nhãn train/test khác cách tính:** train dùng cycle cuối trừ cycle hiện tại. Test đã bị cắt trước khi hỏng, nên nhãn ở cycle cuối lấy từ `RUL_FD001.txt`; không tự gán bằng 0.
4. **Window không đi qua hai engine:** ví dụ, window 141–170 phải thuộc cùng một engine. Kiểm tra local cho thấy test engine ngắn nhất có 31 dòng; nếu chọn window dài hơn, phải công bố cách xử lý chuỗi ngắn và số engine còn lại trong đánh giá.
5. **RUL cap là lựa chọn mô hình:** nếu dùng piecewise-linear label, ghi rõ cap train và cách xử lý nhãn evaluation. Không thay đổi nhãn test âm thầm để có điểm đẹp hơn.
6. **Cảnh báo không phải lịch bảo trì tối ưu:** cảnh báo khi RUL dự đoán ≤ 30 cycles chỉ là quy tắc minh họa. Ngưỡng thực tế cần lead time và chi phí của false alarm/missed alarm; FD001 không cung cấp các thông tin này.
7. **Anomaly detection và RUL khác nhau:** FD001 phù hợp nghiên cứu degradation/RUL, nhưng không có sẵn nhãn anomaly cho mọi cycle. Nếu tự tạo nhãn bằng ngưỡng RUL, cần gọi rõ đó là nhãn dẫn xuất.

Ví dụ về thứ tự áp dụng: sau R1–R3, viết một trang mô tả dữ liệu; sau R4,
mô tả một window mẫu cùng nhãn; sau R5, phác thảo ba biểu đồ gồm sensor theo
cycle, RUL thật–dự đoán và sai số theo động cơ. Chỉ đọc sâu R7–R8 khi phạm vi
báo cáo có streaming hoặc Spark.

## Bổ sung theo phạm vi đã chốt ngày 22/09/2026

Danh sách và mức kiểm chứng của M1–M13 nằm trong [README](README.md).
Các mục áp dụng đề xuất ở R1–R8 là hướng dẫn đọc ban đầu; phạm vi hiện hành
là baseline, Ridge, HistGradientBoosting, batch ETL Pandas/PySpark và ARIMA
phụ. Chưa mở thêm CNN, LSTM hoặc streaming.

### Nguồn cho phần thống kê: M1–M6

- Đọc M4 trước để phân biệt Wilcoxon với kiểm định trung bình, kiểm tra giả
  định đối xứng và ảnh hưởng của ties/zeros. M1 là nguồn gốc phương pháp,
  chưa được đọc toàn văn trong lượt bổ sung.
- Đọc M5 về resampling, ghép cặp và loại khoảng tin cậy. Áp dụng theo engine,
  không resample từng cycle như mẫu độc lập. M2 là bài nền tảng, metadata đã
  đối chiếu nhưng chưa đọc toàn văn.
- Đọc M3 abstract và M6 để phân biệt FDR với xác suất mỗi kết luận đúng.
  BH có điều kiện về phụ thuộc; cấu hình M2.0 dùng
  `scipy.stats.false_discovery_control` với phương pháp BH trên đúng 21 sensors.
  Không suy từ correlation dương quan sát được rằng mọi điều kiện bảo đảm đều đã thỏa.
- Đầu ra cần viết: estimand (đại lượng cần ước lượng), giả thuyết, đơn vị mẫu,
  họ kiểm định, cấu hình và xử lý trường hợp suy biến. Xem
  [protocol](../report/sections/statistical-protocol.tex): ranh giới mean 30
  cycles đầu/cuối và cấu hình thống kê đã được chốt trước khi chạy suy luận.

Ví dụ: engine kết thúc ở cycle 200 dùng 1–30 và 171–200 để phân tích hồi cứu.
Hai số trung bình tạo một cặp, không tạo 60 quan sát độc lập. Chênh lệch cuối
trừ đầu được giữ cùng engine khi bootstrap.

### Đọc trước mô hình: M7–M10

M7 hỗ trợ hàm mục tiêu Ridge và lý do chuẩn hóa đặc trưng. M8 hỗ trợ cơ chế
boosting/binning. M9 giải thích group split; không thay thế việc xác định
các cycle đánh giá trong validation. M10 chỉ hỗ trợ PCA như một phép chiếu;
chiều Health Index và ý nghĩa suy giảm cần kiểm tra trên training data.

Đầu ra: giải thích lựa chọn mô hình, loss và ranh giới preprocessing; sau đó
chốt search space và validation sampling trước huấn luyện. Không lấy cấu hình
ví dụ trong docs làm hyperparameter đã chọn.

Đối chiếu bổ sung ngày 22/09/2026: công thức asymmetric score lấy trực tiếp
từ R4, Section 3.1.4, Equation (5), trang in 9 (trang PDF 10). Với sai số
`predicted - true`, mẫu số nhánh âm là 13 và nhánh không âm là 10; cộng theo
engine. Đã kiểm tra công thức trên bản render local. R1 Section VII có cách
ghi hệ số không nhất quán với mô tả phạt muộn mạnh hơn; dùng rõ quy ước R4
trong báo cáo, không tự chép hệ số từ R1 hoặc đổi dấu sai số âm thầm.

### Đọc khi triển khai ARIMA và ETL: M11–M13

Đọc M11 mục 9.1 về stationarity/differencing, 9.5 về ARIMA, 5.4 và 9.7 về
residual diagnostics. M12 bổ sung cách hiểu ADF; không bác bỏ unit root chưa
chứng minh chuỗi dừng. M13 bổ sung DataFrame/Spark SQL cho nền tảng R8.

Đầu ra: protocol forecast theo thời gian và thiết kế benchmark có parity,
cấu hình máy, chế độ cache và thời gian đo rõ ràng. Chưa có bằng chứng để
tuyên bố Spark nhanh hơn hoặc PCA–ARIMA dự đoán được RUL.

## Đọc kết quả cải thiện ngày 27/09/2026

Đọc mục Nested CV trong tài liệu scikit-learn (`sklearnNestedCV2026`) để phân
biệt vòng chọn cấu hình và vòng đánh giá. Sau đó đọc API
HistGradientBoostingRegressor (`sklearnHGBAPI2026`) để hiểu các tham số trong
manifest. Hai URL chính thức nằm trong README cùng thư mục.

Đối chiếu với `experiments/README.md`: giảm 9,75% low-RUL MAE của `hgb_08`
thuộc inner CV, còn so sánh 3,6676 → 3,6730 cycles thuộc outer evaluation của
cả quy trình lựa chọn. Không ghép hai kết quả này thành kết luận model đã tốt
hơn. Các nguồn được truy cập ngày 27/09/2026; không tải thêm paper trong đợt này.
