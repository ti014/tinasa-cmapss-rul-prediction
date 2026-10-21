# Tài liệu tham khảo

Danh sách đọc phục vụ **Đề tài 8: Phân tích dữ liệu IoT và bảo trì dự đoán**,
tập trung vào **NASA C-MAPSS FD001** đang dùng trong báo cáo cuối kỳ. Nguồn được
kiểm chứng nền tảng ngày **20/09/2026**, bổ sung nguồn phương pháp ngày
**22/09/2026**.

Nên đọc **R1 → R3 → R4 → R5** cho FD001 và đánh giá; R2 hỗ trợ tổng quan nhưng
chưa đọc toàn văn. Các nguồn thống kê M1–M6 bên dưới cung cấp cơ sở cho phần
thống kê. R8 phục vụ phần Spark đã nằm trong phạm vi; R6 và R7 là đọc thêm về
LSTM/streaming. Đây là tuyển chọn theo phương pháp sử dụng, không phải tổng
quan toàn bộ nghiên cứu mới nhất.

## Cấu trúc

```text
references/
├── README.md          # Danh mục và thứ tự đọc
├── reading-guide.md   # Đọc phần nào, áp dụng ra sao, giới hạn
├── bibliography.bib   # Nguồn FD001, bài nền tảng và tài liệu phương pháp
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

## Nguồn phương pháp bổ sung ngày 22/09/2026

| Mã / citation key | Nguồn chính thức | Vai trò và mức kiểm chứng |
| --- | --- | --- |
| M1 `wilcoxon1945` | Wilcoxon (1945), [Individual Comparisons by Ranking Methods](https://doi.org/10.2307/3001968) | Metadata Crossref và mục References của SciPy; chưa đọc toàn văn bài gốc |
| M2 `efron1979bootstrap` | Efron (1979), [Bootstrap Methods: Another Look at the Jackknife](https://doi.org/10.1214/aos/1176344552) | Metadata Crossref và nguồn dẫn; chưa đọc toàn văn bài gốc |
| M3 `benjamini1995fdr` | Benjamini & Hochberg (1995), [Controlling the False Discovery Rate](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) | Metadata và abstract nhà xuất bản; chưa đọc toàn bộ chứng minh |
| M4 `scipyWilcoxon` | [SciPy 1.16.2 — Wilcoxon](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.stats.wilcoxon.html) | Đã đọc định nghĩa, giả định, ties/zeros và lựa chọn phương pháp p-value |
| M5 `scipyBootstrap` | [SciPy 1.16.2 — bootstrap](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.stats.bootstrap.html) | Đã đọc resampling, paired, percentile/BCa và giới hạn suy biến |
| M6 `scipyFDR` | [SciPy 1.16.2 — FDR correction](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.stats.false_discovery_control.html) | Đã đối chiếu BH/BY và điều kiện phụ thuộc; dùng cho cấu hình M2.0 đã chốt |
| M7 `sklearnRidge` | [scikit-learn — Ridge](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification) | Hàm mục tiêu và L2 regularization |
| M8 `sklearnHistBoost` | [scikit-learn — histogram gradient boosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting) | Boosting và binning; chưa có hyperparameter được chọn |
| M9 `sklearnGroupedCV` | [scikit-learn — grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) | Ranh giới group trong cross-validation |
| M10 `sklearnPCA` | [scikit-learn — PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca) | Phép chiếu và phương sai; không chứng minh PCA tự tạo health signal |
| M11 `hyndman2021fpp3` | Hyndman & Athanasopoulos (2021), [Forecasting: Principles and Practice, 3rd ed.](https://otexts.com/fpp3/) | Đã đọc các mục 5.4, 9.1, 9.5, 9.7; dùng lý thuyết, không chuyển pipeline sang R |
| M12 `statsmodelsADF` | [statsmodels — ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) | Giả thuyết unit root và cấu hình diagnostic |
| M13 `sparkSQLGuide` | [Apache Spark — SQL and DataFrames](https://spark.apache.org/docs/latest/sql-programming-guide.html) | Dữ liệu có schema; chưa có benchmark hoặc phiên bản Spark được chọn |

Không tải thêm PDF trong lượt này. Các URL `stable`/`latest` là tài liệu online
được truy cập ngày 22/09/2026; không coi phiên bản trang web là phiên bản đã cài
trong repository. Ba nguồn SciPy được cố định ở 1.16.2, khớp manifest lúc rà
soát. Các bài gốc M1–M3 cung cấp attribution; phần thao tác và giả định được
đối chiếu với tài liệu chính thức đã đọc, không tuyên bố đã đọc chứng minh gốc.

Báo cáo có [cơ sở lý thuyết](../report/sections/theory.tex) và
[protocol thống kê](../report/sections/methodology.tex) (Mục 4.6). Chỉ tài liệu
được trích trong thân báo cáo xuất hiện trong bibliography của PDF; danh mục
đọc rộng hơn không đồng nghĩa tất cả nguồn đã được sử dụng hoặc đọc toàn văn.

## Dữ liệu và phạm vi

- Nguồn dữ liệu: [DATA_SOURCE.md](../data/raw/DATA_SOURCE.md), [README gốc](../data/raw/readme.txt) và [NASA Open Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
- Kết quả đối chiếu file local: train có **20.631 dòng / 100 động cơ**, test có **13.096 dòng / 100 động cơ** và có **100 nhãn RUL**. Mỗi dòng gồm 26 cột: engine ID, cycle, 3 operational settings và 21 sensor readings.
- FD001 là dữ liệu mô phỏng theo **cycle**, không có timestamp lịch. Không chuyển RUL sang giờ/ngày nếu chưa có giả định quy đổi được giải thích.
- Những ví dụ về window, mô hình và ngưỡng cảnh báo là **đề xuất thực nghiệm**, không phải yêu cầu bắt buộc của giảng viên hoặc kết quả đã quan sát.

Ví dụ: nếu một động cơ train kết thúc ở cycle 200 thì RUL tại cycle 170 bằng
30 cycles. Lịch sử cảm biến từ cycle 141 đến 170 có thể tạo thành một window
dài 30 cycles để dự đoán nhãn này. Đây là ví dụ về cách xây dựng dữ liệu, không
phải kết quả của mô hình.

PDF để đọc offline được tập hợp trong [papers/](papers/README.md). Bảng này ghi rõ file đã có, nguồn tải và bài chưa tải được. R1 được sao chép từ PDF gốc trong `data/raw/`; file dữ liệu gốc được giữ nguyên. `references/` không bị loại trừ bởi `.gitignore` hiện tại.

## Nguồn bổ sung ngày 27/09/2026

- `sklearnHGBAPI2026`: [HistGradientBoostingRegressor API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html). Đối chiếu tham số và `early_stopping=False` với môi trường cài scikit-learn 1.9.1.

Nguồn này giải thích phương pháp. Kết quả định lượng lấy từ artifact của
repository, không lấy từ ví dụ của scikit-learn.
