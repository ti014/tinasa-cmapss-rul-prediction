<a href="README.md"><img src="https://flagcdn.com/20x15/us.png" alt="English" width="20" height="15"> English</a> · <a href="README.vi.md"><img src="https://flagcdn.com/20x15/vn.png" alt="Tiếng Việt" width="20" height="15"> Tiếng Việt</a>

<hr>

# Pipeline dự đoán RUL trên NASA C-MAPSS

Repository này phục vụ báo cáo cuối kỳ môn Phân tích dữ liệu lớn. Nội dung chính là xây dựng một quy trình có khả năng tái lập để phân tích dữ liệu cảm biến đa biến và dự đoán Remaining Useful Life (RUL, tuổi thọ hữu ích còn lại) trên NASA C-MAPSS FD001. Báo cáo chú trọng nguồn gốc dữ liệu rõ ràng, phân tích thống kê đúng phương pháp, đánh giá không data leakage (rò rỉ dữ liệu), ETL có khả năng mở rộng và trực quan hóa kết quả dễ giải thích.

## Trạng thái

Repository hiện ở **giai đoạn lập kế hoạch và chuẩn bị dữ liệu**.

Đã có:

- Các file NASA C-MAPSS FD001 nguyên bản và tài liệu mô tả nguồn.
- Kết quả kiểm tra số dòng, số động cơ, schema, khóa và tính liên tục của cycle.
- Danh mục tài liệu tham khảo và hướng dẫn đọc.

Chưa triển khai:

- Pipeline xử lý bằng Pandas và PySpark.
- Phân tích thống kê và xây dựng đặc trưng (feature engineering).
- Các mô hình RUL và phân tích ARIMA.
- Dashboard Streamlit và báo cáo LaTeX hoàn chỉnh.

Các lệnh cài đặt, huấn luyện, đánh giá và chạy dashboard chỉ được bổ sung sau khi code và dependency manifest tương ứng tồn tại.

## Mục tiêu

Báo cáo tập trung trả lời bốn câu hỏi:

1. FD001 đáng tin và phù hợp đến mức nào cho một nghiên cứu học thuật về predictive maintenance?
2. Những tín hiệu cảm biến nào thể hiện sự suy giảm có thể đo lường theo vòng đời động cơ?
3. Có thể dự đoán RUL chính xác đến mức nào từ 30 cycles vận hành gần nhất?
4. Làm thế nào trình bày kết quả mô hình thành bằng chứng cảnh báo sớm minh bạch và tái lập được?

Mục tiêu chính là **dự đoán RUL**. Supervised regression (hồi quy có giám sát) là cách mô hình hóa phù hợp vì biến mục tiêu là số cycle còn lại; đây không phải tên hoặc mục tiêu của báo cáo.

## Dữ liệu

Báo cáo sử dụng **NASA C-MAPSS FD001**, bộ dữ liệu mô phỏng quá trình động cơ turbofan vận hành đến khi hỏng do NASA Ames Prognostics Center of Excellence công bố.

| File | Vai trò | Quy mô local |
| --- | --- | ---: |
| `train_FD001.txt` | Lịch sử đầy đủ đến khi hỏng của 100 động cơ | 20.631 dòng |
| `test_FD001.txt` | Lịch sử bị cắt trước khi hỏng của 100 động cơ khác | 13.096 dòng |
| `RUL_FD001.txt` | RUL thật tại cycle quan sát cuối của từng động cơ trong tập test | 100 nhãn |

Mỗi quan sát gồm:

- Mã động cơ.
- Cycle vận hành.
- Ba thông số vận hành (operational settings).
- Giá trị của 21 cảm biến (sensor measurements).

FD001 đại diện cho một điều kiện vận hành và một dạng suy giảm High-Pressure Compressor. Tín hiệu chứa nhiễu cảm biến mô phỏng.

### Nguồn gốc và tính toàn vẹn

- [Bản ghi NASA Open Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- [Bản lưu có DOI và checksum](https://doi.org/10.5281/zenodo.15346912)
- Checksum ZIP đã ghi nhận: `MD5 79a22f36e80606c69d0e9e4da5bb2b7a`
- Ghi chú nguồn local: [`data/raw/DATA_SOURCE.md`](data/raw/DATA_SOURCE.md)

Đây là dữ liệu mô phỏng, không phải telemetry thu trực tiếp ngoài thực địa. Vì vậy, kết quả hỗ trợ so sánh học thuật tái lập trên FD001 nhưng chưa chứng minh hiệu quả khi triển khai thực tế.

## Kiến trúc dữ liệu dự kiến

```text
Các file text gốc của NASA
        |
        v
Bronze: dữ liệu nguồn bất biến
        |
        v
Silver: observations có kiểu dữ liệu, đã kiểm tra và lưu Parquet
        |
        v
Gold: rolling features, predictions, metrics và bảng SQLite
        |
        +-------------------+
        |                   |
        v                   v
Phân tích thống kê      Mô hình dự đoán RUL
        |                   |
        +---------+---------+
                  v
       Hình minh họa, dashboard Streamlit
       và báo cáo LaTeX cuối kỳ
```

Pandas được dùng cho EDA, thống kê và chuẩn bị dữ liệu mô hình. PySpark tái hiện quy trình ETL và phục vụ thử nghiệm khả năng mở rộng (scalability benchmark). Bộ dữ liệu gốc chỉ có dung lượng vài MB, vì vậy repository không tuyên bố FD001 tự thân chứng minh khả năng xử lý dữ liệu quy mô lớn.

## Phương pháp dự kiến

### Chất lượng dữ liệu

- Kiểm tra schema 26 cột và kiểu dữ liệu số.
- Bảo đảm khóa `(split, engine_id, cycle)` không trùng.
- Kiểm tra cycle liên tục trong từng động cơ.
- Ghi lại số dòng trước và sau mỗi phép biến đổi.
- Giữ phạm vi engine ID riêng biệt giữa train và test.

### Phân tích thống kê

- So sánh hành vi cảm biến ở giai đoạn đầu và cuối vòng đời theo từng động cơ.
- Báo cáo effect size và khoảng tin cậy bootstrap 95%.
- Dùng paired Wilcoxon signed-rank test khi phù hợp.
- Kiểm soát multiple testing (kiểm định nhiều giả thuyết) trên các cảm biến bằng Benjamini-Hochberg FDR.
- Ước lượng độ bất định của các metric đánh giá bằng bootstrap theo động cơ.

### Xây dựng đặc trưng

Biểu diễn chính sử dụng trailing window dài 30 cycles. Với mỗi cảm biến không hằng, các đặc trưng dự kiến gồm:

- Giá trị gần nhất.
- Mean và standard deviation.
- Minimum và maximum.
- Linear trend slope.

Mỗi window chỉ chứa dữ liệu của một động cơ và không sử dụng quan sát tương lai.

### Dự đoán RUL

Các phương pháp dự kiến được so sánh gồm:

1. Naive median-RUL baseline.
2. Ridge regression làm linear baseline có regularization.
3. HistGradientBoostingRegressor làm nonlinear candidate model.

Mô hình được chọn bằng 5-fold GroupKFold, trong đó `engine_id` xác định group. Nhãn test chính thức của NASA được cô lập cho đến lần đánh giá cuối. Các metric đánh giá gồm engine-level MAE, RMSE và NASA asymmetric scoring function.

### Phân tích ARIMA

ARIMA là phân tích time-series phụ, không phải mô hình RUL chính. Mô hình được fit trên phần đầu chuỗi Health Index của một số động cơ validation đại diện, sau đó dự báo phần còn lại và báo cáo:

- Forecast MAE và RMSE.
- Tỷ lệ bao phủ của khoảng dự đoán 95%.
- Augmented Dickey-Fuller stationarity diagnostics.
- Ljung-Box residual autocorrelation diagnostics.

ARIMA không được áp dụng trực tiếp lên nhãn RUL train vốn giảm theo cycle một cách xác định.

## Phạm vi Big Data

Thành phần Big Data tập trung vào vòng đời dữ liệu và khả năng tái lập, không phóng đại dung lượng của bộ dữ liệu:

- Hai phiên bản ETL tương đương bằng Pandas và PySpark.
- Parquet làm định dạng lưu trữ phân tích.
- SQLite làm lớp truy vấn và báo cáo cục bộ, có tính di động.
- Kiểm tra tương đương về số dòng, khóa và giá trị tổng hợp giữa hai công cụ xử lý.
- Thử nghiệm khả năng mở rộng ở các mức dữ liệu 1×, 10× và 100×.

Dữ liệu nhân bản cho thử nghiệm khả năng mở rộng sử dụng engine ID mới và được tách hoàn toàn khỏi phân tích khoa học cũng như quá trình huấn luyện mô hình.

## Quy trình đánh giá

- Dùng trailing window 30 cycles.
- Chia các fold validation theo động cơ, không chia ngẫu nhiên các dòng hoặc window chồng lấn.
- Chỉ fit scaler, feature selection, PCA và mô hình trên các fold training.
- Chọn mô hình theo trung bình engine-level MAE.
- Fit lại mô hình được chọn trên toàn bộ tập train.
- Tạo đúng một dự đoán cuối cho mỗi động cơ trong số 100 động cơ test.
- Chỉ join predictions với `RUL_FD001.txt` trong bước đánh giá cuối.
- Báo cáo các metric tổng hợp, khoảng tin cậy và sai số phân nhóm theo khoảng RUL.

Báo cáo không đặt trước một mức điểm mục tiêu. Kết quả âm vẫn hợp lệ nếu quy trình đánh giá được giữ nguyên và các nhóm lỗi được phân tích trung thực.

## Trực quan hóa dự kiến

- Phân phối vòng đời động cơ.
- Tóm tắt mức biến thiên và xu hướng của cảm biến.
- Quỹ đạo của cảm biến và Health Index theo vòng đời đã chuẩn hóa.
- RUL thật và RUL dự đoán trên các động cơ validation.
- Biểu đồ phân tán giá trị dự đoán và phân phối phần dư trên tập test.
- Phân tích sai số theo động cơ và khoảng RUL.
- Dashboard Streamlit để xem quỹ đạo và kết quả mô hình của từng động cơ.

Dashboard có thể minh họa cảnh báo khi RUL dự đoán không quá 30 cycles. Đây không phải ngưỡng bảo trì tối ưu vì FD001 không cung cấp chi phí bảo trì, downtime hoặc intervention records.

## Cấu trúc repository

Cấu trúc public hiện tại và dự kiến:

```text
.
├── data/
│   ├── raw/                 # FD001 nguyên bản và ghi chú nguồn
│   └── archive/             # Dữ liệu lưu để đối chiếu, không dùng cho mô hình chính
├── references/              # Bibliography, hướng dẫn đọc và các bài báo hiện có
├── notebooks/               # Notebook EDA và thực nghiệm dự kiến
├── src/                     # Module pipeline và xây dựng mô hình dự kiến
├── tests/                   # Kiểm thử data contract và pipeline dự kiến
├── outputs/                 # Metric, hình và artifact dự kiến
├── report/                  # Báo cáo LaTeX dự kiến
├── app.py                   # Entry point Streamlit dự kiến
├── README.md                # Tài liệu tiếng Anh
└── README.vi.md             # Tài liệu tiếng Việt
```

Các file hỗ trợ agent trong môi trường local được loại khỏi GitHub thông qua `.gitignore`.

## Khả năng tái lập

Khi hoàn thành, repository sẽ cung cấp một quy trình được ghi chép rõ từ dữ liệu nguồn bất biến đến dữ liệu đã xử lý, metric, hình minh họa, dữ liệu dashboard và báo cáo cuối kỳ. Các yêu cầu tái lập gồm:

- Dùng random seed cố định `42` cho chia dữ liệu và mô hình ngẫu nhiên.
- Cố định phiên bản Python dependencies.
- Ghi lại số dòng và kết quả validation.
- Lưu cấu hình mô hình và kết quả đánh giá.
- Tách dữ liệu khoa học khỏi dữ liệu tổng hợp dùng kiểm tra scalability.
- Kiểm tra tự động schema, data leakage, ranh giới window và số lượng dự đoán.

## Bắt đầu

Pipeline thực thi chưa được triển khai. Trước khi có dependency manifest và runnable entry points, đây là một repository nghiên cứu đang phát triển, chưa phải package có thể chạy ngay.

Khi bắt đầu triển khai, phần này sẽ mô tả:

1. Phiên bản Python và Java được hỗ trợ.
2. Cách tạo môi trường và cài dependencies.
3. Lệnh kiểm tra dữ liệu và chạy pipeline.
4. Lệnh huấn luyện và đánh giá cuối.
5. Cách khởi động dashboard Streamlit.
6. Cách biên dịch báo cáo bằng XeLaTeX.

## Bối cảnh học thuật và khai báo AI

Repository hỗ trợ báo cáo cuối kỳ môn **Phân tích dữ liệu lớn**. Báo cáo phân biệt rõ kết quả quan sát được với phần giải thích đề xuất, đồng thời khai báo việc sử dụng AI/LLM theo yêu cầu môn học. Mọi phân tích, code, trích dẫn và kết luận được tạo ra với hỗ trợ của AI vẫn phải qua rà soát của con người và kiểm tra khả năng tái lập.

## Tài liệu tham khảo

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). [Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation](https://doi.org/10.1109/PHM.2008.4711414).
- Ramasso, E., & Saxena, A. (2014). [Review and Analysis of Algorithmic Approaches Developed for Prognostics on CMAPSS Dataset](https://doi.org/10.36001/phmconf.2014.v6i1.2512).
- Li, X., Ding, Q., & Sun, J.-Q. (2018). [Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks](https://doi.org/10.1016/j.ress.2017.11.021).
- Xem [`references/reading-guide.md`](references/reading-guide.md) để biết kế hoạch đọc đầy đủ.

## Giới hạn

- FD001 là dữ liệu mô phỏng với một điều kiện vận hành và một dạng lỗi.
- Cycle vận hành không tương đương với giờ hoặc thời gian lịch.
- Dự đoán RUL không tự động xác định lịch bảo trì tối ưu.
- Thử nghiệm khả năng mở rộng không tạo thêm bằng chứng khoa học độc lập.
- Hiệu quả trên FD001 không chứng minh khả năng khái quát hóa ngoài thực tế.

## Giấy phép và sử dụng dữ liệu

Mã nguồn của repository chưa được chọn giấy phép. Bản ghi NASA Open Data hiện không nêu rõ giấy phép của bộ dữ liệu, vì vậy việc phân phối lại và tái sử dụng phải giữ nguyên ghi nhận nguồn và cần được đối chiếu với bản ghi nguồn trước khi công bố. Giấy phép của repository và chính sách phân phối dữ liệu cuối cùng sẽ được bổ sung trước khi repository được giới thiệu như một phần mềm có thể tái sử dụng.
