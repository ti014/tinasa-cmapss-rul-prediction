<a href="README.md"><img src="https://flagcdn.com/20x15/us.png" alt="English" width="20" height="15"> English</a> · <a href="README.vi.md"><img src="https://flagcdn.com/20x15/vn.png" alt="Tiếng Việt" width="20" height="15"> Tiếng Việt</a>

<hr>

# Đề tài 8: Phân tích dữ liệu IoT và bảo trì dự đoán trên NASA C-MAPSS

**Học viên:** Phan Nguyễn Mai Phương · **MSHV:** 25002711.

**Đồng bộ tài liệu 12/10/2026:** mô tả pipeline được đối chiếu với code và artifact
hiện có, gồm HDFS một node, hai nhánh RUL/Health Index độc lập và lịch bảo trì test.
Lượt sửa tài liệu này không chạy lại training hoặc thay đổi kết quả khoa học.
Báo cáo, slide và dashboard trình bày theo rubric môn học; thứ tự trình bày đó
không có nghĩa tín hiệu đã khử nhiễu được đưa vào mô hình RUL.

Repository này phục vụ báo cáo cuối kỳ môn Phân tích dữ liệu lớn, **Đề tài 8 —
Phân tích dữ liệu IoT và bảo trì dự đoán**. Bài toán: phân tích dữ liệu cảm biến
thiết bị để phát hiện dấu hiệu bất thường và dự đoán thời điểm cần bảo trì. Dữ liệu
là bộ mô phỏng động cơ turbofan NASA C-MAPSS. FD002–FD004 chỉ đi qua bước ETL;
mô hình chỉ được huấn luyện và đánh giá trên FD001.

## Trạng thái

Kết quả chính trên FD001:

| Bước | Kết quả | Artifact |
|---|---|---|
| Pipeline dữ liệu | Tải raw từ NASA có kiểm tra MD5/SHA-256; kiểm tra schema, khóa, cycle liên tục; Silver/Gold Parquet và SQLite; Pandas và PySpark khớp tuyệt đối; ở scale 100×, Spark local tính nhanh hơn Pandas khoảng 5,5 lần (không gồm ghi file) | `outputs/acquisition/`, `outputs/etl/`, `outputs/benchmarks/` |
| Lưu trữ phân tán | Hadoop 3.4.3 một node trong Docker: raw FD001 trên HDFS; job MapReduce trên YARN và job Spark đọc HDFS đều tái tạo đúng bảng Gold (200/200 động cơ); `fsck` HEALTHY | `outputs/hdfs/`, `infra/hadoop/` |
| Khử nhiễu | EWMA span 10 giảm khoảng 86% dao động ngắn hạn của 14 sensor có xu hướng | `outputs/degradation/` |
| Cửa sổ thời gian | Window 30 cycle, 21 sensor × 6 thống kê = 126 đặc trưng | `outputs/features/` |
| Xu hướng và bất thường | 15/21 sensor thay đổi có ý nghĩa giữa đầu và cuối vòng đời; Health Index phát hiện bất thường ở 100/100 động cơ train, trung vị 121 cycle trước khi hỏng | `outputs/statistics/`, `outputs/degradation/` |
| Dự đoán RUL | HistGradientBoosting, GroupKFold 5 fold theo động cơ; MAE test 15,35 cycle (CI 95%: 12,73--18,09), RMSE 20,54 | `outputs/modeling/` |
| Cảnh báo sớm | Ngưỡng RUL dự đoán ≤ 30: không bỏ lỡ, lead time trung vị 30 cycle, 49% trễ; ngưỡng 40 giảm trễ còn 6% | `outputs/modeling/tables/` |
| Lịch bảo trì | 21 động cơ test được báo cần bảo trì ngay và cả 21 đều đúng; 4 động cơ còn lại thật sự cần bảo trì được xếp đến hạn sau 1--8 cycle | `outputs/maintenance/` |

Sản phẩm bàn giao:

- [Báo cáo PDF](report/main.pdf) (40 trang, [nguồn LaTeX](report/README.md)).
- [Slide trình bày](slides/cmapss_fd001_presentation.pptx) ([PDF](slides/cmapss_fd001_presentation.pdf)),
  20 slide trên template IUH, dựng bằng [`slides/build_presentation.py`](slides/build_presentation.py).
- [Dashboard Streamlit](app.py) tám trang, chỉ đọc artifact.
- Năm notebook đã chạy (00, 01, 02, 03, 06) trong [`notebooks/`](notebooks/), mỗi notebook ghi họ tên và MSHV.

Lệnh cài đặt và chạy từng bước có ở phần Bắt đầu. Dashboard dùng dependency tùy
chọn `dashboard`.

## Mục tiêu

Báo cáo trả lời năm câu hỏi, sắp theo đúng thứ tự pipeline:

1. Dữ liệu có hợp lệ về schema, khóa và cycle liên tục không, và ETL Pandas--PySpark có cho cùng kết quả không?
2. Nhiễu tín hiệu cảm biến được giảm bao nhiêu sau khử nhiễu, và dữ liệu được tổng hợp theo cửa sổ thời gian ra sao?
3. Sensor suy giảm theo xu hướng nào, và có phát hiện được dấu hiệu bất thường trước khi động cơ hỏng không?
4. Dự đoán RUL đạt độ chính xác bao nhiêu trên tập test độc lập?
5. Dự đoán RUL và tín hiệu bất thường có thể chuyển thành cảnh báo sớm và lịch bảo trì cho từng động cơ với chất lượng ra sao?

Sản phẩm đặc thù theo đề tài là **báo cáo dự đoán thời điểm cần bảo trì kèm biểu
đồ xu hướng cảnh báo sớm**. RUL (Remaining Useful Life, tuổi thọ hữu ích còn lại)
được mô hình hóa bằng hồi quy có giám sát; đây là công cụ, không phải mục tiêu cuối.

## Dữ liệu

Báo cáo dùng bốn subset mô phỏng NASA C-MAPSS. FD001 là subset được phân tích và mô hình hóa; FD002–FD004 chỉ đi qua bước ETL.

| Subset | Dòng train | Dòng test | Động cơ train | Động cơ test và nhãn |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 20.631 | 13.096 | 100 | 100 |
| FD002 | 53.759 | 33.991 | 260 | 259 |
| FD003 | 24.720 | 16.596 | 100 | 100 |
| FD004 | 61.249 | 41.214 | 249 | 248 |

Mỗi subset có các file `train_FD00x.txt`, `test_FD00x.txt` và `RUL_FD00x.txt`. Lịch sử train kết thúc khi hỏng; lịch sử test bị cắt, kèm một nhãn RUL riêng tại cycle quan sát cuối của mỗi động cơ. Quy mô được đối chiếu với [artifact ETL theo subset](outputs/cross_dataset/README.md).

Mỗi quan sát gồm:

- Mã động cơ.
- Cycle vận hành.
- Ba thông số vận hành (operational settings).
- Giá trị của 21 cảm biến (sensor measurements).

FD001 đại diện cho một điều kiện vận hành và một dạng suy giảm High-Pressure Compressor. FD002–FD004 mở rộng số điều kiện vận hành và dạng lỗi. Tín hiệu chứa nhiễu cảm biến mô phỏng.

### Nguồn gốc và tính toàn vẹn

- [Bản ghi NASA Open Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- [Bản lưu có DOI và checksum](https://doi.org/10.5281/zenodo.15346912)
- Checksum ZIP đã ghi nhận: `MD5 79a22f36e80606c69d0e9e4da5bb2b7a`
- Ghi chú nguồn trong repository: [`data/raw/DATA_SOURCE.md`](data/raw/DATA_SOURCE.md)

Đây là dữ liệu mô phỏng, không phải telemetry thu trực tiếp ngoài thực địa. Kết
quả phục vụ phân tích học thuật tái lập trên FD001; kết quả chưa chứng minh hiệu
quả khi triển khai thực tế.

## Kiến trúc dữ liệu

Các lớp mô tả dữ liệu được lưu, không phải chuỗi mà mọi bảng đều là đầu vào của
bảng tiếp theo:

1. **Bronze:** file text NASA bất biến. Reader kiểm tra schema, khóa và cycle liên
   tục trước khi xử lý tiếp.
2. **Silver:** union observations train/test, giữ cột `split` và lưu Parquet
   (33.727 dòng FD001). Nhãn RUL train được lưu riêng.
3. **Gold:** `engine_summary` tổng hợp observations thành 200 dòng động cơ. Một
   nhánh độc lập đọc trực tiếp **Silver observations**, không đọc `engine_summary`,
   để tạo 27.927 dòng rolling features, mỗi dòng có 126 đặc trưng.
4. **Nhánh RUL:** features và targets train đi vào chọn mô hình theo động cơ rồi
   huấn luyện lại. Mô hình dự đoán tại cycle cuối của mỗi động cơ test; chỉ đọc
   nhãn test sau khi đã ghi đủ 100 dự đoán.
5. **Nhánh phân tích tín hiệu:** observations train đã kiểm tra đi vào EDA/thống kê
   và nhánh riêng EWMA → Health Index → bất thường. Các kết quả này không đi vào
   mô hình RUL. Khi lập lịch bảo trì, phép biến đổi Health Index đã fit trên train
   được áp dụng cho test; trạng thái bất thường mới ghép với dự đoán RUL đã lưu.

Hình, dashboard chỉ đọc và báo cáo dùng kết quả đã lưu. Bước HDFS riêng minh họa
lưu trữ Bronze/Silver và đối chiếu bảng tổng hợp động cơ; bước này không đưa quá
trình huấn luyện RUL lên cụm nhiều máy.


### Quan hệ dữ liệu và công cụ

Mỗi dòng được định danh bởi `(split, engine_id, cycle)`. Một động cơ có nhiều
cycle; tại mỗi cycle có 3 operational settings, 21 sensors và nhãn RUL của
train. Với động cơ kết thúc ở cycle 200, observation tại cycle 170 có RUL = 30.
Window 30 cycle tại cycle 170 lấy cycle 141–170 của cùng động cơ để tạo 126
features (6 phép tóm tắt cho mỗi 21 sensor).

| Bước | Công cụ | Đầu ra chính |
| --- | --- | --- |
| Đọc và kiểm tra | Python, Pandas | Schema, khóa, cycle continuity, train RUL |
| ETL | Pandas, PySpark, Parquet, SQLite | Silver/Gold và parity |
| Minh họa HDFS | Docker, Hadoop, YARN, MapReduce, Spark | Bronze/Silver trên HDFS và bảng Gold theo động cơ khớp trên một node |
| Features | Pandas, NumPy; PySpark đối chiếu | Silver observations → window 30 cycle, 126 features |
| Phân tích tín hiệu | Pandas, scikit-learn | EWMA, Health Index fit trên train và trạng thái bất thường; tách khỏi features RUL |
| Modeling | scikit-learn | GroupKFold và HistGradientBoostingRegressor |
| Đánh giá | scikit-learn, SciPy | MAE, RMSE, NASA score, bootstrap CI |
| Trực quan hóa | Matplotlib, Altair, Streamlit | Hình, dashboard và kết quả |

FD001 hiện chưa đủ lớn để tự gọi là Big Data. Báo cáo thể hiện khả năng mở
rộng qua ETL Bronze/Silver/Gold, Parquet, PySpark parity và benchmark 1x/10x/100x.

Pandas đã được dùng trong notebook EDA, làm reference cho ETL, thống kê và
chuẩn bị dữ liệu mô hình. PySpark tái hiện ETL và rolling features để đối chiếu.
Benchmark bổ sung 1x/10x đo cả Pandas và Spark, gồm I/O và RSS của process tree;
Spark ghi qua Arrow trên driver. Bộ dữ liệu gốc chỉ có dung
lượng vài MB, vì vậy repository không tuyên bố FD001 tự thân chứng minh khả
năng xử lý dữ liệu quy mô lớn.

### Cách đọc, truy vấn và biến đổi dữ liệu

Tài liệu walkthrough xử lý dữ liệu giải thích
theo đúng artifact hiện có cách đọc hai file observations, `union` train/test,
giữ `split`, tạo RUL train, tạo rolling window, chọn 126 feature columns và
join theo `(split, engine_id, cycle)`. Tài liệu cũng nêu rõ SQLite chỉ chứa
`engine_summary` và `train_targets`; observations được đọc từ Silver Parquet.

[Notebook 00](notebooks/00_data_pipeline_walkthrough.ipynb) chạy pipeline từ raw,
ghi lại Parquet/SQLite, tạo rolling features và dữ liệu X/y trong một thư mục mới
ở `outputs/notebook00/`. Notebook chạy đối chiếu Pandas/PySpark và job HDFS thật,
cần Java, optional dependency `etl` và Docker Desktop đang chạy. Dữ liệu HDFS
của mỗi lần chạy nằm riêng trong `/cmapss/notebook00/`; các container được giữ
chạy trong Compose project `cmapss-notebook00` để kiểm tra kết quả (NameNode:
`http://localhost:9871`, YARN: `http://localhost:8089`). Notebook không huấn luyện mô hình RUL.

FD001–FD004 được xử lý riêng trong các thư mục subset. Không union bốn subset
thành một bảng chung vì khóa hiện tại chưa có `dataset_id`. Quan hệ chính là
engine--observations một-nhiều; join feature với target và prediction với test
label là một-một. Project chưa triển khai quan hệ nhiều-nhiều, Airflow hoặc
Power BI. [Sơ đồ quan hệ có thể chỉnh sửa](assets/diagrams/data-relations.drawio)
và [bản SVG](assets/diagrams/data-relations.svg) thể hiện các boundary này.

Trong ETL, “chuẩn hóa” nghĩa là kiểm tra schema và chuyển kiểu dữ liệu. Scaling
giá trị sensor là bước riêng: Ridge dùng `StandardScaler` trong từng training
fold, còn Health Index dùng `StandardScaler` trước PCA. HGB dùng feature theo
thang gốc; rolling feature pipeline không fit một scaler chung.

## Phương pháp

### Chất lượng dữ liệu

- Kiểm tra schema 26 cột và kiểu dữ liệu số.
- Bảo đảm khóa `(split, engine_id, cycle)` không trùng.
- Kiểm tra cycle liên tục trong từng động cơ.
- Ghi lại số dòng trước và sau mỗi phép biến đổi.
- Giữ phạm vi ID động cơ riêng biệt giữa train và test.

### Phân tích thống kê

- So sánh hành vi cảm biến ở giai đoạn đầu và cuối vòng đời theo từng động cơ.
- Báo cáo chênh lệch cặp trung bình theo đơn vị gốc và khoảng tin cậy bootstrap 95%.
- Dùng paired Wilcoxon signed-rank test khi phù hợp.
- Kiểm soát multiple testing (kiểm định nhiều giả thuyết) trên các cảm biến bằng Benjamini-Hochberg FDR.
- Ước lượng độ bất định của các metric đánh giá bằng bootstrap theo động cơ.

### Xây dựng đặc trưng

Biểu diễn đặc trưng đã triển khai sử dụng window 30 cycle. Artifact giữ đủ 21 sensors để chưa thực hiện feature selection trên toàn bộ train. Với mỗi sensor, các đặc trưng đã tạo gồm:

- Giá trị gần nhất.
- Trung bình và sample standard deviation (`ddof=1`).
- Giá trị nhỏ nhất và lớn nhất.
- Độ dốc xu hướng tuyến tính theo vị trí 1--30 trong window.

Window dùng giá trị sensor gốc trong Silver, không dùng tín hiệu EWMA hoặc Health
Index. Mỗi window chỉ chứa dữ liệu của một động cơ và không sử dụng quan sát tương
lai. Động cơ dưới 30 cycle bị từ chối thay vì padding, impute hoặc rút ngắn window.
29 cycle đầu mỗi động cơ chưa đủ window, nên còn 17.731 dòng features train và
10.196 dòng test; chỉ 100 dòng test ở cycle cuối được dùng để dự đoán cuối.

### Dự đoán RUL

Pipeline modeling so sánh:

1. Naive baseline theo trung vị RUL.
2. Ridge regression làm baseline tuyến tính có regularization.
3. HistGradientBoostingRegressor (HGB) với 126 đặc trưng window.

Mô hình được chọn bằng GroupKFold 5-fold, trong đó `engine_id` xác định group.
Nhãn test chính thức của NASA được cô lập cho đến khi đã ghi final predictions.
HGB có MAE cross-validation thấp nhất nên là mô hình chính. Metric test gồm
mean engine-level MAE 15,3490, RMSE 20,5413 và NASA score 1.513,2577; khoảng tin
cậy bootstrap được lưu trong `outputs/modeling/tables/final_metrics.json`.

### Khử nhiễu, bất thường và lịch bảo trì

- **Khử nhiễu:** EWMA span 10 làm mượt 14 sensor có xu hướng trong từng động cơ,
  $s_t = \alpha x_t + (1-\alpha) s_{t-1}$ với $\alpha = 2/(\text{span}+1) \approx 0{,}18$;
  chỉ dùng cycle hiện tại và quá khứ; bảy sensor còn lại giữ nguyên.
- **Health Index:** StandardScaler và PCA một thành phần nhận đủ 21 cột sensor sau
  bước làm mượt trên, fit chỉ trên train và không dùng nhãn RUL. Khi lập lịch bảo
  trì, test dùng lại scaler, PCA, chiều dấu và hệ số chuẩn hóa đã fit; không fit
  lại tham số trên test.
- **Bất thường (cảnh báo mức 1):** so mỗi động cơ với 30 cycle đầu của chính nó;
  báo khi `z <= -3` giữ đủ 5 cycle liên tiếp sau đoạn tham chiếu đó. Động cơ có
  dưới 35 cycle quan sát được ghi là chưa đủ dữ liệu đánh giá bất thường.
- **Cảnh báo mức 2:** RUL dự đoán ≤ 30 cycle.
- **Lịch bảo trì:** cycle cần bảo trì = cycle hiện tại + max(⌊RUL dự đoán − 30⌋, 0). Trạng thái
  ưu tiên: cần bảo trì (mức 2), theo dõi tăng cường (mức 1), bình thường. Nhãn test
  chỉ được đọc sau khi lịch đã ghi ra file.

EWMA và Health Index thuộc nhánh phân tích tín hiệu riêng; HGB vẫn nhận 126
rolling features từ sensor gốc. Bất thường của Health Index không phải dự đoán
số cycle còn lại.

Ví dụ: động cơ test 90 ở cycle 146 có RUL dự đoán 31,2, nên cần bảo trì ở cycle 147
và đang ở mức theo dõi tăng cường vì Health Index đã báo bất thường từ cycle 67.

Ma trận đại lượng cần ước lượng, đơn vị phân tích, cách chia dữ liệu/khoảng dự báo, metric và vai trò báo cáo
được lưu trong
[`outputs/modeling/tables/evaluation_protocol.csv`](outputs/modeling/tables/evaluation_protocol.csv).

## Phạm vi Big Data

Thành phần Big Data tập trung vào vòng đời dữ liệu và khả năng tái lập, không phóng đại dung lượng của bộ dữ liệu:

- Thu thập bằng script từ cổng dữ liệu NASA, có kiểm tra checksum.
- Hai phiên bản ETL tương đương bằng Pandas và PySpark.
- Lưu trữ trên HDFS, xử lý bằng job MapReduce (Hadoop Streaming) và job Spark trên cụm Hadoop một node trong Docker.
- Parquet làm định dạng lưu trữ phân tích.
- SQLite làm lớp truy vấn và báo cáo cục bộ, có tính di động.
- Kiểm tra tương đương về số dòng, khóa và giá trị tổng hợp giữa hai công cụ xử lý.
- Thử nghiệm khả năng mở rộng ở các mức dữ liệu 1×, 10× và 100×.

Dữ liệu nhân bản cho thử nghiệm khả năng mở rộng sử dụng ID động cơ mới và được tách hoàn toàn khỏi phân tích khoa học cũng như quá trình huấn luyện mô hình.

Các dịch vụ Hadoop chạy trong Docker trên một máy. Artifact HDFS chứng minh bước
lưu trữ, chạy job và đối chiếu chính xác bảng tổng hợp động cơ, không chứng minh
hiệu năng trên cụm nhiều máy. Benchmark Pandas–Spark 1×/10× là phép đo riêng, dùng
Spark local và xuất Arrow qua driver; thời gian đó không đo writer trên HDFS.

## Quy trình đánh giá

- Dùng window 30 cycle.
- Chia các fold validation theo động cơ, không chia ngẫu nhiên các dòng hoặc window chồng lấn.
- Chỉ fit scaler của Ridge và các mô hình RUL ứng viên trên training fold.
- Giữ đủ 126 features RUL; mô hình chính không có feature selection hoặc PCA.
- Chọn mô hình theo trung bình engine-level MAE.
- Huấn luyện lại mô hình được chọn trên toàn bộ tập train.
- Tạo đúng một dự đoán cuối cho mỗi động cơ trong số 100 động cơ test.
- Chỉ ghép (`join`) predictions với `RUL_FD001.txt` trong bước đánh giá cuối.
- Báo cáo các metric tổng hợp, khoảng tin cậy và sai số phân nhóm theo khoảng RUL.

Báo cáo không đặt trước một mức điểm mục tiêu. Kết quả âm vẫn hợp lệ nếu quy trình đánh giá được giữ nguyên và các nhóm lỗi được phân tích trung thực.

## Trực quan hóa

- Phân phối vòng đời động cơ và quỹ đạo sensor theo cycle.
- Tín hiệu trước và sau khử nhiễu: [`denoise_before_after.png`](outputs/degradation/figures/denoise_before_after.png).
- Health Index, ngưỡng 3σ và lúc phát hiện bất thường: [`anomaly_health_index.png`](outputs/degradation/figures/anomaly_health_index.png).
- RUL thật--dự đoán và phần dư trên test trong [artifact modeling](outputs/modeling/README.md).
- Quỹ đạo cảnh báo RUL ≤ 30 của từng động cơ train.
- Trạng thái đội động cơ test và kế hoạch bảo trì: [`fleet_status.png`](outputs/maintenance/figures/fleet_status.png),
  [`maintenance_planning.png`](outputs/maintenance/figures/maintenance_planning.png).

Ngưỡng 30 cycle chưa phải ngưỡng bảo trì tối ưu vì FD001 không có chi phí bảo
trì, downtime hoặc intervention records.

## Cấu trúc repository

Cấu trúc repository hiện tại:

```text
.
├── data/
│   └── raw/                 # FD001–FD004 nguyên bản và ghi chú nguồn
├── references/              # Tài liệu tham khảo và bối cảnh assignment
├── notebooks/               # Đã chạy: 00 walkthrough dữ liệu, 01 EDA, 02 thống kê, 03 mô hình RUL,
│                            # 06 khử nhiễu, bất thường và lịch bảo trì
├── src/cmapss/              # data/, statistics/, etl/, features/, modeling/, timeseries/ (Health Index),
│                            # degradation/ (EWMA, bất thường), maintenance/ (lịch bảo trì), hdfs/ (HDFS, MapReduce,
│                            # Spark trên HDFS), data/acquire.py (tải và kiểm tra checksum), eda.py, validate.py,
│                            # dashboard.py (đọc artifact), dashboard_app.py (giao diện Streamlit)
├── infra/hadoop/            # Docker Compose cho cụm Hadoop một node và Spark client
├── scripts/                 # Đóng gói, hình báo cáo, chẩn đoán sai số, bằng chứng đối chiếu môn học, xuất slide
├── tests/                   # Kiểm thử data contract, CLI, EDA, thống kê, ETL, HDFS, features, modeling, Health Index,
│                            # khử nhiễu, lịch bảo trì, dashboard và tái lập
├── outputs/validation/      # Báo cáo kiểm tra dữ liệu hiện có
├── outputs/eda/             # Bảng, hình, phương pháp và kiểm tra EDA
├── outputs/statistics/      # Artifact suy luận sensor chỉ dùng train và validation
├── outputs/acquisition/     # Manifest tải dữ liệu: nguồn, MD5 gói, SHA-256 từng file
├── outputs/etl/             # Parquet/SQLite Silver/Gold, metadata và benchmark
├── outputs/hdfs/            # Danh sách file HDFS, fsck, kết quả MapReduce và Spark, ảnh giao diện web
├── outputs/features/        # Parquet Gold rolling features và metadata
├── outputs/modeling/        # Grouped CV, prediction, model artifact và final metrics
├── outputs/degradation/     # Tín hiệu EWMA, Health Index, cycle phát hiện bất thường
├── outputs/maintenance/     # Lịch bảo trì 100 động cơ test, kế hoạch và bảng kiểm tra
├── outputs/cross_dataset/   # Biên bản ETL riêng từng subset FD001–FD004 (FD002–FD004 chỉ đi qua ETL)
├── assets/diagrams/         # Nguồn draw.io, bản xem trước SVG và file URL
├── pyproject.toml           # Thông tin package và dependency trực tiếp
├── requirements.txt         # Cài bằng pip: package và thư viện chạy chương trình
├── requirements-dev.txt     # Thư viện notebook và kiểm thử
├── requirements-etl.txt     # Dependency PySpark tùy chọn cho ETL
├── requirements-dashboard.txt # Dependency Streamlit tùy chọn cho dashboard
├── uv.lock                  # Phiên bản dependencies đã khóa
├── report/                  # Nguồn LaTeX/PDF và hình của báo cáo
├── slides/                  # Deck 20 slide trên template IUH, bản PDF, script trích số liệu và builder
├── app.py                   # Entry point Streamlit (gọi cmapss.dashboard_app)
├── README.md                # Tài liệu tiếng Anh
└── README.vi.md             # Tài liệu tiếng Việt
```

Các file hỗ trợ agent trong môi trường cục bộ được loại khỏi GitHub thông qua `.gitignore`.

## Khả năng tái lập

Repository cung cấp một quy trình được ghi chép rõ từ dữ liệu nguồn bất biến đến dữ liệu đã xử lý, metric, hình minh họa, dữ liệu dashboard và báo cáo cuối kỳ. Các yêu cầu tái lập gồm:

- Dùng seed `42` cho model ngẫu nhiên và bootstrap; GroupKFold không shuffle.
- Cố định phiên bản các dependency Python.
- Ghi lại số dòng và kết quả validation.
- Lưu cấu hình mô hình và kết quả đánh giá.
- Tách dữ liệu khoa học khỏi dữ liệu tổng hợp dùng kiểm tra khả năng mở rộng.
- Kiểm tra tự động schema, data leakage, ranh giới window và số lượng dự đoán.

## Bắt đầu

Cần dùng Python 3.12. Từ thư mục gốc, tạo môi trường và cài dependency một lần:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Sau đó chạy kiểm tra dữ liệu:

```powershell
.venv\Scripts\python -m cmapss.validate
```

Lệnh đọc FD001, kiểm tra dữ liệu, tạo và kiểm tra train RUL trong bộ nhớ,
rồi ghi [báo cáo validation](outputs/validation/fd001.json).
Test labels không được đọc trong luồng này.

[`requirements.txt`](requirements.txt) chứa các dependency cần để chạy chương
trình và cài gói hiện tại qua `-e .`. Nếu cần notebook, phát triển hoặc chạy test,
thay bằng [`requirements-dev.txt`](requirements-dev.txt) ở bước cài đặt.

Với uv, `uv sync --frozen --no-dev` cài môi trường chạy chương trình, không kèm
dependency cho notebook hoặc test. Môi trường này có `psutil` để đo bộ nhớ khi
benchmark; Spark cần thêm `--extra etl` và Java. Ví dụ,
`python -m cmapss.etl.scalability --help` vẫn chạy được trong môi trường này.

Khi đổi dependency, cập nhật `uv.lock` rồi chạy lại lệnh export ở đầu cả bốn file
`requirements*.txt`. Chạy `python -m pytest tests/test_dependency_exports.py` để
đối chiếu các file với lock, kể cả những phiên bản chỉ được cài trên Linux.

Đặc tả dữ liệu và luồng dữ liệu ghép cặp chỉ dùng train nằm trong `src/cmapss/`:

```text
contracts.py     Tên cột, kiểu dữ liệu và ngoại lệ
readers.py       Đọc file thành DataFrame
validation.py    Kiểm tra schema, khóa và cycle
targets.py       Tạo và kiểm tra train RUL
statistics/
  config.py       Cấu hình suy luận đã chốt
  pairs.py        Tạo và kiểm tra cặp động cơ/sensor
  inference.py    Bootstrap, Wilcoxon, FDR và tóm tắt theo sensor
  validation.py   Kiểm tra bảng suy luận sensor
  artifact_io.py  CSV/JSON, hash và reader round-trip
  metadata.py     Metadata lần chạy và validation payload
  artifacts.py    Ghi CSV/JSON và kiểm tra round-trip
  reproducibility.py
                  Validation chạy lặp, runtime audit và so sánh artifact
statistics_artifacts.py
                  Compatibility import cho bộ ghi artifact
```

Ví dụ: động cơ train kết thúc tại cycle 200 thì `rul = 200 - cycle`.
Hàm tạo nhãn từ chối quan sát test. Xem [data contract](data/CONTRACTS.md)
để biết các điều kiện và giới hạn kiểm tra.

### Pipeline dữ liệu tổng thể

Sơ đồ dưới đây tách nhánh RUL khỏi phân tích tín hiệu. Rolling features đọc trực
tiếp Silver observations; `engine_summary` là bảng tổng hợp riêng, không phải đầu
vào của features. EDA/thống kê chỉ dùng train. EWMA và Health Index fit trên train
phục vụ phân tích bất thường, rồi transform test để lập lịch; chúng không đi vào
HGB. Dự đoán RUL và trạng thái bất thường chỉ được ghép ở báo cáo bảo trì. Nhãn
test được đọc sau khi ghi predictions hoặc lịch. Trên hình, bước 1--3 chuẩn bị
dữ liệu; nhánh A dự đoán RUL, nhánh B theo dõi tín hiệu. Ô nét đứt (chạy lại trên
Hadoop) chỉ kiểm tra lại bảng tổng hợp động cơ và không đi vào mô hình RUL. [Sơ đồ trong báo cáo](report/figures/pipeline.pdf)
thể hiện cùng các ranh giới này.

![Pipeline dữ liệu tổng thể FD001](assets/diagrams/fd001-pipeline.png)

[Nguồn Draw.io có thể chỉnh sửa](assets/diagrams/fd001-pipeline.drawio) ·
[Hình trong báo cáo (PDF)](report/figures/pipeline.pdf)

### Chạy notebook EDA

Sau khi cài `requirements-dev.txt`, mở notebook từ thư mục gốc:

```powershell
.venv\Scripts\python -m jupyterlab notebooks/01_fd001_eda.ipynb
```

Chọn kernel Python của `.venv` và **Restart Kernel and Run All Cells**.
Notebook chỉ phân tích train, tạo 12 bảng CSV và 14 hình PNG trong thư mục mới
`outputs/notebook01/run-*/`. Các kiểm tra tái lập EDA trước đây áp dụng cho bộ
artifact gốc ở `outputs/eda/`, được giữ nguyên để đối chiếu.
Bộ kiểm thử ngày 21/10/2026 có **272 kiểm thử đạt**.

![Quy trình EDA train FD001](assets/diagrams/fd001-eda.svg)

[Nguồn sơ đồ](assets/diagrams/fd001-eda.drawio) ·
[Phương pháp, kết quả và giới hạn](outputs/eda/README.md)

### Chạy notebook thống kê

Sau khi cài `requirements-dev.txt`, mở notebook suy luận chỉ dùng train:

```powershell
.venv\Scripts\python -m jupyterlab notebooks/02_fd001_inference.ipynb
```

Chạy toàn bộ cell tạo thư mục mới `outputs/notebook02/run-*/`, chứa hai bảng
`tables/engine_sensor_pairs.csv`, `tables/sensor_inference.csv`, `run_info.json`,
`validation.json` và hai hình trong `figures/`. Ví dụ, kết quả có 2.100 cặp
động cơ--sensor và 21 dòng suy luận theo
sensor. Sáu sensor hằng toàn tập được giữ với trạng thái
`all_zero_difference`; 15 sensor kiểm định được đều bác bỏ giả thuyết không sau
Benjamini-Hochberg FDR theo protocol đã chốt. Đây là kết quả chỉ dùng train,
không phải lựa chọn đặc trưng (feature selection).

![Quy trình thống kê train FD001](assets/diagrams/fd001-statistics.svg)

[Nguồn draw.io có thể chỉnh sửa](assets/diagrams/fd001-statistics.drawio) ·
[Phương pháp, bảng, validation và giới hạn](outputs/statistics/README.md)

Ví dụ: trung vị tuổi thọ của 100 động cơ train là 199 cycle. Sáu sensor hằng
trên toàn train; `sensor_6` biến thiên trên toàn tập nhưng hằng trong 38 động cơ.
Hai hình suy luận dùng panel riêng cho từng sensor vì các sensor khác thang đo.
`sensor_6` cho thấy p-value nhỏ không đồng nghĩa với thay đổi lớn theo đơn vị
đo; `sensor_9` cho thấy cần đọc phân phối theo động cơ cùng mean và CI. Đây là
kết quả train-only, chưa phải lựa chọn đặc trưng cho mô hình. Kiểm tra chạy lặp
notebook hai lần từ fresh kernel cho hai CSV và hai PNG giống nhau; runtime
audit đã chặn thử nghiệm mở `test_FD001.txt` và `RUL_FD001.txt`.

Lệnh kiểm tra tái lập bên dưới ghi vào `outputs/statistics/` để so sánh hai lần
chạy. Chạy notebook thông thường vẫn tạo thư mục riêng trong `outputs/notebook02/`.

Có thể chạy lại kiểm tra thống kê bằng:

```powershell
.venv\Scripts\python -m cmapss.statistics.reproducibility `
  --project-root . `
  --notebook notebooks/02_fd001_inference.ipynb `
  --output-dir outputs/statistics `
  --work-dir tmp/statistics-reproducibility `
  --timeout 1800
```

Đây là audit cục bộ được nạp qua `PYTHONPATH`, không phải filesystem sandbox,
CI enforcement hoặc branch protection.

### Thu thập dữ liệu raw

`cmapss.data.acquire` tải `CMAPSSData.zip` từ cổng dữ liệu mở của NASA, kiểm tra MD5 của gói và
SHA-256 của từng file, rồi ghi `outputs/acquisition/manifest.json`. File đã có trong `data/raw` chỉ
được giữ khi trùng từng byte; file khác nội dung làm lệnh dừng thay vì bị ghi đè.

```powershell
.venv\Scripts\python -m cmapss.data.acquire
```

### Chạy bước HDFS, MapReduce và Spark

Cần Docker. `infra/hadoop/docker-compose.yml` dựng cụm Hadoop 3.4.3 một node (NameNode, DataNode,
ResourceManager, NodeManager) và một Spark 4.0.1 client. Lệnh dưới đây đưa raw FD001 lên HDFS, chạy
job MapReduce viết bằng Hadoop Streaming trên YARN và job Spark ghi lớp Silver dạng Parquet lên HDFS,
rồi so hai bảng tổng hợp theo động cơ với bảng Gold của Pandas:

```powershell
.venv\Scripts\python -m cmapss.hdfs            # thêm --keep-running để xem http://localhost:9870
```

Bằng chứng được ghi vào `outputs/hdfs/` (tóm tắt, danh sách file trên HDFS, báo cáo `fsck`, log job,
bảng kết quả và ảnh giao diện NameNode, YARN).

### Chạy ETL

Pipeline Pandas tham chiếu đọc và kiểm tra hai file observations FD001, ghi
artifact Silver/Gold và đo benchmark 1x/10x/100x:

```powershell
.venv\Scripts\python -m cmapss.etl `
  --raw-dir data/raw `
  --output-dir outputs/etl `
  --benchmark-scales 1,10,100 `
  --benchmark-repetitions 2
```

Cài extra PySpark và cung cấp Java/JAVA_HOME trước khi yêu cầu đối chiếu kết quả:

```powershell
uv sync --frozen --extra etl
.venv\Scripts\python -m cmapss.etl `
  --with-spark `
  --raw-dir data/raw `
  --output-dir outputs/etl
```

Xem [artifact và validation ETL](outputs/etl/README.md) để biết schema, số
dòng, thời gian benchmark và kết quả đối chiếu Pandas/PySpark.

Pipeline lưu `raw_dir` tuyệt đối trong metadata và bảo vệ thư mục này ngay cả
khi đổi thư mục chạy. ETL CLI ghi Silver/Gold, benchmark và parity được yêu cầu
vào staging trước khi publish; lỗi ghi hoặc publish sẽ khôi phục bộ cũ. Manifest
được publish cuối, liên kết benchmark/parity với checksum input và thời điểm ETL.
`auxiliary_checks.parity.status=not_requested` nghĩa là lần hiện tại không chạy
Spark, dù thư mục còn `parity.json` lịch sử. Writer giữ các file ngoài phạm vi
của nó, ví dụ README. Publication nhiều file không bảo đảm reader/writer đồng
thời hoặc mất điện; đóng ứng dụng đang mở SQLite trước khi ghi lại trên Windows.
Nếu rollback cũng lỗi, thông báo trả đường dẫn backup để phục hồi.

### Tạo rolling features

Sau khi ETL đã ghi Silver observations, tạo Gold rolling features:

```powershell
.venv\Scripts\python -m cmapss.features `
  --silver-observations outputs/etl/silver/observations.parquet `
  --output-dir outputs/features
```

Artifact hiện tại có 27.927 dòng: 17.731 train windows và 10.196 test windows.
Xem [artifact và validation rolling features](outputs/features/README.md) để
biết contract cột, số dòng, kiểm tra leakage và giới hạn.

Rolling pipeline kế thừa Bronze boundary từ metadata ETL. Với Silver cũ chưa
lưu `raw_dir`, truyền thêm `--raw-dir` thực tế; ví dụ `--raw-dir D:/datasets/CMAPSS`.
Đường dẫn này bổ sung bảo vệ, không bỏ qua boundary đã lưu. Rolling writer cũng
dùng staging/rollback và từ chối ghi đè Silver input hoặc manifest của nguồn.

### Chạy modeling RUL

Sau khi đã có ETL và rolling features, chạy huấn luyện theo group và đánh giá
cuối:

```powershell
.venv\Scripts\python -m cmapss.modeling `
  --rolling-features outputs/features/gold/rolling_features.parquet `
  --train-targets outputs/etl/silver/train_targets.parquet `
  --test-labels data/raw/RUL_FD001.txt `
  --output-dir outputs/modeling
```

Lần chạy hiện tại chọn `hist_gradient_boosting`, đạt mean engine-level MAE
15,3490, RMSE 20,5413 và NASA score 1.513,2577 trên 100 động cơ test.
Xem [artifact và validation modeling](outputs/modeling/README.md) để biết fold
metrics, final predictions, final evaluation, hình và giới hạn.

### Chạy ETL cho FD001–FD004

FD002–FD004 chỉ đi qua bước ETL; mô hình chỉ được huấn luyện và đánh giá trên
FD001. Để tạo lại biên bản ETL theo subset (tổng 265.256 bản ghi của bốn subset),
chạy ETL CLI cho từng subset, ví dụ FD002:

```powershell
.venv\Scripts\python -m cmapss.etl `
  --dataset-id FD002 `
  --output-dir outputs/cross_dataset/FD002/etl
```

Xem [artifact ETL theo subset](outputs/cross_dataset/README.md).

### Chạy khử nhiễu và phát hiện bất thường

Chỉ đọc `data/raw/train_FD001.txt`, không dùng nhãn RUL:

```powershell
.venv\Scripts\python -m cmapss.degradation
```

Lệnh ghi `outputs/degradation/`: bảng mức giảm dao động, cycle phát hiện bất
thường của 100 động cơ train, tín hiệu gốc/EWMA và Health Index theo cycle, hai hình.

### Chạy lịch bảo trì

Sau khi đã có `outputs/modeling/predictions/final_predictions.parquet`:

```powershell
.venv\Scripts\python -m cmapss.maintenance
```

Lệnh fit Health Index trên train, chỉ transform test, lập lịch bảo trì và ghi ra
file, rồi mới đọc `RUL_FD001.txt` để tạo bảng kiểm tra `status_vs_actual.csv`.

### Chạy notebook phân tích kết quả

Notebook 01, 02, 03 và 06 giải thích từng bước qua một ví dụ, rồi mở rộng sang toàn
bộ dữ liệu. Mỗi lần chạy ghi kết quả riêng trong `outputs/notebookNN/run-*/`.
Notebook 03 chạy lại cross-validation và đánh giá RUL. Notebook 06 tính lại EWMA,
Health Index, lưới độ nhạy và lịch, dùng RUL dự đoán từ baseline đã lưu. Các
notebook dùng artifact đầu vào có sẵn trong repo và có thể chạy bằng kernel riêng.

| Notebook | Nội dung | Dữ liệu đọc |
|---|---|---|
| `03_fd001_modeling_evaluation.ipynb` | Chạy 5-fold CV, chọn mô hình, đánh giá test và cảnh báo RUL ≤ 30 | `outputs/features/gold/rolling_features.parquet`, train targets; nhãn test đọc sau khi ghi dự đoán |
| `06_fd001_degradation_maintenance.ipynb` | EWMA, bất thường, lưới độ nhạy, lịch bảo trì và đối chiếu nhãn test | Raw observations, `outputs/modeling/predictions/final_predictions.parquet`; nhãn test đọc sau khi lập lịch |

Mở notebook trong JupyterLab rồi chạy tất cả các ô, hoặc chạy từ dòng lệnh:

```powershell
.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace notebooks/06_fd001_degradation_maintenance.ipynb
```

### Chạy dashboard tái lập

Cài dependency tùy chọn và khởi động Streamlit sau khi đã có artifact modeling,
`cmapss.degradation` và `cmapss.maintenance`:

```powershell
uv sync --extra dashboard
uv run --extra dashboard streamlit run app.py
```

Dashboard có tám trang, sắp theo thứ tự pipeline. Có thể mở thẳng một trang bằng
query string, ví dụ `http://localhost:8501/?page=Lịch bảo trì`.

| Trang | Artifact đọc | Nội dung |
|---|---|---|
| Tổng quan | `outputs/modeling`, `outputs/degradation`, `outputs/maintenance` | Bốn con số chính, pipeline 7 bước, gợi ý thứ tự demo |
| Dữ liệu & EDA | `outputs/eda`, `outputs/features` | Phân phối tuổi thọ, Spearman sensor–RUL, quỹ đạo sensor |
| Khử nhiễu | `outputs/degradation` | Tín hiệu gốc và sau EWMA theo động cơ/sensor, mức giảm dao động |
| Xu hướng & bất thường | `outputs/degradation`, `outputs/statistics` | Health Index, ngưỡng 3σ, lúc phát hiện, phân phối thời gian báo trước |
| Dự đoán RUL | `outputs/modeling` | So sánh model trong CV, một động cơ test, quỹ đạo OOF |
| Cảnh báo sớm | `outputs/modeling/tables` | Ngưỡng 30 cycle, phân loại thời điểm, quỹ đạo cảnh báo |
| Lịch bảo trì | `outputs/maintenance` | Bảng 100 động cơ, kế hoạch, chi tiết từng động cơ, kiểm tra bằng nhãn test |
| Provenance | `run_info.json` | Schema và isolation contract của lần chạy |

Lớp đọc artifact nằm trong `src/cmapss/dashboard.py`; giao diện nằm trong
`src/cmapss/dashboard_app.py`. `app.py` chỉ là entry point và vẫn export
`load_dashboard_data`, vì `cmapss.reproducibility` import hàm này; do đó import
`app` không cần Streamlit. Dashboard không huấn luyện lại model và không đọc
`data/raw/RUL_FD001.txt`. Trang Dữ liệu & EDA dùng cột `sensor_*__last` và
`sensor_*__mean` của Gold rolling features, nên quỹ đạo bắt đầu từ cycle 30.

### Kiểm tra pipeline end-to-end

Tái tạo pipeline khoa học từ raw FD001 bất biến vào một thư mục tạm mới, rồi so
sánh nội dung bảng, metric, hình, checksum raw và dashboard loader với artifact
đã có:

```powershell
uv run python -m cmapss.reproducibility `
  --output-root tmp/e2e-repro-final `
  --repo-root . `
  --evidence-path tmp/e2e-repro-final/validation.json
```

Lệnh chạy ETL Pandas với benchmark 1x, rolling features và modeling RUL, rồi
kiểm tra dashboard loader. Lệnh yêu cầu output root mới, xác nhận raw hash không đổi
và so sánh bảng với `rtol=1e-9`, `atol=1e-9`. Parity PySpark và tái lập notebook
là các kiểm tra riêng, đã được ghi trong evidence.

<details>
<summary>Dành cho phát triển và kiểm tra riêng</summary>

- Bản vá bảo mật: pytest 9.0.3, JupyterLab 4.5.10 và nbconvert 7.17.1.
  Cài lại `requirements-dev.txt` nếu đã có môi trường. Xem
  [nội dung bản vá và bằng chứng kiểm thử hồi quy](outputs/validation/dependencies.md).
- Sau khi cài `requirements-dev.txt`, chạy bộ kiểm thử: `.venv\Scripts\python -m pytest`.
- Kiểm tra thêm tính toàn vẹn test labels: thêm `--check-test-labels` vào lệnh
  chính. Tùy chọn này không ghép nhãn hoặc tính metric mô hình.
- `--raw-dir` và `--output` chỉ cần khi muốn thay đường dẫn mặc định.
- `requirements*.txt` được xuất từ `uv.lock` và giữ nguyên phiên bản đã khóa,
  kể cả dependency bắc cầu. `pyproject.toml` vẫn khai báo metadata của gói.
  Môi trường đã kiểm chứng là Python 3.12.14 trên Windows.
- Nếu đã dùng uv, có thể chạy `uv run cmapss-validate`; không bắt buộc cài uv
  khi dùng cách cài bằng pip ở trên.

Nếu pytest không có quyền truy cập Temp trên Windows, dùng thư mục riêng:

```powershell
New-Item -ItemType Directory -Force tmp | Out-Null
.venv\Scripts\python -m pytest --basetemp=tmp/pytest-data-contract
```

Pytest dọn thư mục `--basetemp` khi chạy lại; chỉ dành thư mục đó cho test.

</details>

[Báo cáo PDF](report/main.pdf) trình bày các phân tích FD001 đã kiểm chứng. Xem [hướng dẫn báo cáo](report/README.md) để biết
cấu trúc nguồn và lệnh build PDF.

## Bối cảnh học thuật

Repository hỗ trợ báo cáo cuối kỳ môn **Phân tích dữ liệu lớn**. Báo cáo phân
biệt kết quả quan sát được, diễn giải và giới hạn; mọi số liệu đều được đối chiếu
với dữ liệu và quy trình tái lập tương ứng.

## Quy trình thực hiện

Tác giả xác định câu hỏi nghiên cứu, chọn bộ dữ liệu, phương pháp và đọc lại kết
quả. Trợ lý lập trình AI (Claude và Codex) được dùng để soạn mã, test, tài liệu,
slide và chạy các bước kiểm tra nhất quán. Mọi số liệu trong báo cáo và slide lấy
từ file trong `outputs/` do mã của repository sinh ra, và tác giả đã đối chiếu
với các file đó. Các commit đầu có trailer đồng tác giả AI; trailer này được bỏ
khi dọn lịch sử để công khai, và mục này là bản ghi về việc sử dụng AI.

## Tài liệu tham khảo

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). [Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation](https://doi.org/10.1109/PHM.2008.4711414).
- Ramasso, E., & Saxena, A. (2014). [Review and Analysis of Algorithmic Approaches Developed for Prognostics on CMAPSS Dataset](https://doi.org/10.36001/phmconf.2014.v6i1.2512).
- Li, X., Ding, Q., & Sun, J.-Q. (2018). [Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks](https://doi.org/10.1016/j.ress.2017.11.021).
- Xem [`references/reading-guide.md`](references/reading-guide.md) để biết hướng dẫn tài liệu tham khảo.

## Giới hạn

- FD001 là dữ liệu mô phỏng với một điều kiện vận hành và một dạng lỗi.
- Cycle vận hành không tương đương với giờ hoặc thời gian lịch.
- Lịch bảo trì dùng ngưỡng 30 cycle cố định, chưa hiệu chuẩn theo chi phí; mức theo dõi tăng cường còn rộng (41/59 động cơ có RUL thật trên 60).
- Phát hiện bất thường chưa đo được tỷ lệ báo động giả vì không có nhãn thời điểm bắt đầu suy giảm.
- Thử nghiệm khả năng mở rộng không tạo thêm bằng chứng khoa học độc lập.
- Hiệu quả trên FD001 không chứng minh khả năng khái quát hóa ngoài thực tế.

## Giấy phép và sử dụng dữ liệu

Mã nguồn được phát hành theo [giấy phép MIT](LICENSE). Dữ liệu NASA C-MAPSS trong `data/raw/` là tài liệu của NASA; bản ghi NASA Open Data không nêu giấy phép của bộ dữ liệu, nên dữ liệu được phân phối lại nguyên vẹn kèm ghi nhận nguồn (xem [`data/raw/DATA_SOURCE.md`](data/raw/DATA_SOURCE.md)) và không thuộc phạm vi giấy phép MIT. Các bài báo của bên thứ ba không được đưa vào repository; tải theo liên kết trong [`references/papers/README.md`](references/papers/README.md).
