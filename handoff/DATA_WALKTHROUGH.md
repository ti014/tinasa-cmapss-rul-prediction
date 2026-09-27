# Tổ chức, truy xuất và biến đổi dữ liệu C-MAPSS

Pipeline hiện có đọc từng subset, nối các dòng train/test có cùng schema, lưu
observations trong Parquet, tạo nhãn train và tổng hợp theo engine. Từ lịch sử
của từng engine, pipeline tạo 126 đặc trưng trên window 30 cycles để dự đoán RUL.
Tài liệu này giải thích hành vi đã triển khai, đối chiếu với code và artifact
ngày 28/09/2026. Không có kết quả training hoặc benchmark mới trong lượt này.

## Đối chiếu yêu cầu môn học

Brief Đề tài 8 yêu cầu phân tích dữ liệu cảm biến, xử lý nhiều tín hiệu,
chuẩn hóa và tổng hợp theo time window, trực quan hóa suy giảm, dự đoán và cảnh
báo bảo trì. [Brief gốc](../references/context/assignment-brief.png) là nguồn
yêu cầu; transcript nhận dạng tiếng nói có lỗi chỉ bổ sung bối cảnh trao đổi.

| Yêu cầu | Đã thực hiện | Giới hạn cần nói rõ |
|---|---|---|
| Chuỗi cảm biến, nhiều tín hiệu | 21 sensor theo engine/cycle, EDA và thống kê train | Dữ liệu mô phỏng batch, chưa có streaming hoặc thiết bị thật |
| Chuẩn hóa và time window | Schema/type ở ETL; scaler riêng cho Ridge và Health Index; 30-cycle rolling features | HGB dùng feature theo thang gốc; không có một bước scaling chung trước mọi model |
| Xu hướng suy giảm | Quỹ đạo sensor, so sánh đầu/cuối, Health Index | Thay đổi tín hiệu chưa chứng minh quan hệ nhân quả hoặc điểm bắt đầu suy giảm |
| Dự đoán/cảnh báo | RUL test, OOF warning trajectory và timing 44/7/49 | Ngưỡng nghiên cứu 30 cycles, chưa suy ra lịch bảo trì tối ưu |
| Nội dung xử lý dữ liệu lớn | PySpark ETL local, Parquet, nhiều tầng dữ liệu, parity và benchmark Pandas 1×/10×/100× | Chưa đo cluster, Spark runtime hay partition-aware query |

## 1. Nhiều file được tổ chức và gom thế nào?

```text
data/raw/                         Dữ liệu NASA bất biến
├── train_FD001.txt               26 trường trên mỗi dòng, không header
├── test_FD001.txt                Cùng schema observations
├── RUL_FD001.txt                 Một nhãn endpoint cho mỗi test engine
└── ...                          Ba file tương tự cho FD002–FD004
outputs/etl/                     Kết quả chính FD001
├── silver/observations.parquet   33.727 dòng, 27 cột sau khi thêm split
├── silver/train_targets.parquet  20.631 dòng, 4 cột
└── gold/
    ├── engine_summary.parquet    200 dòng, 5 cột
    └── etl.sqlite               engine_summary và train_targets
outputs/features/gold/
└── rolling_features.parquet     27.927 dòng, 129 cột
outputs/cross_dataset/FD00x/      ETL và features riêng cho mỗi subset
```

Reader tách mỗi dòng theo whitespace, yêu cầu đúng 26 trường và gán tên cột:
`engine_id`, `cycle`, ba `setting_*`, 21 `sensor_*`. Sau đó thêm `split`.
`engine_id/cycle` có kiểu int64; settings/sensors có kiểu float64.

Trong **một subset**, Pandas concat hai bảng theo chiều dòng. PySpark thực hiện
`unionByName`. Cả hai giữ `split` để phân biệt train engine 1 với test engine 1.
Đây là union các observations, không phải join lịch sử train/test với nhau.
File RUL test được giữ riêng đến đánh giá cuối; dòng thứ i của file thuộc
test engine i theo quy ước bộ dữ liệu.

FD001–FD004 hiện lưu và xử lý **riêng**, không union thành một bảng toàn cục.
Khóa `(split, engine_id, cycle)` chỉ duy nhất trong một subset. Nếu xây bảng
chung trong tương lai, cần giữ `dataset_id` trong khóa để tránh trùng engine
giữa các subset. Schema hiện tại chưa có cột đó.

Nguồn: [reader](../src/cmapss/data/readers.py),
[ETL Pandas](../src/cmapss/etl/pandas_pipeline.py),
[ETL PySpark](../src/cmapss/etl/spark_pipeline.py),
[cross-dataset](../src/cmapss/cross_dataset.py).

## 2. Lấy dữ liệu và truy vấn ở đâu?

SQLite hiện chỉ có `engine_summary` và `train_targets`, có index trên khóa.
Ví dụ SQL dưới đây chạy trên `outputs/etl/gold/etl.sqlite`:

```sql
SELECT split, engine_id, n_cycles, first_cycle, last_cycle
FROM engine_summary
WHERE split = 'train' AND engine_id = 1;

SELECT cycle, rul
FROM train_targets
WHERE split = 'train' AND engine_id = 1 AND cycle = 100;
```

Kết quả FD001 đã đối chiếu: engine train 1 có 192 cycles, từ 1 đến 192;
target tại cycle 100 là 92. Không có bảng `observations` trong SQLite này.

Để lấy các giá trị sensor, module features đọc
`outputs/etl/silver/observations.parquet` bằng Pandas, kiểm tra contract rồi
lọc `split/engine_id/cycle`. Chọn cột khóa cùng sensor cần xem là thao tác
projection (lấy một số cột). Đọc Parquet rồi lọc trong bộ nhớ khác với việc đã
triển khai query engine, SQL trên Parquet hoặc partition pruning.

Ví dụ: lấy `sensor_11` của train engine 1 ở cycles 71–100 sẽ nhận 30 dòng.
Giá trị trung bình là 47,395 theo thang gốc. Kết quả này đối chiếu được với
`sensor_11__mean` tại cycle 100 trong artifact features.
Dashboard hiện đọc predictions/metrics/Health Index đã sinh, không cung cấp
giao diện truy vấn raw sensor hoặc chạy lại ETL.

## 3. Transform diễn ra qua những bước nào?

| Bước | Input → output | Phép biến đổi và kiểm tra |
|---|---|---|
| Reader | Text → typed observations | Tách trường, gán schema/type, thêm split; lỗi width/type/key/cycle bị từ chối |
| Silver | Hai bảng train/test → observations Parquet | Concat/union, sắp theo khóa; giữ 33.727 dòng FD001 |
| Train target | Train observations → train_targets | Group theo engine lấy cycle cuối, RUL = max_cycle − cycle |
| Engine summary | Observations → 200 engine rows | Group theo split/engine, count/min/max cycle |
| Features | Observations → rolling feature rows | Sort, group theo split/engine, window 30, tính 6 thống kê/sensor |
| Model input | Train features + targets → X/y | Join one-to-one bằng 3 cột khóa, X gồm 126 features, y là rul |
| Final evaluation | Predictions đã ghi + test labels → errors/metrics | Join one-to-one theo split/engine, một prediction mỗi engine |

Transform (biến đổi dữ liệu) ở đây gồm đổi kiểu, gom dòng, tổng hợp, tạo nhãn
và tạo đặc trưng. Nó không phải kiến trúc mô hình Transformer. Việc tạo RUL
train sử dụng endpoint đã biết để tạo **nhãn**, không đưa endpoint đó vào X.
Không dùng cycle cuối test để suy ra RUL thật vì lịch sử test bị cắt.

Không có bước imputation, xóa outlier hoặc lọc nhiễu chuyên dụng trong đường
feature hiện tại. EDA FD001 xác nhận không có missing values/khóa trùng;
validation từ chối đầu vào không đạt contract. Rolling mean tổng hợp tín hiệu
trong window, nhưng không nên gọi toàn pipeline là đã khử nhiễu có kiểm chứng.

Chuẩn hóa schema/type ở ETL khác scaling giá trị: Ridge fit StandardScaler
trên training fold, Health Index fit StandardScaler/PCA trên 90 train engines
không thuộc 10 engine đánh giá. HGB được chọn dùng features theo thang gốc.
Không có scaler chung nằm trước bước rolling features trong pipeline RUL.

## 4. Columns, features và feature selection

Observation có 27 cột: ba cột khóa và 24 cột đo lường. Rolling features giữ
ba khóa cùng **21 × 6 = 126** cột theo mẫu `<sensor>__<stat>`: `last`, `mean`,
`std` với ddof=1, `min`, `max`, `slope`. Ba `setting_*` không đi vào X của
model RUL hiện tại. `split`, `engine_id`, `cycle` dùng quản lý và chia fold,
không phải các feature columns đưa vào estimator.

Tạo 126 feature columns là feature engineering (xây dựng đặc trưng).
Model hiện nhận đủ 126 cột, chưa thực hiện feature selection để giảm số cột.
Danh sách sensor có correlation cao hoặc có p-value nhỏ trong EDA/statistics
không được dùng để chọn sensor cho model.

Ví dụ thực tế FD001: train engine 1 kết thúc ở cycle 192. Tại cycle 100,
window lấy 30 observations từ 71 đến 100. `sensor_11__last = 47,64`,
`sensor_11__mean = 47,395`; train target là `192 − 100 = 92` cycles.
Cycle 101 không tham gia feature tại cycle 100. Join features/targets giữ
17.731 train rows, không nhân lên thành các tổ hợp cycles.

Một engine dài T cycles có T−29 feature rows với window 30.
Tổng FD001: 20.631−100×29 = 17.731 train rows;
13.096−100×29 = 10.196 test rows. Engine quá ngắn khiến pipeline báo lỗi,
không tự loại engine, padding hoặc rút ngắn window.

## 5. Quan hệ dữ liệu và join

![Quan hệ và ranh giới dữ liệu trong một subset](../assets/diagrams/data-relations.svg)

[Nguồn draw.io](../assets/diagrams/data-relations.drawio).

| Hai đối tượng | Quan hệ trong một subset | Cách sử dụng |
|---|---|---|
| Engine summary – observations | 1:N theo split/engine | Một engine có nhiều cycles |
| Train observation – train target | 1:1 theo split/engine/cycle | Một nhãn cho mỗi observation train |
| Observation ở endpoint – feature row | 1:0..1 theo split/engine/cycle | 29 cycles đầu chưa đủ window; từ cycle 30 có một row |
| Train feature – train target tương ứng | 1:1 trên các khóa đủ window | Join có `validate="one_to_one"` trong modeling |
| Final prediction – test label | 1:1 theo split/engine | Nhãn endpoint, đọc sau khi ghi prediction |

Một window phụ thuộc 30 observations; các window kế tiếp chồng lấn 29
observations. Đây là dependency tính toán theo thời gian, không phải một bảng
many-to-many join đã triển khai. 126 feature columns cũng không tạo quan hệ
1:N giữa các bản ghi. Không cần thêm bảng liên kết nhiều-nhiều nếu bài toán
không có thực thể yêu cầu quan hệ đó.

Ví dụ join sai: engine 1 có 163 train feature rows và 192 target rows.
Join chỉ bằng engine ID sẽ ghép mọi feature cycle với mọi target cycle,
tạo 163×192 = 31.296 dòng. Join đầy đủ ba khóa trả đúng 163 dòng.
Index trong SQLite hỗ trợ truy vấn; unique key validation nằm ở code,
không có foreign key constraint được khai báo trong SQLite hiện tại.

## 6. Phần Big Data được thể hiện thế nào?

Phần xử lý dữ liệu phải trình bày được input/output, schema, grain (đơn vị
của một dòng), khóa, union/join, group/window, lưu trữ và truy xuất trước khi
nói metric model. Project minh họa các nội dung này bằng batch ETL nhiều file,
Parquet, ba boundary PySpark tương đương Pandas và phép đo tổng hợp theo scale.

PySpark hiện chạy `local[1]`, parity collect về Pandas; output khoa học được
ghi bởi đường Pandas. Không có distributed cluster/HDFS, Spark rolling
features, Airflow DAG hoặc Power BI. Không gọi giữ Parquet theo thư mục subset
là đã partitionBy engine. Benchmark 100× có 3.372.700 observations **nhân bản**,
chỉ đo hàm Pandas `build_engine_summary` trong bộ nhớ, không đo I/O, Spark hoặc
rolling feature workload. Không dùng số đó để kết luận Spark nhanh hơn.

Cách trình bày phù hợp: báo cáo áp dụng các thao tác tổ chức và xử lý dữ liệu
trong môn Phân tích dữ liệu lớn cho bài toán cảm biến time series; evidence
phân tán hiện giới hạn ở ETL PySpark local và parity. Chưa có tiêu chí chấm đủ
để khẳng định phần này đáp ứng mọi yêu cầu của giảng viên.

## 7. Câu trả lời ngắn khi trình bày

“Trong một subset, chương trình đọc hai file observations cùng schema rồi
union theo dòng và giữ split. Dữ liệu sensor nằm trong Parquet, bảng tổng hợp
và nhãn train nằm trong SQLite. Chương trình group theo engine, lấy 30 cycles
gần nhất, tạo 126 features rồi join đúng cycle với RUL train. Engine train 1
ở cycle 100 dùng cycles 71–100 và có target 92. Quan hệ chính là một engine
có nhiều observations; join features/targets là một–một. PySpark hiện kiểm
tra ETL tương đương Pandas trên local, còn modeling và rolling features dùng
Pandas/scikit-learn.”

## Nguồn kiểm tra

- Schema/khóa: [data contract](../data/CONTRACTS.md).
- Feature/model input: [rolling](../src/cmapss/features/rolling.py),
  [modeling](../src/cmapss/modeling/pipeline.py),
  [model columns](../src/cmapss/modeling/contracts.py).
- Row counts/parity: [ETL](../outputs/etl/run_info.json),
  [parity](../outputs/etl/parity.json), [features](../outputs/features/run_info.json).
- Model/cảnh báo: [modeling artifacts](../outputs/modeling/README.md).
- Health Index: [timeseries artifacts](../outputs/timeseries/README.md).
- Walkthrough demo: [DEMO_SCRIPT.md](DEMO_SCRIPT.md).
