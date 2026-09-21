# EDA FD001 trên tập train

Mốc 1 đã hoàn thành: khảo sát 20.631 observations của 100 engines, xuất 12 bảng
CSV và 11 hình PNG. Tuổi thọ quan sát có median 199 cycles; sáu sensors hằng
trên toàn train. Đây là kết quả mô tả, chưa phải kiểm định giả thuyết hoặc
bằng chứng về hiệu quả dự đoán RUL.

## Input, cách chạy và đầu ra

[Notebook](../../notebooks/01_fd001_eda.ipynb) chỉ đọc
`data/raw/train_FD001.txt`, dùng reader, validator và train targets hiện có.
Không đọc test observations hoặc test labels. RUL bằng `max_cycle - cycle`
trong từng engine; ví dụ engine kết thúc ở cycle 200 có RUL 30 tại cycle 170.

Sau khi tạo `.venv` với Python 3.12 theo README, chạy từ thư mục gốc:

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m jupyterlab notebooks/01_fd001_eda.ipynb
```

Chọn kernel Python của `.venv`, sau đó chọn **Restart Kernel and Run All Cells**.
Notebook tìm thư mục gốc từ vị trí hiện tại, tạo lại bảng/hình tại `outputs/eda/`
và ghi đè các artifact cùng tên. Raw input được giữ nguyên. Notebook không
cần Spark, Java hoặc môi trường mô hình ở mốc này.

![Quy trình EDA FD001](../../assets/diagrams/fd001-eda.svg)

[Nguồn draw.io có thể chỉnh sửa](../../assets/diagrams/fd001-eda.drawio).

```text
notebooks/01_fd001_eda.ipynb   Phân tích, biểu đồ và diễn giải
src/cmapss/data/              Reader, validator và train RUL dùng lại
src/cmapss/eda.py             Thống kê, chọn ví dụ và correlation theo engine
outputs/eda/
├── tables/                  12 bảng CSV
├── figures/                 11 hình PNG
├── run_info.json            SHA-256 input, phiên bản và quy ước tính
└── validation.json          Bằng chứng kiểm tra tại thời điểm bàn giao
```

`run_info.json` được notebook tạo lại; `validation.json` là biên bản của lần
kiểm tra độc lập, không tự cập nhật khi người dùng sửa hoặc chạy lại notebook.

## Phương pháp

1. Kiểm tra data contract, tạo RUL và ghép theo khóa
   `(split, engine_id, cycle)` với quan hệ một–một.
2. Mô tả tuổi thọ với một giá trị mỗi engine. Phân phối RUL gộp observations
   được trình bày riêng vì engine sống lâu đóng góp nhiều dòng hơn.
3. Mô tả 3 settings và 21 sensors: missing, số giá trị khác nhau, min/max,
   mean, median, quartiles, sample std (`ddof=1`). Quartiles nội suy tuyến tính.
   Cột hằng được xác định bằng `n_unique == 1`; không tự đặt ngưỡng near-constant.
4. Chọn engine gần Q25, Q50, Q75 tuổi thọ nhất; nếu bằng khoảng cách, chọn ID
   nhỏ nhất. Vẽ giá trị gốc theo cycle, không smooth. Đây là ví dụ minh họa,
   không phải mẫu đại diện thống kê. Mỗi ô dùng thang đo riêng.
5. Tính Spearman sensor–sensor gộp train và sensor–RUL riêng từng engine.
   Spearman dùng average ranks khi có ties. Sensor/RUL hằng cho `NaN`, không
   thay bằng 0. Tóm tắt các hệ số theo engine với trọng số bằng nhau.
6. Kiểm tra số dòng, khóa, miền correlation, tính đối xứng và checksum raw;
   xuất bảng, hình và metadata.

Histogram dùng 20 bins để trình bày. Các điểm ngoài whisker của boxplot vẫn
được giữ. Tên sensor được giữ theo schema; chưa gán đơn vị vật lý khi nguồn
raw chưa mô tả đầy đủ. Sample std của một cột hằng đôi khi có sai số số thực
cỡ `1e-15`; `n_unique`, không phải std, quyết định trạng thái hằng.

## Kết quả quan sát

| Nội dung | Kết quả trên train | Bằng chứng |
| --- | --- | --- |
| Input | 20.631 observations, 100 engines; không missing hoặc trùng khóa | [Input](tables/input_summary.csv) |
| Tuổi thọ | 128–362 cycles; mean 206,31; median 199; Q25–Q75: 177–229,25 | [Thống kê](tables/lifetime_summary.csv), [hình](figures/engine_lifetimes.png) |
| RUL theo observation | 0–361 cycles; median 103 | [Thống kê](tables/rul_summary.csv), [hình](figures/rul_distribution.png) |
| Cột hằng toàn train | `setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19` | [Mô tả](tables/measurement_summary.csv) |
| `sensor_6` | Hai giá trị 21,60/21,61; hằng trong 38/100 engines | [Theo engine](tables/constant_engine_counts.csv) |
| Engines minh họa | 87: 178 cycles; 26: 199 cycles; 75: 229 cycles | [Quy tắc chọn](tables/example_engines.csv) |

Ba sensor có trị tuyệt đối của median Spearman sensor–RUL lớn nhất trong
khảo sát này là `sensor_11` (−0,819), `sensor_12` (+0,808) và `sensor_4`
(−0,784). Mỗi giá trị tóm tắt 100 hệ số theo engine; xem
[bảng đầy đủ](tables/sensor_rul_summary.csv) và
[boxplot](figures/sensor_rul_by_engine.png). Đây không phải danh sách feature
được chọn cho mô hình.

Trong ba engines minh họa, `sensor_11` nhìn chung tăng và `sensor_12` giảm về
cuối chuỗi. Tuy nhiên, `sensor_9` và `sensor_14` ở engine 26 có dạng khác với
engines 87 và 75; xem [quỹ đạo trang 2](figures/sensor_trajectories_2.png).
Bảng correlation trên 100 engines cũng cho thấy dấu của hệ số hai sensor này
không đồng nhất. Vì vậy, một đường minh họa hoặc một hệ số gộp không mô tả hết
sự khác biệt giữa engines.

## Danh mục bảng và hình

| Bảng CSV | Một dòng biểu diễn |
| --- | --- |
| `input_summary.csv` | Toàn bộ input train |
| `schema.csv` | Một cột sau khi ghép RUL, gồm split và khóa |
| `engine_lifetimes.csv` | Một engine, 100 dòng |
| `lifetime_summary.csv` | Một thống kê tuổi thọ |
| `rul_summary.csv` | Một thống kê RUL theo observation |
| `measurement_summary.csv` | Một setting/sensor, 24 dòng |
| `within_engine_variability.csv` | Một engine và cột đo lường, 2.400 dòng |
| `constant_engine_counts.csv` | Số engines hằng của một cột đo lường |
| `example_engines.csv` | Một phân vị dùng chọn engine minh họa |
| `sensor_spearman_pooled.csv` | Một hàng của ma trận 21 × 21 sensors |
| `sensor_rul_by_engine.csv` | Một engine và sensor, 2.100 dòng |
| `sensor_rul_summary.csv` | Phân phối correlation của một sensor qua engines |

Hình gồm phân phối tuổi thọ, RUL và settings; ba trang phân phối sensors; ba
trang quỹ đạo sensors; một heatmap sensor–sensor và một boxplot sensor–RUL.
Các cặp không xác định được ghi `NaN` trong CSV, ô xám trên heatmap hoặc hàng
trống có số lượng `n=0` trên boxplot.

## Kiểm tra và giới hạn

- Toàn bộ suite đạt **69 tests**, gồm ví dụ tính tay mean/std/quartiles,
  Spearman có ties, tách engines, giữ constants và từ chối dữ liệu không phải train.
- Cài lại `requirements-dev.txt` bằng pip trong môi trường kiểm tra riêng:
  `pip check` đạt. JupyterLab mở notebook qua entry point đã hướng dẫn, trả
  HTTP 200; instance tạm đã được dừng sau kiểm tra.
- Chạy notebook từ kernel mới, sau đó chạy độc lập trong thư mục chỉ chứa
  train input. Runtime audit của lần kiểm tra chặn việc mở cả hai file test;
  đã thử đọc để xác nhận cơ chế chặn hoạt động. Đây là bằng chứng local của
  lần chạy, không phải filesystem sandbox hoặc CI enforcement.
- Cả 12 CSV và 11 PNG của hai lần chạy giống nhau theo byte trong cùng môi
  trường. SHA-256 của ba file raw không đổi. Mọi hình và sơ đồ đã được xem
  bản render; thông tin nằm trong [validation.json](validation.json).
- [run_info.json](run_info.json) ghi phiên bản và SHA-256 train. Việc PNG giống
  từng byte chỉ được kiểm chứng trong môi trường Windows/Python 3.12.14 này;
  font/rendering trên môi trường khác có thể khác.

Các observations liên tiếp phụ thuộc thời gian; correlation mô tả quan hệ
đơn điệu, không chứng minh nhân quả. Quartiles giữa engines không phải khoảng
tin cậy. Mốc này chưa có p-value, bootstrap, feature selection hoặc model.
Việc chọn sensor, scaler hoặc mô hình sau này phải được học trong training
folds. Dữ liệu mô phỏng FD001 không trực tiếp xác định lịch bảo trì thực tế.

Mốc 2 tiếp theo sẽ chốt định nghĩa đầu/cuối vòng đời rồi thực hiện paired
analysis, effect size, bootstrap và hiệu chỉnh nhiều kiểm định. Các giả thuyết
phát sinh từ EDA cần được ghi rõ là khám phá.
