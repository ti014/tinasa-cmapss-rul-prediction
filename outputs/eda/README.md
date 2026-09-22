# EDA FD001 trên tập train

Mốc 1 đã hoàn thành: khảo sát 20.631 quan sát của 100 động cơ, xuất 12 bảng
CSV và 11 hình PNG. Tuổi thọ quan sát có trung vị 199 cycles; sáu sensor hằng
trên toàn train. Đây là kết quả mô tả, chưa phải kiểm định giả thuyết hoặc
bằng chứng về hiệu quả dự đoán RUL.

## Input, cách chạy và đầu ra

[Notebook](../../notebooks/01_fd001_eda.ipynb) chỉ đọc
`data/raw/train_FD001.txt`, dùng trình đọc, bộ kiểm tra và nhãn RUL train hiện
có. Notebook không đọc quan sát test hoặc nhãn test. RUL bằng `max_cycle - cycle`
trong từng động cơ; ví dụ động cơ kết thúc ở cycle 200 có RUL 30 tại cycle 170.

Sau khi tạo `.venv` với Python 3.12 theo README, chạy từ thư mục gốc:

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m jupyterlab notebooks/01_fd001_eda.ipynb
```

Chọn kernel Python của `.venv`, sau đó chọn **Restart Kernel and Run All Cells**.
Notebook tìm thư mục gốc từ vị trí hiện tại, tạo lại bảng/hình tại `outputs/eda/`
và ghi đè các tệp đầu ra cùng tên. Dữ liệu raw được giữ nguyên. Notebook không
cần Spark, Java hoặc môi trường mô hình ở mốc này.

![Quy trình EDA FD001](../../assets/diagrams/fd001-eda.svg)

[Nguồn draw.io có thể chỉnh sửa](../../assets/diagrams/fd001-eda.drawio).

```text
notebooks/01_fd001_eda.ipynb   Phân tích, biểu đồ và diễn giải
src/cmapss/data/              Trình đọc, bộ kiểm tra và nhãn RUL train dùng lại
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
2. Mô tả tuổi thọ với một giá trị mỗi động cơ. Phân phối RUL gộp các quan sát
   được trình bày riêng vì động cơ sống lâu đóng góp nhiều dòng hơn.
3. Mô tả 3 thông số vận hành và 21 sensor: giá trị thiếu, số giá trị khác nhau,
   min/max, trung bình, trung vị, các tứ phân vị và độ lệch chuẩn mẫu (`ddof=1`).
   Các tứ phân vị dùng nội suy tuyến tính. Cột hằng được xác định bằng
   `n_unique == 1`; không tự đặt ngưỡng near-constant.
4. Chọn động cơ gần Q25, Q50, Q75 tuổi thọ nhất; nếu bằng khoảng cách, chọn ID
   nhỏ nhất. Vẽ giá trị gốc theo cycle, không smooth. Đây là ví dụ minh họa,
   không phải mẫu đại diện thống kê. Mỗi ô dùng thang đo riêng.
5. Tính Spearman sensor–sensor gộp train và sensor–RUL riêng từng động cơ.
   Spearman dùng average ranks khi có ties. Sensor/RUL hằng cho `NaN`, không
   thay bằng 0. Tóm tắt các hệ số theo động cơ với trọng số bằng nhau.
6. Kiểm tra số dòng, khóa, miền correlation, tính đối xứng và checksum của raw;
   xuất bảng, hình và metadata.

Histogram dùng 20 bin để trình bày. Các điểm ngoài whisker của boxplot vẫn
được giữ. Tên sensor được giữ theo schema; chưa gán đơn vị vật lý khi nguồn
raw chưa mô tả đầy đủ. Độ lệch chuẩn mẫu của một cột hằng đôi khi có sai số số
thực cỡ `1e-15`; `n_unique`, không phải std, quyết định trạng thái hằng.

## Kết quả quan sát

| Nội dung | Kết quả trên train | Bằng chứng |
| --- | --- | --- |
| Input | 20.631 quan sát, 100 động cơ; không có giá trị thiếu hoặc khóa trùng | [Input](tables/input_summary.csv) |
| Tuổi thọ | 128–362 cycles; trung bình 206,31; trung vị 199; Q25–Q75: 177–229,25 | [Thống kê](tables/lifetime_summary.csv), [hình](figures/engine_lifetimes.png) |
| RUL theo quan sát | 0–361 cycles; trung vị 103 | [Thống kê](tables/rul_summary.csv), [hình](figures/rul_distribution.png) |
| Cột hằng toàn train | `setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19` | [Mô tả](tables/measurement_summary.csv) |
| `sensor_6` | Hai giá trị 21,60/21,61; hằng trong 38/100 động cơ | [Theo động cơ](tables/constant_engine_counts.csv) |
| Động cơ minh họa | 87: 178 cycles; 26: 199 cycles; 75: 229 cycles | [Quy tắc chọn](tables/example_engines.csv) |

Ba sensor có trị tuyệt đối của trung vị Spearman sensor–RUL lớn nhất trong
khảo sát này là `sensor_11` (−0,819), `sensor_12` (+0,808) và `sensor_4`
(−0,784). Mỗi giá trị tóm tắt 100 hệ số theo engine; xem
[bảng đầy đủ](tables/sensor_rul_summary.csv) và
[boxplot](figures/sensor_rul_by_engine.png). Đây không phải danh sách feature
được chọn cho mô hình.

Trong ba động cơ minh họa, `sensor_11` nhìn chung tăng và `sensor_12` giảm về
cuối chuỗi. Tuy nhiên, `sensor_9` và `sensor_14` ở động cơ 26 có dạng khác với
động cơ 87 và 75; xem [quỹ đạo trang 2](figures/sensor_trajectories_2.png).
Bảng correlation trên 100 động cơ cũng cho thấy dấu của hệ số hai sensor này
không đồng nhất. Vì vậy, một đường minh họa hoặc một hệ số gộp không mô tả hết
sự khác biệt giữa engines.

## Danh mục bảng và hình

| Bảng CSV | Một dòng biểu diễn |
| --- | --- |
| `input_summary.csv` | Toàn bộ input train |
| `schema.csv` | Một cột sau khi ghép RUL, gồm split và khóa |
| `engine_lifetimes.csv` | Một động cơ, 100 dòng |
| `lifetime_summary.csv` | Một thống kê tuổi thọ |
| `rul_summary.csv` | Một thống kê RUL theo observation |
| `measurement_summary.csv` | Một thông số vận hành hoặc sensor, 24 dòng |
| `within_engine_variability.csv` | Một động cơ và cột đo lường, 2.400 dòng |
| `constant_engine_counts.csv` | Số động cơ hằng của một cột đo lường |
| `example_engines.csv` | Một phân vị dùng để chọn động cơ minh họa |
| `sensor_spearman_pooled.csv` | Một hàng của ma trận 21 × 21 sensor |
| `sensor_rul_by_engine.csv` | Một động cơ và sensor, 2.100 dòng |
| `sensor_rul_summary.csv` | Phân phối correlation của một sensor qua các động cơ |

Hình gồm phân phối tuổi thọ, RUL và thông số vận hành; ba trang phân phối
sensor; ba trang quỹ đạo sensor; một heatmap sensor–sensor và một boxplot
sensor–RUL.
Các cặp không xác định được ghi `NaN` trong CSV, ô xám trên heatmap hoặc hàng
trống có số lượng `n=0` trên boxplot.

## Kiểm tra và giới hạn

- Toàn bộ bộ kiểm thử đạt **69 tests**, gồm ví dụ tính tay trung bình/độ lệch
  chuẩn/tứ phân vị, Spearman có ties, tách các động cơ, giữ cột hằng và từ chối
  dữ liệu không phải train.
- Cài đặt lại các dependency từ `requirements-dev.txt` bằng pip trong môi
  trường kiểm tra riêng: `pip check` đạt. JupyterLab mở notebook qua điểm khởi
  chạy đã hướng dẫn, trả
  HTTP 200; instance tạm đã được dừng sau kiểm tra.
- Chạy notebook từ kernel mới, sau đó chạy độc lập trong thư mục chỉ chứa
  train input. Runtime audit của lần kiểm tra chặn việc mở cả hai file test;
  đã thử đọc để xác nhận cơ chế chặn hoạt động. Đây là bằng chứng cục bộ của
  lần chạy, không phải filesystem sandbox hoặc CI enforcement.
- Cả 12 CSV và 11 PNG của hai lần chạy giống nhau theo byte trong cùng môi
  trường. SHA-256 của ba file raw không đổi. Mọi hình và sơ đồ đã được xem
  bản render; thông tin nằm trong [validation.json](validation.json).
- [run_info.json](run_info.json) ghi phiên bản và SHA-256 train. Việc PNG giống
  từng byte chỉ được kiểm chứng trong môi trường Windows/Python 3.12.14 này;
  font/rendering trên môi trường khác có thể khác.

Các quan sát liên tiếp phụ thuộc thời gian; correlation mô tả quan hệ đơn điệu,
không chứng minh nhân quả. Các tứ phân vị giữa các động cơ không phải khoảng tin
cậy. Mốc này không bao gồm p-value, bootstrap, lựa chọn đặc trưng hoặc mô hình.
Việc chọn sensor, scaler hoặc mô hình sau này phải được học trong training folds.
Dữ liệu mô phỏng FD001 không trực tiếp xác định lịch bảo trì thực tế.

Mốc M2.3a đã thực hiện phân tích ghép cặp, tính chênh lệch cặp trung bình,
bootstrap và hiệu chỉnh nhiều kiểm định trên train. M2.3b còn cần tạo hình phân
phối, biểu đồ chênh lệch kèm khoảng tin cậy và bổ sung diễn giải sâu trong báo cáo. Các giả thuyết
phát sinh từ EDA vẫn phải được ghi rõ là khám phá.
