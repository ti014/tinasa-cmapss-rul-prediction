# Data contract cho NASA C-MAPSS FD001

Contract hiện thực thi việc đọc quan sát, kiểm tra dữ liệu, tạo nhãn RUL train,
tạo bảng cặp sensor đầu/cuối vòng đời, tóm tắt suy luận theo sensor và xuất
tệp đầu ra M2.3a chỉ dùng train. Mốc M2 đã có notebook suy luận, hai bảng CSV,
metadata lần chạy, hai hình M2.3b và biên bản validation M2.4 tại
`outputs/statistics/`. Mốc M3.1 đã bổ sung ETL Pandas cho Silver observations,
Silver train targets và Gold engine summary tại `outputs/etl/`; M3.2 đã bổ sung
Gold rolling features tại `outputs/features/`; mốc modeling đã bổ sung
cross-validation, final predictions và final metrics tại `outputs/modeling/`.
PySpark adapter đã chạy parity với Pandas reference trong môi trường local có
JRE.

## Nguồn quy tắc và phạm vi

- [README](../README.vi.md): chất lượng dữ liệu, window, chia theo engine và đánh
  giá cuối; đây là nguồn public của các quy tắc pipeline.
- [Nguồn FD001](raw/DATA_SOURCE.md) và [mô tả NASA](raw/readme.txt): 26 cột,
  ý nghĩa cycle, engine, train/test và ánh xạ nhãn.
- Kế hoạch cục bộ `docs/plans/active/2026-09-21-cmapss-rul-final-report.md`:
  raw RUL, window 30 cycles, 100 final predictions và phạm vi triển khai.
- Quy tắc đã được yêu cầu triển khai trong đợt data contract ngày 2026-09-22.

Các kiểm tra không tự điền giá trị thiếu, loại dòng lỗi, nối engine giữa split,
cap RUL hoặc sửa raw. Sai dữ liệu được báo bằng `ContractError` có mã quy tắc,
đối tượng vi phạm, nguồn và hướng xử lý. Các kiểm tra chung áp dụng cho bảng
observations trước khi lọc window. Profile FD001 áp dụng riêng cho file đầy đủ,
không dùng cho fold, window hoặc dữ liệu benchmark.

## Schema hiện có

| Bảng | Cột theo thứ tự | Kiểu và khóa |
| --- | --- | --- |
| Raw text | `engine_id`, `cycle`, `setting_1`–`setting_3`, `sensor_1`–`sensor_21` | 26 cột whitespace, không header; hai cột đầu là số nguyên |
| Observations | `split`, sau đó 26 cột raw | Reader trả `split` dạng Pandas string; ID/cycle `int64`; settings/sensors `float64`; khóa `(split, engine_id, cycle)` |
| Train targets | `split`, `engine_id`, `cycle`, `rul` | Khóa khớp observations train; `rul` nguyên, không âm, đơn vị cycle |
| Test labels | `split`, `engine_id`, `rul` | `split=test`; ID và RUL `int64`; khóa `(split, engine_id)` |
| Engine-sensor pairs | `split`, `engine_id`, `sensor`, `n_cycles`, ranh giới/số phần tử hai đoạn, `early_mean`, `late_mean`, `difference`, `pair_status` | Chỉ `train`; khóa `(split, engine_id, sensor)`; 21 dòng theo sensor cho mỗi engine; output canonical theo engine rồi `sensor_1`–`sensor_21` |
| Sensor inference | `sensor`, số cặp/dấu/zero, chênh lệch cặp trung bình và trung vị, CI, bootstrap status, Wilcoxon statistic/raw p, adjusted p, FDR và `test_status` | Một dòng mỗi sensor theo đúng thứ tự `sensor_1`–`sensor_21`; giữ all-zero là trạng thái riêng; không dùng để chọn feature trước cross-validation |
| Fold assignments | `split`, `engine_id`, `fold` | Chỉ `train`; mỗi engine thuộc đúng một validation fold |
| OOF predictions | Khóa train window, `fold`, `model_name`, `y_true`, `y_pred` và các cột lỗi | Một dòng cho mỗi window validation và mỗi candidate model |
| Final predictions | `split`, `engine_id`, `cycle`, `model_name`, `rul_pred` | Chỉ `test`; đúng 100 dòng, một prediction ở cycle cuối mỗi test engine; ghi trước khi join `RUL_FD001.txt` |
| Final evaluation | Final predictions đã join test labels, thêm `y_true`, `y_pred` và các cột lỗi | Chỉ tạo sau khi final predictions đã được ghi; dùng để tính metric cuối |

Reader observations nhận `split` do bên gọi xác định; CLI gán train/test từ tên
file cố định. Không thể suy ra split chỉ từ 26 giá trị của một dòng. Validator
chấp nhận hai engine cùng ID nếu khác split. Nó không thể phát hiện một bên gọi
cố ý gán sai split cho toàn bộ file có nội dung hợp lệ.

## Quy tắc và bằng chứng

### RAW_WIDTH

Mỗi dòng observations có đúng 26 trường; mỗi dòng nhãn có đúng một trường.
Whitespace đầu/cuối, nhiều spaces và tabs được chấp nhận; dòng trống bị báo lỗi
để không làm lệch ánh xạ dòng nhãn sang engine. Ví dụ: một dòng observations
chỉ có 25 trường bị báo lỗi ngay tại số dòng đó, không được pad bằng null.

### SCHEMA

DataFrame phải có đúng tên và thứ tự cột trong bảng schema. Không nhận thêm
cột `rul` vào observations. Fixture bỏ `sensor_21` hoặc thêm `rul` phải bị từ chối.

### NONEMPTY

File hoặc bảng cần ít nhất một dòng. Fixture nhỏ được dùng cho kiểm thử chung;
file rỗng không được xem là validation thành công.

### NUMERIC

ID/cycle là số nguyên dương; RUL là số nguyên không âm. Trong raw text, các
trường này được Pandas chuyển trực tiếp từ chuỗi sang `int64`, không làm tròn
`1.5` hoặc ép `1.0` thành ID. Settings/sensors là số thực hữu hạn, không null. Không áp đặt ngưỡng vật lý
cho sensor khi chưa có nguồn. Ví dụ: `nan`, `inf` hoặc `bad` trong sensor bị
từ chối; sensor âm hữu hạn vẫn được chấp nhận.

### SPLIT

Observations chỉ nhận `train` hoặc `test`, không null. Profile của từng file
chỉ nhận split tương ứng; test labels chỉ thuộc `test`. Validation folds sau
này lấy từ train, không tạo split raw mới có tên `validation`.

### UNIQUE_KEY

Khóa observations và train targets là `(split, engine_id, cycle)`; khóa test
labels là `(split, engine_id)`. Ví dụ `(train, 1, 1)` và `(test, 1, 1)` được
chấp nhận cùng nhau; lặp hai lần `(train, 1, 1)` phải bị phát hiện.

### CYCLE_CONTINUITY

Sau khi sắp xếp theo khóa, chênh lệch giữa hai cycle liên tiếp trong cùng engine
và split phải bằng 1. Validator không sửa thứ tự dữ liệu đầu vào. `1, 2, 4`
không hợp lệ; các dòng `3, 1, 2` hợp lệ vì chuỗi đủ cycle sau khi sắp xếp.

### FD001_PROFILE

Chỉ áp dụng cho bản FD001 đầy đủ đã xác minh:

| Dữ liệu | Số dòng | Engine ID | Cycle đầu |
| --- | ---: | --- | --- |
| Train | 20.631 | 1–100 | 1 trong mỗi engine |
| Test | 13.096 | 1–100 | 1 trong mỗi engine |
| Test labels | 100 | 1–100 | Không áp dụng |

Fixture hai engines có thể đạt contract chung nhưng phải trượt profile này.
Row count và tính liên tục không chứng minh rằng giá trị sensor hoặc phần cuối
chuỗi chưa bị thay đổi; báo cáo lưu SHA-256 để đối chiếu phiên bản đầu vào,
không thay thế việc xác minh provenance.

### LABEL_MAPPING

Dòng thứ `i` của `RUL_FD001.txt` được reader gán cho test engine `i`. Không sort
nhãn theo giá trị RUL. Nhãn `[30, 10, 0]` phải ánh xạ lần lượt đến engine
`[1, 2, 3]`. Validator kiểm tra ID 1–N và khóa; nếu giá trị bị hoán đổi trước
khi đọc mà cấu trúc vẫn đúng thì cần đối chiếu checksum nguồn để phát hiện.

### TRAIN_ONLY

Chỉ tạo nhãn theo max cycle cho lịch sử train đầy đủ đến khi hỏng. Hàm
`make_train_targets` từ chối test và bảng trộn train/test, không đọc file nhãn
test. Bên gọi phải cung cấp toàn bộ lịch sử trước khi lọc window: một chuỗi
train bị cắt đuôi nhưng vẫn liên tục không thể được nhận diện chỉ từ công thức.

`build_engine_sensor_pairs` cũng chỉ nhận observations thuộc `train`, sau khi
chạy contract observations chung. Nó không đọc test observations hoặc test
labels. Ví dụ bảng trộn engine train với engine test bị từ chối trước khi tính
mean; đây là ranh giới phân tích hồi cứu train, không phải cơ chế filesystem
access control cho mọi code trong repository.

### TARGET_KEYS

Train targets phải có đúng tập khóa observations, không thiếu/thêm dòng. Bảng
targets được đổi thứ tự vẫn hợp lệ; validator đối chiếu bằng khóa, không ghép
theo vị trí dòng.

### TRAIN_RUL

`rul = max_cycle - cycle` trong từng train engine, không cap ở đợt này. Ví dụ
engine có cycle `1, 2, 3, 4` nhận RUL `3, 2, 1, 0`; engine khác kết thúc ở
cycle 2 nhận `1, 0`. Test kiểm tra các giá trị tính tay và trường hợp sai RUL.

### CYCLE_START

Mỗi engine dùng tạo cặp phải bắt đầu ở cycle 1 và liên tục. Không suy diễn
đoạn đầu từ vị trí dòng input; cycle 2–61 dù liên tục vẫn bị từ chối.

### PAIR_WINDOW

Mỗi engine phải có ít nhất 60 cycles. Đoạn đầu cố định là 1–30; nếu `T_i` là
cycle cuối thì đoạn cuối là `T_i-29`–`T_i`. Hai đoạn có đúng 30 observations
và không chồng nhau. Engine 59 cycles bị từ chối, không rút ngắn window hoặc
bỏ engine để tiếp tục chạy. Ví dụ engine dài 200 cycles dùng 1–30 và 171–200.

### PAIR_SENSOR

Mỗi engine phải có đúng một dòng cho từng `sensor_1` đến `sensor_21`; không
lọc sensor từ EDA, correlation hoặc kết quả suy luận. Một dòng thiếu sensor
hoặc tên ngoài tập này làm validator dừng.

### PAIR_KEY

Khóa bảng cặp là `(split, engine_id, sensor)`. Không nhận hai dòng có cùng
khóa, kể cả khi các giá trị mean trùng nhau.

### PAIR_ORDER

Bảng canonical luôn sắp engine ID tăng dần, rồi đúng thứ tự `sensor_1` đến
`sensor_21`; tránh thứ tự chuỗi làm `sensor_10` đứng trước `sensor_2`.

### PAIR_NUMERIC

`early_mean`, `late_mean` và `difference` là `float64` hữu hạn; không ghi
null, NaN hoặc Infinity.

### PAIR_DIFFERENCE

`difference = late_mean - early_mean` trước khi làm tròn để trình bày. Ví dụ
sensor bằng cycle trong engine 60 cycles có `early_mean=15,5`,
`late_mean=45,5` và `difference=30`. Validator yêu cầu đúng kết quả `float64`
của phép trừ này, không dùng relative tolerance có thể chấp nhận chênh lệch
khác ở thang giá trị lớn.

### PAIR_STATUS

Mọi dòng đã tạo có `pair_status=valid`; input lỗi dừng ở contract thay vì tạo
dòng trạng thái lỗi.

Bảng cặp là API trong bộ nhớ do `src/cmapss/statistics/pairs.py` tạo và kiểm tra.
`src/cmapss/statistics/artifacts.py` dùng API này để xuất
`outputs/statistics/tables/engine_sensor_pairs.csv`,
`outputs/statistics/tables/sensor_inference.csv`, `run_info.json` và
`validation.json`; `src/cmapss/statistics_artifacts.py` vẫn giữ compatibility
import cho caller cũ. Cả hai CSV được đọc lại và kiểm tra sau khi xuất. Hình phân
phối, biểu đồ chênh lệch kèm khoảng tin cậy và diễn giải sâu thuộc M2.3b. M2.4
dùng `src/cmapss/statistics/reproducibility.py` để chạy focused tests, full
suite, notebook hai lần từ fresh kernel, so sánh hash artifact và runtime audit.

### INFERENCE_CONFIG

Cấu hình M2.0 đã chốt trước khi xem kết quả: mean paired difference là chênh lệch
cặp trung bình chính theo đơn vị gốc; bootstrap percentile 95% dùng 10.000 resamples theo engine và seed
42; Wilcoxon là two-sided, zero method `wilcox`, không continuity correction,
không dùng `auto`, và dùng permutation 100.000 resamples; FDR dùng
Benjamini-Hochberg với `q=0,05` trên đúng 21 sensors. Implementation dùng SciPy
đã khóa trong môi trường; không thêm `statsmodels` chỉ để hiệu chỉnh p-value.

### INFERENCE_SENSOR

Bảng suy luận phải có đủ 21 dòng theo thứ tự `sensor_1` đến `sensor_21`. Không
lọc sensor theo p-value, CI, EDA hoặc correlation. Ba operational settings không
nằm trong family kiểm định này.

### INFERENCE_COUNTS

Với mỗi sensor, `n_positive + n_negative + n_zero = n_pairs`. Các đếm này dựa
trên `difference` gốc `float64`; zero được xác định bằng `difference == 0.0`,
không làm tròn bằng ngưỡng tùy ý để đổi trạng thái kiểm định.

### INFERENCE_CI

`ci_lower <= ci_upper`, phương pháp là percentile và mức CI là 0,95 theo cấu
hình M2.0. Bootstrap suy biến được ghi bằng `bootstrap_status=degenerate`;
không diễn giải khoảng bằng 0 là chắc chắn tuyệt đối.

### INFERENCE_PVALUE

`raw_p_value` và `p_value_adjusted` phải thuộc `[0, 1]` khi sensor kiểm định
được. Với all-zero, output public giữ p-value là NA; sentinel `1,0` chỉ dùng
bên trong vector FDR để giữ family đủ 21 sensors. Lỗi số bất ngờ phải làm lần
chạy thất bại, không chuyển thành all-zero hoặc p-value hợp lệ.

### INFERENCE_STATUS

`test_status` hiện nhận `ok` hoặc `all_zero_difference`. Sensor all-zero vẫn ở
trong bảng, có Wilcoxon không áp dụng và FDR public là NA. Trạng thái này chỉ
nói protocol đầu/cuối không quan sát được biến thiên trong train, không chứng
minh sensor vô dụng cho dự đoán RUL.

### RAW_IMMUTABLE

CLI không cho ghi báo cáo vào thư mục raw được truyền qua `--raw-dir`. Kiểm thử
ghi đè file nguồn phải thất bại và giữ nguyên bytes. Đây là bảo vệ tại CLI,
không phải filesystem permission áp dụng cho mọi chương trình khác.

### ETL_SILVER_GOLD

Pipeline ETL phải đọc raw qua reader/validator hiện có, không sửa Bronze và
ghi artifact ra ngoài `data/raw/`. Silver observations giữ đúng schema
`OBSERVATION_COLUMNS`, khóa `(split, engine_id, cycle)` và thứ tự canonical theo
`split`, `engine_id`, `cycle`. Silver train targets giữ đúng
`TRAIN_TARGET_COLUMNS`, có cùng tập khóa với train observations và công thức
`rul = max_cycle - cycle`.

Gold engine summary có đúng một dòng cho mỗi `(split, engine_id)` với các cột
`n_cycles`, `first_cycle`, `last_cycle`. Gold M3.1 là catalog engine-level để
kiểm tra boundary và truy vấn cục bộ; nó chưa phải rolling feature hoặc
prediction artifact. Parquet là định dạng analytical artifact; SQLite là lớp
truy vấn portable, cả hai phải đọc lại được sau khi ghi.

Pipeline Pandas là reference implementation. PySpark adapter phải tạo cùng ba
boundary; parity checker so sánh schema, row count, khóa duy nhất và các cột số
trong tolerance đã ghi ở API (`rtol=1e-9`, `atol=1e-12`). Nếu PySpark hoặc Java
không sẵn sàng, phải báo giới hạn runtime và không được claim parity đã đạt.

Benchmark 1×, 10× và 100× chỉ dùng observations nhân bản trong bộ nhớ, remap
engine ID để không trùng khóa, và đánh dấu `scientific_artifacts_isolated=true`.
Bản nhân bản không được ghi vào artifact khoa học, không được dùng tạo kết luận
EDA hoặc huấn luyện mô hình. Metadata phải ghi scale, số lần lặp, cold/warm
run, runtime và row counts.


### ROLLING_FEATURES

Rolling features M3.2 đọc Silver observations đã đạt contract và ghi Gold
`outputs/features/gold/rolling_features.parquet`. Khóa của mỗi dòng là
`(split, engine_id, cycle)`, trong đó `cycle` là cycle cuối của trailing window.
Mỗi window có đúng 30 observations liên tục, cùng split và cùng engine; không
được dùng cycle tương lai, không nối qua engine khác và không đọc hoặc join
`RUL_FD001.txt`.

Artifact giữ đủ 21 sensors để chưa thực hiện feature selection trên toàn bộ
train. Mỗi sensor tạo sáu cột theo mẫu `<sensor>__<stat>` với `stat` thuộc
`last`, `mean`, `std`, `min`, `max`, `slope`. `std` là sample standard
deviation (`ddof=1`) trên 30 giá trị trong window. `slope` là hệ số hồi quy
tuyến tính least-squares theo vị trí 1--30 trong chính window. Ví dụ nếu
`sensor_1` bằng cycle và window là cycles 1--30 thì `sensor_1__last=30`,
`mean=15,5`, `std` là độ lệch chuẩn mẫu của 1--30, `min=1`, `max=30` và
`slope=1`.

Mỗi engine phải có ít nhất 30 cycles để tạo rolling features. Engine dưới 30
cycles bị từ chối bằng rule `ROLLING_WINDOW`; pipeline không tự rút ngắn window,
không padding và không impute. Với FD001 hiện tại, artifact có 17.731 train
windows, 10.196 test windows, 126 feature columns và 27.927 dòng tổng cộng.
Scaler, PCA, lựa chọn sensor và model fit vẫn thuộc mốc mô hình; nếu loại bỏ
sensor hằng thì quyết định đó phải được học trong training fold, không lấy từ
artifact toàn train.

### MODELING_GROUP_SPLIT

Modeling dùng 5-fold GroupKFold theo `engine_id` trên 17.731 train windows. Mỗi
engine train thuộc đúng một validation fold; các window của cùng engine không bị
chia giữa train và validation trong cùng fold. `fold_assignments.csv` có 100
dòng, mỗi dòng là một engine train và fold validation tương ứng.

### MODELING_TARGET_KEYS

Train rolling features phải join được đúng một RUL target theo khóa
`(split, engine_id, cycle)`. Mọi train feature thiếu target làm pipeline dừng.
Test features không được join `RUL_FD001.txt` trong bước chuẩn bị dataset.

### MODELING_SAMPLE_WEIGHT

Training fold dùng weighting theo engine: tổng sample weight của mỗi engine bằng
1. Quy tắc này giảm việc engine sống lâu có nhiều window lấn át engine ngắn.
Nó không thay đổi metric báo cáo; metric chính vẫn là mean engine-level MAE.

### MODELING_MODEL_SELECTION

Các ứng viên hiện tại là `median_baseline`, `ridge_alpha_1` và
`hist_gradient_boosting`. Model được chọn bằng `mean_engine_mae` trung bình qua
5 folds. Scaler của Ridge nằm trong pipeline và được fit trong training fold.
Test labels không được đọc hoặc dùng để chọn model.

### MODELING_FINAL_PREDICTIONS

Sau khi chọn model, pipeline refit model trên toàn bộ train windows và tạo đúng
một prediction tại cycle cuối của mỗi test engine. `final_predictions.parquet`
phải có 100 dòng, cột `rul_pred` hữu hạn và được ghi trước khi đọc
`RUL_FD001.txt`.

### MODELING_FINAL_EVALUATION

Chỉ bước final evaluation mới đọc `RUL_FD001.txt` và join theo
`(split, engine_id)`. Metric cuối hiện gồm mean engine-level MAE, RMSE, NASA
asymmetric score và bootstrap percentile CI 95% theo 100 test engines. File
`outputs/modeling/run_info.json` lưu checksum của rolling features, train
targets và test labels.

## Test labels và phạm vi cô lập

Lệnh validation mặc định chỉ đọc observations, không mở `RUL_FD001.txt`.
`--check-test-labels` đọc riêng để kiểm tra tính toàn vẹn; báo cáo chỉ lưu số
nhãn, checksum và trạng thái, không lưu giá trị nhãn, không join observations
và không tính metric. Kiểm thử chạy mặc định trong thư mục không có file nhãn.

`read_test_labels` là API riêng, không được bảo vệ bằng quyền truy cập hệ điều
hành. Pipeline modeling kiểm tra ranh giới bằng cách ghi `final_predictions`
trước khi đọc test labels và lưu trạng thái join nhãn sau cùng trong
`outputs/modeling/run_info.json`.

Validation M2.4 của thống kê tạo runtime guard cục bộ với rule
`STATISTICS_TRAIN_ONLY`. Guard cho phép đọc `train_FD001.txt` và chặn probe mở
`test_FD001.txt` hoặc `RUL_FD001.txt` trong tiến trình notebook. Đây là bằng
chứng local của notebook thống kê, không phải filesystem sandbox, hook, CI
enforcement hoặc branch protection cho toàn repository.

## Contract đã chốt cho giai đoạn sau

| Artifact | Quy tắc đã chốt | Phần chưa triển khai |
| --- | --- | --- |
| Rolling features | Đã có artifact M3.2 tại `outputs/features/gold/rolling_features.parquet`: khóa observation tại cycle cuối window, trailing window đủ 30 cycles, cùng engine/split, không dùng tương lai; giữ 21 sensors; cột `<sensor>__<stat>` với `last`, `mean`, sample `std` (`ddof=1`), `min`, `max`, slope | Chưa có PCA hoặc feature selection; nếu thêm, phải fit trong training fold |
| Final predictions | Đã có artifact tại `outputs/modeling/predictions/final_predictions.parquet`: khóa `(split, engine_id, cycle)` tại cycle cuối test; một giá trị RUL dự đoán mỗi engine, đúng 100 dòng; tạo trước khi join test labels | Chưa có dashboard hoặc serving API; chưa clip prediction vì lần chạy hiện tại không sinh giá trị âm |

Ví dụ window tại cycle 170 là cycle 141–170 của cùng engine. M3.2 đã hiện
thực hóa feature transformer và validator cho artifact rolling features. Mốc
modeling đã tạo metrics và prediction cho FD001; ARIMA, dashboard và kiểm tra
tái lập toàn bộ vẫn chưa hoàn thành.

## Chạy kiểm tra

Hướng dẫn cài bằng `venv` và `pip install -r requirements.txt` nằm trong
[README](../README.vi.md#bắt-đầu). `requirements-dev.txt` bổ sung công cụ test;
hai file này giữ nguyên phiên bản từ `uv.lock`. Không cần uv khi cài bằng pip.

Nếu đã dùng uv, từ thư mục gốc chỉ cần:

```powershell
uv run cmapss-validate
```

Lệnh tự chuẩn bị môi trường trước khi kiểm tra dữ liệu. Thêm
`--check-test-labels` nếu cần kiểm tra riêng tính toàn vẹn nhãn test; dùng
`uv run pytest` cho kiểm thử khi phát triển.

Dependency trực tiếp nằm trong `pyproject.toml`; `uv.lock` khóa cả dependency
bắc cầu. Môi trường được kiểm chứng ở đợt đầu là Python 3.12.14 trên Windows.
CLI trả mã 0 khi đạt, 1 khi lỗi đầu vào/contract, 2 khi đường output không hợp
lệ hoặc không ghi được. Lỗi dữ liệu tạo báo cáo `status=failed`, thay báo cáo
cũ tại đường output được chọn để tránh đọc nhầm kết quả thành công trước đó.

Nếu gặp lỗi quyền thư mục tạm mặc định của pytest trên Windows, dùng hướng dẫn
`--basetemp` trong [README](../README.vi.md#bắt-đầu). Thư mục đó chỉ dành cho
pytest và được pytest dọn khi chạy lại.

Thành phần chịu trách nhiệm validation là pytest và CLI trong package `cmapss`.
Chưa có optional hook hoặc CI gọi các kiểm tra này; branch protection chưa được
xác minh.
