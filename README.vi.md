<a href="README.md"><img src="https://flagcdn.com/20x15/us.png" alt="English" width="20" height="15"> English</a> · <a href="README.vi.md"><img src="https://flagcdn.com/20x15/vn.png" alt="Tiếng Việt" width="20" height="15"> Tiếng Việt</a>

<hr>

# Pipeline dự đoán RUL trên NASA C-MAPSS

Repository này phục vụ báo cáo cuối kỳ môn Phân tích dữ liệu lớn. Báo cáo xây dựng quy trình có khả năng tái lập để phân tích dữ liệu cảm biến đa biến và dự đoán Remaining Useful Life (RUL, tuổi thọ hữu ích còn lại) trên NASA C-MAPSS FD001. Trọng tâm gồm nguồn gốc dữ liệu rõ ràng, phân tích thống kê phù hợp, đánh giá tránh data leakage (rò rỉ dữ liệu), ETL có khả năng mở rộng và trực quan hóa kết quả dễ diễn giải.

## Trạng thái

Repository đã có **gói kiểm tra dữ liệu FD001, nhãn RUL train, notebook EDA chỉ dùng train đã hoàn thành ở mốc 1 và các tệp đầu ra suy luận sensor M2.3a--M2.3b chỉ dùng train cho mốc 2**.

Đã có:

- Các file NASA C-MAPSS FD001 nguyên bản và tài liệu mô tả nguồn.
- Trình đọc (reader), bộ kiểm tra (validator) và kiểm thử số dòng, số động cơ, schema, khóa, cycle và nhãn RUL train.
- [Data contract (đặc tả dữ liệu)](data/CONTRACTS.md), CLI xuất báo cáo JSON và tệp khóa phiên bản dependency.
- [Notebook EDA đã chạy](notebooks/01_fd001_eda.ipynb), 12 bảng CSV,
  11 hình và [diễn giải kết quả, bằng chứng kiểm tra](outputs/eda/README.md).
- Danh mục tài liệu tham khảo và hướng dẫn đọc.
- [Báo cáo LaTeX cập nhật theo tiến độ](report/README.md) và [bản PDF](report/main.pdf),
  có data dictionary (từ điển dữ liệu), bảng thuật ngữ và chữ viết tắt, nhật ký
  công việc, lý do lựa chọn và nguồn đối chiếu kết quả validation/EDA/M2.3a.
  Mốc 1 EDA và bảng suy luận M2.3a đã được kiểm tra độc lập; hình M2.3b đã có,
  còn validation tái lập và mô hình là bước tiếp theo.
- Cơ sở lý thuyết có trích dẫn và [protocol thống kê mốc 2](report/sections/statistical-protocol.tex).
  Đã chốt so sánh giá trị trung bình của 30 cycles đầu và 30 cycles cuối theo từng động cơ.
  Cấu hình thống kê M2.0 đã được chốt trước suy luận; notebook M2.3a đã chạy bảng suy luận train-only.
- Helper `build_engine_sensor_pairs` chỉ nhận train, tạo và kiểm tra một cặp
  30 cycles đầu/cuối cho từng động cơ và đủ 21 sensor.
- Helper M2.2 để tính chênh lệch cặp trung bình theo đơn vị gốc, percentile bootstrap CI, Wilcoxon
  permutation và Benjamini-Hochberg FDR theo cấu hình M2.0 đã chốt.
- [Notebook suy luận M2.3a đã chạy](notebooks/02_fd001_inference.ipynb), hai
  bảng CSV, metadata (siêu dữ liệu lần chạy) và [ghi chú validation](outputs/statistics/README.md).
  Đầu ra giữ đủ 21 sensor và không đọc quan sát test hoặc nhãn test.
- Hình M2.3b được tạo từ các artifact trên: [phân phối chênh lệch](outputs/statistics/figures/difference_distributions.png)
  và [effect size kèm bootstrap CI](outputs/statistics/figures/effect_size_ci.png).

Chưa triển khai:

- Pipeline ETL Silver/Gold bằng Pandas và PySpark.
- Validation tái lập hai lần và audit train-only chủ động cho notebook thống kê,
  cùng pipeline ETL Silver/Gold và xây dựng đặc trưng (feature engineering).
- Các mô hình RUL và phân tích ARIMA.
- Dashboard Streamlit và báo cáo hoàn chỉnh (đã bắt đầu bản thảo LaTeX).

Lệnh cài đặt, kiểm tra dữ liệu và mở notebook có ở phần Bắt đầu. Huấn luyện, đánh giá mô hình và dashboard chưa có điểm khởi chạy.

## Mục tiêu

Báo cáo tập trung trả lời bốn câu hỏi:

1. FD001 đáng tin và phù hợp đến mức nào cho một nghiên cứu học thuật về predictive maintenance?
2. Những tín hiệu cảm biến nào thể hiện sự suy giảm có thể đo lường theo vòng đời động cơ?
3. Có thể dự đoán RUL chính xác đến mức nào từ 30 cycles vận hành gần nhất?
4. Làm thế nào trình bày kết quả mô hình thành bằng chứng cảnh báo sớm minh bạch và tái lập được?

Mục tiêu chính là **dự đoán RUL**. Supervised regression (hồi quy có giám sát) là cách mô hình hóa phù hợp vì biến mục tiêu là số cycle còn lại; đây không phải tên hoặc mục tiêu của báo cáo.

## Dữ liệu

Báo cáo sử dụng **NASA C-MAPSS FD001**, bộ dữ liệu mô phỏng quá trình động cơ turbofan vận hành đến khi hỏng do NASA Ames Prognostics Center of Excellence công bố.

| File | Vai trò | Quy mô tệp |
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
- Ghi chú nguồn trong repository: [`data/raw/DATA_SOURCE.md`](data/raw/DATA_SOURCE.md)

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

Pandas đã được dùng trong notebook EDA và sẽ tiếp tục phục vụ thống kê, cũng như chuẩn bị dữ liệu cho mô hình. PySpark sẽ tái hiện quy trình ETL và phục vụ thử nghiệm khả năng mở rộng (scalability benchmark). Bộ dữ liệu gốc chỉ có dung lượng vài MB, vì vậy repository không tuyên bố FD001 tự thân chứng minh khả năng xử lý dữ liệu quy mô lớn.

## Phương pháp dự kiến

### Chất lượng dữ liệu

- Kiểm tra schema 26 cột và kiểu dữ liệu số.
- Bảo đảm khóa `(split, engine_id, cycle)` không trùng.
- Kiểm tra cycle liên tục trong từng động cơ.
- Ghi lại số dòng trước và sau mỗi phép biến đổi.
- Giữ phạm vi engine ID riêng biệt giữa train và test.

### Phân tích thống kê

- So sánh hành vi cảm biến ở giai đoạn đầu và cuối vòng đời theo từng động cơ.
- Báo cáo chênh lệch cặp trung bình theo đơn vị gốc và khoảng tin cậy bootstrap 95%.
- Dùng paired Wilcoxon signed-rank test khi phù hợp.
- Kiểm soát multiple testing (kiểm định nhiều giả thuyết) trên các cảm biến bằng Benjamini-Hochberg FDR.
- Ước lượng độ bất định của các metric đánh giá bằng bootstrap theo động cơ.

### Xây dựng đặc trưng

Biểu diễn chính sử dụng trailing window dài 30 cycles. Với mỗi cảm biến không hằng, các đặc trưng dự kiến gồm:

- Giá trị gần nhất.
- Trung bình và độ lệch chuẩn.
- Giá trị nhỏ nhất và lớn nhất.
- Độ dốc xu hướng tuyến tính.

Mỗi window chỉ chứa dữ liệu của một động cơ và không sử dụng quan sát tương lai.

### Dự đoán RUL

Các phương pháp dự kiến được so sánh gồm:

1. Naive baseline theo trung vị RUL.
2. Ridge regression làm baseline tuyến tính có regularization.
3. HistGradientBoostingRegressor làm mô hình ứng viên phi tuyến.

Mô hình dự kiến được chọn bằng GroupKFold 5-fold, trong đó `engine_id` xác định group. Nhãn test chính thức của NASA được cô lập cho đến lần đánh giá cuối. Các metric đánh giá gồm engine-level MAE, RMSE và NASA asymmetric scoring function.

### Phân tích ARIMA

ARIMA là phân tích chuỗi thời gian phụ, không phải mô hình RUL chính. Mô hình dự kiến được huấn luyện (`fit`) trên phần đầu chuỗi Health Index của một số động cơ validation đại diện, sau đó dự báo phần còn lại và báo cáo:

- MAE và RMSE của dự báo.
- Tỷ lệ bao phủ của khoảng dự đoán 95%.
- Chẩn đoán tính dừng Augmented Dickey-Fuller.
- Chẩn đoán tự tương quan phần dư Ljung-Box.

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
- Chỉ huấn luyện scaler, feature selection, PCA và mô hình trên các fold training.
- Chọn mô hình theo trung bình engine-level MAE.
- Huấn luyện lại mô hình được chọn trên toàn bộ tập train.
- Tạo đúng một dự đoán cuối cho mỗi động cơ trong số 100 động cơ test.
- Chỉ ghép (`join`) predictions với `RUL_FD001.txt` trong bước đánh giá cuối.
- Báo cáo các metric tổng hợp, khoảng tin cậy và sai số phân nhóm theo khoảng RUL.

Báo cáo không đặt trước một mức điểm mục tiêu. Kết quả âm vẫn hợp lệ nếu quy trình đánh giá được giữ nguyên và các nhóm lỗi được phân tích trung thực.

## Trực quan hóa

- Phân phối vòng đời động cơ (đã có ở mốc 1).
- Biến thiên và quỹ đạo sensor theo cycle gốc (đã có ở mốc 1).
- Dự kiến: quỹ đạo sensor và Health Index theo vòng đời đã chuẩn hóa.
- Dự kiến: RUL thật và RUL dự đoán trên các động cơ validation.
- Dự kiến: biểu đồ phân tán giá trị dự đoán và phân phối phần dư trên tập test.
- Dự kiến: phân tích sai số theo động cơ và khoảng RUL.
- Dự kiến: dashboard Streamlit để xem quỹ đạo và kết quả mô hình của từng động cơ.

Dashboard có thể minh họa cảnh báo khi RUL dự đoán không quá 30 cycles. Đây không phải ngưỡng bảo trì tối ưu vì FD001 không cung cấp chi phí bảo trì, downtime hoặc intervention records.

## Cấu trúc repository

Cấu trúc hiện có và dự kiến của repository:

```text
.
├── data/
│   ├── raw/                 # FD001 nguyên bản và ghi chú nguồn
│   └── archive/             # Dữ liệu lưu để đối chiếu, không dùng cho mô hình chính
├── references/              # Danh mục tài liệu, hướng dẫn đọc và các bài báo hiện có
├── notebooks/               # Notebook EDA và thống kê train FD001 đã chạy
├── src/cmapss/              # data/: contract; statistics/: pairs/inference/artifacts; eda.py; validate.py
├── tests/                   # Kiểm thử data contract, CLI, EDA và thống kê
├── outputs/validation/      # Báo cáo kiểm tra dữ liệu hiện có
├── outputs/eda/             # Bảng, hình, phương pháp và kiểm tra EDA
├── outputs/statistics/      # Bảng suy luận sensor M2.3a và metadata
├── assets/diagrams/         # Nguồn draw.io, bản xem trước SVG và file URL
├── pyproject.toml           # Thông tin package và dependency trực tiếp
├── requirements.txt         # Cài bằng pip: package và thư viện chạy chương trình
├── requirements-dev.txt     # Thư viện notebook và kiểm thử
├── uv.lock                  # Phiên bản dependencies đã khóa
├── report/                  # Bản thảo LaTeX/PDF, tiến độ và bằng chứng
├── app.py                   # Entry point Streamlit dự kiến
├── README.md                # Tài liệu tiếng Anh
└── README.vi.md             # Tài liệu tiếng Việt
```

Các file hỗ trợ agent trong môi trường cục bộ được loại khỏi GitHub thông qua `.gitignore`.

## Khả năng tái lập

Khi hoàn thành, repository sẽ cung cấp một quy trình được ghi chép rõ từ dữ liệu nguồn bất biến đến dữ liệu đã xử lý, metric, hình minh họa, dữ liệu dashboard và báo cáo cuối kỳ. Các yêu cầu tái lập gồm:

- Dùng random seed cố định `42` cho chia dữ liệu và mô hình ngẫu nhiên.
- Cố định phiên bản các dependency Python.
- Ghi lại số dòng và kết quả validation.
- Lưu cấu hình mô hình và kết quả đánh giá.
- Tách dữ liệu khoa học khỏi dữ liệu tổng hợp dùng kiểm tra scalability.
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

Đặc tả dữ liệu và luồng dữ liệu ghép cặp chỉ dùng train nằm trong `src/cmapss/`:

```text
contracts.py     Tên cột, kiểu dữ liệu và ngoại lệ
readers.py       Đọc file thành DataFrame
validation.py    Kiểm tra schema, khóa và cycle
targets.py       Tạo và kiểm tra train RUL
statistics/
  config.py       Cấu hình suy luận M2.0 đã chốt
  pairs.py        Tạo và kiểm tra cặp engine/sensor
  inference.py    Bootstrap, Wilcoxon, FDR và tóm tắt theo sensor
  validation.py   Kiểm tra bảng suy luận sensor
  artifact_io.py  CSV/JSON, hash và reader round-trip
  metadata.py     Metadata lần chạy và validation payload
  artifacts.py    Ghi CSV/JSON và kiểm tra round-trip
statistics_artifacts.py
                  Compatibility import cho bộ ghi artifact
```

Ví dụ: engine train kết thúc tại cycle 200 thì `rul = 200 - cycle`.
Hàm tạo nhãn từ chối quan sát test. Xem [data contract](data/CONTRACTS.md)
để biết các điều kiện và giới hạn kiểm tra.

### Chạy notebook EDA

Sau khi cài `requirements-dev.txt`, mở notebook từ thư mục gốc:

```powershell
.venv\Scripts\python -m jupyterlab notebooks/01_fd001_eda.ipynb
```

Chọn kernel Python của `.venv` và **Restart Kernel and Run All Cells**.
Notebook chỉ phân tích train và tạo lại 12 bảng CSV, 11 hình PNG.
Các lần kiểm tra dùng kernel mới đã tạo tệp đầu ra giống nhau trong cùng môi
trường; bộ kiểm thử tại mốc nghiệm thu EDA đạt **69 tests**. Toàn bộ bộ kiểm
thử hiện tại, gồm kiểm thử xuất tệp đầu ra M2.3a, đạt **94 tests**.

![Quy trình EDA train FD001](assets/diagrams/fd001-eda.svg)

[Nguồn sơ đồ](assets/diagrams/fd001-eda.drawio) ·
[Phương pháp, kết quả và giới hạn](outputs/eda/README.md)

### Chạy notebook thống kê

Sau khi cài `requirements-dev.txt`, mở notebook suy luận chỉ dùng train:

```powershell
.venv\Scripts\python -m jupyterlab notebooks/02_fd001_inference.ipynb
```

Chạy toàn bộ cell sẽ tạo lại `outputs/statistics/tables/engine_sensor_pairs.csv`,
`outputs/statistics/tables/sensor_inference.csv`, `run_info.json` và
`validation.json`, cùng hai hình trong `outputs/statistics/figures/`. Ví dụ,
tệp đầu ra hiện tại có 2.100 cặp động cơ--sensor và 21 dòng suy luận theo
sensor. Sáu sensor hằng toàn tập được giữ với trạng thái
`all_zero_difference`; 15 sensor kiểm định được đều bác bỏ giả thuyết không sau
Benjamini-Hochberg FDR theo protocol đã chốt. Đây là kết quả chỉ dùng train,
không phải lựa chọn đặc trưng (feature selection).

![Quy trình thống kê train FD001](assets/diagrams/fd001-statistics.svg)

[Nguồn draw.io có thể chỉnh sửa](assets/diagrams/fd001-statistics.drawio) ·
[File URL mở bằng trình duyệt](assets/diagrams/fd001-statistics.drawio-url.txt) ·
[Phương pháp, bảng, validation và giới hạn](outputs/statistics/README.md)

Ví dụ: trung vị tuổi thọ của 100 động cơ train là 199 cycles. Sáu sensor hằng
trên toàn train; `sensor_6` biến thiên trên toàn tập nhưng hằng trong 38 động cơ.
Hai hình M2.3b dùng panel riêng cho từng sensor vì các sensor khác thang đo.
`sensor_6` cho thấy p-value nhỏ không đồng nghĩa với thay đổi lớn theo đơn vị
đo; `sensor_9` cho thấy cần đọc phân phối theo engine cùng mean và CI. Đây là
kết quả train-only, chưa phải lựa chọn đặc trưng cho mô hình.

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

Validation tái lập M2.4, ETL Silver/Gold, rolling features, PySpark, mô hình
và dashboard chưa được triển khai. [Bản thảo báo cáo](report/main.pdf) được cập nhật cùng công việc
đã kiểm chứng: nhật ký ghi ngày, lý do, tệp đầu ra, cách kiểm tra và phần còn lại.
Xem [hướng dẫn báo cáo](report/README.md) để biết cấu trúc nguồn và lệnh build
PDF. Bản hiện tại có kết quả validation và một số quan sát EDA ban đầu đã
đối chiếu, chưa có kết quả đánh giá mô hình.

## Bối cảnh học thuật và khai báo AI

Repository hỗ trợ báo cáo cuối kỳ môn **Phân tích dữ liệu lớn**. Báo cáo phân biệt rõ kết quả quan sát được với phần giải thích đề xuất, đồng thời khai báo việc sử dụng AI/LLM theo yêu cầu môn học. Mọi phân tích, mã nguồn, trích dẫn và kết luận có hỗ trợ của AI vẫn phải qua rà soát của con người và kiểm tra khả năng tái lập.

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
