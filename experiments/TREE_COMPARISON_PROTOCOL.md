**Phan Nguyễn Mai Phương — MSHV 25002711**

# Protocol so sánh HGB và XGBoost cho dự đoán RUL

## Kết luận trước

Đây là protocol (quy trình thực nghiệm) cố định để so sánh bốn arm trên cùng một ma trận **169 features** và cùng cách chia theo `engine_id`:

- `hgb_uncapped`: `HistGradientBoostingRegressor`, huấn luyện trên raw RUL.
- `hgb_cap125`: HGB, chỉ cap nhãn huấn luyện ở 125 cycles.
- `xgb_uncapped`: XGBoost, chọn một candidate trong grid đã khai báo trước, huấn luyện trên raw RUL.
- `xgb_cap125`: XGBoost, chọn một candidate trong cùng grid nhưng cap nhãn huấn luyện ở 125 cycles.

Mỗi arm được đánh giá bằng nested grouped cross-validation: **5 outer folds và 3 inner folds** trong cấu hình khoa học. Mọi phép đo dùng raw RUL của validation. Nhãn test chỉ được đọc sau khi đã lưu selection, bốn model cuối, prediction cuối và các checksum liên quan. Kết quả smoke run chỉ kiểm tra luồng chạy; chưa được dùng để kết luận arm nào tốt hơn.

Outer folds và test của FD001 đã được xem trong các đợt nghiên cứu trước. Đợt này là nghiên cứu tiếp nối; chưa phải xác nhận độc lập trên dữ liệu mới. HGB dùng cấu hình cố định, còn XGBoost được tìm trong 12 cấu hình mỗi cap; kết quả chỉ áp dụng cho các quy trình huấn luyện này.

Protocol không tự thay thế model hoặc registry đã công bố. Một arm chỉ được mô tả là phù hợp để promotion khi đạt đầy đủ promotion gate trên outer predictions của cấu hình khoa học.

## Mục tiêu và phạm vi

Protocol trả lời câu hỏi sau:

> Khi giữ nguyên dữ liệu, 169 features, trọng số, engine folds và thang đo raw RUL, XGBoost có cải thiện dự đoán toàn quỹ đạo và vẫn giữ các kiểm soát endpoint, NASA score và cảnh báo so với HGB hay không?

Phạm vi gồm:

1. Xác nhận dữ liệu và targets train.
2. Xây và đóng băng một feature matrix duy nhất.
3. Huấn luyện bốn arm với engine-disjoint folds.
4. Chọn XGBoost candidate trong inner OOF, sau đó chọn procedure trong inner OOF.
5. Đánh giá outer predictions bằng các metric đã định trước.
6. Fit bốn model trên toàn bộ train sau khi selection cuối được đóng băng.
7. Báo cáo test theo từng arm sau cùng, với mục đích reporting-only.

TCN, LSTM và LightGBM không thuộc protocol này. Có thể thực hiện ở một protocol riêng nếu giữ nguyên quy tắc về frozen features, engine split, raw-RUL evaluation và test ordering.

## Dữ liệu và feature contract

### Nguồn dữ liệu

Runner mặc định đọc:

```text
outputs/etl/silver/observations.parquet
outputs/etl/silver/train_targets.parquet
data/raw/RUL_FD001.txt
```

`observations.parquet` phải có đúng hai split `train` và `test`. `train_targets.parquet` phải chứa raw RUL cho toàn bộ observation train và được kiểm tra bằng `validate_train_targets`. Test observations không được ghép với test RUL trong giai đoạn feature building hoặc model selection.

### Frozen feature matrix

`audit_study.build_study_features` được gọi một lần trước khi khởi động worker. Matrix train được lưu với:

```text
KEY = (split, engine_id, cycle)
FEATURES = 169 features
rul = raw train target
```

`cycle` nằm trong `KEY` và cũng là một feature; khi lưu parquet, cột này chỉ xuất hiện một lần. Matrix test có `KEY` và 169 features, không có `rul`.

Feature set gồm rolling features hiện hành và nhóm history features theo `cycle` cùng các thống kê sensor tương đối với 30 cycles đầu. Các feature phải hữu hạn, có khóa duy nhất và giữ đúng thứ tự đã khai báo trong `cmapss.modeling.tree_comparison.contracts.FEATURES`.

Mọi worker chỉ đọc `frozen_train_features.parquet`. Worker không được tự xây lại features, đọc test labels hoặc dùng lifetime để tạo input model. Feature matrix, targets, source files, dependency lock, project metadata, phiên bản thư viện và outer fold assignments được hash trong `manifest.json`.

Ví dụ contract train sau khi đóng băng:

```text
split | engine_id | cycle | ...169 FEATURES... | rul
train | 1         | 30    | ...                 | 190
```

Cap `125` chỉ áp dụng cho `rul` dùng trong `fit`. `y_true` của validation và test luôn là raw RUL.

## Cấu hình khoa học

`StudyConfig` mặc định có các giá trị sau:

| Nhóm | Giá trị | Ý nghĩa |
|---|---:|---|
| `outer_splits` | `5` | Outer folds theo engine |
| `inner_splits` | `3` | Inner folds theo engine |
| `seed` | `42` | Seed cho model và bootstrap |
| `fit_threads` | `4` | Native fitting threads mỗi worker |
| `max_workers` | `2` | Process workers chạy song song |
| `hgb_max_iter` | `200` | Số iteration HGB |
| `xgb_n_estimators` | `300` | Số cây XGBoost |
| `bootstrap_resamples` | `10,000` | Số paired engine bootstrap draws |
| `endpoint_fractions` | `(0.5, 0.7, 0.9)` | Panel endpoint retrospective |
| `thresholds` | `(30, 35, 40, 45, 50)` | Grid predicted-RUL warning |
| `persistences` | `(1, 3, 5)` | Số lần xác nhận liên tiếp |
| `lead_requirement` | `30` cycles | Mốc lead minh họa |
| `target_on_time_rate` | `0.9` | Mục tiêu on-time trong policy calibration |
| các margin | `0.0` | So sánh không cho phép suy giảm |

`StudyConfig.scientific` chỉ là `True` khi giữ đầy đủ protocol trên. `--dry-run` cố ý dùng cấu hình rút gọn và không được dùng cho kết luận khoa học.

## Candidate grid và cách fit

### HGB reference

Hai candidate HGB là fixed reference:

- `hgb_uncapped`: `target_cap=None`.
- `hgb_cap125`: `target_cap=125`.

Cả hai dùng:

```text
loss = squared_error
learning_rate = 0.05
max_iter = 200 trong scientific run
max_leaf_nodes = 31
max_depth = None
min_samples_leaf = 20
l2_regularization = 0.0
early_stopping = False
```

### XGBoost grid

Mỗi family XGBoost có 12 candidate:

```text
max_depth     ∈ {3, 5, 7}
learning_rate ∈ {0.03, 0.1}
reg_lambda    ∈ {1.0, 10.0}
```

Hai family là `xgb_uncapped` và `xgb_cap125`, tổng cộng **24 XGBoost candidates**. Cộng hai HGB reference, protocol có **26 specs**. Candidate ID có dạng:

```text
xgb_uncapped_d3_lr0.1_l21
xgb_cap125_d7_lr0.03_l210
```

Các tham số chung của XGBoost gồm `tree_method="hist"`, `device="cpu"`, `objective="reg:squarederror"`, `subsample=1.0`, `colsample_bytree=1.0`, `reg_alpha=0.0`, `gamma=0.0`, `n_jobs=fit_threads`, `random_state=42` và `eval_metric="rmse"`.

Sample weights của mọi tree model dùng `engine_total_one`: tổng trọng số của mỗi engine bằng một. Với XGBoost, `min_child_weight` và `reg_lambda` nominal được scale theo `n_engines / n_rows` để tương thích với thang trọng số. Metadata của mỗi fit ghi cả nominal và effective parameters.

Runtime pin XGBoost là **3.4.1**. Tài liệu stable của XGBoost đang tham chiếu API **3.4.2** có thể chứa khác biệt nhỏ ở bản mới hơn; khi tái lập, ưu tiên runtime pin trong `uv.lock` và artifact metadata:

- [XGBoost 3.4.2: Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [XGBoost 3.4.2: tree booster parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [XGBoost 3.4.1: package và wheels trên PyPI](https://pypi.org/project/xgboost/3.4.1/)

## Nested grouped evaluation

### Outer folds

Scientific run dùng assignments đã tồn tại tại:

```text
outputs/modeling/tables/fold_assignments.csv
```

Mỗi `engine_id` thuộc đúng một outer fold. Nếu cấu hình không dùng 5 outer folds, runner tạo assignments bằng `GroupKFold` cho smoke/custom run. Không có observation của cùng engine đi qua cả fit và validation.

### Inner OOF

Với mỗi outer fold:

1. Lấy riêng outer-train engines.
2. Chia chúng thành 3 inner folds bằng `GroupKFold`.
3. Fit toàn bộ candidates trên cùng các inner fitting engines.
4. Gom inner validation predictions thành inner OOF.
5. Đánh giá HGB references và từng XGBoost candidate trên cùng vector raw RUL.

XGBoost candidate trong mỗi family được chọn theo thứ tự:

1. Giữ candidate thỏa endpoint engine MAE và endpoint engine NASA score không xấu hơn HGB tương ứng trong margin đã khai báo.
2. Trong nhóm còn lại, tối thiểu hóa pooled trajectory RMSE.
3. Hòa thì dùng endpoint MAE rồi `candidate_id`.
4. Nếu không có candidate thỏa numeric controls, vẫn lưu winner unconstrained để báo cáo, nhưng đánh dấu `numeric_controls_met=False`; candidate đó không đủ điều kiện promotion.

Sau khi có bốn arm đã chọn (`hgb_uncapped`, `hgb_cap125`, `xgb_uncapped`, `xgb_cap125`), runner calibrate warning policy trên inner OOF. Grid policy dùng threshold và persistence đã đóng băng. Procedure cuối được chọn bằng trajectory RMSE, endpoint MAE/NASA và warning controls so với `hgb_uncapped`; XGBoost còn phải giữ family numeric control.

Outer validation chỉ dùng để đo hiệu năng của procedure đã chọn từ inner data. Không dùng outer metric để chọn candidate, policy hoặc procedure.

### Metric và warning readout

Mỗi prediction vector được tính lại:

- trajectory RMSE trên toàn bộ raw-RUL trajectory;
- trajectory mean engine MAE và NASA score;
- endpoint mean engine MAE và NASA score tại fractions `0.5`, `0.7`, `0.9` của lifetime retrospective;
- on-time rate, early alarm burden, transitions và retractions.

Endpoint fractions chỉ định vị cycle đánh giá sau khi có trajectory hoàn chỉnh. Chúng không đi vào model input và không đại diện cho một test distribution chưa thấy.

Warning policy là minh họa cho đánh giá protocol, không phải maintenance SLA. `lead_requirement=30` nghĩa là kiểm tra khả năng báo trước tối thiểu 30 cycles theo định nghĩa trong audit implementation.

## Bootstrap và promotion gate

`paired_comparisons` resample theo `engine_id`, giữ nguyên multiplicity khi một engine được chọn nhiều lần. Khoảng tin cậy là conditional trên outer predictions đã lưu; nó **không** bao gồm biến thiên do refit model.

Trajectory RMSE được tính bằng căn bậc hai của tổng SSE chia tổng số dòng. Khi bootstrap lấy lặp một engine, cả SSE và số dòng của engine đó được tính lặp tương ứng. Các metric endpoint và warning dùng trung bình theo engine. Đây là các khoảng tin cậy riêng lẻ, chưa hiệu chỉnh multiplicity cho nhiều arm hoặc nhiều metric.

Promotion gate của `selected_pipeline` so với `hgb_uncapped` yêu cầu đồng thời:

- trajectory RMSE cải thiện: upper 95% CI của delta `< 0`;
- endpoint engine MAE không xấu hơn: upper 95% CI `<= 0`;
- endpoint engine NASA không xấu hơn: upper 95% CI `<= 0`;
- warning on-time không xấu hơn: lower 95% CI `>= 0`;
- early alarm burden không xấu hơn: upper 95% CI `<= 0`;
- cấu hình phải là scientific protocol đầy đủ.

`hgb_cap125` là capped reference riêng. Capped-reference controls được ghi trong decision nhưng không thay đổi primary gate. Test metrics không tham gia gate.

## Final fit và test ordering

Sau khi hoàn tất outer readouts:

1. Chạy final inner grouped CV trên toàn bộ train engines.
2. Đóng băng `selection.json`, gồm bốn selected specs, policy, procedure và outer decision.
3. Fit đúng bốn selected arm trên toàn bộ frozen train matrix.
4. Lưu từng `model.joblib` và `model.json`.
5. Reload từng model và yêu cầu prediction replay exact (`np.array_equal`).
6. Tạo `final_predictions.parquet` cho cycle cuối của từng test engine và lưu receipt/checksum.
7. Chỉ sau đó mới đọc `data/raw/RUL_FD001.txt` để tạo reporting-only test evaluation.

Test labels được đọc theo procedure sau khi các bước trên đã hoàn tất. Nếu thiếu model, selection, prediction receipt hoặc checksum không khớp, runner dừng trước khi mở test labels.

## Cây package và luồng module

```text
src/cmapss/modeling/tree_comparison/
├── __init__.py       # public contracts và estimator exports
├── __main__.py       # python -m entrypoint
├── contracts.py      # StudyConfig, TreeSpec, FEATURES, specs()
├── estimators.py     # FittedTree, weights, HGB/XGB fit
├── cache.py          # frozen matrix LRU và process-safe fit cache
├── analysis.py       # readout, XGB selection, policy, gate
└── runner.py         # freeze, nested scheduling, final fit, reporting

scripts/
└── run_tree_comparison.py  # wrapper đặt src/ vào PYTHONPATH
```

Luồng chính:

```text
observations + train targets
        ↓ validate và build_study_features một lần
frozen_train_features.parquet (KEY + 169 features + raw rul)
        ↓ retained ProcessPool, worker chỉ nhận path + engine IDs
fit cache → inner OOF → XGB family controls → warning policies
        ↓
outer predictions/events → paired comparisons → promotion decision
        ↓
final inner selection → 4 final models → final test predictions
        ↓
test labels đọc sau cùng → reporting-only evaluation
```

## Lệnh chạy trên Linux

Từ repository root:

```bash
uv sync --frozen
```

Smoke run rút gọn (không dùng để kết luận):

```bash
uv run --frozen python -m cmapss.modeling.tree_comparison \
  --output-dir outputs/experiments/tree_comparison/20261005_linux_smoke \
  --dry-run \
  --skip-test-evaluation \
  --fit-threads 4 \
  --max-workers 2
```

Scientific run đầy đủ:

```bash
uv run --frozen python -m cmapss.modeling.tree_comparison \
  --output-dir outputs/experiments/tree_comparison/20261005_linux_xgb \
  --fit-threads 4 \
  --max-workers 2
```

`--output-dir` là bắt buộc để tránh ghi đè run khác. Có thể dùng wrapper tương đương:

```bash
uv run --frozen python scripts/run_tree_comparison.py \
  --output-dir outputs/experiments/tree_comparison/20261005_linux_xgb
```

Kiểm tra artifacts sau khi run hoàn tất:

```bash
uv run --frozen python scripts/verify_tree_comparison.py \
  --output-dir outputs/experiments/tree_comparison/20261005_linux_xgb
```

Dùng run ID mới trên Linux; không resume cache Windows khi source hoặc lock identity khác.

Runner hỗ trợ resume khi manifest, source, inputs, versions, frozen matrix, folds và configuration còn tương thích. Không tiếp tục một thư mục non-empty không có manifest hợp lệ.

## Bản đồ output

Một run riêng dưới `outputs/experiments/tree_comparison/<run_id>/` có các nhóm artifact sau:

| Artifact | Nội dung |
|---|---|
| `manifest.json` | author, config, specs, versions, hashes, split contract |
| `source_snapshot/` | mã nguồn local và `uv.lock`/`pyproject.toml` dùng trong run |
| `input_snapshot/` | bản sao observations và train targets đã hash |
| `frozen_train_features.parquet` | ma trận train duy nhất cho mọi worker |
| `frozen_test_features.parquet` | test features không có RUL |
| `outer_assignments.csv` | mapping engine → outer fold |
| `fit_cache/` | prediction parquet và metadata cho từng candidate/fold |
| `stages/outer_<n>/` | inner assignments, candidate metrics, policy metrics, selection |
| `outer_predictions.parquet` | predictions của bốn arm và `selected_pipeline` |
| `outer_endpoints.parquet` | endpoint rows tại fractions đã định trước |
| `warning_events.parquet` | event readout theo arm/fold |
| `aggregate_metrics.csv` | trajectory, endpoint và warning summaries |
| `engine_metrics.csv` | metric theo engine dùng cho paired bootstrap |
| `paired_comparison.csv` | delta, CI và scope của bootstrap |
| `decision.json` | promotion controls và giới hạn đánh giá |
| `selection.json` | final inner specs, policies và procedure |
| `models/<arm>/` | `model.joblib` và metadata/checksum |
| `final_predictions.parquet` | endpoint prediction cuối trên test features |
| `final_predictions_receipt.json` | checksum và bằng chứng replay trước test read |
| `test/<arm>/` | test predictions/evaluation/metrics, reporting-only |
| `test_evaluation.json` | test labels hash và cờ reporting-only |

Các artifact cũ trong `outputs/modeling/` không bị thay thế. Run tree comparison phải được review riêng trước khi đưa bất kỳ model nào vào registry hoặc báo cáo chính thức.

## Giới hạn và quy tắc diễn giải

- Không gọi smoke run là scientific result.
- Không chọn model bằng một số endpoint đẹp nếu trajectory RMSE và các controls không đạt.
- Không diễn giải conditional bootstrap như uncertainty của quá trình retraining.
- Không xem endpoint retrospective là đánh giá online tại thời điểm chưa biết lifetime.
- Không dùng test labels để chọn candidate, policy, procedure hoặc thay đổi outer gate.
- Không gộp HGB cap125 và uncapped thành một kết luận về kiến trúc; cap nhãn là một biến can thiệp riêng.
- Không so sánh XGBoost với HGB nếu thay đổi đồng thời features, folds, weights hoặc raw-RUL evaluation.

Tài liệu này mô tả protocol và artifact contract. Kết luận mô hình nào được ưu tiên chỉ được ghi sau khi scientific run hoàn tất và các file `decision.json`, `paired_comparison.csv`, test reporting và provenance đã được kiểm tra.

