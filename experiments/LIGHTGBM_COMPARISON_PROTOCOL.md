# Protocol so sánh HGB và LightGBM cho dự đoán RUL

**Tác giả:** Phan Nguyễn Mai Phương — **MSHV 25002711**  
**Ngày đóng băng:** 2026-10-05  
**Mục đích:** kiểm tra LightGBM như một đối chứng tree-based độc lập trên đúng ma trận features, engine folds và cách đánh giá đã dùng cho HGB/XGBoost.

## Phạm vi

Study dùng bốn arm:

- `hgb_uncapped`: `HistGradientBoostingRegressor`, huấn luyện trên raw RUL.
- `hgb_cap125`: HGB, chỉ cắt nhãn huấn luyện ở 125.
- `lgbm_uncapped`: `LGBMRegressor`, raw RUL.
- `lgbm_cap125`: LightGBM, chỉ cắt nhãn huấn luyện ở 125.

Mọi validation và test readout đều dùng raw RUL. Việc cắt nhãn không được áp dụng cho truth hoặc metrics.

## Dữ liệu và chia fold

Tất cả arm dùng cùng frozen causal matrix gồm **169 features**: 126 rolling-window features, `cycle` và 42 độ lệch sensor so với 30 cycles đầu. Feature không sử dụng nhãn tương lai. Các engine không giao nhau giữa fit và validation. Outer protocol có 5 folds theo published engine assignments; inner protocol có 3 `GroupKFold` folds; seed là 42.

Sample weight là `engine_total_one`: tổng trọng số của mỗi engine bằng một. Với LightGBM, `reg_lambda` và `min_child_weight` hiệu dụng được ghi cùng giá trị nominal và scale theo `n_fit_engines / n_fit_rows` để giữ quy ước objective mean-one; mỗi fit lưu cả nominal và effective values.

## Cấu hình ứng viên

HGB giữ cấu hình reference: 200 iterations, learning rate 0.05, tối đa 31 leaves, `min_samples_leaf=20`, L2 bằng 0, squared error và không early stopping.

LightGBM dùng `LGBMRegressor` CPU với:

- 200 trees, `learning_rate=0.05`, `num_leaves=31`, `max_depth=-1`;
- `min_child_samples=20`, `min_child_weight=0.001` nominal, `max_bin=255`;
- `reg_lambda` nominal `{0, 10}`, `reg_alpha=0`, `min_split_gain=0`;
- `subsample=1`, `colsample_bytree=1`, `n_jobs=4`, `random_state=42`;
- `deterministic=True`, `force_col_wise=True`, objective `regression`.

Mỗi cap có hai ứng viên LightGBM. Full study có 6 candidates: 18 inner fits cho mỗi outer fold, 20 outer fits, 18 final-inner fits và 4 final models, tổng cộng **132 fits**.

## Chọn ứng viên và gate

Trong từng cap, LightGBM chọn theo pooled trajectory RMSE nhỏ nhất trong các ứng viên có endpoint engine MAE và endpoint mean-engine NASA không xấu hơn HGB cùng cap. Nếu không có ứng viên đạt controls, giữ diagnostic winner nhưng đánh dấu `numeric_controls_met=false`.

Sau đó calibrate warning policy trên inner predictions với threshold `{30,35,40,45,50}`, persistence `{1,3,5}`, lead requirement 30 và target on-time 0.9. Chọn pipeline theo trajectory RMSE với safeguards so với `hgb_uncapped`: endpoint MAE/NASA, on-time rate và early-alarm burden.

Promotion gate được đóng băng trước training. Gate yêu cầu upper 95% paired engine bootstrap CI của delta trajectory RMSE `< 0`, endpoint MAE/NASA và early-alarm burden `<= 0`, đồng thời lower CI của on-time rate `>= 0`. Bootstrap có 10.000 resamples, giữ multiplicity của engine và tính RMSE từ pooled SSE/số dòng. CI là conditional trên predictions cố định, không bao gồm biến thiên do retraining. Các safeguards dùng zero margin. Nhiều arm và metric được báo cáo nhưng không áp dụng multiplicity adjustment across arms; CI phải được đọc trong phạm vi protocol đã đóng băng.

## Thứ tự artifact và test

Runner đóng băng source, lock, input hashes, feature matrix và assignments trước fit. Sau khi final selection và final models được lưu, model được reload và replay prediction exact. Chỉ sau bước đó mới đọc test labels; test là reporting-only và không tham gia selection hoặc gate. Output chính nằm trong `outputs/experiments/lightgbm_comparison/20261005_study01/`; pilot và exploratory runs giữ namespace riêng.

Study này không tự động thay published model hoặc registry. Kết luận chỉ được ghi sau khi full run, verification và paired readouts hoàn tất.

## Giới hạn diễn giải

Engine folds và test labels đã từng được xem trong các nghiên cứu trước, nên đây là follow-up có historical inspection chứ không phải xác nhận độc lập hoàn toàn. Endpoint panel là hồi cứu tại fractions 50%, 70% và 90%; nó không đại diện cho đánh giá online trước khi biết lifetime. Không dùng một số test đẹp để nới gate hoặc chọn lại procedure. Nếu LightGBM không đạt gate thì không cần rerun chỉ để quảng bá ứng viên; chỉ rerun trên Linux khi cần kiểm tra reproducibility với run ID mới.
