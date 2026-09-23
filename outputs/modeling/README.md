# RUL modeling FD001

Mốc modeling huấn luyện và đánh giá ba ứng viên dự đoán RUL từ artifact rolling
features M3.2. Pipeline dùng 5-fold GroupKFold theo `engine_id`, chọn model theo
mean engine-level MAE trên out-of-fold validation và chỉ đọc `RUL_FD001.txt` sau
khi đã ghi `final_predictions.parquet`.

## Cách chạy

Từ thư mục gốc, sau khi đã có ETL M3.1 và rolling features M3.2:

```powershell
.venv\Scripts\python -m cmapss.modeling `
  --rolling-features outputs/features/gold/rolling_features.parquet `
  --train-targets outputs/etl/silver/train_targets.parquet `
  --test-labels data/raw/RUL_FD001.txt `
  --output-dir outputs/modeling
```

Lệnh hiện tại chọn `hist_gradient_boosting` và in:

```text
selected_model=hist_gradient_boosting final_engine_mae=15.349047 rmse=20.541267 nasa_score=1513.257656
```

## Artifact

| Tệp | Nội dung | Số dòng |
| --- | --- | ---: |
| `tables/fold_assignments.csv` | Engine train thuộc fold validation nào | 100 |
| `tables/fold_metrics.csv` | Metric theo model và fold | 15 |
| `tables/model_metrics.csv` | Metric trung bình để chọn model | 3 |
| `predictions/oof_predictions.parquet` | Out-of-fold predictions cho 3 model | 53.193 |
| `predictions/final_predictions.parquet` | 100 prediction test, ghi trước khi join nhãn | 100 |
| `predictions/final_evaluation.parquet` | Prediction đã join test labels để đánh giá cuối | 100 |
| `tables/final_metrics.json` | Metric cuối và bootstrap CI | 1 |
| `models/selected_model.joblib` | Model đã refit trên toàn bộ train | 1 |
| `figures/final_true_vs_predicted.png` | Scatter RUL thật và dự đoán | 1 |
| `figures/final_residuals_by_rul.png` | Sai số dự đoán theo RUL thật | 1 |

## Kết quả hiện tại

Cross-validation dùng 17.731 train windows, giữ 21 sensors và 126 feature
columns. Tổng trọng số của mỗi engine trong training fold bằng 1 để engine dài
không lấn át engine ngắn.

| Model | Mean engine MAE | RMSE trung bình | Được chọn |
| --- | ---: | ---: | --- |
| `median_baseline` | 46,9543 | 62,9006 | Không |
| `ridge_alpha_1` | 23,9922 | 35,9467 | Không |
| `hist_gradient_boosting` | 20,2143 | 33,5479 | Có |

Đánh giá cuối trên 100 test engines:

| Metric | Giá trị | Bootstrap CI 95% |
| --- | ---: | --- |
| Mean engine MAE | 15,3490 | [13,2594; 17,3941] |
| RMSE | 20,5413 | [17,3442; 23,5850] |
| NASA score | 1.513,2577 | [743,4999; 2.476,2743] |

Sai số trung bình thấp nhất ở nhóm RUL thật 0--30 cycles và cao hơn ở các engine
còn nhiều RUL. Đây là kết quả trên FD001 mô phỏng, chưa chứng minh khả năng khái
quát hóa sang điều kiện vận hành hoặc fault mode khác.

## Kiểm tra chống leakage

- Fold assignment dùng `engine_id` làm group; mỗi engine chỉ xuất hiện trong một
  validation fold.
- Scaler của Ridge nằm trong pipeline và chỉ fit trên training fold.
- Test labels không được đọc trong cross-validation hoặc chọn model.
- `final_predictions.parquet` có đúng 100 dòng và được ghi trước bước join nhãn.
- `run_info.json` lưu input SHA-256, model được chọn, cấu hình và trạng thái
  `completed_after_final_predictions` cho bước đọc test labels.

## Giới hạn

- Phân tích PCA--ARIMA train-only đã được ghi tại
  [`outputs/timeseries`](../timeseries/README.md); dashboard đọc các artifact
  modeling và timeseries tại `app.py`. Phân tích lỗi theo sensor vẫn chưa có.
- Có thể chạy `python -m cmapss.reproducibility` để tái tạo toàn bộ chuỗi từ raw
  vào output root mới và đối chiếu metric, bảng, hình cùng dashboard loader.
- Chưa cắt prediction âm vì lần chạy hiện tại không sinh giá trị âm; config ghi
  rõ `prediction_clipping=none`.
- NASA score trong cross-validation là tổng theo window validation nên chỉ dùng
  để so sánh tương đối giữa model trong cùng protocol. Metric cuối dùng 100 dòng
  test engine-level.
