# Phân tích PCA--ARIMA trên Health Index FD001

Artifact này là phân tích chuỗi thời gian phụ, không phải mô hình dự đoán RUL.
Pipeline đọc Silver observations và fold assignment của M3.3, lọc train, fit
`StandardScaler` và PCA một thành phần trên 21 sensor, rồi định hướng PC1 để
Health Index có xu hướng giảm theo cycle. Mười engine được chọn trước forecast
bằng quota cân bằng theo GroupKFold fold và vị trí lifetime cách đều trong từng
fold.

## Cách chạy

```powershell
.venv\Scripts\python -m cmapss.timeseries `
  --observations outputs/etl/silver/observations.parquet `
  --fold-assignments outputs/modeling/tables/fold_assignments.csv `
  --output-dir outputs/timeseries
```

Mỗi chuỗi được chia theo thứ tự cycle: 70% đầu để fit và 30% cuối để forecast.
ARIMA order được chọn bằng grid AIC chỉ trên fit segment với `p,q` từ 0 đến 2
và `d` thuộc `{0,1}`. Artifact ghi MAE, RMSE, coverage của prediction interval
95%, ADF trên fit segment và Ljung--Box trên residuals. Ví dụ, `engine_id=39`
có 89 cycles fit, 39 cycles holdout và order `(1, 0, 1)`.

## Artifact

| Tệp | Nội dung |
| --- | --- |
| `tables/selected_engines.csv` | 10 engine train được chọn trước forecast |
| `tables/health_index.csv` | 20.631 Health Index train observations |
| `tables/forecast_metrics.csv` | MAE, RMSE, coverage và trạng thái theo engine |
| `tables/diagnostics.csv` | ADF và Ljung--Box theo engine |
| `predictions/arima_forecasts.parquet` | 704 dòng holdout forecast và khoảng dự đoán |
| `figures/health_index_selected_engines.png` | Quỹ đạo Health Index của 10 engine |
| `figures/arima_forecasts.png` | Actual, forecast và prediction interval |
| `models/health_index_scaler.joblib` | Scaler fit trên train |
| `models/health_index_pca.joblib` | PCA fit trên train |
| `run_info.json` | Cấu hình, checksum, row counts và ranh giới cô lập |

## Kết quả đã kiểm chứng

Cả 10 engine đều fit thành công. Coverage 95% theo engine nằm trong khoảng
21,43%--97,03%, còn MAE Health Index nằm trong khoảng 0,1518--1,0396. ADF
không bác bỏ tính không dừng trên cả 10 fit segment (`p-value` từ 0,8609 đến
0,9990); Ljung--Box không cho thấy bằng chứng mạnh về tự tương quan residual ở
mức 5% trong các chuỗi đã chọn. Đây là chẩn đoán của Health Index forecast,
không phải kết luận về RUL hoặc lịch bảo trì.

Pipeline không đọc test labels và không dùng forecast để đổi danh sách engine.
Health Index cũng không được đánh đồng với RUL: PC1 là biểu diễn tổng hợp của
sensor sau chuẩn hóa, còn RUL là số cycle còn lại theo nhãn FD001.
