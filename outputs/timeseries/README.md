# Phân tích PCA--ARIMA trên Health Index FD001

Artifact này là phân tích chuỗi thời gian phụ, không phải mô hình dự đoán RUL.
Pipeline đọc Silver observations và fold assignment của M3.3, lọc train, chọn
10 engine trước forecast bằng quota cân bằng theo GroupKFold fold và vị trí
lifetime cách đều trong từng fold. `StandardScaler`, PCA một thành phần, hướng
PC1 và tham số chuẩn hóa Health Index được fit trên 90 train engines còn lại;
10 engine được chọn chỉ được transform.

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
95%, ADF trên fit segment và Ljung--Box trên residuals. Hai point-forecast
baseline được chạy trên đúng prefix/holdout: `last_value` giữ giá trị cuối
prefix, còn `drift` kéo dài mức thay đổi trung bình từ đầu đến cuối prefix.
Baseline không có prediction interval nên chỉ so MAE và RMSE với ARIMA. Ví dụ,
`engine_id=39` có 89 cycles fit, 39 cycles holdout và order `(0, 1, 1)`.

## Artifact

| Tệp | Nội dung |
| --- | --- |
| `tables/selected_engines.csv` | 10 engine train được chọn trước forecast |
| `tables/health_index.csv` | 20.631 Health Index train observations |
| `tables/forecast_metrics.csv` | MAE, RMSE, coverage và trạng thái theo engine |
| `tables/baseline_metrics.csv` | MAE/RMSE của `last_value` và `drift` trên cùng holdout |
| `tables/diagnostics.csv` | ADF và Ljung--Box theo engine |
| `predictions/arima_forecasts.parquet` | 704 dòng holdout forecast và khoảng dự đoán |
| `figures/health_index_selected_engines.png` | Quỹ đạo Health Index của 10 engine |
| `figures/arima_forecasts.png` | Actual, forecast và prediction interval |
| `models/health_index_scaler.joblib` | Scaler fit trên 90 engine train không được chọn |
| `models/health_index_pca.joblib` | PCA fit trên 90 engine train không được chọn |
| `models/health_index_parameters.json` | Hướng, mean/scale chuẩn hóa và danh sách engine fit |
| `run_info.json` | Cấu hình, checksum, row counts và ranh giới cô lập |

## Kết quả đã kiểm chứng

Cả 10 engine đều có trạng thái fit hội tụ. Coverage 95% theo engine nằm trong
khoảng 21,43%--97,03%, còn MAE Health Index của ARIMA nằm trong khoảng
0,1515--1,0372. Trung bình theo 10 engine, ARIMA đạt MAE 0,4110 và RMSE
0,5027; `drift` đạt 0,7469 và 0,9128; `last_value` đạt 0,9478 và 1,1553.
ARIMA có MAE thấp hơn `drift` trên 7/10 engine và thấp hơn `last_value` trên
8/10 engine; đây là so sánh mô tả trên cùng holdout, chưa phải kiểm định ưu thế
thống kê.
ADF không bác bỏ tính không dừng trên cả 10 fit segment (`p-value` từ 0,8606
đến 0,9990). Ljung--Box được điều chỉnh theo `p+q` và cho p-value dưới 0,05 ở
engine 69 và 70; vì vậy không thể kết luận residual của toàn bộ chuỗi đã chọn
không còn tự tương quan. Đây là chẩn đoán của Health Index forecast, không phải
kết luận về RUL hoặc lịch bảo trì.

Pipeline không đọc test labels và không dùng forecast để đổi danh sách engine.
Health Index cũng không được đánh đồng với RUL: PC1 là biểu diễn tổng hợp của
sensor sau chuẩn hóa, còn RUL là số cycle còn lại theo nhãn FD001.
