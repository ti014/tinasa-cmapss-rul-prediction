# Gói bàn giao cuối — NASA C-MAPSS FD001

## Phạm vi

Repository hiện có pipeline Pandas từ raw FD001 đến artifact ETL, rolling
features, grouped RUL modeling, PCA--ARIMA Health Index và dashboard
read-only. Bộ slide trình bày nằm tại
`slides/cmapss_fd001_presentation.pptx`.

Kết quả chỉ được diễn giải trong phạm vi FD001: 100 train engines, 100 test
engines, một điều kiện vận hành và một dạng lỗi. Health Index forecast là phân
tích phụ, không phải dự báo RUL.

## Cấu trúc bàn giao

```text
slides/
├── cmapss_fd001_presentation.pptx
└── README.md
handoff/
├── FINAL_HANDOFF.md
└── DEMO_SCRIPT.md
report/main.pdf
outputs/{etl,features,modeling,timeseries}/
src/cmapss/{etl,features,modeling,timeseries,reproducibility.py}
tests/
```

## Lệnh kiểm tra chính

Kịch bản trình bày trực tiếp nằm tại [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md).

Tái lập toàn bộ pipeline vào output root mới:

```powershell
uv run python -m cmapss.reproducibility `
  --output-root tmp/e2e-repro-final `
  --repo-root . `
  --evidence-path tmp/e2e-repro-final/validation.json
```

Kiểm tra test và cú pháp:

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp=tmp/pytest-handoff --tb=short
.venv\Scripts\python.exe -m compileall -q src tests
uv lock --check
```

Mốc end-to-end ngày 2026-09-23 đã đối chiếu 15 bảng, metrics, bốn hình
byte-identical, raw hashes và dashboard loader. Full suite đạt 123 tests.

## Kết quả cần trình bày

Model được chọn là `hist_gradient_boosting` bằng 5-fold `GroupKFold` theo
`engine_id`. Trên 100 test engines:

| Metric | Giá trị |
| --- | ---: |
| Mean engine MAE | 15.3490 |
| RMSE | 20.5413 |
| NASA score | 1513.2577 |
| Bootstrap 95% CI của MAE | [13.2594, 17.3941] |

Rolling features dùng window 30 cycles, tạo 27.927 windows và 126 feature
columns. PCA--ARIMA dùng 10 validation engines, split 70/30 và tạo 704 forecast
rows; diagnostics gồm ADF, Ljung--Box và prediction interval coverage.

## Câu hỏi phản biện và câu trả lời ngắn

1. **Vì sao chia theo engine?** Vì các window trong cùng engine có phụ thuộc
   theo thời gian; chia ngẫu nhiên theo row sẽ làm metric lạc quan.
2. **Có data leakage không?** Scaler, PCA, model selection và GroupKFold chỉ
   học trên train/fold; test labels chỉ được join sau final predictions.
3. **Vì sao dùng HistGradientBoosting?** Nó đạt mean engine MAE thấp nhất trong
   protocol đã chốt: 20.2143 ở cross-validation, so với Ridge 23.9922 và
   median baseline 46.9543.
4. **ARIMA có dự báo RUL không?** Không. ARIMA chỉ forecast Health Index và
   được trình bày như phân tích time-series phụ.
5. **Kết quả có áp dụng cho động cơ thực tế không?** Chưa thể kết luận. FD001
   chỉ đại diện một điều kiện vận hành và một dạng lỗi mô phỏng.
6. **PySpark có thay thế Pandas trong modeling không?** PySpark được dùng cho
   parity và benchmark ETL; modeling khoa học chính vẫn chạy bằng Pandas và
   scikit-learn.
7. **Tái lập được bằng chứng nào?** Lệnh reproducibility dựng lại output trong
   thư mục mới, so sánh nội dung bảng với tolerance `1e-9`, kiểm raw hashes và
   đọc lại artifact bằng dashboard loader.

## Giới hạn còn lại

- Chưa có serving API.
- Chưa đánh giá trên bộ dữ liệu độc lập hoặc nhiều fault mode.
- PySpark parity và notebook reproducibility là các kiểm tra riêng với lệnh
  end-to-end Pandas.
- Dashboard đọc artifact đã sinh; không tự huấn luyện lại model.
