# Gói bàn giao cuối — NASA C-MAPSS FD001--FD004

## Phạm vi

Repository hiện có pipeline Pandas từ raw FD001--FD004 đến artifact ETL, rolling
features, grouped RUL modeling, PCA--ARIMA Health Index và dashboard
read-only. Bộ slide trình bày nằm tại
`slides/cmapss_fd001_presentation.pptx`.

Deck 15 slide được cập nhật ngày 25/09/2026 theo mạch bài thuyết trình học
thuật, gồm bài toán, dữ liệu, protocol, phương pháp, kết quả RUL, phân tích lỗi,
Health Index, cảnh báo sớm, tái lập, giới hạn và ma trận transfer FD001--FD004.
Finalizer đạt package integrity, layout, font và import; 15 slide đã được render
để kiểm tra nội dung và bố cục.
Bản 10 slide trước redesign được giữ tại
`slides/archive/decks/cmapss_fd001_presentation_before_academic_redesign.pptx`.

Kết quả RUL chính vẫn được diễn giải trong phạm vi FD001: 100 train engines, 100
test engines, một điều kiện vận hành và một dạng lỗi. Ma trận bổ sung dùng
window 10 cycles để kiểm tra bốn subset; Health Index forecast là phân tích phụ,
không phải dự báo RUL.

## Cấu trúc bàn giao

```text
slides/
├── cmapss_fd001_presentation.pptx
├── archive/decks/                 # Bản deck cũ để đối chiếu
├── cmapss_fd001_presentation.tex
└── cmapss_fd001_presentation.pdf
├── build_academic_presentation.mjs
├── finalize_academic_presentation.mjs
├── archive/builders/              # Script build cũ
└── README.md
handoff/
├── FINAL_HANDOFF.md
└── DEMO_SCRIPT.md
report/main.pdf
handoff/SPEECH_SCRIPT_VI.tex
handoff/SPEECH_SCRIPT_VI.pdf
outputs/{etl,features,modeling,timeseries,cross_dataset}/
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

Mốc end-to-end ngày 2026-09-24 đã đối chiếu 18 bảng, metrics, năm hình
byte-identical, raw hashes và dashboard loader. Lượt nghiệm thu timing tiếp
theo tại `tmp/e2e-review-20260924` cũng đạt; full suite hiện tại đạt 154 tests.

Kiểm tra ngày 2026-09-24 gồm loader và trang Cảnh báo sớm trên browser thật
ở localhost:8517; đã quan sát bảng timing và phân phối lead time. Không gọi
kiểm tra loader là rehearsal toàn bộ năm view.
Case demo Engine 023 hiển thị RUL thật 113, dự đoán 170.4768 và absolute error
57.4768; Engine 070 hiển thị Health Index MAE 1.0372, RMSE 1.2205 và coverage
95% là 21.43%. View Provenance hiển thị 100 evaluation rows, 704 forecast rows
và `test_label_join.status=completed_after_final_predictions`.

## Kết quả cần trình bày

Model được chọn là `hist_gradient_boosting` bằng 5-fold `GroupKFold` theo
`engine_id`. Trên 100 test engines:

| Metric | Giá trị |
| --- | ---: |
| Mean engine MAE | 15.3490 |
| RMSE | 20.5413 |
| NASA score | 1513.2577 |
| Bootstrap 95% CI của MAE | [12.7286, 18.0866] |

Rolling features dùng window 30 cycles, tạo 27.927 windows và 126 feature
columns. PCA--ARIMA dùng 10 validation engines, split 70/30 và tạo 704 forecast
rows; diagnostics gồm ADF, Ljung--Box và prediction interval coverage.
Cảnh báo RUL 30 cycles: 100/100 engine từng cảnh báo; timing là 44 trước,
7 đúng và 49 sau ngưỡng. Lead time: median 30, Q1--Q3 26--35, min--max
19--55 cycles. OOF cũng dùng chọn model; đây là mô tả nội bộ. False-alarm
rate cấp engine là null theo định nghĩa event toàn trajectory, không phải 0%.

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
   chỉ đại diện một điều kiện vận hành và một dạng lỗi mô phỏng; transfer sang
   FD002--FD004 có MAE 47,3949--60,3215 cycles trong matrix window 10.
6. **PySpark có thay thế Pandas trong modeling không?** PySpark được dùng cho
   parity ETL; runtime benchmark chỉ đo Pandas. Modeling vẫn chạy bằng Pandas và
   scikit-learn.
7. **Tái lập được bằng chứng nào?** Lệnh reproducibility dựng lại output trong
   thư mục mới, so sánh nội dung bảng với tolerance `1e-9`, kiểm raw hashes và
   đọc lại artifact bằng dashboard loader.

## Giới hạn còn lại

- Chưa có serving API.
- Chưa có telemetry thực tế; transfer C-MAPSS đã được đánh giá trong
  `outputs/cross_dataset/` nhưng vẫn là dữ liệu mô phỏng.
- PySpark parity và notebook reproducibility là các kiểm tra riêng với lệnh
  end-to-end Pandas.
- Dashboard đọc artifact đã sinh; không tự huấn luyện lại model.
