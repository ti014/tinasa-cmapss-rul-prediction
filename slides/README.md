# Slide trình bày C-MAPSS FD001--FD004

Deck chính: [`cmapss_fd001_presentation.pptx`](cmapss_fd001_presentation.pptx)
([bản xuất riêng để đối chiếu](archive/decks/cmapss_fd001_presentation_transfer.pptx))
Nguồn Beamer: [`cmapss_fd001_presentation.tex`](cmapss_fd001_presentation.tex) ·
[PDF Beamer](cmapss_fd001_presentation.pdf)
Kịch bản demo dashboard: [`handoff/DEMO_SCRIPT.md`](../handoff/DEMO_SCRIPT.md).
Kịch bản thuyết trình LaTeX: [`SPEECH_SCRIPT_VI.pdf`](../handoff/SPEECH_SCRIPT_VI.pdf)
và [nguồn LaTeX](../handoff/SPEECH_SCRIPT_VI.tex).

Deck gồm 18 slide, đi theo mạch bài thuyết trình học thuật: bài toán, dữ liệu,
nhãn RUL, pipeline, phân tích sensor, rolling features, evaluation protocol,
model selection, kết quả RUL, phân tích lỗi, Health Index--ARIMA, cảnh báo sớm,
ETL/tái lập, kết luận và ma trận transfer FD001--FD004.

Nguồn số liệu lấy từ các artifact đã kiểm chứng trong `outputs/`, đặc biệt:

- `outputs/modeling/tables/final_metrics.json`
- `outputs/modeling/tables/model_metrics.csv`
- `outputs/modeling/tables/warning_summary.json`
- `outputs/timeseries/run_info.json`
- `outputs/timeseries/tables/baseline_metrics.csv`
- `outputs/etl/benchmark.json`
- `outputs/cross_dataset/matrix.json`
- `outputs/cross_dataset/README.md`
- `tmp/e2e-review-20260924/validation.json`
- `report/sections/conclusion.tex`

Deck hiện hành được xây dựng bằng `build_academic_presentation.mjs` với font
Arial, giữ tỷ lệ 16:9 và các bảng native editable. `finalize_academic_presentation.mjs`
kiểm tra bản xuất bằng Artifact Tool. Hai script dùng runtime đã được cung cấp
trong môi trường Codex; chúng không sinh lại số liệu khoa học.

Slide RUL ghi MAE 15,3490 cycles và khoảng tin cậy bootstrap theo engine 95%
[12,7286; 18,0866]. Slide cảnh báo ghi timing 44/7/49 và median lead time
30 cycles trên selected-model OOF train. OOF cũng được dùng để chọn model, nên
đây là bằng chứng mô tả nội bộ. Benchmark chỉ đo Pandas in-memory; chưa có phép
đo tốc độ PySpark. Slide 18 ghi rõ ma trận window 10: transfer từ FD001 sang
FD002, FD003 và FD004 có MAE lần lượt 60,3215; 47,3949; 58,5731 cycles. Metric
chính FD001 dùng window 30 và không so trực tiếp với ma trận window 10.

Bản cập nhật đã qua kiểm tra package integrity, layout, Arial, import và tám
bảng editable trên slide 3, 6, 7, 8, 11, 15, 16 và 18. Cả 18 slide đã được render để soát
clipping, wrapping, hình, bảng và số trang. Biên bản finalizer ở
`tmp/slides-rul-20260927/finalizer/` (thư mục tạm, không thuộc gói Git).

Bản cũ trước redesign được giữ trong `archive/decks/` để đối chiếu hoặc khôi
phục khi cần.

## Cập nhật ngày 28/09/2026

Slide 4 bổ sung boundary đọc file, `union`, Silver Parquet và SQLite. Slide 6
ghi rõ one-to-one join bằng ba cột khóa, 126 feature columns và việc ba
operational settings chưa nằm trong `X`. Nội dung này khớp với
[`handoff/DATA_WALKTHROUGH.md`](../handoff/DATA_WALKTHROUGH.md). Finalizer mới
pass package integrity, layout, font và import cho 18 slide.

Slide 14–16 trình bày protocol cải thiện, kết quả vòng ngoài và đánh đổi trong
inner CV. Slide 17 là kết luận, slide 18 là ma trận transfer. Nguồn số liệu mới:
`outputs/experiments/rul_improvement/20260927_full/`. Không thay baseline;
MAE test vẫn 15,3490 cycles. Ví dụ low-RUL MAE giảm 9,75% ở một ứng viên inner
CV phải được đọc cùng mức tăng MAE toàn bộ 0,32 cycles.

Hai script build/finalize dùng đường dẫn runtime Codex cục bộ ở đầu file.
Trước khi chạy finalizer trong môi trường này, đặt:

```powershell
$env:RUNTIME_NODE_MODULES='C:/Users/timewo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'
$env:RUNTIME_NODE='C:/Users/timewo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
& $env:RUNTIME_NODE slides/build_academic_presentation.mjs
& $env:RUNTIME_NODE slides/finalize_academic_presentation.mjs
```

Kết quả finalizer nằm trong `tmp/slides-rul-20260927/final/`; chỉ sao chép sang
deck chính sau khi validation đạt. Máy khác cần điều chỉnh đường dẫn runtime,
hoặc biên dịch nguồn Beamer bằng TeX distribution có hỗ trợ tiếng Việt.
