# Slide trình bày C-MAPSS FD001--FD004

Deck chính: [`cmapss_fd001_presentation.pptx`](cmapss_fd001_presentation.pptx)
([bản xuất riêng để đối chiếu](archive/decks/cmapss_fd001_presentation_transfer.pptx))
Nguồn Beamer: [`cmapss_fd001_presentation.tex`](cmapss_fd001_presentation.tex) ·
[PDF Beamer](cmapss_fd001_presentation.pdf)
Kịch bản demo dashboard: [`handoff/DEMO_SCRIPT.md`](../handoff/DEMO_SCRIPT.md).
Kịch bản thuyết trình LaTeX: [`SPEECH_SCRIPT_VI.pdf`](../handoff/SPEECH_SCRIPT_VI.pdf)
và [nguồn LaTeX](../handoff/SPEECH_SCRIPT_VI.tex).

Deck gồm 15 slide, đi theo mạch bài thuyết trình học thuật: bài toán, dữ liệu,
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
đo tốc độ PySpark. Slide 15 ghi rõ ma trận window 10: transfer từ FD001 sang
FD002, FD003 và FD004 có MAE lần lượt 60,3215; 47,3949; 58,5731 cycles. Metric
chính FD001 dùng window 30 và không so trực tiếp với ma trận window 10.

Bản cập nhật đã qua kiểm tra package integrity, layout, Arial, import và sáu
bảng editable trên slide 3, 6, 7, 8, 11 và 15. Cả 15 slide đã được render để soát
clipping, wrapping, hình, bảng và số trang. Biên bản finalizer ở
`tmp/slides-transfer-20260925/finalizer/` (thư mục tạm, không thuộc gói Git).

Bản cũ trước redesign được giữ trong `archive/decks/` để đối chiếu hoặc khôi
phục khi cần.
