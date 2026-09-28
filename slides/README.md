# Slide trình bày C-MAPSS FD001--FD004

Deck chính: [`cmapss_fd001_presentation.pptx`](cmapss_fd001_presentation.pptx) ·
[bản PDF](cmapss_fd001_presentation.pdf) (xuất từ PowerPoint).
Kịch bản nói 30 phút: [`SPEECH_SCRIPT_VI.pdf`](../handoff/SPEECH_SCRIPT_VI.pdf)
([nguồn LaTeX](../handoff/SPEECH_SCRIPT_VI.tex)).
Kịch bản demo dashboard: [`handoff/DEMO_SCRIPT.md`](../handoff/DEMO_SCRIPT.md).

## Cấu trúc bài nói

Deck gồm 20 slide, bằng giới hạn 20 slide, cho bài nói khoảng 29 phút và một
phút dự phòng. Mỗi slide có tiêu đề nêu kết luận, một thành phần trực quan chính,
dòng nguồn và speaker notes ghi thời lượng.

| Phần | Slide | Thời lượng | Nội dung |
|---|---|---|---|
| Bìa | 1 | 0:30 | Đề tài, tác giả, bốn con số chính |
| 1. Bài toán và dữ liệu | 2--5 | 5:30 | Câu hỏi nghiên cứu, RUL, đối chiếu yêu cầu đề tài 8, bốn subset, ranh giới nhãn test |
| 2. Pipeline Big Data | 6--7 | 4:00 | Bronze/Silver/Gold, parity Pandas--PySpark, benchmark Pandas 1×/10×/100× |
| 3. Phân tích chuỗi thời gian | 8--11 | 6:00 | EDA, xu hướng suy giảm sensor, trailing window 30, Health Index--ARIMA |
| 4. Mô hình và đánh giá | 12--16 | 8:00 | GroupKFold, chọn model, kết quả test, phân tích lỗi, cảnh báo sớm |
| 5. Mở rộng và kết luận | 17--20 | 5:15 | Nested CV, transfer FD001→FD002--FD004, trả lời câu hỏi, giới hạn |

Slide làm tròn số liệu đến hai chữ số thập phân; speaker notes và kịch bản nói
dùng số đầy đủ. Metric chính FD001 dùng window 30; slide 18 dùng window 10 và
không so trực tiếp với metric chính.

## Luồng dựng deck

```text
outputs/ (JSON, CSV, Parquet, PNG đã nghiệm thu)
  │  slides/extract_slide_data.py   chỉ đọc artifact, không huấn luyện lại
  ▼
slides/slide_data.json              số liệu dùng trên slide
  │  slides/build_presentation.cjs  pptxgenjs + sharp (crop hình sensor)
  ▼
slides/cmapss_fd001_presentation.pptx
  │  PowerPoint (COM) SaveAs PDF
  ▼
slides/cmapss_fd001_presentation.pdf
```

Biểu đồ trên slide 7, 11, 13, 16 và 18 là chart native của PowerPoint, dữ liệu
lấy từ `slide_data.json`. Hình còn lại là PNG sẵn có trong `outputs/`; slide 9
crop hai panel `sensor_4` và `sensor_7` từ
`outputs/eda/figures/sensor_trajectories_1.png` lúc build.

Nguồn số liệu chính:

- `outputs/modeling/tables/{final_metrics.json,model_metrics.csv,warning_summary.json}`
- `outputs/modeling/predictions/final_evaluation.parquet`
- `outputs/etl/{parity.json,benchmark.json}`, `outputs/features/run_info.json`
- `outputs/eda/tables/`, `outputs/statistics/tables/sensor_inference.csv`
- `outputs/timeseries/tables/{forecast_metrics.csv,baseline_metrics.csv}`
- `outputs/cross_dataset/matrix.json`, `outputs/cross_dataset/FD00x/etl/run_info.json`
- `outputs/experiments/rul_improvement/20260927_full/`

## Cách build lại

```powershell
uv run --frozen python slides/extract_slide_data.py
$env:NODE_PATH='C:/Users/timewo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'
& 'C:/Users/timewo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' slides/build_presentation.cjs
```

Đường dẫn Node ở trên là runtime cục bộ trên máy tác giả, có sẵn `pptxgenjs` và
`sharp`. Máy khác cần Node và hai package này trong `NODE_PATH`. Biến
`DECK_OUT` cho phép ghi deck ra đường dẫn khác để kiểm tra trước khi thay deck
chính. PDF được xuất bằng PowerPoint; repository không có script xuất PDF.

## Kiểm tra đã chạy (28/09/2026)

- `validate.py` của pptx skill: `All validations PASSED!`.
- Render 20 slide bằng PowerPoint ở 1600×900 và soát chữ tràn, chồng lấn, nhãn
  biểu đồ, dấu thập phân và số trang. Lần render đầu phát hiện chồng chữ ở slide
  5, 9, 10, 13, nhãn phần tràn khung và dấu thập phân kiểu Anh trong chart; đã
  sửa bằng cách đưa giá trị vào nhãn trục hoặc dùng số nguyên (ms, cycles làm tròn).
- PDF xuất từ PowerPoint có 20 trang; kịch bản nói build bằng Tectonic, 25 trang,
  log không có lỗi, overfull hoặc underfull.

Giới hạn: chưa trình chiếu thử trên máy của phòng báo cáo. Font là Arial; nếu máy
trình chiếu thiếu Arial, bố cục có thể lệch. Chart native hiển thị theo locale
của PowerPoint, nên các giá trị thập phân được đặt trong nhãn text thay vì data
label.

## Bản cũ

Deck 18 slide trước ngày 28/09/2026 (PPTX, nguồn Beamer và PDF Beamer) nằm trong
[`archive/decks/`](archive/decks/) với hậu tố `_18slides_20260928`; script
build/finalize cũ nằm trong [`archive/builders/`](archive/builders/).
