# Slide trình bày C-MAPSS FD001--FD004

Học viên: **Phan Nguyễn Mai Phương** · MSHV: **25002711**.

Deck chính: [`cmapss_fd001_presentation.pptx`](cmapss_fd001_presentation.pptx) ·
[bản PDF](cmapss_fd001_presentation.pdf) (xuất từ PowerPoint).
Kịch bản nói 18--20 phút: [`SPEECH_SCRIPT_VI.pdf`](../handoff/SPEECH_SCRIPT_VI.pdf)
([nguồn LaTeX](../handoff/SPEECH_SCRIPT_VI.tex)).
Kịch bản demo dashboard: [`handoff/DEMO_SCRIPT.md`](../handoff/DEMO_SCRIPT.md).

## Cấu trúc bài nói

Deck gồm 20 slide cho phần nói khoảng 17 phút và demo dashboard khoảng 3 phút.
Mỗi slide có tiêu đề nêu kết luận, một thành phần trực quan chính,
dòng nguồn và speaker notes ghi thời lượng.

| Phần | Slide | Thời lượng | Nội dung |
|---|---|---|---|
| Bìa | 1 | 0:20 | Đề tài, học viên, bốn con số chính |
| 1. Bài toán và dữ liệu | 2--5 | 2:45 | Sáu câu hỏi, RUL, đối chiếu yêu cầu, bốn subset, ranh giới nhãn test |
| 2. Pipeline Big Data | 6--7 | 2:00 | Bronze/Silver/Gold, parity Pandas--PySpark, benchmark |
| 3. Phân tích chuỗi thời gian | 8--11 | 2:30 | EDA, xu hướng sensor, trailing window 30, Health Index--ARIMA |
| 4. Mô hình và đánh giá | 12--16 | 5:00 | GroupKFold, HGB126/HGB169, alignment, warning policy |
| 5. Mở rộng và kết luận | 17--20 | 4:25 | Nested CV, transfer, Q1--Q6, giới hạn |
| Demo dashboard | -- | 3:00 | Trình diễn artifact đã tạo |

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

Biểu đồ trên slide 7, 11, 13 và 18 là chart native của PowerPoint, dữ liệu
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
- `outputs/experiments/rul_improvement/20260928_rebuild/`

## Cách build lại

```powershell
uv run --frozen python slides/extract_slide_data.py
pnpm --dir slides install --frozen-lockfile
pnpm --dir slides build
```

Cần Node.js và pnpm 11.25.0. `package.json` ghim `pptxgenjs` 4.0.1 và `sharp`
0.35.4; `pnpm-lock.yaml` khóa cả dependency gián tiếp. Không cần `NODE_PATH`
hay đường dẫn runtime riêng của máy thực hiện. Biến `DECK_OUT` cho phép ghi deck
ra đường dẫn khác. Ví dụ: `$env:DECK_OUT='tmp/review.pptx'` trước khi build.
PDF xuất bằng `scripts/export_slides.ps1` trên máy Windows có PowerPoint.

## Kiểm tra và xuất bản cuối (07/10/2026)

- Dựng thành công với dependency lockfile; alignment HGB126/HGB169, warning policy
  và Q1--Q6 đã đồng bộ. Bản cuối có 20 slide; chart native được kiểm tra dữ liệu
  cache và PPTX được xuất lại bằng PowerPoint.
- Benchmark mới: 1×/10×, ba lượt mỗi engine, local[4], heap 4 GiB; có I/O và
  process-tree RSS. Lượt heap mặc định bị OOM được lưu riêng, không gộp vào median.

- Kiểm tra gói, bố cục, font, chart native và import Artifact Tool ngày 07/10 đều đạt;
  chart ở slide 7, 11, 13 và 18 giữ dạng editable.
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
