# Slide trình bày Đề tài 8

Học viên: **Phan Nguyễn Mai Phương** · MSHV: **25002711**.

Deck chính: [`cmapss_fd001_presentation.pptx`](cmapss_fd001_presentation.pptx) ·
[bản PDF](cmapss_fd001_presentation.pdf) (xuất từ PowerPoint). Deck dựng trên
template PowerPoint 16:9 của IUH tại
[`template/`](template/); logo, dải header, footer và slogan là master gốc của
template nên chỉnh sửa được trong PowerPoint.

[Sổ tay thuyết trình](../handoff/SPEECH_SCRIPT_VI.pdf) ([nguồn LaTeX](../handoff/SPEECH_SCRIPT_VI.tex))
đặt hình từng slide kèm chỗ cần chỉ, lời đọc và câu chuyển slide; phần demo dashboard
(sau slide 18) có ảnh chụp từng bước. Ảnh slide trong sổ tay nằm ở
`handoff/assets/slides/`; khi sửa deck, xuất lại ảnh bằng
`scripts/export_slides.ps1 -PreviewDirectory handoff/assets/slides` rồi đổi tên thành
`slide01.png`...`slide20.png`.

## Cấu trúc bài nói

Deck gồm 20 slide theo mạch trong [`handoff/STORYLINE.md`](../handoff/STORYLINE.md);
mốc thời gian lấy từ kế hoạch thời gian của sổ tay thuyết trình.

| Phần | Slide | Hết phần lúc | Nội dung |
|---|---|---|---|
| Giới thiệu | 1–3 | 1:25 | Bìa, bài toán và định nghĩa RUL, quy trình năm bước |
| Dữ liệu và hạ tầng | 4–12 | 8:05 | Bốn tập con C-MAPSS, cấu trúc bản ghi, chất lượng dữ liệu, Bronze–Silver–Gold, HDFS/MapReduce/YARN, PySpark đối chiếu Pandas, truy vấn SQL và Spark SQL, sáu bước biến đổi, mở rộng quy mô 1×–200× |
| Phân tích và dự đoán | 13–18 | 12:15 | Làm mượt EWMA, xu hướng suy giảm, Health Index và cảnh báo mức một, mô hình RUL, cảnh báo mức hai và ngưỡng 40, lịch bảo trì |
| Demo dashboard | sau 18 | 14:15 | Trình diễn artifact đã tạo |
| Kết luận | 19 | 14:50 | Kết luận, hạn chế, hướng phát triển |
| Cảm ơn | 20 | 15:00 | Mời đặt câu hỏi |

Slide 3 chia quy trình thành năm bước: slide 4–12 trình bày bước 1–3 (thu thập,
lưu trữ, xử lý), slide 13–18 trình bày bước 4–5 (phân tích, cảnh báo và lịch bảo
trì). FD002–FD004 chỉ đi qua bước ETL; mô hình chỉ được huấn luyện và đánh giá
trên FD001.

## Luồng dựng deck

```text
outputs/ (JSON, CSV, PNG đã nghiệm thu)
  │  cmapss.degradation / cmapss.maintenance tạo artifact nếu cần tái tính
  │  slides/extract_slide_data.py   đọc artifact đã có
  │  slides/slide_figures.py        vẽ hình benchmark quy mô (chưa có module vẽ)
  ▼
slides/slide_data.json + slides/figures/benchmark_scaling.png + hình gốc trong outputs/
  │  slides/build_presentation.py   python-pptx trên template IUH
  ▼
slides/cmapss_fd001_presentation.pptx
  │  scripts/export_slides.ps1      PowerPoint (COM) SaveAs PDF
  ▼
slides/cmapss_fd001_presentation.pdf
```

Hình trên slide là chính file hình do module và notebook sinh ra, cùng file mà báo
cáo dùng: `outputs/degradation/figures/`, `outputs/modeling/figures/`,
`outputs/maintenance/figures/`, hình EDA của notebook 01 và ảnh chụp dashboard trong
`handoff/assets/dashboard/`. Slide không vẽ lại hình. Riêng hình benchmark quy mô do
`slides/slide_figures.py` vẽ từ `outputs/benchmarks/20261021/compute_only.csv`.

Mỗi slide chính có một hình và vài con số; tiêu đề là câu kết luận của hình. Chi tiết,
công thức, giới hạn và nguồn số liệu nằm trong ghi chú người nói của từng slide.

## Cách build lại

Từ thư mục gốc repository:

```powershell
uv run --frozen python -m cmapss.degradation
uv run --frozen python -m cmapss.maintenance
uv run --frozen python slides/extract_slide_data.py
uv run --frozen python slides/slide_figures.py
uv run --no-project --with python-pptx==1.0.2 --with pillow==12.3.0 python slides/build_presentation.py
powershell -File scripts/export_slides.ps1
```

Bước cuối cần Windows có PowerPoint. Font dùng trong deck là Arial để hiển thị
đúng tiếng Việt trên mọi máy; font gốc của template (Bai Jamjuree, SF Pro Display)
không được cài trên máy dựng.

## Bản cũ

Script pptxgenjs của deck 07/10/2026 nằm tại
[`archive/build_presentation_pptxgenjs.cjs`](archive/build_presentation_pptxgenjs.cjs);
`package.json` và `pnpm-lock.yaml` chỉ còn phục vụ script này. Deck 18 slide trước
ngày 28/09/2026 (PPTX, nguồn Beamer và PDF Beamer) nằm trong
[`archive/decks/`](archive/decks/); script build/finalize cũ nằm trong
[`archive/builders/`](archive/builders/).
