# Slide trình bày Đề tài 8

Học viên: **Phan Nguyễn Mai Phương** · MSHV: **25002711**.

Deck chính: [`cmapss_fd001_presentation.pptx`](cmapss_fd001_presentation.pptx) ·
[bản PDF](cmapss_fd001_presentation.pdf) (xuất từ PowerPoint). Deck dựng trên
template PowerPoint 16:9 của IUH tại
[`template/`](template/); logo, dải header, footer và slogan là master gốc của
template nên chỉnh sửa được trong PowerPoint.

[Sổ tay thuyết trình](../handoff/SPEECH_SCRIPT_VI.pdf) ([nguồn LaTeX](../handoff/SPEECH_SCRIPT_VI.tex))
đặt hình từng slide kèm chỗ cần chỉ, lời đọc và câu chuyển slide; phần demo dashboard
(sau slide 19) có ảnh chụp từng bước. Ảnh slide trong sổ tay nằm ở
`handoff/assets/slides/`; khi sửa deck, xuất lại ảnh bằng
`scripts/export_slides.ps1 -PreviewDirectory handoff/assets/slides` rồi đổi tên thành
`slide01.png`...`slide20.png`.

## Cấu trúc bài nói

Deck gồm 20 slide, thứ tự bám rubric và cùng thứ tự với mục Kết quả của báo cáo.

| Phần | Slide | Thời lượng | Nội dung |
|---|---|---|---|
| Bìa và lộ trình | 1--2 | 0:45 | Đề tài, bốn con số chính, năm câu hỏi |
| 1. Bài toán và dữ liệu | 3--5 | 2:10 | RUL, đối chiếu yêu cầu đề tài, C-MAPSS, ranh giới nhãn test |
| 2. Pipeline dữ liệu | 6--8 | 2:25 | Bronze/Silver/Gold, parity và benchmark Pandas--PySpark, EDA |
| 3. Khử nhiễu và cửa sổ thời gian | 9--10 | 1:45 | EWMA span 10, window 30 → 126 đặc trưng |
| 4. Xu hướng và bất thường | 11--12 | 2:00 | Kiểm định đầu/cuối, Health Index phát hiện bất thường |
| 5. Dự đoán RUL và cảnh báo | 13--16 | 4:30 | MAE test (GroupKFold trong notes), phân tích sai số, hai mức cảnh báo, lịch bảo trì 100 động cơ test |
| 6. Mở rộng và kết luận | 17--20 | 3:00 | Thử nghiệm mở rộng, Q1--Q5, khó khăn và bài học, cảm ơn |
| Demo dashboard | -- | 3:00 | Trình diễn artifact đã tạo |

## Luồng dựng deck

```text
outputs/ (JSON, CSV, PNG đã nghiệm thu)
  │  slides/extract_slide_data.py   chỉ đọc artifact, không huấn luyện lại
  │  python -m cmapss.degradation   khử nhiễu và phát hiện bất thường
  │  python -m cmapss.maintenance   lịch bảo trì cho 100 động cơ test
  ▼
slides/slide_data.json + outputs/degradation/
  │  slides/build_presentation.py   python-pptx trên template IUH
  ▼
slides/cmapss_fd001_presentation.pptx
  │  scripts/export_slides.ps1      PowerPoint (COM) SaveAs PDF
  ▼
slides/cmapss_fd001_presentation.pdf
```

Biểu đồ ở slide 7 và 17 là chart native của PowerPoint; các hình còn lại là PNG
sẵn có trong `outputs/`. Slide 11 crop hai panel `sensor_4` và `sensor_7` từ
`outputs/eda/figures/sensor_trajectories_1.png` lúc build.

## Cách build lại

Từ thư mục gốc repository:

```powershell
uv run --frozen python slides/extract_slide_data.py
uv run --frozen python -m cmapss.degradation
uv run --frozen python -m cmapss.maintenance
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
