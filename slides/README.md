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
| Bìa và lộ trình | 1--2 | 0:55 | Đề tài, bốn con số chính, năm câu hỏi |
| 1. Bài toán và dữ liệu | 3--5 | 2:25 | RUL, đối chiếu yêu cầu đề tài, lý do chọn C-MAPSS, ranh giới nhãn test |
| 2. Pipeline dữ liệu | 6--8 | 2:15 | Pipeline 1–3, A1–A5, B1–B4 và năm bước quy trình dữ liệu; đối chiếu Pandas--PySpark--Hadoop; EDA |
| 3. Khử nhiễu và cửa sổ thời gian | 9--10 | 1:50 | EWMA span 10, cửa sổ 30 cycle → 126 đặc trưng, chuẩn hóa |
| 4. Xu hướng và bất thường | 11--12 | 2:00 | Kiểm định đầu/cuối, Health Index phát hiện bất thường |
| 5. Dự đoán RUL và cảnh báo | 13--16 | 4:25 | MAE test và GroupKFold, phân tích sai số, hai mức cảnh báo (OOF train), lịch bảo trì 100 động cơ test |
| 6. Mở rộng và kết luận | 17--19 | 2:30 | Thử nghiệm mở rộng, Q1--Q5, khó khăn, bài học và giới hạn |
| Demo dashboard | sau 19 | 2:30 | Trình diễn artifact đã tạo |
| Cảm ơn | 20 | 0:10 | Mời đặt câu hỏi; tổng thời gian 19:00 |

Slide 6 dùng cùng cách gọi với Hình 5 của báo cáo: bước 1–3 chuẩn bị dữ liệu,
nhánh A dự đoán RUL (A1–A5), nhánh B theo dõi tín hiệu chỉ học trên train
(B1–B4), ô nét đứt là bước chạy lại trên Hadoop để đối chiếu bảng tổng hợp động
cơ. Lịch bảo trì nhận RUL dự đoán từ A4 và trạng thái bất thường từ B4.
Pandas/PySpark đối chiếu ETL và rolling features (A1–A2).

## Luồng dựng deck

```text
outputs/ (JSON, CSV, PNG đã nghiệm thu)
  │  cmapss.degradation / cmapss.maintenance tạo artifact nếu cần tái tính
  │  slides/extract_slide_data.py   đọc artifact đã có
  │  slides/slide_figures.py        vẽ lại hình cho slide từ dữ liệu đã lưu
  ▼
slides/slide_data.json + slides/figures/ + outputs/degradation/ + outputs/maintenance/ + outputs/hdfs/
  │  slides/build_presentation.py   python-pptx trên template IUH
  ▼
slides/cmapss_fd001_presentation.pptx
  │  scripts/export_slides.ps1      PowerPoint (COM) SaveAs PDF
  ▼
slides/cmapss_fd001_presentation.pdf
```

Biểu đồ ở slide 7 và 17 là chart native của PowerPoint, nhãn số dùng dấu phẩy thập
phân ở mọi máy chiếu. Tám hình matplotlib (slide 8, 9, 11–16) được
`slides/slide_figures.py` vẽ lại vào `slides/figures/` từ dữ liệu raw và artifact
trong `outputs/`: đúng kích thước ô trên slide, chữ từ 9,5 pt, nhãn tiếng Việt, dấu
phẩy thập phân, có chú giải màu. Script không huấn luyện lại mô hình và không ghi đè
hình gốc của pipeline trong `outputs/`.

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
