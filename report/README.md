# Báo cáo cuối kỳ

**Học viên:** Phan Nguyễn Mai Phương · **MSHV:** 25002711.

[Bản PDF](main.pdf) được biên dịch từ [main.tex](main.tex), cập nhật ngày
21/10/2026. Báo cáo trình bày Đề tài 8 — phân tích dữ liệu IoT và bảo trì dự đoán
trên NASA C-MAPSS FD001, theo thứ tự rubric: pipeline dữ liệu (6.1), khử nhiễu và
cửa sổ thời gian (6.2), xu hướng và bất thường (6.3), dự đoán RUL (6.4), cảnh báo
sớm (6.5), báo cáo dự đoán thời điểm cần bảo trì (6.6) và thử nghiệm mở rộng (6.7).

Đợt đồng bộ ngày 12/10/2026 làm rõ rolling features đọc Silver observations,
engine_summary là nhánh tổng hợp riêng, và lịch bảo trì kết hợp RUL với Health
Index fit trên train rồi transform test. Kích thước bảng trong sơ đồ tổng thể
thuộc FD001; Hadoop/HDFS là bước đối chiếu trên một node Docker. Bản nguồn và
bản xuất pipeline trong README, báo cáo và slide được cập nhật cùng lượt.

Đợt sửa ngày 08/10/2026:

- Đổi tựa về Đề tài 8; viết lại Tóm tắt, câu hỏi Q1--Q5 và Kết luận.
- Mô hình chính là HistGradientBoosting với 126 đặc trưng window; bản 169 đặc
  trưng, nested CV, similarity, chuyển miền FD002--FD004 và ARIMA được gom vào
  Mục 6.7 "Thử nghiệm mở rộng".
- Thêm Mục 5.2--5.3 và 6.2--6.3 (khử nhiễu EWMA, Health Index, phát hiện bất
  thường), Mục 5.7 và 6.6 (lịch bảo trì cho 100 động cơ test), Mục 8 "Khó khăn và
  bài học rút ra".
- Sơ đồ `figures/pipeline.tex` thay nhánh PCA--ARIMA bằng nhánh khử nhiễu → Health
  Index → bất thường.
- `sections/followup-tables.tex` không còn được ghép vào PDF; benchmark local được
  chuyển vào Mục 6.1.

Hình mới đọc từ `outputs/degradation/figures/` và `outputs/maintenance/figures/`.

Ngày 28/09/2026, phần thân báo cáo được viết lại theo cấu trúc Giới thiệu --
Dữ liệu -- Cơ sở lý thuyết -- Tổ chức, truy vấn và biến đổi dữ liệu -- Phương
pháp -- Kết quả -- Thảo luận -- Kết luận. Bản viết lại giữ nguyên toàn bộ số
liệu, gom các giới hạn về Mục 7.6 thay vì lặp lại ở từng đoạn, và đưa thử
nghiệm nested CV vào PDF (Mục 5.7 và 6.6).

Mục 4 là section riêng về xử lý dữ liệu, trả lời sáu câu hỏi: gom nhiều file
thế nào (union có cột `split`), lưu ở đâu (Parquet, SQLite), truy vấn ra sao
(SQL trên SQLite, Spark SQL và Pandas trên Parquet), biến đổi qua những bước nào
(sáu bước, kèm kích thước bảng sau mỗi bước), cột nào vào mô hình, và quan hệ
giữa các bảng (1:N, 1:1, 1:0..1; không có N:N). Mục này có hai sơ đồ TikZ mới,
bốn đoạn mã nguồn trích từ `src/cmapss` và một hình minh họa window vẽ từ dữ
liệu. Mọi ví dụ chạy lại được trong
[`notebooks/00_data_pipeline_walkthrough.ipynb`](../notebooks/00_data_pipeline_walkthrough.ipynb).

Nội dung của `data-walkthrough.tex`, `statistical-protocol.tex` và hai file
`rul-improvement-*.tex` được gộp vào `data-processing.tex`, `methodology.tex` và
`current-results.tex`. Hình TikZ `statistical-protocol` không còn dùng nên đã
được xóa; có thể khôi phục từ lịch sử Git nếu cần.

## Cấu trúc

```text
report/
├── main.tex                  # Bìa, tóm tắt, mục lục, cấu hình listings, ghép các phần
├── main.pdf                  # Bản PDF đã biên dịch (40 trang)
├── sections/
│   ├── glossary.tex          # Thuật ngữ và chữ viết tắt
│   ├── problem-data.tex      # Mục 1 Giới thiệu, Mục 2 Dữ liệu
│   ├── theory.tex            # Mục 3 Cơ sở lý thuyết
│   ├── data-processing.tex   # Mục 4 Gom file, lưu trữ, truy vấn, biến đổi,
│   │                         # cột cho mô hình, quan hệ giữa các bảng
│   ├── methodology.tex       # Mục 5 Phương pháp: khử nhiễu, Health Index, bất thường,
│   │                         # kiểm định, mô hình, cảnh báo, lịch bảo trì, mở rộng
│   ├── current-results.tex   # Mục 6 Kết quả theo thứ tự rubric
│   ├── discussion.tex        # Mục 7 Thảo luận và Mục 8 Khó khăn và bài học
│   ├── conclusion.tex        # Mục 9 Kết luận theo câu hỏi Q1--Q5
│   ├── followup-tables.tex   # Bảng follow-up cũ; không còn ghép vào PDF
│   └── appendix.tex          # Phụ lục nguồn; không đưa vào PDF chính
└── figures/
    ├── diagram-style.tex           # Màu, cỡ ô, mũi tên và chú giải dùng chung cho mọi sơ đồ
    ├── data-flow.tex / .pdf        # Luồng dữ liệu: bảng, kích thước, phép biến đổi
    ├── data-relations.tex / .pdf   # Quan hệ giữa các bảng, khóa và bậc quan hệ
    ├── pipeline.tex / .pdf         # Pipeline tổng thể gồm nhánh train-only
    └── rul-improvement.tex / .pdf  # Sơ đồ nested CV
```

Các biểu đồ vẽ từ dữ liệu (minh họa window, dự đoán OOF theo RUL thật, cảnh báo
theo tuổi thọ, dịch chuyển sensor khi chuyển miền, ma trận MAE chuyển miền) được
sinh bởi `scripts/report_figures.py` vào `outputs/diagnostics/figures/`. Script
chỉ đọc dữ liệu raw và artifact có sẵn, không huấn luyện lại mô hình:

```powershell
.venv/Scripts/python.exe scripts/report_figures.py
```

Hình kết quả được tham chiếu trực tiếp từ `outputs/` để tránh sao chép và lệch
nguồn. Bibliography dùng chung tại `../references/bibliography.bib`. Sơ đồ
pipeline trong báo cáo có nguồn TikZ tại `figures/pipeline.tex`; sơ đồ tổng thể
cùng nội dung cho README có nguồn draw.io tại
`../assets/diagrams/fd001-pipeline.drawio`. Mỗi bản xuất được sinh từ nguồn tương ứng.
Mọi sơ đồ TikZ nạp `figures/diagram-style.tex`; các sơ đồ draw.io dùng cùng quy ước:
xanh lá là dữ liệu, xanh dương là bước xử lý, cam là kết quả, viền xám nét đứt là
kiểm tra phụ, mũi tên cam nét đứt là dữ liệu chỉ được mở sau cùng.

## Biên dịch

Với TeX distribution có `latexmk`, từ thư mục gốc repository:

```powershell
latexmk -cd -xelatex -interaction=nonstopmode report/figures/pipeline.tex
latexmk -cd -xelatex -interaction=nonstopmode report/figures/data-flow.tex
latexmk -cd -xelatex -interaction=nonstopmode report/figures/data-relations.tex
latexmk -cd -xelatex -interaction=nonstopmode report/figures/rul-improvement.tex
latexmk -cd -xelatex -interaction=nonstopmode report/main.tex
```

Môi trường hiện tại dùng Tectonic portable tại
`tmp/report-tools/tectonic/tectonic.exe`:

```powershell
Push-Location report
../tmp/report-tools/tectonic/tectonic.exe --keep-logs main.tex
Pop-Location
```

Lần build ngày 21/10/2026 tạo PDF 40 trang (đúng giới hạn 40 trang), không có
lỗi biên dịch, citation hoặc reference chưa phân giải, và không có overfull box.
Để đạt 40 trang, thân bài dùng cỡ chữ 11pt và giãn dòng 1.0, mục lục chỉ đến
subsection, và ba listing Python ở Mục 4 được thay bằng tên file nguồn. Tectonic vẫn ghi cảnh báo môi trường về đường dẫn font Windows;
đây không phải lỗi nội dung. Phần bibliography dùng line spacing 1.0 và cỡ chữ
nhỏ hơn phần thân để giữ toàn bộ tài liệu tham khảo trong giới hạn trang.

## Số liệu chẩn đoán cho phần Thảo luận

Mục 7 dùng thêm các số liệu chẩn đoán được tính lại từ artifact đã có, không
huấn luyện lại mô hình. Chạy lại bằng:

```powershell
.venv/Scripts/python.exe scripts/report_error_diagnostics.py
```

Script ghi `outputs/diagnostics/error_diagnostics.json`, gồm:

- sai số OOF của HGB theo nhóm RUL thật (Bảng 24, Hình 15): dự đoán trung bình gần như
  không đổi, khoảng 145--147 cycles, khi RUL thật trên 150;
- tỷ trọng của mười sai số test lớn nhất trong NASA score (75,5%) và MAE test
  theo nhóm RUL thật;
- tuổi thọ trung bình theo nhóm thời điểm cảnh báo (trễ: 182,8; trước ngưỡng:
  229,1 cycles) và tương quan giữa tuổi thọ và độ lệch cảnh báo (−0,56);
- chẩn đoán chuyển miền (Bảng 25, Hình 18): tỷ lệ giá trị sensor test nằm trong miền
  train FD001, sai số có dấu và độ lệch chuẩn của dự đoán.

Phần so sánh với Li và cộng sự dùng RMSE FD001 trong Table 4 của bài báo
(`references/papers/R4_Li_Ding_Sun_2018_CNN_RUL.pdf`): 12,61 cycles khi cắt nhãn
ở 125 và 13,32 cycles khi không cắt nhãn.

## Thực nghiệm cải thiện RUL

Nested GroupKFold 5×3 không tìm được cấu hình đáp ứng tiêu chí đặt trước: MAE
vùng RUL thấp ở vòng ngoài tăng từ 3,6676 lên 3,6730 cycles, số động cơ cảnh
báo trễ tăng từ 49 lên 53. Cấu hình cuối trùng mô hình gốc, nên test MAE vẫn là
15,3490 cycles. Nguồn và lệnh chạy lại: [thực nghiệm](../experiments/README.md).
