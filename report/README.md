# Báo cáo cuối kỳ

[Bản PDF](main.pdf) được biên dịch từ [main.tex](main.tex), cập nhật ngày
28/09/2026. Báo cáo trình bày pipeline dự đoán RUL trên NASA C-MAPSS FD001 và
ma trận đánh giá chuyển miền sang FD002--FD004 (window 10 cycles).

Ngày 28/09/2026, phần thân báo cáo được viết lại theo cấu trúc Giới thiệu --
Dữ liệu -- Cơ sở lý thuyết -- Tổ chức, truy vấn và biến đổi dữ liệu -- Phương
pháp -- Kết quả -- Thảo luận -- Kết luận. Bản viết lại giữ nguyên toàn bộ số
liệu, gom các giới hạn về Mục 7.6 thay vì lặp lại ở từng đoạn, và đưa thử
nghiệm nested CV vào PDF (Mục 5.7 và 6.6).

Mục 4 là section riêng về xử lý dữ liệu, trả lời năm câu hỏi: gom nhiều file
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
├── main.pdf                  # Bản PDF đã biên dịch (49 trang)
├── sections/
│   ├── glossary.tex          # Thuật ngữ và chữ viết tắt
│   ├── problem-data.tex      # Mục 1 Giới thiệu, Mục 2 Dữ liệu
│   ├── theory.tex            # Mục 3 Cơ sở lý thuyết
│   ├── data-processing.tex   # Mục 4 Gom file, lưu trữ, truy vấn, biến đổi,
│   │                         # cột cho mô hình, quan hệ giữa các bảng
│   ├── methodology.tex       # Mục 5 Phương pháp: mô hình, cảnh báo, phân tích
│   │                         # ghép cặp, ARIMA, chuyển miền, nested CV
│   ├── current-results.tex   # Mục 6 Kết quả
│   ├── discussion.tex        # Mục 7 Thảo luận: nguồn gốc sai số, cảnh báo trễ,
│   │                         # chuyển miền, giới hạn, hướng phát triển
│   ├── conclusion.tex        # Mục 8 Kết luận theo câu hỏi Q1--Q6
│   └── appendix.tex          # Phụ lục nguồn; không đưa vào PDF chính
└── figures/
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
pipeline có nguồn Draw.io tại `../assets/diagrams/fd001-pipeline.drawio`.

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

Lần build ngày 28/09/2026 tạo PDF 49 trang (dưới giới hạn 50 trang), không có
lỗi biên dịch, citation hoặc reference chưa phân giải, và không có
overfull/underfull box. Tectonic vẫn ghi cảnh báo môi trường về đường dẫn font
Windows; đây không phải lỗi nội dung.

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
