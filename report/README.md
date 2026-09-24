# Báo cáo cuối kỳ

[Bản PDF](main.pdf) được biên dịch từ [main.tex](main.tex), cập nhật ngày
24/09/2026. Báo cáo trình bày các kết quả đã có trên NASA C-MAPSS FD001 và
được biên tập theo cấu trúc mục tiêu--phương pháp--kết quả--thảo luận:
kiểm tra dữ liệu, EDA train-only, ETL Silver/Gold, đối chiếu Pandas--PySpark,
rolling features 30 cycles, suy luận chênh lệch sensor đầu/cuối vòng đời,
phân tích PCA--ARIMA train-only, đánh giá mô hình RUL, cảnh báo sớm OOF và phân
tích lỗi theo engine.

Báo cáo không đưa vào timeline, kế hoạch tiếp theo, đường dẫn artifact nội bộ
hoặc khai báo AI/LLM. Phần dữ liệu giải thích rõ nhãn kết cục, ý nghĩa cột,
phân bố và cách xử lý; phần phương pháp giải thích lý do, đơn vị và nguồn của
các công thức. Dashboard Streamlit đã được bổ sung như một consumer read-only
của artifact; PCA--ARIMA vẫn được trình bày như phân tích phụ trên Health Index,
không phải RUL model. Lệnh tái lập end-to-end đã chạy lại pipeline Pandas từ
raw và đối chiếu artifact cuối.
Checklist bàn giao nằm tại `handoff/FINAL_HANDOFF.md` đã đồng bộ với báo cáo.
Slide tại `slides/` là artifact của lượt trước, chưa được biên tập hoặc
nghiệm thu lại sau cập nhật timing; không coi đây là bản trình chiếu mới nhất.

## Cấu trúc

```text
report/
├── main.tex                     # Bìa, định dạng, mục lục và ghép các phần
├── main.pdf                     # Bản PDF đã biên dịch
├── sections/
│   ├── glossary.tex             # Thuật ngữ và chữ viết tắt dùng trong báo cáo
│   ├── problem-data.tex         # Bài toán, data dictionary và chất lượng dữ liệu
│   ├── theory.tex               # Cơ sở lý thuyết của các phương pháp đã dùng
│   ├── methodology.tex          # Luồng xử lý, EDA, feature và suy luận
│   ├── statistical-protocol.tex # Protocol ghép cặp đầu/cuối vòng đời
│   ├── current-results.tex      # Bảng, hình và diễn giải kết quả
│   ├── discussion.tex           # Thảo luận, giới hạn và hướng phát triển
│   ├── conclusion.tex           # Kết luận và phạm vi khái quát hóa
│   └── appendix.tex             # Cấu hình, lệnh tái lập và mapping artifact
└── figures/
    ├── pipeline.tex             # Bản TeX giữ cho report PDF hiện tại
    ├── pipeline.pdf             # Hình pipeline dùng trong báo cáo
    ├── statistical-protocol.tex # Nguồn TikZ của protocol thống kê
    └── statistical-protocol.pdf # Bản xuất của protocol
```

Hình EDA và thống kê được tham chiếu trực tiếp từ thư mục `outputs/` để tránh
sao chép và lệch nguồn. Bibliography dùng chung tại
`../references/bibliography.bib`.

Sơ đồ pipeline có nguồn Draw.io chỉnh sửa được tại
`../assets/diagrams/fd001-pipeline.drawio`. File này là nguồn để mở và chỉnh
trong Draw.io; bản PDF trong report vẫn được giữ riêng để biên dịch LaTeX ổn
định.

## Biên dịch

Với TeX distribution có `latexmk`, từ thư mục gốc repository:

```powershell
latexmk -cd -xelatex -interaction=nonstopmode report/figures/pipeline.tex
latexmk -cd -xelatex -interaction=nonstopmode report/figures/statistical-protocol.tex
latexmk -cd -xelatex -interaction=nonstopmode report/main.tex
```

Môi trường hiện tại dùng Tectonic portable tại
`tmp/report-tools/tectonic/tectonic.exe`:

```powershell
Push-Location report/figures
../../tmp/report-tools/tectonic/tectonic.exe --keep-logs pipeline.tex
../../tmp/report-tools/tectonic/tectonic.exe --keep-logs statistical-protocol.tex
Pop-Location
Push-Location report
../tmp/report-tools/tectonic/tectonic.exe --keep-logs main.tex
Pop-Location
```

Lần build ngày 24/09/2026 tạo PDF 45 trang, không có lỗi biên dịch, citation
undefined, reference undefined hoặc overfull/underfull box. Tectonic vẫn ghi
cảnh báo môi trường về Fontconfig và font Windows; đây không phải lỗi nội dung.
