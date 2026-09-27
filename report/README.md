# Báo cáo cuối kỳ

[Bản PDF](main.pdf) được biên dịch từ [main.tex](main.tex), cập nhật ngày
27/09/2026. Báo cáo trình bày các kết quả chính trên NASA C-MAPSS FD001 và ma
trận đánh giá bổ sung trên toàn bộ FD001--FD004. Kết quả RUL chi tiết vẫn thuộc
FD001; FD002--FD004 dùng protocol window 10 cycles riêng để kiểm tra chuyển miền.
Báo cáo được biên tập theo cấu trúc mục tiêu--phương pháp--kết quả--thảo luận:
kiểm tra dữ liệu, EDA train-only, ETL Silver/Gold, đối chiếu Pandas--PySpark,
rolling features 30 cycles, suy luận chênh lệch sensor đầu/cuối vòng đời,
phân tích PCA--ARIMA train-only, đánh giá mô hình RUL, cảnh báo sớm OOF và phân
tích lỗi theo engine.

Báo cáo tập trung vào nội dung học thuật, kết quả đã kiểm chứng và giới hạn; không
đưa timeline hoặc kế hoạch cá nhân vào PDF. Phần dữ liệu giải thích rõ nhãn kết cục, ý nghĩa cột,
phân bố và cách xử lý; phần phương pháp giải thích lý do, đơn vị và nguồn của
các công thức. Dashboard Streamlit chỉ đọc các tệp kết quả; PCA--ARIMA vẫn được
trình bày như phân tích phụ trên Health Index, không phải mô hình RUL. Lệnh tái lập end-to-end đã chạy lại pipeline Pandas từ
raw và đối chiếu artifact cuối.
Checklist bàn giao nằm tại `handoff/FINAL_HANDOFF.md` đã đồng bộ với báo cáo.
Deck 18 slide tại `slides/` đã được cập nhật theo mạch học thuật: bài toán,
dữ liệu, protocol, phương pháp, kết quả, phân tích lỗi, Health Index, cảnh báo
sớm, tái lập và giới hạn. Bản cập nhật dùng các metric đã nghiệm thu và được
render, finalizer kiểm tra ngày 27/09/2026.

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
│   ├── rul-improvement-method.tex  # Nested CV và tiêu chí lựa chọn
│   ├── rul-improvement-results.tex # Kết quả và đánh đổi đã quan sát
│   ├── discussion.tex           # Thảo luận, giới hạn và hướng phát triển
│   ├── conclusion.tex           # Kết luận và phạm vi khái quát hóa
│   └── appendix.tex             # Cấu hình, lệnh tái lập và mapping artifact
└── figures/
    ├── pipeline.tex             # Bản TeX giữ cho report PDF hiện tại
    ├── pipeline.pdf             # Hình pipeline dùng trong báo cáo
    ├── rul-improvement.tex       # Nguồn TikZ của thực nghiệm cải thiện
    ├── rul-improvement.pdf       # Bản xuất đã kiểm tra
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
latexmk -cd -xelatex -interaction=nonstopmode report/figures/rul-improvement.tex
latexmk -cd -xelatex -interaction=nonstopmode report/main.tex
```

Môi trường hiện tại dùng Tectonic portable tại
`tmp/report-tools/tectonic/tectonic.exe`:

```powershell
Push-Location report/figures
../../tmp/report-tools/tectonic/tectonic.exe --keep-logs pipeline.tex
../../tmp/report-tools/tectonic/tectonic.exe --keep-logs statistical-protocol.tex
../../tmp/report-tools/tectonic/tectonic.exe --keep-logs rul-improvement.tex
Pop-Location
Push-Location report
../tmp/report-tools/tectonic/tectonic.exe --keep-logs main.tex
Pop-Location
```

Lần build ngày 27/09/2026 tạo PDF 53 trang, không có lỗi biên dịch, citation
undefined, reference undefined hoặc overfull/underfull box. Tectonic vẫn ghi
cảnh báo môi trường về Fontconfig và font Windows; đây không phải lỗi nội dung.

## Thực nghiệm cải thiện RUL

Đã bổ sung nested GroupKFold 5×3, bảng so sánh bốn giai đoạn, khoảng tin cậy
chênh lệch, trajectory tốt/xấu và bảng đánh đổi trong inner CV. Kết quả chưa
đạt tiêu chí thay baseline: MAE RUL thấp ở vòng ngoài 3,6676 → 3,6730 cycles,
cảnh báo trễ 49 → 53/100 engines. Cấu hình cuối trùng baseline; test MAE vẫn
15,3490 cycles. Ví dụ `hgb_08` giảm sai số RUL thấp khoảng 9,75% trong inner CV
nhưng tăng MAE toàn bộ khoảng 0,32 cycles; đây chưa phải kết quả ngoài độc lập.
Nguồn và lệnh chạy lại: [thực nghiệm](../experiments/README.md).
