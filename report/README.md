# Báo cáo cuối kỳ

**Học viên:** Phan Nguyễn Mai Phương · **MSHV:** 25002711
**Giảng viên hướng dẫn:** TS. Võ Thị Thanh Vân

[Bản PDF](main.pdf) được biên dịch từ [main.tex](main.tex). Báo cáo trình bày Đề tài 8,
phân tích dữ liệu IoT và bảo trì dự đoán trên NASA C-MAPSS FD001, trong giới hạn 40 trang.

## Cấu trúc

```text
report/
├── main.tex                  # Bìa, tóm tắt, mục lục, cấu hình listings, ghép các phần
├── main.pdf                  # Bản PDF đã biên dịch
├── sections/
│   ├── glossary.tex          # Thuật ngữ và chữ viết tắt
│   ├── problem-data.tex      # Mục 1 Giới thiệu, Mục 2 Dữ liệu
│   ├── theory.tex            # Mục 3 Cơ sở lý thuyết
│   ├── data-processing.tex   # Mục 4 Tổ chức, truy vấn và biến đổi dữ liệu
│   ├── methodology.tex       # Mục 5 Phương pháp
│   ├── current-results.tex   # Mục 6 Kết quả
│   ├── discussion.tex        # Mục 7 Thảo luận, Mục 8 Khó khăn và bài học
│   └── conclusion.tex        # Mục 9 Kết luận
└── figures/
    ├── diagram-style.tex           # Màu, cỡ ô, mũi tên dùng chung cho các sơ đồ TikZ
    ├── data-flow.tex / .pdf        # Luồng dữ liệu: bảng, kích thước, phép biến đổi
    ├── data-relations.tex / .pdf   # Quan hệ giữa các bảng, khóa và bậc quan hệ
    └── pipeline.tex / .pdf         # Pipeline tổng thể
```

## Nguồn hình

- Sơ đồ TikZ nằm trong `figures/`, nạp chung `figures/diagram-style.tex`.
- Hình kết quả được tham chiếu trực tiếp từ `outputs/` (EDA, khử nhiễu, Health Index,
  mô hình, cảnh báo, lịch bảo trì, HDFS) để không lệch nguồn.
- Ba hình chẩn đoán (minh họa window, dự đoán OOF theo RUL thật, cảnh báo theo tuổi thọ)
  do `scripts/report_figures.py` sinh vào `outputs/diagnostics/figures/`.
- Số liệu chẩn đoán ở Mục 7 do `scripts/report_error_diagnostics.py` tính lại từ artifact
  đã có, ghi vào `outputs/diagnostics/error_diagnostics.json`.
- Hình benchmark mở rộng quy mô do `slides/slide_figures.py` sinh vào
  `slides/figures/benchmark_scaling.png`.

Các script trên chỉ đọc dữ liệu và artifact có sẵn, không huấn luyện lại mô hình:

```powershell
uv run --frozen python scripts/report_figures.py
uv run --frozen python scripts/report_error_diagnostics.py
```

## Biên dịch

Với Tectonic portable của môi trường hiện tại, chạy từ thư mục gốc repository:

```powershell
Push-Location report
../tmp/report-tools/tectonic/tectonic.exe --keep-logs main.tex
Pop-Location
```

Với TeX distribution có `latexmk`:

```powershell
latexmk -cd -xelatex -interaction=nonstopmode report/figures/pipeline.tex
latexmk -cd -xelatex -interaction=nonstopmode report/figures/data-flow.tex
latexmk -cd -xelatex -interaction=nonstopmode report/figures/data-relations.tex
latexmk -cd -xelatex -interaction=nonstopmode report/main.tex
```

Danh mục tài liệu tham khảo dùng chung tại `../references/bibliography.bib`.
