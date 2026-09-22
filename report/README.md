# Báo cáo cuối kỳ đang thực hiện

[Bản PDF](main.pdf) được cập nhật cùng tiến độ triển khai; đây chưa phải bản
nộp cuối cùng. Nguồn chính là [main.tex](main.tex). Ngày cập nhật hiện tại:
22/09/2026.

Bản hiện tại có bài toán RUL, dữ liệu và contract, cơ sở lý thuyết có trích
dẫn, protocol thống kê mốc 2, phương pháp kèm lý do lựa chọn, kết quả
validation, một số quan sát EDA và bảng suy luận sensor đầu/cuối vòng đời trên
train, khó khăn, giới hạn, khai báo AI/LLM, nhật ký tiến độ và bảng đối chiếu
bằng chứng. Chưa có kết quả mô hình, ARIMA, ETL Silver/Gold hoặc dashboard.
Mốc EDA mô tả đã hoàn thành theo [biên bản nghiệm thu](../outputs/eda/validation.json);
bảng thống kê M2.3a có biên bản tại [outputs/statistics/validation.json](../outputs/statistics/validation.json).

Ngày 22/09/2026 bổ sung lý thuyết window, Ridge, histogram gradient boosting,
grouped validation, paired analysis, bootstrap, FDR, PCA–ARIMA và Spark.
Đã chốt so sánh mean 30 cycles đầu với 30 cycles cuối của từng engine;
helper tạo cặp và helper suy luận train theo engine--sensor đã có và được kiểm
thử. Notebook M2.3a đã xuất 2.100 cặp, 21 dòng suy luận, `run_info.json` và
`validation.json`; hình phân phối và biểu đồ chênh lệch kèm khoảng tin cậy vẫn là bước tiếp theo.
Chênh lệch cặp trung bình, cấu hình bootstrap, Wilcoxon permutation, ties/zeros và mức FDR đã
được chốt ở M2.0 trong
[protocol](sections/statistical-protocol.tex). Ví dụ engine sống 200 cycles
được ghép cặp đoạn 1–30 với 171–200. Không dùng hai đoạn này làm feature
dự đoán tại một cycle chưa quan sát đến cuối đời.

## Cấu trúc

```text
report/
├── main.tex                     # Bìa, định dạng, mục lục và ghép các phần
├── main.pdf                     # Bản đọc đã biên dịch
├── sections/
│   ├── progress.tex             # Nhật ký công việc và điều kiện hoàn thành
│   ├── problem-data.tex         # Bài toán, dữ liệu và chất lượng
│   ├── theory.tex               # Cơ sở lý thuyết và nghiên cứu liên quan
│   ├── methodology.tex          # Thiết kế, lý do, giả định và phần còn mở
│   ├── statistical-protocol.tex # Mốc 2: cặp và cấu hình M2.0 đã chốt
│   ├── current-results.tex      # Chỉ ghi kết quả có artifact
│   ├── discussion.tex           # Khó khăn, giới hạn và khai báo AI
│   └── evidence.tex             # Đường dẫn đối chiếu và thông tin còn thiếu
├── figures/
│   ├── pipeline.tex             # Nguồn TikZ phân biệt hiện tại/dự kiến
│   ├── pipeline.pdf             # Hình vector dùng trong báo cáo
│   ├── statistical-protocol.tex # Sơ đồ TikZ phân tích đầu/cuối đời
│   └── statistical-protocol.pdf # Bản xuất của sơ đồ mốc 2
└── evidence/
    └── report-checks.txt        # Bằng chứng đối chiếu số liệu được trích
```

Bibliography dùng chung tại `../references/bibliography.bib`. Hai hình EDA
được tham chiếu trực tiếp từ `../outputs/eda/figures/`, không tạo bản sao
dễ lệch nguồn. Hai sơ đồ phân biệt luồng dữ liệu hiện có/dự kiến và protocol
mốc 2; bảng suy luận M2.3a đã chạy nhưng chưa có hình thống kê trong báo cáo. Mức đọc/kiểm chứng tài liệu nằm ở
[danh mục nguồn](../references/README.md); không đồng nhất tài liệu đã có DOI
với tài liệu đã đọc toàn văn.

## Cập nhật sau mỗi mốc

- Ghi ngày, công việc, lý do, đầu ra, kiểm tra và phần còn lại trong
  `sections/progress.tex`. Giữ các mốc cũ; cập nhật trạng thái hiện hành và
  thêm mốc mới khi có bằng chứng. Không tự ước lượng phần trăm hoặc giờ công.
- Cập nhật phần nội dung liên quan ngay khi artifact đã có. Mỗi quyết định
  phải giải thích mục tiêu, cách thực hiện, lý do, cách kiểm tra và giới hạn.
  Quyết định chưa chốt phải ghi rõ; không lấy default của thư viện làm quyết định.
- Đối chiếu số liệu với CSV/JSON/log, ghi đường dẫn ở `sections/evidence.tex`.
  Phân biệt quan sát, diễn giải và kế hoạch; không điền metric giả hoặc số 0
  cho kết quả chưa chạy. Không chạy lại training chỉ để viết báo cáo.
- Cập nhật `reportdate` ở `main.tex`, ngày ở trang bìa khi đổi tháng và trạng
  thái trong README này. Giữ nguồn TikZ và PDF nhất quán khi luồng thay đổi.
- Rà docs liên quan, active plan hiện có và cả hai README ở gốc repository.
  Biên dịch lại PDF, kiểm tra log, citation, liên kết và bản render trước bàn giao.

Ví dụ: sau khi hoàn tất ETL, ghi rõ input raw, output Parquet, row counts,
kết quả parity và lý do dùng Parquet. Nếu benchmark chưa chạy, không ghi
“PySpark nhanh hơn Pandas”.

Báo cáo dùng văn phong khách quan và hình thức thực hiện cá nhân, không xưng
ngôi thứ nhất hoặc ngụ ý có nhóm thực hiện. Họ tên và MSHV chỉ nằm trên bìa;
không đặt MSHV vào PDF metadata. Tên giảng viên, mã lớp, mẫu bìa và thông tin
đăng ký chưa được cung cấp đầy đủ nên cần bổ sung trước khi nộp.

## Biên dịch

Cần XeLaTeX hoặc Tectonic, cùng các font Times New Roman, Arial và Consolas
(có trên máy Windows đã dùng để build). Python không thay thế TeX engine.

Với TeX distribution có `latexmk`, từ thư mục gốc repository:

```powershell
latexmk -cd -xelatex -interaction=nonstopmode report/figures/pipeline.tex
latexmk -cd -xelatex -interaction=nonstopmode report/figures/statistical-protocol.tex
latexmk -cd -xelatex -interaction=nonstopmode report/main.tex
```

Lượt khởi tạo dùng Tectonic 0.17.0 portable từ
[release chính thức](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0),
đặt tại `tmp/report-tools/tectonic/tectonic.exe` (không đưa binary vào Git).
Từ gốc repository, chạy:

```powershell
Push-Location report/figures
../../tmp/report-tools/tectonic/tectonic.exe --keep-logs pipeline.tex
../../tmp/report-tools/tectonic/tectonic.exe --keep-logs statistical-protocol.tex
Pop-Location
Push-Location report
../tmp/report-tools/tectonic/tectonic.exe --keep-logs main.tex
Pop-Location
```

Tectonic tự chạy các lượt LaTeX/BibTeX cần thiết; lần đầu cần mạng để tải
TeX packages. Bản portable trong `tmp/` chỉ phục vụ máy hiện tại; máy khác
cần cài Tectonic hoặc TeX distribution. Lệnh `latexmk` là cách dùng tương
đương được hướng dẫn, chưa được thực thi trong lượt này.

Log cần rà: lỗi biên dịch, glyph bị thiếu, citation/reference undefined,
overfull/underfull box và duplicate destination. Sau khi kiểm tra, có thể
dọn aux/log riêng của report; giữ PDF và nguồn. Không đưa thư mục `tmp/`
vào gói nộp.

Bản M2.3a hiện tại đã kiểm chứng PDF **31 trang**, **19 nguồn được trích** trong
nội dung từ danh mục 22 nguồn. Log LaTeX/BibTeX không có overfull/underfull,
glyph thiếu, citation/reference chưa giải quyết hoặc cảnh báo bibliography.
Đã rà bản render và hai sơ đồ TikZ có kiểm tra độc lập; bằng chứng mới nhất nằm
trong `evidence/report-checks.txt`. Tectonic có diagnostic
Fontconfig thiếu cấu hình mặc định và cảnh báo dùng font ở đường dẫn Windows;
build vẫn trả mã 0, chữ tiếng Việt và font được kiểm tra trên bản render.
Đây là giới hạn môi trường đã ghi nhận, không bảo đảm bố cục giống từng byte
trên máy khác.
