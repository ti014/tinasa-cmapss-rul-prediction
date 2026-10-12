# Bàn giao bài báo cáo — Đề tài 8

**Học viên:** Phan Nguyễn Mai Phương · **MSHV:** 25002711 · **Đồng bộ:** 12/10/2026

## Đọc gì trước

Chỉ cần một file: **[`SPEECH_SCRIPT_VI.pdf`](SPEECH_SCRIPT_VI.pdf)** (sổ tay thuyết trình).
Mỗi trang là một slide, ghi rõ nhìn vào đâu, đọc câu gì và khi nào chuyển slide. Phần
demo dashboard có ảnh chụp từng bước. Cuối sổ tay có câu hỏi phản biện và một trang số
liệu tra nhanh.

## Bài làm gồm những gì

- **Báo cáo:** [`report/main.pdf`](../report/main.pdf), 50 trang.
- **Slide:** [`slides/cmapss_fd001_presentation.pptx`](../slides/cmapss_fd001_presentation.pptx),
  20 slide trên template IUH; bản PDF dự phòng nằm cùng thư mục.
- **Dashboard:** mở bằng lệnh bên dưới, chín trang, dùng để demo.
- **Notebook:** bảy notebook trong `notebooks/`, đã chạy sẵn, ghi họ tên và MSHV ở đầu.
- **Mã nguồn và dữ liệu đã xử lý:** `src/cmapss/` và `outputs/`.

## Ngày báo cáo

Làm theo trang "Trước buổi báo cáo" trong sổ tay. Lệnh duy nhất cần gõ:

```powershell
uv run --extra dashboard streamlit run app.py
```

Mang theo bản PDF của slide trong USB, phòng khi máy chiếu không có PowerPoint.

## Nộp bài

Đề yêu cầu một file zip tên `Nhom<X>_TenDeTai_BaoCaoCuoiKy.zip`, gồm báo cáo, mã nguồn,
slide và dữ liệu đã xử lý (hoặc script tạo lại dữ liệu). File nộp là
`dist/Nhom4_PhanTichDuLieuIoTBaoTriDuDoan_BaoCaoCuoiKy.zip`, tạo lại bằng `.venv\Scripts\python.exe scripts/package_submission.py`.
Không đưa vào zip: các lần chạy thử nghiệm không dùng trong `outputs/experiments/`, `tmp/`, `.venv/`,
`slides/node_modules/`, `slides/template/` và các file cấu hình cá nhân ở thư mục gốc
(`auth.json`, `*.sqlite`, `config.toml`).

## Nếu cần sửa số liệu hoặc slide

Pipeline đã đồng bộ: rolling đọc Silver observations; engine_summary là bảng
tổng hợp riêng. EWMA/HI dùng cho bất thường và được áp dụng lên test bằng tham
số fit trên train khi lập lịch. Pandas/PySpark đối chiếu ETL/rolling;
MapReduce/Spark trên HDFS đối chiếu engine_summary trên một node Docker.

Nếu chỉ sửa cách trình bày, dựng lại deck từ artifact hiện có bằng builder,
xuất PDF/preview rồi build sổ tay. Chỉ tái tạo degradation/maintenance khi cần
tính lại artifact; sửa câu chữ không yêu cầu chạy lại model RUL.

Chạy lại theo thứ tự, từ thư mục gốc project:

```powershell
uv run --frozen python -m cmapss.degradation
uv run --frozen python -m cmapss.maintenance
uv run --frozen python slides/extract_slide_data.py
uv run --no-project --with python-pptx==1.0.2 --with pillow==12.3.0 python slides/build_presentation.py
powershell -File scripts/export_slides.ps1 -PreviewDirectory handoff/assets/slides
```

Sau lệnh cuối, đổi tên ảnh trong `handoff/assets/slides/` thành `slide01.png` đến
`slide20.png`, rồi build lại sổ tay:

```powershell
cd handoff
..\tmp\report-tools\tectonic\tectonic.exe SPEECH_SCRIPT_VI.tex
```

Kiểm tra toàn bộ: `.venv\Scripts\python.exe -m pytest -q` (lần gần nhất: 453 đạt, 1 bỏ qua
vì máy không có `torch`).

## Các file khác trong thư mục này

- [`COURSE_ALIGNMENT.md`](COURSE_ALIGNMENT.md): đối chiếu từng yêu cầu của đề và rubric với
  bằng chứng trong bài. Đọc khi muốn chắc không sót yêu cầu nào.
- [`DATA_WALKTHROUGH.md`](DATA_WALKTHROUGH.md): giải thích chi tiết cách gom, lưu, truy vấn
  và biến đổi dữ liệu. Đọc khi bị hỏi sâu về phần dữ liệu.
- `archive/`: biên bản và kịch bản cũ, không cần đọc.
