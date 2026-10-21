# Rà soát tính nhất quán toàn dự án

**Học viên:** Phan Nguyễn Mai Phương — **MSHV:** 25002711

Ngày rà soát: 21/10/2026, sau commit `1f10c85`. Phạm vi gồm `src/`, `tests/`,
`notebooks/`, `report/`, `slides/`, `handoff/`, `README.md` và `README.vi.md`.
Lượt rà soát này chỉ đọc file, không sửa nội dung nào.

## Tiến độ sửa

Cách phối hợp: Claude sửa phần chữ (`report/`, `handoff/`, `README*.md`). Codex
sửa phần code (`src/`, `scripts/`, `slides/*.py`, `notebooks/`, `tests/`). Sandbox
của Codex không ghi được `.git`, nên Codex chỉ sửa file và ghi tiến độ vào
`tmp/codex_progress.md`. Claude chạy test, review rồi commit thay. Cập nhật bảng
này sau mỗi commit.

| Việc | Mục audit | Người làm | Trạng thái | Commit |
|---|---|---|---|---|
| Commit phần notebook và HDFS Codex làm trước đó | — | Codex sửa, Claude commit | Xong | `f390abf`, `1f10c85` |
| Báo cáo audit và quy ước trong `AGENTS.md` | — | Claude | Xong | `85693ac` |
| Speech: 12 sensor đổi cùng hướng | A1 | Claude | Xong | `b46ca07` |
| Một hàm định dạng số và một hàm đọc dữ liệu thô dùng chung | C4 | Codex | Xong | `fcbd076` |
| Một bộ bins nhóm RUL dùng chung | A7 | Codex | Xong | `80c5b31` |
| Thuật ngữ trong report, speech, handoff: cycle, cross-validation, động cơ train N | B1, B3 | Claude | Xong | `ef6f50c` |
| Một bảng màu, một font, trục dấu phẩy, nhãn hình tiếng Việt | E1, E2, D2, D3 | Codex sửa, Claude test và commit | Xong (485 test đạt) | `cf68703` |
| Slide đọc số từ `slide_data.json`, sửa câu 145 cycle, đổi "kiểm định chéo" | A3, A9, B3 | Codex sửa; Claude sửa lỗi NaN, test và commit | Xong | `3c45afe` |
| Notebook: phẩy thập phân, mục "Lưu kết quả", gọi `cmapss.*` | D1, D4, D5, B2 | Codex | Đang làm | |
| Docstring, comment, thông báo lỗi tiếng Việt; return type | C1, C2, C3 | Codex | Chờ | |
| Report: 12 sensor, dung lượng 44,9 MB, ghi chú run dựng lại | A2, A6, A8 | Claude | Xong | `f7b84d3` |
| Số test trong report và handoff | A4 | Claude | Chờ Codex xong để chạy test lần cuối | |
| Rút report từ 51 xuống 40 trang (11pt, giãn dòng 1.0, mục lục đến subsection, bỏ 3 listing Python; thêm HI và RUL cap vào glossary) | A5, B4, B6 | Claude | Xong | `4754428` |
| 47,40; thứ tự build deck; FD001–FD004 trong README | E3, E5, E7 | Claude | Xong | `f7b84d3` |
| Ngày cập nhật tài liệu | E4 | Claude | Làm cuối cùng | |
| Macro `\num` trong speech | E6 | — | Bỏ qua: chỉ ảnh hưởng cách in đậm, sửa 100+ chỗ không đổi nội dung | |
| Vẽ lại 6 PNG degradation, maintenance, diagnostics mà report nhúng; nhãn 'cycle' trong script hình và dashboard; thay nốt chữ 'engine' trong report; ngày report 21/10 | E1, B1, B2, E4 | Claude | Xong (bảng số trong outputs/ không đổi) | `db5c1d1` |
| Vẽ lại 3 hình modeling và 1 hình EDA | E1 | Claude | Chờ: hình modeling cần chạy lại huấn luyện; hình EDA chờ Codex xong `eda.py` | |

## Kết luận

Số liệu chính khớp nhau ở mọi tài liệu. Ví dụ: MAE test 15,35, RMSE 20,54,
NASA score 1513, 126 đặc trưng, window 30 cycle, cảnh báo 44/7/49, lịch bảo trì
21/59/20, ARIMA 0,411/0,747/0,948. Mô hình chính là HGB và không dùng RUL cap.

Lệch chủ yếu nằm ở cách viết: tên gọi thuật ngữ, ngôn ngữ docstring, dấu thập
phân và bảng màu của hình. Có đúng một lỗi số liệu cần sửa trước khi thuyết
trình (mục A1).

## A. Số liệu

| # | Mức | Vấn đề | Vị trí | Nguồn đúng |
|---|---|---|---|---|
| A1 | Đã sửa 21/10 | Speech nói "5 cảm biến đổi cùng một hướng ở cả 100 động cơ"; đúng là 12 | `handoff/SPEECH_SCRIPT_VI.tex:357` | `outputs/statistics/tables/sensor_inference.csv`; slide 11 và notebook 02 cell 21 đã đúng |
| A2 | Trung bình | Report chỉ nêu 5 sensor (4, 7, 11, 12, 20) nên người đọc dễ hiểu là chỉ có 5 | `report/sections/current-results.tex:137`, `conclusion.tex` (Q3) | như A1 |
| A3 | Trung bình | Lời nói slide 14: "dự đoán hiếm khi vượt quá khoảng 145 cycle". Thực tế 16,5% dự đoán OOF lớn hơn 145; 145–147 là giá trị **trung bình** khi RUL thật > 150 | `slides/build_presentation.py:875-876` | `outputs/diagnostics/error_diagnostics.json`; câu đúng ở `discussion.tex:15` |
| A4 | Trung bình | Report nêu hai số test khác nhau: 453 đạt (09/10) và 426 đạt (07/10) | `data-processing.tex:103` và `appendix.tex:51`, `DATA_WALKTHROUGH.md:253` | `handoff/FINAL_HANDOFF.md:69` (453) |
| A5 | Thấp | Số trang report: 50 / 51 / "dưới 50" | `README.md:39`, `README.vi.md:38`, `FINAL_HANDOFF.md:14`, `report/main.tex:23`, `report/README.md` | cần đếm lại PDF |
| A6 | Thấp | Raw data ghi "khoảng 46 MB"; thực tế 44,9 MB | `report/sections/discussion.tex:130` | `outputs/acquisition/manifest.json` |
| A7 | Thấp | MAE theo nhóm RUL: report dùng nhóm (60,90] / >90, notebook 03 dùng 3 nhóm khác nên ra 21,53 / 21,77 / 20,09; script dùng thêm 2 cách chia khác | `current-results.tex:283`, notebook 03 cell 18-19, `scripts/report_error_diagnostics.py:21, 49` | cần một hằng số bins dùng chung |
| A8 | Thấp | Appendix trỏ run `20260927_full`, notebook 05 và slide dùng `20260928_rebuild` (số liệu hai run trùng nhau) | `report/sections/appendix.tex:97` | `outputs/experiments/rul_improvement/20260928_rebuild/` |
| A9 | Thấp | Số viết cứng trong code slide, đúng nhưng không tự cập nhật: 4,38; 46,95; 3,78; 41/59; "4 động cơ 31–38"; "12 động cơ 4–27" | `slides/build_presentation.py:870, 892, 1046` và lân cận | nên đọc từ `slide_data.json` |

Trong notebook, phần chữ khớp với output do chính notebook in ra (notebook
00–06).

## B. Thuật ngữ

| # | Khái niệm | Các biến thể đang dùng | Vị trí chính |
|---|---|---|---|
| B1 (đã sửa phần chữ 21/10) | cycle | "30 cycles" trong report/handoff/README (khoảng 111 lần); "30 cycle" trong notebook/slide/speech; speech còn gọi là "chuyến bay" khoảng 50 lần | `SPEECH_SCRIPT_VI.tex:185` trở đi |
| B2 | động cơ | "train engine 1", "động cơ train 1", "Engine {id}", "Số engines" | notebook 01 cell 11, 29, 42; notebook 04 cell 16 |
| B3 (đã sửa speech 21/10; slide do Codex) | cross-validation | "kiểm định chéo" trong slide và speech, trùng nghĩa với kiểm định giả thuyết ở Mục 3.5 | `build_presentation.py:660, 828, 838, 1001`; `SPEECH_SCRIPT_VI.tex:409` |
| B4 | Health Index | HI chỉ định nghĩa ở `methodology.tex:70`, glossary không có; "chỉ số sức khỏe" xuất hiện rải rác | `pipeline.tex:27`, slide `:435, 534`, speech `:272, 345` |
| B5 | window | "trailing window", "cửa sổ thời gian", "window trượt", "rolling window", "window nhân quả" | theory/glossary, `current-results.tex`, README.md, COURSE_ALIGNMENT |
| B6 | RUL cap | glossary không định nghĩa; report chỉ nhắc một lần "cắt nhãn ở 125" | `discussion.tex:58` |
| B7 | Câu chào | "Kính chào quý thầy cô" và "Kính chào thầy/cô và các bạn" | `build_presentation.py:360`, `SPEECH_SCRIPT_VI.tex:177` |

## C. Code

| # | Vấn đề | Chi tiết |
|---|---|---|
| C1 | Ngôn ngữ docstring chia theo package | Tiếng Việt ở các package lõi (data, etl, features/rolling, degradation, maintenance, timeseries, statistics, modeling/baseline, metrics). Tiếng Anh ở các package thí nghiệm (gated_blend, tcn_comparison, lightgbm_comparison, multiscale_comparison, tree_comparison) và vài module lẻ. Có package lẫn cả hai (audit_study, comparison_runner, followup, improvement). Tổng: 280 docstring hàm tiếng Anh, 155 tiếng Việt |
| C2 | Thông báo lỗi và comment | Lỗi: 579 tiếng Anh, 178 tiếng Việt. Comment: 77 tiếng Anh, 55 tiếng Việt |
| C3 | Thiếu kiểu trả về | `hdfs/lifetime_job.py:17-56`, `hdfs/spark_job.py:22`, `tcn_comparison/model.py:14-60`; `hdfs/__main__.py` thiếu docstring module |
| C4 | Hàm bị viết lại | Đọc raw: `slides/slide_figures.py:_train_raw`, `scripts/report_figures.py:read_raw` không dùng `cmapss.data.readers`. Định dạng số kiểu Việt có 3 bản: `build_presentation.vn`, `slide_figures._vn`, `report_figures.vn` |
| C5 | Lệnh chạy | README dùng `.venv/Scripts/python.exe`; slide và handoff dùng `uv run --frozen python` |

## D. Notebook

| # | Vấn đề | Vị trí |
|---|---|---|
| D1 | Dấu chấm thập phân trong markdown ("MAE test là 15.35") | notebook 03 cell 13, 16, 19, 24, 31; notebook 04 cell 19, 23, 26 |
| D2 | Ba kiểu style hình: seaborn colorblind + dpi 100 (01); rcParams dpi 110 + grid (03–06); không đặt style (00, 02) | cell setup từng notebook |
| D3 | Nhãn trục tiếng Anh "Mean engine MAE (cycle)" | notebook 03 cell 12 |
| D4 | Phần kết không đồng nhất: 01, 04, 05, 06 kết bằng mục "Lưu …"; 03 lưu rải rác | notebook 03 |
| D5 | Logic viết lại trong notebook thay vì gọi `cmapss.*` | 01 cell 10 (tuổi thọ), 01 cell 19 (sensor hằng), 04 cell 9 (Spearman), 03 cell 18 và 23 (bins RUL) |
| D6 | Khối tên và MSHV có 3 dạng: `project.AUTHOR_MARKDOWN` xuống hai dòng; notebook một dòng, ngăn bằng "—"; README ngăn bằng "·" | `src/cmapss/project.py` |

## E. Report, slide, speech

| # | Vấn đề | Vị trí |
|---|---|---|
| E1 | Hình report (PNG từ `src`, trục dùng dấu chấm, màu `#2166ac`) và hình slide (`#21409A`, Arial, dấu phẩy) khác palette; `report_figures.py` dùng palette thứ ba | `modeling/baseline/artifact_io.py`, `timeseries/pipeline.py`, `slides/slide_figures.py`, `scripts/report_figures.py` |
| E2 | Nhãn hình tiếng Anh: "actual", "forecast", "95% interval", "validation engines" | `src/cmapss/timeseries/pipeline.py:84-129` |
| E3 | "47,395" đặt cạnh "47,64" dễ đọc thành 47 nghìn | `report/sections/data-processing.tex:206, 289` |
| E4 | Ngày cập nhật lệch: `main.tex` ghi 14/10; `report/README.md`, COURSE_ALIGNMENT, DATA_WALKTHROUGH ghi 12/10 | các file đã nêu |
| E5 | Thứ tự build deck khác nhau; `FINAL_HANDOFF.md` thiếu bước `slide_figures.py` | `slides/README.md`, `handoff/FINAL_HANDOFF.md` |
| E6 | Macro `\num` của speech dùng để in đậm, kể cả cho chữ (`\num{Đề tài 8}`) | `SPEECH_SCRIPT_VI.tex:36` |
| E7 | Markdown viết "FD002--FD004" kiểu LaTeX | `README.md:72` |

## Chưa kiểm

- Số trang thực của `report/main.pdf`.
- Nội dung từng slide trong file PPTX (chỉ kiểm qua `build_presentation.py`).
- Số liệu nằm bên trong ảnh PNG.
- `report/sections/theory.tex`, `glossary.tex` và các file `experiments/*.md`.

Các số đếm thuật ngữ dùng regex, nên có thể lệch vài đơn vị.
