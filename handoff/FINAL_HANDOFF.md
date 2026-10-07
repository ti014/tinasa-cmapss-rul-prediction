# Gói bàn giao — NASA C-MAPSS FD001--FD004

**Học viên:** Phan Nguyễn Mai Phương · **MSHV:** 25002711.
**Rà soát đồng bộ:** 07/10/2026.

Cập nhật 29/09: [báo cáo thực nghiệm bổ sung](../experiments/FOLLOWUP_20260929.md)
ghi đầy đủ kết quả, protocol và cách tái lập. Đã bổ sung capped RUL/history,
normalization theo điều kiện vận hành, rolling features bằng Spark và benchmark
local theo stage. Thông tin học viên có ở đầu cả sáu notebook; slide có dependency
lockfile. Dashboard vẫn dùng HGB126 ở run lịch sử `20260928_rebuild`; deck và
speech dùng số liệu alignment `20261007` để trình bày HGB126/HGB169; các notebook
follow-up đọc artifact `followup/20260929` khi cần. Các vai trò này được giữ riêng,
không gộp thành một metric triển khai duy nhất.

Đợt cải thiện mã nguồn ngày 07/10/2026 đã hoàn tất bảo vệ Bronze theo đường dẫn
thực tế, staging/rollback cho ETL và rolling, đồng bộ dependency và gom lifecycle
hai runner XGBoost/LightGBM. Toàn bộ suite đạt **426 passed, 1 skipped** (thiếu
torch cho TCN). CLI FD001 chạy từ CWD khác giữ nguyên checksum raw; môi trường
no-dev chạy benchmark được; hai worker spawn thật save/load và resume đúng.
Xem [kế hoạch hoàn tất](CODE_RELIABILITY_PLAN_20261007.md) và
[bằng chứng](CODE_RELIABILITY_20261007.json). Run cũ giữ snapshot lịch sử; source
mới phải dùng run folder mới nếu chạy tiếp, không sửa receipt cũ để vượt guard.

Bản báo cáo cuối bổ sung ranh giới xử lý outlier/nhiễu trong bảng chất lượng dữ
liệu: rolling statistics và Health Index không được gọi là denoising. Dashboard
hiển thị rõ đây là view legacy của HGB126; HGB169 chỉ là predictor candidate có
điều kiện từ alignment replay và không thay thế metric test HGB126.

## Phạm vi

Repository hiện có pipeline Pandas từ raw FD001--FD004 đến artifact ETL, rolling
features, grouped RUL modeling, PCA--ARIMA Health Index và dashboard
read-only. Bộ slide trình bày nằm tại
`slides/cmapss_fd001_presentation.pptx`.

Deck hiện có 20 slide; bản đồng bộ cuối dành 18–20 phút **đã gồm demo
khoảng 3 phút**, chia năm phần: bài toán và dữ liệu; pipeline Big Data; phân tích
chuỗi thời gian; mô hình và đánh giá; mở rộng và kết luận. Mỗi slide có tiêu đề nêu kết luận, một thành
phần trực quan chính, dòng nguồn và speaker notes có thời lượng. Validator của
pptx skill và chart native gate đạt ngày 07/10. Bản PPTX/PDF cuối đã được xuất lại
bằng PowerPoint và render đủ 20 slide để soát các slide trọng yếu.
Kịch bản nói đồng bộ: `handoff/SPEECH_SCRIPT_VI.pdf` (thời lượng mục tiêu
18–20 phút, gồm demo).
Deck 18 slide trước đó (PPTX và Beamer) được giữ tại
`slides/archive/decks/cmapss_fd001_presentation_18slides_20260928*`.

Kết quả RUL chính vẫn được diễn giải trong phạm vi FD001: 100 train engines, 100
test engines, một điều kiện vận hành và một dạng lỗi. Ma trận bổ sung dùng
window 10 cycles để kiểm tra bốn subset; Health Index forecast là phân tích phụ,
không phải dự báo RUL.

## Cấu trúc bàn giao

Walkthrough hỏi đáp về gom file, truy vấn, transform, feature columns và quan hệ
dữ liệu nằm tại [`DATA_WALKTHROUGH.md`](DATA_WALKTHROUGH.md). Sơ đồ có nguồn
chỉnh sửa tại [`assets/diagrams/data-relations.drawio`](../assets/diagrams/data-relations.drawio)
và bản xem trước SVG/PNG.

```text
slides/
├── cmapss_fd001_presentation.pptx # Deck chính 20 slide
├── cmapss_fd001_presentation.pdf  # Bản PDF xuất từ PowerPoint
├── extract_slide_data.py          # Đọc số liệu từ outputs/ → slide_data.json
├── slide_data.json
├── build_presentation.cjs         # Dựng deck bằng pptxgenjs
├── archive/decks/                 # Deck cũ, gồm bản 18 slide PPTX/Beamer
├── archive/builders/              # Script build cũ
└── README.md
handoff/
├── FINAL_HANDOFF.md
├── COURSE_ALIGNMENT_20261007.md
├── DEMO_SCRIPT.md
├── DATA_WALKTHROUGH.md
└── SPEECH_SCRIPT_VI.tex/.pdf
report/main.pdf
handoff/SPEECH_SCRIPT_VI.tex
handoff/SPEECH_SCRIPT_VI.pdf
outputs/{etl,features,modeling,timeseries,cross_dataset,experiments}/
src/cmapss/{etl,features,modeling,timeseries,reproducibility.py}
tests/
```

## Lệnh kiểm tra chính

Kịch bản trình bày trực tiếp nằm tại [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md).

Tái lập toàn bộ pipeline vào output root mới:

```powershell
uv run python -m cmapss.reproducibility `
  --output-root tmp/e2e-repro-final `
  --repo-root . `
  --evidence-path tmp/e2e-repro-final/validation.json
```

Kiểm tra test và cú pháp:

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp=tmp/pytest-handoff --tb=short
.venv\Scripts\python.exe -m compileall -q src tests
uv lock --check
```

Lần kiểm tra end-to-end ngày 2026-09-24 đã đối chiếu 18 bảng, metrics, năm hình
byte-identical, raw hashes và dashboard loader. Lượt nghiệm thu timing tiếp
theo tại `tmp/e2e-review-20260924` cũng đạt; lượt kiểm tra lịch sử ngày 27/09/2026 đạt 175 tests.

Kiểm tra ngày 2026-09-24 gồm loader và trang Cảnh báo sớm trên browser thật
ở localhost:8517; đã quan sát bảng timing và phân phối lead time. Không gọi
kiểm tra loader là rehearsal toàn bộ năm view.
Case demo Engine 023 hiển thị RUL thật 113, dự đoán 170.4768 và absolute error
57.4768; Engine 070 hiển thị Health Index MAE 1.0372, RMSE 1.2205 và coverage
95% là 21.43%. View Provenance hiển thị 100 evaluation rows, 704 forecast rows
và `test_label_join.status=completed_after_final_predictions`.

## Trạng thái kết quả cần trình bày

Artifact alignment đã được tạo tại
`outputs/course_alignment/20261007/summary.json` ở chế độ read-only replay. Không
fit model mới và không đọc test labels để chọn. Artifact khuyến nghị **HGB169**
làm predictor tham chiếu cho whole-trajectory/endpoint; **HGB126** được giữ là
baseline lịch sử và là view dashboard legacy. HGB169 gồm 126 rolling features,
`cycle` và 42 history features. Đây là khuyến nghị có điều kiện vì prediction đã
lưu được so sánh bằng paired bootstrap conditional, chưa bao gồm refit/seed/
selection uncertainty và chưa phải blind confirmation độc lập.

Các panel chung (100 train engines, 17.731 trajectory rows; endpoint lấy các
fraction 0,5/0,7/0,9) của alignment là:

| Model/panel | Mean engine MAE | RMSE | NASA score |
|---|---:|---:|---:|
| HGB169 — trajectory | 17,0009 | 29,8068 | 24.032.889,07 (pooled) |
| HGB169 — endpoint 50/70/90 | 9,8913 | 14,9913 | 2.065,13 (pooled) |

Đây là prediction trên train OOF/outer đã lưu, không phải test FD001 và không
được thay thế cho bảng test lịch sử bên dưới. Paired bootstrap so HGB169 với
HGB126 trên cùng khóa; khoảng tin cậy có điều kiện trên prediction đã lưu.

**Bảng test lịch sử HGB126 (dashboard legacy, FD001, window 30):**

| Metric | Giá trị |
| --- | ---: |
| Mean engine MAE (HGB126 baseline lịch sử) | 15.3490 |
| RMSE | 20.5413 |
| NASA score | 1513.2577 |
| Bootstrap 95% CI của MAE (HGB126 baseline lịch sử) | [12.7286, 18.0866] |

Rolling features dùng window 30 cycles, tạo 27.927 windows và 126 feature
columns. PCA--ARIMA dùng 10 validation engines, split 70/30 và tạo 704 forecast
rows; diagnostics gồm ADF, Ljung--Box và prediction interval coverage.
Warning policy là quyết định riêng. Summary giữ policy calibrated169 lịch sử và
hai sensitivity policy threshold 30/40; dưới threshold 30 HGB169 kém timely hơn,
còn threshold 40 chỉ nhỉnh hơn về on-time cùng burden tăng nhẹ. Không claim
unconditional whole-process improvement. Bảng lịch sử của baseline cố định
`RUL ≤ 30` là 100/100 engine từng cảnh báo, 44 trước, 7 đúng và 49 sau ngưỡng;
lead time median 30, Q1--Q3 26--35, min--max 19--55 cycles. Đây là OOF nội bộ;
false-alarm rate cấp engine là null theo định nghĩa event toàn trajectory, không
phải 0%.

Sensitivity cố định trên cùng 100 train engines:

| Predictor | Threshold | On-time | Early-alarm burden | Median lead |
|---|---:|---:|---:|---:|
| HGB169 | 30 | 45/100 | 0,0107 | 29 cycles |
| HGB169 | 40 | 96/100 | 0,0555 | 38 cycles |

Các policy này không được fit/chọn trên outer result. Calibrated169 historical
readout đạt on-time 94/100, burden 0,0522, median lead 37 cycles, nhưng dùng
inner-selected policy theo outer fold và không phải xác nhận mù độc lập.

## Câu hỏi phản biện và câu trả lời ngắn

1. **Vì sao chia theo engine?** Vì các window trong cùng engine có phụ thuộc
   theo thời gian; chia ngẫu nhiên theo row sẽ làm metric lạc quan.
2. **Có data leakage không?** Scaler, PCA, model selection và GroupKFold chỉ
   học trên train/fold; test labels chỉ được join sau final predictions.
3. **Vì sao dùng HGB169?** Alignment replay giữ cùng 17.731 khóa train, 100
   engine, raw truth và grouped fold; HGB169 thêm cycle và 42 history features
   vào 126 rolling features và tốt hơn trên panel whole-trajectory/endpoint.
   Kết luận là predictor tham chiếu có điều kiện; HGB126 vẫn được giữ làm
   baseline audit. Warning policy được đánh giá riêng.
4. **ARIMA có dự báo RUL không?** Không. ARIMA chỉ forecast Health Index và
   được trình bày như phân tích time-series phụ.
5. **Kết quả có áp dụng cho động cơ thực tế không?** Chưa thể kết luận. FD001
   chỉ đại diện một điều kiện vận hành và một dạng lỗi mô phỏng; transfer sang
   FD002--FD004 có MAE 47,3949--60,3215 cycles trong matrix window 10.
6. **PySpark có thay thế Pandas trong modeling không?** PySpark được dùng cho
   parity ETL/rolling và benchmark local có I/O. Modeling vẫn chạy bằng Pandas và
   scikit-learn.
7. **Tái lập được bằng chứng nào?** Lệnh reproducibility dựng lại output trong
   thư mục mới, so sánh nội dung bảng với tolerance `1e-9`, kiểm raw hashes và
   đọc lại artifact bằng dashboard loader.

## Giới hạn còn lại

- Chưa có serving API.
- Chưa có telemetry thực tế; transfer C-MAPSS đã được đánh giá trong
  `outputs/cross_dataset/` nhưng vẫn là dữ liệu mô phỏng.
- PySpark parity, benchmark local và notebook reproducibility là các kiểm tra
  riêng; không gọi chúng là một phép chạy cluster end-to-end.
- Dashboard đọc artifact đã sinh; không tự huấn luyện lại model.

## Thực nghiệm cải thiện đã hoàn tất

Nested GroupKFold 5×3 thử window 10/20/30, 24 cấu hình HGB và ba cách đặt
trọng số. Kết quả outer của đợt follow-up lịch sử: MAE RUL thấp 3,6676 →
3,6730 cycles; cảnh báo trễ/bỏ lỡ 49 → 53 engines, nên gate không đạt. Đây là
bằng chứng để không promote biến thể đó; quyết định model/protocol nộp bài phải
đọc từ alignment artifact mới.

Ví dụ `hgb_08` giảm low-RUL MAE khoảng 9,75% trong inner CV nhưng tăng MAE toàn
bộ 0,32 cycles. Đây là đánh đổi cần xác định mục tiêu trước đợt tiếp theo;
không gọi nó là kết quả ngoài mẫu độc lập hoặc mô hình đã được thay thế.

Bằng chứng: `outputs/experiments/rul_improvement/20260928_rebuild/validation.json`
đối chiếu 511 fit records, split, source/input hashes, baseline không đổi và
predictions của model đã lưu. Run đầy đủ mất 78,61 phút, thêm một fit cuối.
Run rebuild ngày 28/09/2026 đã tạo manifest theo source bytes tại thời điểm chạy;
source snapshot lưu nguyên các bytes này để đối chiếu sau khi code thay đổi.
Verifier pass với 511 fit records, 100 outer engines, 17.731 dòng mỗi stage,
baseline không đổi, paired CI được tính lại và 100 test predictions khớp model đã
lưu. Các bảng kết quả chính khớp run `20260927_full`; chỉ `fit_seconds` khác do
thời gian chạy. Lệnh kiểm tra dùng run mới:
`uv run --frozen python scripts/verify_rul_improvement.py`.
Full suite ngày 29/09 đạt 198 tests; đợt reliability ngày 07/10 đạt 426 passed,
1 skipped như đã ghi ở đầu tài liệu. Báo cáo có 49 trang, dưới giới hạn 50 trang;
deck PPTX có 20 slide. AppTest kiểm
tra trang **Thử nghiệm cải thiện** và chọn engine 98; đây là kiểm tra giao diện
bằng Streamlit testing API, không phải browser rehearsal của toàn dashboard.

Không chạy lại ETL/reproducibility cũ trong lượt này vì không sửa ETL hoặc
thay artifact `outputs/modeling`. Kiểm tra mới xác nhận baseline giữ nguyên.
Các kiểm tra end-to-end cũ ở trên được giữ rõ ngày thực hiện.

Bản ghi kiểm tra và SHA-256 của các tài liệu bàn giao ngày 29/09 nằm tại
[`VALIDATION_20260929.json`](VALIDATION_20260929.json).
