# Kế hoạch cải thiện độ tin cậy mã nguồn

Ngày: 07/10/2026. Trạng thái: hoàn tất.

Học viên: Phan Nguyễn Mai Phương. MSHV: 25002711.

## Kết quả cần đạt

Hoàn tất bốn cải thiện đã rà soát: bảo vệ đúng thư mục Bronze, giữ artifact
nhất quán sau lỗi ghi, đồng bộ dependency và giảm logic trùng giữa runner
XGBoost/LightGBM. Kết quả được chứng minh bằng kiểm thử và chạy pipeline nhỏ.

## Phạm vi và bằng chứng ban đầu

- `pandas_pipeline.py` và `features/rolling.py` dùng `data/raw` theo CWD;
  đường dẫn tùy chỉnh chưa được bảo vệ.
- Fault injection lần ghi Parquet thứ hai đã tái hiện observations mới đi
  cùng targets và manifest cũ trong thư mục tạm.
- Benchmark import `psutil` nhưng dependency chỉ nằm trong nhóm dev.
- Bốn requirements export chưa khớp lock.
- Hai runner lặp orchestration, provenance, folds và receipts.

Giữ giao thức thực nghiệm, kết quả khoa học và các source snapshot lịch sử.
Không cần chạy training đầy đủ để kiểm chứng đợt sửa này.

## Cách thực hiện

1. Lưu raw boundary đã resolve từ pipeline; chuyển tiếp vào rolling features.
   Kiểm tra cả output và các file đích thực tế, kể cả symlink/junction.
2. Ghi artifact vào staging cùng filesystem. Chỉ publish khi đã ghi đủ.
   Backup file hiện có và rollback nếu publication lỗi; metadata được publish
   cuối. Giữ file ngoài phạm vi writer. Không gọi thay nhiều file là atomic.
3. Khai báo dependency runtime đúng, cập nhật lock, export đủ bốn requirements;
   kiểm tra CLI benchmark trong môi trường không cài dev.
4. Tách orchestration dùng chung; giữ worker có thể pickle và adapter từng
   mô hình. Kiểm chứng cache, nguồn, receipts, folds và luồng final bằng fixture.
5. Chạy kiểm thử liên quan rồi toàn bộ suite, kiểm tra diff và ghi bằng chứng
   hoàn tất ở kế hoạch này.

## Rủi ro và khôi phục

- Output hiện có có thể chứa README, benchmark hoặc parity: chỉ thay file
  writer sở hữu; kiểm thử việc giữ các file còn lại.
- Windows có thể chặn replace khi file đang mở: rollback sau lỗi; nếu rollback
  cũng lỗi, giữ backup và báo đường dẫn phục hồi.
- Publication nhiều file không bảo đảm reader đồng thời hoặc mất điện đột
  ngột. Phạm vi bảo đảm là lỗi trong quá trình ghi/publish mà writer xử lý được.
- Source thay đổi làm run cũ không còn phù hợp để resume; giữ run cũ và tạo
  run mới, không chỉnh snapshot để vượt provenance guard.

## Tiến độ

- [x] Raw boundary đúng, không phụ thuộc CWD; có positive/negative proof.
- [x] Staging, rollback và bảo toàn file ngoài phạm vi; có fault injection.
- [x] Dependency/lock/requirements khớp; runtime tối thiểu chạy được.
- [x] Hai runner dùng orchestration chung; hành vi từng mô hình được kiểm chứng.
- [x] Kiểm thử tích hợp, toàn bộ suite và tài liệu sử dụng đã cập nhật.

## Kiểm chứng cần có

- Output bằng hoặc bên trong custom raw bị từ chối trước mọi ghi; raw hash
  nguyên vẹn khi thay CWD và khi chạy CLI.
- ETL và rolling round trip thành công. Lỗi Parquet/SQLite/metadata ở staging
  không đổi artifact cũ; lỗi publish khôi phục toàn bộ artifact cũ.
- Rerun thành công cập nhật đủ artifact, manifest và giữ ancillary files.
- Export mới so với lock không có chênh lệch; CLI benchmark --help và worker
  Pandas nhỏ chạy trong môi trường no-dev.
- Tests runner phân biệt estimator/grid của XGBoost và LightGBM, kiểm tra
  nguồn bất biến, receipts và điều kiện chuyển sang final.
- Toàn bộ pytest suite đạt; ghi rõ skip và giới hạn thực tế.

## Kết quả

- Đã thêm `src/cmapss/artifacts.py`; ETL/rolling dùng staging, backup và
  rollback, bảo vệ actual raw + override + input files. SQLite được đóng
  connection trước khi publish trên Windows. Cleanup lỗi giữ exception gốc;
  cleanup sau publish thành công phát warning rõ trạng thái.
- ETL CLI staging bao cả benchmark/parity được yêu cầu; manifest đánh dấu
  auxiliary checks hiện tại và receipts liên kết đúng input/time.
- `psutil` đã chuyển vào runtime. `uv.lock` và bốn requirements đã đồng bộ;
  5 kiểm thử export/lock đạt. Môi trường `--no-dev` có psutil nhưng không có
  pytest/PySpark/JupyterLab; CLI benchmark và worker Pandas thực chạy được.
- Lifecycle chung ở `modeling/comparison_runner.py`; runner riêng còn 170 và
  174 dòng. Các scientific contracts, selectors, estimators và cache của hai
  study giữ nguyên so với snapshot cũ. Tests kiểm chứng folds, raw evaluation
  labels, source freeze, receipts và cache resume; 2 worker spawn thật fit HGB
  2 iterations trên 80 dòng/169 features, save/load và reuse checksum đúng.
- Targeted ETL/features: **53 passed**. Toàn bộ suite: **426 passed, 1 skipped**
  trong 79,43 giây; 14 deprecation warnings của dependency. Skip duy nhất do
  không có torch cho TCN, không nằm trong bốn phần cải thiện này.
- CLI thật chạy từ CWD khác: 33.727 observations, 20.631 targets, 200 engine
  summaries và 27.927 feature rows (17.731 train, 10.196 test). SHA-256 toàn bộ
  raw files trước/sau giống nhau. Benchmark receipt khớp manifest.
- README EN/VI và `data/CONTRACTS.md` đã mô tả boundary, recovery và receipts.
  `git diff --check` sạch. Bằng chứng tổng hợp:
  [CODE_RELIABILITY_20261007.json](CODE_RELIABILITY_20261007.json).

Lệnh full suite đã dùng:

```powershell
.venv\Scripts\python -m pytest tests -q --tb=short `
  --ignore=rules --ignore=skills --ignore=vendor_imports `
  --basetemp=tmp/pytest-code-reliability-20261007-full01 `
  --junitxml=tmp/code-reliability-pytest-20261007.xml
```

Basetemp mới đã được kiểm tra nằm trong workspace trước khi chạy. Ignore xử lý
junction không truy cập được trong môi trường Windows này; không bỏ qua test
trong `tests/`. Linux được kiểm tra dependency markers/pins từ lock, chưa chạy
trực tiếp trên máy Linux của người dùng. Không chạy lại nested scientific study
và không thay các metric hay snapshot lịch sử.
