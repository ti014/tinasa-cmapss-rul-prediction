# Tổ chức package modeling

Phan Nguyễn Mai Phương — MSHV 25002711

Mỗi workflow nằm trong một package riêng. Các module dùng chung ở cấp `modeling`
chỉ chứa contract, metrics và warning events. `pipeline.py` giữ import cũ;
implementation của baseline nằm trong `baseline/`.

```text
modeling/
├── __init__.py               # API mặc định: ModelingConfig, run_modeling_pipeline
├── __main__.py               # python -m cmapss.modeling
├── contracts.py              # Schema và hằng số dùng chung
├── metrics.py                # Sai số RUL, NASA score, bootstrap
├── early_warning.py          # Warning events theo engine
├── pipeline.py               # Import tương thích với code/model cũ
├── baseline/                 # Pipeline RUL mặc định
│   ├── config.py             # ModelingConfig, ModelingResult
│   ├── dataset.py            # Join targets, trọng số, GroupKFold
│   ├── training.py           # Estimator, CV, final predictions
│   ├── evaluation.py         # Protocol và final label join
│   ├── artifact_io.py        # Bảng, model, figures, run_info
│   ├── pipeline.py           # Điều phối train → chọn model → predict
│   └── cli.py                # Tham số command line
└── audit_study/              # Ablation và warning policy
    ├── config.py
    ├── features.py
    ├── estimators.py
    ├── training.py
    ├── evaluation.py
    ├── artifact_io.py
    ├── provenance.py         # Đóng băng source/input và kiểm tra resume
    ├── runner.py
    └── cli.py
```

## Chọn đúng nơi sửa

- Sửa feature hoặc preprocessing: `dataset.py` hoặc `features.py` của workflow.
- Sửa estimator: `estimators.py` hoặc `training.py`; giữ việc fit preprocessing trong training fold.
- Sửa tiêu chí chọn candidate hoặc ghép nhãn cuối: `evaluation.py` của workflow.
- Sửa đường dẫn artifact, cache, metadata: `artifact_io.py` hoặc `provenance.py`.
- Sửa thứ tự các bước: `pipeline.py` hoặc `runner.py`; không đưa logic fit vào CLI.

Ví dụ, thay đổi cách chia GroupKFold theo động cơ thuộc `baseline/dataset.py`.
Thay đổi cách tính NASA score dùng chung thuộc `metrics.py`.

## Hướng phụ thuộc và API

CLI gọi runner. Runner kết hợp cấu hình, dữ liệu, training, evaluation và artifact I/O.
Các module bên dưới không import runner hoặc CLI của chính workflow. Khi dùng lại
logic của workflow khác, import trực tiếp module sở hữu logic đó.

Code mới dùng `cmapss.modeling.baseline` hoặc module cụ thể, chẳng hạn
`cmapss.modeling.baseline.dataset.prepare_modeling_dataset`.
`cmapss.modeling.pipeline` vẫn cung cấp API cũ để notebook và pickle cũ tiếp tục dùng được.
`audit_study` giữ các symbol cũ trong `__init__.py`, nhưng implementation chỉ nằm
ở các module con.

## Chạy và kiểm tra

Các lệnh `cmapss-modeling`, `python -m cmapss.modeling` và
`python -m cmapss.modeling.audit_study` giữ nguyên tên và tham số. Sau khi cập nhật checkout, cài lại editable package
để console entry points nhận đường dẫn module mới: `uv pip install --no-deps -e .`.

Test cấu trúc package và tính tương thích nằm ở `tests/test_modeling_packages.py`.
Test pipeline baseline và cảnh báo sớm nằm trong `tests/test_modeling.py` và
`tests/test_early_warning.py`.

Phải ghi model, lựa chọn cuối và predictions trước khi đọc test labels.
Source snapshot phải bao gồm module con được dùng trong run. Thay đổi source làm
identity của run thay đổi; hãy tạo run mới, không sửa manifest cũ để ép resume.
