# Bản vá dependencies ngày 22/09/2026

Đã cập nhật ba development dependencies được Dependabot cảnh báo. Runtime
dependencies, code phân tích và dữ liệu không đổi. Cài lại môi trường hiện
có bằng `.venv\Scripts\python -m pip install -r requirements-dev.txt`.

| Package | Phiên bản cũ | Phiên bản đã vá | Nguồn upstream |
| --- | --- | --- | --- |
| pytest | 8.4.2 | 9.0.3 | [Sửa temporary directory không an toàn](https://github.com/pytest-dev/pytest/releases/tag/9.0.3) |
| JupyterLab | 4.4.9 | 4.5.10 | [Bản vá XSS và các lỗi liên quan](https://github.com/jupyterlab/jupyterlab/releases/tag/v4.5.10) |
| nbconvert | 7.16.6 | 7.17.1 | [Bản vá path traversal](https://github.com/jupyter/nbconvert/releases/tag/v7.17.1) |

Thông báo khi push ban đầu chỉ có hai alerts: cùng lỗi pytest
`GHSA-6w46-j5rx-g56g` trong `requirements-dev.txt` và `uv.lock`. Khi kiểm tra
trực tiếp qua GitHub API, có 24 alerts mở, tương ứng 12 advisories riêng biệt
trên ba packages. Ví dụ, phiên bản pytest cũ thuộc khoảng ảnh hưởng `<9.0.3`;
phiên bản đã cài, lock và export đều là `9.0.3`, nằm ngoài khoảng này.

`pyproject.toml`, `uv.lock` và `requirements-dev.txt` đã đồng bộ. Đã export
lại `requirements.txt` nhưng nội dung không đổi vì bản vá chỉ ảnh hưởng nhóm
dev. Các dependency bắc cầu không đổi phiên bản.

Kiểm tra trên Windows/Python 3.12.14:

- Đối chiếu cả 24 khoảng phiên bản từ GitHub: phiên bản cũ bị phát hiện;
  phiên bản mới trong môi trường, lock và requirements đều nằm ngoài khoảng
  ảnh hưởng. [Biên bản JSON](dependency-security.json) lưu package, advisory,
  phiên bản và kết quả local; đây không phải kiểm tra toàn bộ advisory database.
- `uv pip check` và cài `requirements-dev.txt` bằng pip trong môi trường
  kiểm tra riêng đều đạt; `pip check` không có dependency hỏng.
- Toàn bộ suite đạt **69 passed in 6.15s** với pytest 9.0.3.
- Chạy notebook bằng kernel mới, chỉ có train input; 12 CSV và 11 PNG giống
  từng byte với artifact mốc 1. Ba raw checksums không đổi. Runtime audit
  kiểm tra không đọc test observations hoặc labels vẫn đạt.
- JupyterLab trả HTTP 200, instance kiểm tra đã dừng. nbconvert xuất HTML
  thành công; có cảnh báo accessibility về 11 hình thiếu alternative text,
  không phải lỗi bảo mật hoặc lỗi export. Chưa kiểm tra export PDF trong lượt này.

Notebook và `outputs/eda/validation.json` được giữ nguyên như bằng chứng của
mốc 1. Kết quả kiểm tra sau nâng dependency được lưu riêng ở biên bản trên.
Sơ đồ và data contract không cần sửa vì data flow, schema và RUL không đổi.
Không thêm CI, hook hoặc dismiss cảnh báo. Dependabot phải quét commit trên
default branch để cập nhật trạng thái phía GitHub.
