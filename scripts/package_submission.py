"""Đóng gói bài nộp cuối kỳ thành một file zip.

Học viên: Phan Nguyễn Mai Phương · MSHV 25002711.

Đề yêu cầu nộp một file zip gồm báo cáo, toàn bộ mã nguồn, slide và dữ liệu đã xử lý
(hoặc script tái tạo dữ liệu). Script chỉ gom các file thuộc bài làm; bỏ môi trường ảo,
file tạm, cache, cấu hình cá nhân ở thư mục gốc và các lần chạy thử nghiệm không còn
được sử dụng.

Chạy từ thư mục gốc repository:

    .venv\\Scripts\\python.exe scripts/package_submission.py
    .venv\\Scripts\\python.exe scripts/package_submission.py --name Nhom05_BaoTriDuDoan_BaoCaoCuoiKy
"""

from __future__ import annotations

import argparse
import fnmatch
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "PhanNguyenMaiPhuong_DeTai8_BaoTriDuDoan_BaoCaoCuoiKy"

# File và thư mục đưa vào gói, tính từ thư mục gốc.
INCLUDE = [
    "README.vi.md", "README.md", "app.py", "pyproject.toml", "uv.lock",
    "requirements.txt", "requirements-dev.txt", "requirements-etl.txt", "requirements-dashboard.txt",
    ".streamlit/config.toml",
    "src", "notebooks", "scripts", "tests",
    "data/raw", "data/CONTRACTS.md",
    "report/main.pdf", "report/main.tex", "report/README.md", "report/sections", "report/figures",
    "references/bibliography.bib", "references/README.md", "references/reading-guide.md",
    "slides/cmapss_fd001_presentation.pptx", "slides/cmapss_fd001_presentation.pdf",
    "slides/build_presentation.py", "slides/extract_slide_data.py", "slides/slide_data.json", "slides/README.md",
    "handoff/COURSE_ALIGNMENT.md", "handoff/DATA_WALKTHROUGH.md",
    "assets/diagrams",
    "experiments",
    "outputs",
]

# Bỏ trong mọi thư mục.
EXCLUDE_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "fit_cache", "node_modules", "bin"}
EXCLUDE_PATTERNS = ["*.pyc", "*.log", "*.aux", "*.xdv", "*.out", "*.toc", "*.blg", "*.bbl", "*.synctex.gz"]

# Bỏ trong outputs/: lần chạy thử nghiệm không còn dùng và dữ liệu trung gian của FD002--FD004.
OUTPUT_KEEP_EXPERIMENTS = ("outputs/experiments/rul_improvement/20260928_rebuild/",
                           "outputs/experiments/followup/20260929/", "outputs/experiments/README.md")
CROSS_DATASET_KEEP = ("outputs/cross_dataset/matrix.json", "outputs/cross_dataset/README.md",
                      "outputs/cross_dataset/evaluations/")


def keep(rel: str) -> bool:
    parts = rel.split("/")
    if EXCLUDE_NAMES.intersection(parts) or any(part.endswith(".egg-info") for part in parts):
        return False
    if any(fnmatch.fnmatch(parts[-1], pattern) for pattern in EXCLUDE_PATTERNS):
        return False
    if rel.startswith("outputs/experiments/"):
        return rel.startswith(OUTPUT_KEEP_EXPERIMENTS)
    if rel.startswith("outputs/cross_dataset/"):
        return rel.startswith(CROSS_DATASET_KEEP)
    return True


def collect() -> list[Path]:
    files: list[Path] = []
    for item in INCLUDE:
        path = ROOT / item
        if not path.exists():
            raise FileNotFoundError(f"thiếu {item}")
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        files.extend(p for p in candidates if keep(p.relative_to(ROOT).as_posix()))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Đóng gói bài nộp cuối kỳ")
    parser.add_argument("--name", default=DEFAULT_NAME, help="tên file zip, không có đuôi .zip")
    parser.add_argument("--output-dir", default="dist")
    args = parser.parse_args()

    files = collect()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.name}.zip"
    listing = []
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            archive.write(path, f"{args.name}/{rel}")
            listing.append(f"{path.stat().st_size:>12,}  {rel}")
        archive.writestr(f"{args.name}/DANH_SACH_FILE.txt",
                         "Phan Nguyễn Mai Phương - MSHV 25002711\n"
                         "Danh sách file trong gói nộp (kích thước byte, đường dẫn)\n\n" + "\n".join(listing) + "\n")
    total = sum(p.stat().st_size for p in files)
    print(f"wrote {target.relative_to(ROOT)}: {len(files)} file, "
          f"{total / 1e6:.1f} MB trước nén, {target.stat().st_size / 1e6:.1f} MB sau nén")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
