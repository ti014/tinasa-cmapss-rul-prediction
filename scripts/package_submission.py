"""Đóng gói bài nộp cuối kỳ thành một file zip.

Học viên: Phan Nguyễn Mai Phương · MSHV 25002711.

Đề yêu cầu nộp một file zip gồm báo cáo, toàn bộ mã nguồn, slide và dữ liệu đã xử lý
(hoặc script tái tạo dữ liệu). Script chỉ gom các file thuộc bài làm; bỏ môi trường ảo,
file tạm, cache, cấu hình cá nhân ở thư mục gốc và các lần chạy thử nghiệm không còn
được sử dụng.

Chạy từ thư mục gốc repository:

    .venv\\Scripts\\python.exe scripts/package_submission.py
    .venv\\Scripts\\python.exe scripts/package_submission.py --name Nhom4_TenKhac_BaoCaoCuoiKy
"""

from __future__ import annotations

import argparse
import fnmatch
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAME = "Nhom4_PhanTichDuLieuIoTBaoTriDuDoan_BaoCaoCuoiKy"

# File và thư mục đưa vào gói, tính từ thư mục gốc.
INCLUDE = [
    "README.vi.md", "README.md", "app.py", "pyproject.toml", "uv.lock",
    "requirements.txt", "requirements-dev.txt", "requirements-etl.txt", "requirements-dashboard.txt",
    ".streamlit/config.toml",
    "src", "notebooks", "scripts", "tests", "infra",
    "data/raw", "data/CONTRACTS.md",
    "report/main.pdf", "report/main.tex", "report/README.md", "report/sections", "report/figures",
    "references/bibliography.bib", "references/README.md", "references/reading-guide.md",
    "slides/cmapss_fd001_presentation.pptx", "slides/cmapss_fd001_presentation.pdf",
    "slides/build_presentation.py", "slides/extract_slide_data.py", "slides/slide_data.json", "slides/README.md",
    "slides/slide_figures.py", "slides/figures",
    "handoff/COURSE_ALIGNMENT.md", "handoff/DATA_WALKTHROUGH.md",
    "assets/diagrams",
    "outputs",
]

# Bỏ trong mọi thư mục.
EXCLUDE_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "fit_cache", "node_modules", "bin"}
EXCLUDE_PATTERNS = ["*.pyc", "*.log", "*.aux", "*.xdv", "*.out", "*.toc", "*.blg", "*.bbl", "*.synctex.gz"]

# Thử nghiệm phụ không thuộc báo cáo: notebook 04, 05 và kết quả của chúng.
EXCLUDE_PREFIXES = ("notebooks/04_", "notebooks/05_", "outputs/notebook04/", "outputs/notebook05/",
                    "outputs/experiments/", "outputs/timeseries/")
# Đầu vào của scripts/build_course_alignment_evidence.py (nguồn kết quả ngưỡng 30/40).
EVIDENCE_INPUTS = {f"outputs/experiments/multiscale_comparison/20261006_study02/{name}"
                   for name in ("outer_predictions.parquet", "manifest.json", "aggregate_metrics.csv")}
# FD002--FD004 chỉ đi qua ETL: giữ biên bản ETL làm nguồn của bảng bốn subset, bỏ dữ liệu trung gian.
CROSS_DATASET_KEEP = ("outputs/cross_dataset/README.md",)


def keep(rel: str) -> bool:
    parts = rel.split("/")
    if EXCLUDE_NAMES.intersection(parts) or any(part.endswith(".egg-info") for part in parts):
        return False
    if any(fnmatch.fnmatch(parts[-1], pattern) for pattern in EXCLUDE_PATTERNS):
        return False
    if rel in EVIDENCE_INPUTS:
        return True
    if rel.startswith(EXCLUDE_PREFIXES):
        return False
    if rel.startswith("outputs/cross_dataset/"):
        return (rel.startswith(CROSS_DATASET_KEEP)
                or fnmatch.fnmatch(rel, "outputs/cross_dataset/FD00?/etl/run_info.json"))
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
