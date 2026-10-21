"""Thu thập bộ NASA C-MAPSS từ nguồn công khai và ghi manifest cho tầng Bronze.

Ví dụ chạy từ thư mục gốc repository:: uv run --frozen python -m cmapss.data.acquire
--dest data/raw Gói ZIP được tải về thư mục tạm, kiểm tra MD5, rồi so SHA-256 của từng
file với giá trị đã ghi nhận. File đã có trong ``--dest`` chỉ được giữ nguyên khi trùng
nội dung; file khác nội dung làm lệnh dừng chứ không bị ghi đè.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SOURCE_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
SOURCE_PAGE = "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
SOURCE_MD5 = "79a22f36e80606c69d0e9e4da5bb2b7a"

# SHA-256 của các file dữ liệu trong gói, ghi nhận lúc tải lần đầu (09/2026).
EXPECTED_SHA256 = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
    "train_FD002.txt": "dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f",
    "test_FD002.txt": "de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02",
    "RUL_FD002.txt": "c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b",
    "train_FD003.txt": "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2",
    "test_FD003.txt": "299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790",
    "RUL_FD003.txt": "df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb",
    "train_FD004.txt": "27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218",
    "test_FD004.txt": "1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5",
    "RUL_FD004.txt": "196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1",
}


class AcquisitionError(RuntimeError):
    """Gói tải về hoặc file đích không khớp với giá trị đã ghi nhận."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str = SOURCE_URL, *, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "cmapss-acquire/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def extract_members(archive: bytes, expected: dict[str, str] = EXPECTED_SHA256) -> dict[str, bytes]:
    """Lấy các file dữ liệu trong ZIP theo tên file, bỏ qua thư mục con trong gói.

    Mỗi file phải có đúng SHA-256 đã ghi nhận; thiếu file hoặc lệch hash đều báo lỗi.
    """
    found: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for info in bundle.infolist():
            name = Path(info.filename).name
            if name in expected and not info.is_dir():
                found[name] = bundle.read(info)
    missing = sorted(set(expected) - set(found))
    if missing:
        raise AcquisitionError(f"gói thiếu file: {', '.join(missing)}")
    for name, data in found.items():
        if _sha256(data) != expected[name]:
            raise AcquisitionError(f"{name}: SHA-256 không khớp giá trị đã ghi nhận")
    return found


def place_files(files: dict[str, bytes], dest: Path) -> dict[str, str]:
    """Ghi file vào ``dest``; file đã có thì phải trùng nội dung. Trả về trạng thái từng file."""
    dest.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    for name, data in sorted(files.items()):
        target = dest / name
        if target.exists():
            if _sha256(target.read_bytes()) != _sha256(data):
                raise AcquisitionError(f"{target} đã có nhưng khác nội dung; không ghi đè")
            status[name] = "unchanged"
        else:
            target.write_bytes(data)
            status[name] = "written"
    return status


def acquire(
    dest: Path,
    manifest_path: Path,
    *,
    archive: bytes | None = None,
    source_md5: str = SOURCE_MD5,
    expected: dict[str, str] = EXPECTED_SHA256,
) -> dict:
    archive = download() if archive is None else archive
    md5 = hashlib.md5(archive).hexdigest()
    if md5 != source_md5:
        raise AcquisitionError(f"MD5 của gói là {md5}, cần {source_md5}")
    files = extract_members(archive, expected)
    status = place_files(files, dest)
    manifest = {
        "author": "Phan Nguyễn Mai Phương — MSHV 25002711",
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive_md5": md5,
        "archive_bytes": len(archive),
        "files": [
            {"name": name, "bytes": len(data), "sha256": _sha256(data), "status": status[name]}
            for name, data in sorted(files.items())
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tải NASA C-MAPSS và kiểm tra checksum")
    parser.add_argument("--dest", default="data/raw", help="Thư mục chứa file raw")
    parser.add_argument("--manifest", default="outputs/acquisition/manifest.json", help="File manifest ghi ra")
    parser.add_argument("--archive", help="Dùng file ZIP đã tải sẵn thay vì tải lại")
    args = parser.parse_args(argv)
    archive = Path(args.archive).read_bytes() if args.archive else None
    manifest = acquire(Path(args.dest), Path(args.manifest), archive=archive)
    for item in manifest["files"]:
        print(f"{item['name']:<16} {item['bytes']:>10,} bytes  {item['status']}")
    print(f"MD5 gói: {manifest['archive_md5']} (khớp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
