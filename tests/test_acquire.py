"""Kiểm thử bước thu thập bằng gói ZIP giả, không cần mạng."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from cmapss.data import acquire as acq


def _bundle(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in files.items():
            bundle.writestr(f"CMAPSSData/{name}", data)
    return buffer.getvalue()


FILES = {"train_FD001.txt": b"1 1 0.1\n", "RUL_FD001.txt": b"112\n"}
EXPECTED = {name: hashlib.sha256(data).hexdigest() for name, data in FILES.items()}


def test_extract_reads_nested_members_and_checks_hash() -> None:
    assert acq.extract_members(_bundle(FILES), EXPECTED) == FILES
    with pytest.raises(acq.AcquisitionError, match="SHA-256"):
        acq.extract_members(_bundle({**FILES, "RUL_FD001.txt": b"999\n"}), EXPECTED)
    with pytest.raises(acq.AcquisitionError, match="thiếu"):
        acq.extract_members(_bundle({"train_FD001.txt": FILES["train_FD001.txt"]}), EXPECTED)


def test_place_files_never_overwrites_different_content(tmp_path: Path) -> None:
    assert acq.place_files(FILES, tmp_path) == {"RUL_FD001.txt": "written", "train_FD001.txt": "written"}
    assert acq.place_files(FILES, tmp_path)["train_FD001.txt"] == "unchanged"
    (tmp_path / "RUL_FD001.txt").write_bytes(b"edited\n")
    with pytest.raises(acq.AcquisitionError, match="không ghi đè"):
        acq.place_files(FILES, tmp_path)
    assert (tmp_path / "RUL_FD001.txt").read_bytes() == b"edited\n"


def test_acquire_checks_md5_and_writes_manifest(tmp_path: Path) -> None:
    archive = _bundle(FILES)
    md5 = hashlib.md5(archive).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    with pytest.raises(acq.AcquisitionError, match="MD5"):
        acq.acquire(tmp_path / "raw", manifest_path, archive=archive, source_md5="0" * 32, expected=EXPECTED)
    assert not manifest_path.exists()

    acq.acquire(tmp_path / "raw", manifest_path, archive=archive, source_md5=md5, expected=EXPECTED)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["archive_md5"] == md5
    assert {item["name"]: item["sha256"] for item in manifest["files"]} == EXPECTED


def test_recorded_hashes_match_the_raw_files_in_the_repository() -> None:
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    for name, digest in acq.EXPECTED_SHA256.items():
        path = raw / name
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
