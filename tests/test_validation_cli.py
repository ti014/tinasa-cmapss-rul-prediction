"""Kiểm tra CLI qua subprocess, báo cáo lỗi và ranh giới đọc test labels."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def run_cli(raw_dir: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    """Gọi entry point thật bằng Python của môi trường kiểm thử."""
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "cmapss.validate", "--raw-dir", str(raw_dir),
         "--output", str(output), *extra], capture_output=True, text=True, encoding="utf-8",
    )


def test_default_validation_works_without_test_labels(raw_dir: Path, tmp_path: Path) -> None:
    isolated = tmp_path / "raw"
    isolated.mkdir()
    for split in ("train", "test"):
        shutil.copyfile(raw_dir / f"{split}_FD001.txt", isolated / f"{split}_FD001.txt")
    output = tmp_path / "report.json"
    result = run_cli(isolated, output)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["checks"][-1]["status"] == "not_requested"
    assert "RUL_FD001.txt" not in report["input_sha256"]
    assert report["test_labels_joined"] is False


def test_full_validation_preserves_raw(raw_dir: Path, tmp_path: Path) -> None:
    paths = list(raw_dir.glob("*FD001.txt"))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output = tmp_path / "report.json"
    result = run_cli(raw_dir, output, "--check-test-labels")
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["input_sha256"] == before
    assert report["checks"][-1]["rows"] == 100
    assert report["test_labels_joined"] is False
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_failure_replaces_previous_success_report(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "train_FD001.txt").write_text("1 1 0\n", encoding="utf-8")
    output = tmp_path / "report.json"
    output.write_text('{"status":"passed"}', encoding="utf-8")
    result = run_cli(raw, output)
    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "[RAW_WIDTH]" in report["error"]


def test_missing_file_has_failure_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = run_cli(tmp_path / "missing", output)
    assert result.returncode == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


@pytest.mark.parametrize("name", ["train_FD001.txt", "new-report.json"])
def test_output_cannot_write_inside_raw(tmp_path: Path, name: str) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    original = raw / "train_FD001.txt"
    original.write_bytes(b"original content")
    result = run_cli(raw, raw / name)
    assert result.returncode == 2
    assert "[RAW_IMMUTABLE]" in result.stderr
    assert original.read_bytes() == b"original content"
    assert list(raw.iterdir()) == [original]
