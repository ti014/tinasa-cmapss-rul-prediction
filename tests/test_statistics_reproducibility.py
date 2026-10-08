"""Kiểm tra runtime guard cho validation tái lập thống kê."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from cmapss.statistics.reproducibility import (
    BLOCKED_TEST_FILENAMES,
    TRAIN_FILENAME,
    TRAIN_ONLY_AUDIT_RULE,
    TRAIN_ONLY_AUTHORITY,
    guarded_python_env,
    write_train_only_audit_guard,
)


def _run_python(code: str, *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Chạy Python con để chứng minh `sitecustomize` được nạp thật."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_train_only_runtime_guard_allows_train_and_blocks_test_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / TRAIN_FILENAME).write_text("train\n", encoding="utf-8")
    for filename in BLOCKED_TEST_FILENAMES:
        (raw_dir / filename).write_text("blocked\n", encoding="utf-8")

    guard_dir = tmp_path / "guard"
    write_train_only_audit_guard(guard_dir)
    env = guarded_python_env(guard_dir, base_env=os.environ)

    positive = _run_python(
        f"from pathlib import Path; print(Path('data/raw/{TRAIN_FILENAME}').read_text())",
        cwd=tmp_path,
        env=env,
    )
    assert positive.returncode == 0
    assert "train" in positive.stdout

    for filename in BLOCKED_TEST_FILENAMES:
        negative = _run_python(
            f"from pathlib import Path; Path('data/raw/{filename}').open('r', encoding='utf-8').read()",
            cwd=tmp_path,
            env=env,
        )
        combined_output = negative.stdout + negative.stderr
        assert negative.returncode != 0
        assert TRAIN_ONLY_AUDIT_RULE in combined_output
        assert TRAIN_ONLY_AUTHORITY in combined_output
        assert filename in combined_output
