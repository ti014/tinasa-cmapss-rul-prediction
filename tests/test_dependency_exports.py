"""Check that pip installations use the same dependency lock as uv."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = (
    ("requirements.txt", ("--no-dev",)),
    ("requirements-dev.txt", ("--group", "dev")),
    ("requirements-etl.txt", ("--extra", "etl", "--no-dev")),
    ("requirements-dashboard.txt", ("--extra", "dashboard", "--no-dev")),
)


@pytest.fixture
def uv_executable() -> str:
    """Use the project's lock tool when it is available to the test runner."""
    executable = shutil.which("uv")
    if executable is None:
        pytest.skip("uv is required to verify dependency lock exports")
    return executable


def _requirements(text: str) -> list[str]:
    """Ignore generated comments while preserving versions and platform markers."""
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def test_dependency_lock_matches_project(uv_executable: str) -> None:
    """Reject a stale lock before comparing frozen exports derived from it."""
    result = subprocess.run(
        [uv_executable, "lock", "--check", "--offline"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("filename", "selection"), EXPORTS)
def test_requirements_match_frozen_lock(
    uv_executable: str, filename: str, selection: tuple[str, ...]
) -> None:
    """Detect missing transitive pins, including requirements for another OS."""
    result = subprocess.run(
        [
            uv_executable,
            "export",
            "--frozen",
            "--offline",
            *selection,
            "--no-hashes",
            "--no-annotate",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert _requirements((ROOT / filename).read_text(encoding="utf-8")) == _requirements(result.stdout)
