"""Đọc, ghi artifact và metadata của study."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import platform
import shutil
import subprocess

import numpy as np
import pandas as pd

from cmapss.modeling.improvement.config import (
    ImprovementConfig,
    WEIGHT_SCHEMES,
    WINDOWS,
    make_hgb_candidates,
)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Không thể serialize {type(value)!r}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prepare_run(output: Path, observations: Path, targets: Path, config: ImprovementConfig) -> None:
    """Ghi manifest và snapshot baseline; từ chối resume nếu đầu vào thay đổi."""
    import sklearn
    identity = {
        "config": asdict(config),
        "inputs": {str(path): _sha256(path) for path in (observations, targets)},
        "source_sha256": {str(path): _sha256(path) for path in sorted(Path("src/cmapss").rglob("*.py"))},
        "dependencies_sha256": _sha256(Path("uv.lock")),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    manifest_path = output / "search_manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("Resume không khớp: cấu hình, đầu vào, mã nguồn hoặc dependency đã đổi; dùng thư mục output mới")
        for name, digest in saved["baseline_sha256"].items():
            if _sha256(output / "baseline_snapshot" / name) != digest:
                raise ValueError("Checksum snapshot baseline không khớp")
        return
    if any(output.iterdir()):
        raise ValueError("Thư mục output không rỗng nhưng thiếu search manifest")
    shutil.copytree("outputs/modeling", output / "baseline_snapshot")
    baseline_hashes = {str(path.relative_to(output / "baseline_snapshot")): _sha256(path)
                       for path in (output / "baseline_snapshot").rglob("*") if path.is_file()}
    git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (output / "working_tree.patch").write_text(subprocess.check_output(["git", "diff", "HEAD"], text=True, encoding="utf-8"), encoding="utf-8")
    (output / "git_status.txt").write_text(subprocess.check_output(["git", "status", "--short"], text=True, encoding="utf-8"), encoding="utf-8")
    source_dir = output / "source_snapshot"
    source_dir.mkdir()
    for name in identity["source_sha256"]:
        path = Path(name)
        destination = source_dir / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shutil.copy2("uv.lock", source_dir / "uv.lock")
    _write_json(manifest_path, {
        "identity": identity, "git_revision": git_revision, "baseline_sha256": baseline_hashes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": make_hgb_candidates(config), "weights": WEIGHT_SCHEMES, "windows": WINDOWS,
        "selection_rule": "low-RUL engine MAE, then late/missed rate, overall engine MAE, candidate ID; all three controls no worse than stage baseline",
        "test_history": "FD001 test labels and baseline scores were examined before this study; this is not a pristine test set",
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "fit_threads": config.fit_threads},
    })
