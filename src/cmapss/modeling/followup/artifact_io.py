"""Đọc, ghi artifact và metadata của study."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import shutil

from cmapss.features.enhanced import FeatureRecipe
from cmapss.modeling.followup.config import StudyConfig
from cmapss.modeling.improvement.artifact_io import _write_json


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_manifest(
    repo: Path, output: Path, dataset_id: str, observations: Path, targets: Path,
    candidates: tuple[FeatureRecipe, ...], config: StudyConfig,
) -> str:
    """Hash mã nguồn, đầu vào và cấu hình; từ chối resume khi bất kỳ phần nào thay đổi."""
    source_paths = sorted(path for folder in ("data", "features", "modeling") for path in (repo / "src/cmapss" / folder).rglob("*.py"))
    input_paths = [observations, targets]
    if dataset_id == "FD001" and config.outer_splits == 5:
        input_paths.append(repo / "outputs/modeling/tables/fold_assignments.csv")
    identity = {
        "dataset_id": dataset_id, "config": asdict(config), "candidates": [asdict(recipe) for recipe in candidates],
        "inputs": {str(path.relative_to(repo).as_posix()): _digest(path) for path in input_paths},
        "runtime_versions": {package: version(package) for package in ("numpy", "pandas", "scikit-learn", "joblib")},
        "source_sha256": {path.relative_to(repo).as_posix(): _digest(path) for path in source_paths},
        "dependencies_sha256": _digest(repo / "uv.lock"),
    }
    path = output / "search_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("Resume không khớp: mã nguồn, đầu vào, cấu hình hoặc dependency đã đổi; dùng thư mục lần chạy mới")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _write_json(path, {"created_at_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
                           "test_history": "Historical test results were already inspected; this is an exploratory follow-up.",
                           "interval_scope": "Paired engine bootstrap conditional on stored outer predictions; excludes refit variability."})
        for source in [*source_paths, repo / "uv.lock"]:
            target = output / "source_snapshot" / source.relative_to(repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
