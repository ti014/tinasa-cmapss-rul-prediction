"""Đóng băng nguồn, đầu vào và kiểm tra tính toàn vẹn khi resume."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import platform
import shutil

import joblib
import numpy as np
import pandas as pd
import sklearn

from cmapss.modeling.audit_study.artifact_io import _json_value, _relative, _sha256, _write_json
from cmapss.modeling.audit_study.config import AuditStudyConfig, make_candidate_specs
from cmapss.provenance import local_source_files


SOURCE_FILES = (
    "src/cmapss/__init__.py",
    "src/cmapss/data/__init__.py",
    "src/cmapss/data/contracts.py",
    "src/cmapss/data/readers.py",
    "src/cmapss/data/targets.py",
    "src/cmapss/data/validation.py",
    "src/cmapss/features/__init__.py",
    "src/cmapss/features/contracts.py",
    "src/cmapss/features/rolling.py",
    "src/cmapss/modeling/__init__.py",
    "src/cmapss/modeling/metrics.py",
    "src/cmapss/modeling/audit_study/__init__.py",
)


BASELINE_FILES = (
    "outputs/modeling/models/selected_model.joblib",
    "outputs/modeling/predictions/oof_predictions.parquet",
    "outputs/modeling/predictions/final_evaluation.parquet",
    "outputs/modeling/tables/fold_assignments.csv",
    "outputs/modeling/tables/final_metrics.json",
)


def _initialize_run(output: Path, repo_root: Path, inputs: tuple[Path, ...],
                    config: AuditStudyConfig) -> tuple[dict[str, Any], str]:
    """Ghi manifest và snapshot mã nguồn hoặc kiểm tra chúng khi resume."""
    allowed = repo_root / "outputs/experiments/audit_remediation"
    if output == allowed or allowed not in output.parents:
        raise ValueError("Output phải là lần chạy mới dưới outputs/experiments/audit_remediation")
    source_names = local_source_files(repo_root, SOURCE_FILES)
    sources = {name: _sha256(repo_root / name) for name in source_names}
    baseline = {name: _sha256(repo_root / name) for name in BASELINE_FILES}
    identity = _json_value({
        "config": asdict(config),
        "candidates": [asdict(spec) for spec in make_candidate_specs()
                       if spec.candidate_id in config.candidate_ids],
        "inputs_sha256": {_relative(path, repo_root): _sha256(path) for path in inputs},
        "source_sha256": sources,
        "dependency_lock_sha256": _sha256(repo_root / "uv.lock"),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "baseline_sha256": baseline,
    })
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("Resume không khớp: mã nguồn, đầu vào, cấu hình, dependency hoặc baseline đã đổi")
        for name, digest in sources.items():
            if _sha256(output / "source_snapshot" / name) != digest:
                raise ValueError("Checksum snapshot mã nguồn không khớp")
        if _sha256(output / "source_snapshot/uv.lock") != identity["dependency_lock_sha256"]:
            raise ValueError("Checksum snapshot dependency không khớp")
        manifest = saved
    else:
        if any(output.iterdir()):
            raise ValueError("Thư mục lần chạy không rỗng nhưng thiếu manifest; chọn output mới")
        for name in source_names + ("uv.lock",):
            destination = output / "source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo_root / name, destination)
        manifest = {
            "schema_version": 1, "identity": identity,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "path_convention": "repository-relative POSIX; snapshot paths mirror repository",
            "endpoint_selection": "mean engine raw-RUL MAE at 50/70/90% retrospective lifetime prefixes; NASA/engine then candidate ID ties",
            "endpoint_limit": "lifetime used retrospectively only to position origins, never in X; this panel is not an unseen real-world test distribution",
            "target_cap": "training labels only; all evaluation truth remains raw RUL",
            "history": "cycle age and last/mean relative to first30 sensor means; causal for retained cycle>=30 rows",
            "weighting": "HGB engine total1 (published baseline); Ridge engine equal and mean1 with weighted train-only scaler",
            "warning_selection": "inner OOF; feasible90% lead>=30 minimum early burden; else maximum on-time rate then burden; deterministic cutoff/persistence ties",
            "warning_limit": "illustrative30-cycle lead and90% target, not a maintenance requirement; no negative-engine false-alarm denominator",
            "procedures": ["baseline_fixed", "baseline_calibrated", "selected_calibrated"],
            "test_history": "FD001 test labels/results examined in historical baseline and improvement runs; reporting only, never a selection or promotion gate",
            "promotion_gate": "paired outer endpoint-MAE delta upper95% CI<0; test is not a gate; all NASA/trajectory/policy tradeoffs are reported",
            "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                         "fit_threads": config.fit_threads},
        }
        _write_json(path, manifest)
    cache_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return manifest, cache_identity
