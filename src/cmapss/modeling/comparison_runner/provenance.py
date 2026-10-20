"""Đóng băng nguồn, đầu vào và kiểm tra tính toàn vẹn khi resume."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any
import hashlib
import json
import platform
import shutil

import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.data.targets import validate_train_targets
from cmapss.modeling import audit_study as audit
from cmapss.modeling.comparison_runner.artifact_io import _read_json, _sha, _write_json
from cmapss.modeling.comparison_runner.contracts import AUTHOR, ComparisonBackend, ComparisonConfig
from cmapss.provenance import local_source_files


def source_files(root: Path, study_name: str, entrypoint: str) -> tuple[str, ...]:
    """Study source files plus their cmapss import closure and the entrypoint."""
    folder = root / "src/cmapss/modeling" / study_name
    entries = tuple(path.relative_to(root).as_posix() for path in folder.glob("*.py"))
    return local_source_files(root, (*entries, entrypoint))



def installed_versions(model_package: str) -> dict[str, str]:
    names = ("numpy", "pandas", "scikit-learn", model_package, "pyarrow", "joblib",
             "scipy", "threadpoolctl")
    return {"python": platform.python_version(),
            **{name: version(name) for name in names}}


def _assert_immutable(root: Path, output: Path, manifest: dict[str, Any]) -> None:
    """Fail if sources, lock file, inputs, folds or frozen features changed."""
    identity = manifest["identity"]
    for field in ("source_sha256", "inputs_sha256", "outer_fold_source_sha256"):
        for name, digest in identity[field].items():
            if _sha(root / name) != digest:
                raise ValueError(f"immutable {field} changed during run: {name}")
    if _sha(root / "uv.lock") != identity["dependency_lock_sha256"]:
        raise ValueError("dependency lock changed during run")
    if _sha(root / "pyproject.toml") != identity["project_sha256"]:
        raise ValueError("project metadata changed during run")
    for name, digest in identity["frozen_features_sha256"].items():
        if _sha(output / name) != digest:
            raise ValueError(f"frozen feature artifact changed: {name}")


def freeze_inputs(
    root: Path, output: Path, observations_path: Path, targets_path: Path,
    config: ComparisonConfig, backend: ComparisonBackend,
) -> tuple[dict[str, Any], str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Snapshot sources and inputs and save the frozen feature matrix before fitting.

    On resume, the stored matrix is reused after checking every hash against
    the original manifest.
    """
    allowed = root / "outputs/experiments" / backend.study_name
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError(f"output must be a new run descendant of {backend.study_name}")
    source_names = backend.source_files(root)
    provenance = audit._json_value({
        "config": asdict(config), "candidates": [asdict(item) for item in backend.specs(config)],
        "source_sha256": {name: _sha(root / name) for name in source_names},
        "inputs_sha256": {path.relative_to(root).as_posix(): _sha(path)
                          for path in (observations_path, targets_path)},
        "dependency_lock_sha256": _sha(root / "uv.lock"), "project_sha256": _sha(root / "pyproject.toml"),
        "versions": backend.versions(),
        "outer_fold_source_sha256": ({
            "outputs/modeling/tables/fold_assignments.csv": _sha(
                root / "outputs/modeling/tables/fold_assignments.csv")}
            if config.outer_splits == 5 else {}),
        "feature_columns": backend.features, "target_column": "rul",
    })
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    train_path = output / "frozen_train_features.parquet"
    test_path = output / "frozen_test_features.parquet"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        expected = {key: manifest["identity"][key] for key in provenance}
        if expected != provenance:
            raise ValueError("resume mismatch: config/source/input/versions/folds changed")
        _assert_immutable(root, output, manifest)
        for field, directory in (("source_sha256", "source_snapshot"),
                                 ("inputs_sha256", "input_snapshot")):
            for name, digest in manifest["identity"][field].items():
                if _sha(output / directory / name) != digest:
                    raise ValueError(f"snapshot checksum mismatch: {name}")
        if _sha(output / "source_snapshot/uv.lock") != provenance["dependency_lock_sha256"]:
            raise ValueError("dependency snapshot changed")
        if _sha(output / "source_snapshot/pyproject.toml") != provenance["project_sha256"]:
            raise ValueError("project snapshot changed")
        train, test = pd.read_parquet(train_path), pd.read_parquet(test_path)
        assignments = pd.read_csv(output / "outer_assignments.csv")
        if _sha(output / "outer_assignments.csv") != manifest["identity"]["outer_assignments_sha256"]:
            raise ValueError("frozen outer fold assignments changed")
    else:
        if any(output.iterdir()):
            raise ValueError("nonempty run folder has no manifest; choose a fresh output")
        for name in (*source_names, "uv.lock", "pyproject.toml"):
            destination = output / "source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination)
        for path in (observations_path, targets_path):
            destination = output / "input_snapshot" / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        observations = pd.read_parquet(observations_path)
        targets = pd.read_parquet(targets_path)
        if set(observations.split) != {"train", "test"}:
            raise ValueError("observations require exactly train and test splits")
        validate_train_targets(observations[observations.split.eq("train")], targets)
        feature_columns = list(dict.fromkeys((*KEY, *backend.features)))
        features = audit.build_study_features(observations).loc[:, feature_columns]
        train = features[features.split.eq("train")].merge(
            targets, on=list(KEY), how="left", validate="one_to_one")
        train = train.sort_values(["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
        test = features[features.split.eq("test")].sort_values(
            ["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
        if train.rul.isna().any() or not np.isfinite(train[list(backend.features)]).all().all():
            raise ValueError("missing train labels or nonfinite frozen features")
        if len(backend.features) != 169 or not np.isfinite(test[list(backend.features)]).all().all():
            raise ValueError("comparison requires the complete finite 169-feature matrix")
        assignments = audit._outer_assignments(train, config.audit_config(), root)
        assignments.to_csv(output / "outer_assignments.csv", index=False)
        audit._write_parquet(train, train_path)
        audit._write_parquet(test, test_path)
        pd.testing.assert_frame_equal(train, pd.read_parquet(train_path), check_exact=True)
        pd.testing.assert_frame_equal(test, pd.read_parquet(test_path), check_exact=True)
        provenance["frozen_features_sha256"] = {
            path.name: _sha(path) for path in (train_path, test_path)}
        provenance["outer_assignments_sha256"] = _sha(output / "outer_assignments.csv")
        manifest = {
            "schema_version": 1, "author": AUTHOR, "identity": provenance,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_identity": hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
            "procedures": list(backend.arms),
            "primary_selection_metric": "inner full-trajectory RMSE, subject to endpoint/NASA/warning controls",
            "split_contract": "same inner and outer engine folds for every candidate; no engine crosses fit/validation",
            "frozen_feature_replay_exact": True,
            "test_history": "historically inspected FD001 test; reporting only after all choices and predictions are saved",
            "target_scale": "cap affects fitting labels only; every evaluation uses raw RUL",
            "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        }
        _write_json(manifest_path, manifest)
        raw_train = observations[observations.split.eq("train")]
        expected = len(raw_train) - 29 * raw_train.engine_id.nunique()
        if len(train) != expected:
            raise ValueError("frozen train matrix violates complete window30 boundary")
        _write_json(output / "data_boundaries.json", {
            "author": AUTHOR, "window_cycles": 30, "feature_count": len(backend.features),
            "train_rows": len(train), "test_feature_rows": len(test),
            "train_engines": int(train.engine_id.nunique()),
            "test_engines": int(test.engine_id.nunique()),
            "train_prefix_rows_excluded": len(raw_train) - len(train),
            "invalid_raw_rows_removed": 0, "raw_label_evaluation": True,
            "endpoint_fractions": config.endpoint_fractions,
        })
    identity = manifest["source_identity"]
    _assert_immutable(root, output, manifest)
    return manifest, identity, train, test, assignments
