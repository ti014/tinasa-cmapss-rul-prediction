"""Run the nested gated HGB study on top of an existing LightGBM/HGB study run.

No model is fitted: 46 cached CV predictions and the 2 final HGB models are reused.
Each fold's choice is saved before its outer predictions are read, and test labels
are read only after the final predictions are written.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cmapss.data.contracts import KEY
from cmapss.data.readers import read_test_labels
from cmapss.modeling import audit_study as audit
from cmapss.modeling.metrics import add_prediction_errors, summarize_prediction_errors

from .analysis import (
    calibrate_policy, candidate_readout, choose_procedure, select_gate_candidate,
    write_readouts,
)
from .config import StudyConfig, specs
from .contracts import GateSpec
from .gating import blend_predictions
from .inference import load_inference

BASE_ARMS = ("hgb_uncapped", "hgb_cap125")
ARMS = BASE_ARMS + ("gated_blend",)
AUTHOR = {"full_name": "Phan Nguyễn Mai Phương", "student_id": "25002711"}
DEFAULT_SOURCE = "outputs/experiments/lightgbm_comparison/20261005_study01"
DEFAULT_OUTPUT = "outputs/experiments/gated_blend_comparison/20261005_study01"
PROTOCOL = "experiments/GATED_BLEND_PROTOCOL.md"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"JSON object required: {path}")
    return result


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via a temp file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(audit._json_value(payload), sort_keys=True,
                                    ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    # On Windows a reader holding the file can block os.replace for a moment;
    # retry for about 1.6 s before giving up.
    delays = (.02, .04, .08, .16, .32, .5, .5)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def _resolve(root: Path, value: str | Path) -> Path:
    """Resolve a path under the repo root; reject paths outside it."""
    path = Path(value)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path must remain inside repository: {resolved}")
    return resolved


def _versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow", "joblib",
                "scipy", "threadpoolctl")
    return {"python": platform.python_version(), **{name: version(name) for name in packages}}


def _source_files(root: Path) -> tuple[str, ...]:
    """Source files of this package, their cmapss import closure, script and protocol."""
    pending = [path.relative_to(root).as_posix()
               for path in (root / "src/cmapss/modeling/gated_blend").glob("*.py")]
    pending.extend(["scripts/run_gated_blend_study.py", PROTOCOL])
    found: set[str] = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"derived study source missing: {name}")
        found.add(name)
        if not name.startswith("src/") or path.suffix != ".py":
            continue
        package = Path(name).parent.relative_to("src").as_posix().replace("/", ".")
        parent = Path(name).parent
        while parent != Path("src"):
            initializer = (parent / "__init__.py").as_posix()
            if (root / initializer).is_file():
                pending.append(initializer)
            parent = parent.parent
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".")
                    base = ".".join(parts[:len(parts) - node.level + 1])
                    module = ".".join(filter(None, (base, node.module)))
                else:
                    module = node.module or ""
                modules = [module, *(f"{module}.{alias.name}" for alias in node.names)]
            for module in modules:
                if not module.startswith("cmapss"):
                    continue
                candidate = Path("src") / Path(*module.split("."))
                for local in (candidate.with_suffix(".py"), candidate / "__init__.py"):
                    if (root / local).is_file():
                        pending.append(local.as_posix())
    return tuple(sorted(found))


def _frame_digest(frame: pd.DataFrame) -> str:
    """Same frame hash as the source study (includes column order and dtypes)."""
    digest = hashlib.sha256()
    digest.update(repr(tuple(frame.columns)).encode("utf-8"))
    digest.update(repr(tuple(str(dtype) for dtype in frame.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _guard_source(root: Path, source: Path) -> dict[str, Any]:
    """Check that the source run is complete, verified and unchanged on disk."""
    manifest = _read_json(source / "manifest.json")
    identity = manifest["identity"]
    computed = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if computed != manifest.get("source_identity"):
        raise ValueError("source manifest identity checksum mismatch")
    for field, snapshot in (("source_sha256", "source_snapshot"),
                             ("inputs_sha256", "input_snapshot")):
        for name, digest in identity[field].items():
            if (_sha(_resolve(root, name)) != digest
                    or _sha(source / snapshot / name) != digest):
                raise ValueError(f"source current/snapshot checksum mismatch: {name}")
    for name, digest in identity["outer_fold_source_sha256"].items():
        if _sha(_resolve(root, name)) != digest:
            raise ValueError("source engine fold table checksum mismatch")
    for name, field in (("uv.lock", "dependency_lock_sha256"),
                         ("pyproject.toml", "project_sha256")):
        if (_sha(root / name) != identity[field]
                or _sha(source / "source_snapshot" / name) != identity[field]):
            raise ValueError(f"source dependency/project checksum mismatch: {name}")
    for name, digest in identity["frozen_features_sha256"].items():
        if _sha(source / name) != digest:
            raise ValueError(f"source frozen-feature checksum mismatch: {name}")
    if _sha(source / "outer_assignments.csv") != identity["outer_assignments_sha256"]:
        raise ValueError("source frozen outer assignments checksum mismatch")
    verification = _read_json(source / "verification.json")
    if (verification.get("status") != "passed"
            or verification.get("source_identity") != manifest["source_identity"]
            or not verification.get("checks")
            or not all(value is True for value in verification["checks"].values())
            or verification.get("model_fit_performed") is not False
            or verification.get("raw_test_labels_read") is not False):
        raise ValueError("source independent verification must have passed all checks")
    if _sha(root / "scripts/verify_lightgbm_study.py") != verification.get("source_script_sha256"):
        raise ValueError("source verification script checksum mismatch")
    run = _read_json(source / "run_info.json")
    if run.get("status") != "completed" or run.get("source_identity") != manifest["source_identity"]:
        raise ValueError("source study must be complete")
    baseline = StudyConfig()
    for field in ("outer_splits", "inner_splits", "seed", "fit_threads", "hgb_max_iter",
                  "endpoint_fractions", "thresholds", "persistences", "lead_requirement",
                  "target_on_time_rate", "endpoint_mae_margin", "nasa_score_margin",
                  "warning_rate_margin", "early_alarm_burden_margin"):
        if identity["config"].get(field) != audit._json_value(getattr(baseline, field)):
            raise ValueError(f"source HGB/evaluation configuration mismatch: {field}")
    if identity["versions"] != _versions():
        raise ValueError("source numeric/serialization runtime versions differ")
    for arm, cap in (("hgb_uncapped", None), ("hgb_cap125", 125)):
        matching = [spec for spec in identity["candidates"] if spec.get("candidate_id") == arm]
        expected = {"family": arm, "model_kind": "hgb", "target_cap": cap,
                    "max_depth": None, "learning_rate": .05,
                    "reg_lambda_nominal": 0., "num_leaves": 31}
        if len(matching) != 1 or any(matching[0].get(key) != value for key, value in expected.items()):
            raise ValueError("source fixed HGB candidate parameters mismatch")
    columns = identity["feature_columns"]
    if (len(columns) != 169 or len(set(columns)) != 169
            or {"y_true", "rul"} & set(columns)):
        raise ValueError("source requires complete 169-feature label-free matrix")
    return manifest


def _validate_frozen_frames(
    source: Path, manifest: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Load the source's frozen train/test features and outer fold assignments."""
    columns = manifest["identity"]["feature_columns"]
    train = pd.read_parquet(source / "frozen_train_features.parquet")
    test = pd.read_parquet(source / "frozen_test_features.parquet")
    for frame, split in ((train, "train"), (test, "test")):
        if (frame.empty or not frame.columns.is_unique or frame.duplicated(list(KEY)).any()
                or not frame.split.eq(split).all()
                or not np.isfinite(frame[columns].to_numpy(dtype="float64")).all()):
            raise ValueError(f"invalid source frozen {split} feature frame")
    if {"rul", "y_true"} & set(test.columns):
        raise ValueError("source frozen test features must not contain labels")
    if not np.isfinite(train.rul).all() or (train.rul < 0).any():
        raise ValueError("source frozen train truth must be finite raw RUL")
    assignments = pd.read_csv(source / "outer_assignments.csv")
    if (assignments.engine_id.duplicated().any()
            or set(assignments.engine_id) != set(train.engine_id)
            or set(assignments.outer_fold) != set(range(1, 6))):
        raise ValueError("source outer assignments must cover every engine exactly once")
    ordered = train.loc[:, list(dict.fromkeys((*KEY, *columns, "rul")))].copy()
    return train, test, assignments, _frame_digest(ordered)


def _inventory(
    root: Path, source: Path, manifest: dict[str, Any], train: pd.DataFrame,
    assignments: pd.DataFrame, frame_digest: str,
) -> tuple[dict[str, Any], set[Path]]:
    """Map the 46 HGB CV caches and 2 final models of the source run; verify each one."""
    source_id, identity = manifest["source_identity"], manifest["identity"]
    candidates = {item["candidate_id"]: item for item in identity["candidates"]
                  if item["candidate_id"] in BASE_ARMS}
    if set(candidates) != set(BASE_ARMS):
        raise ValueError("source requires exactly the two HGB reference specs")
    all_ids = tuple(sorted(map(int, train.engine_id.unique())))
    metadata_by_task = {}
    artifacts = {source / name for name in (
        "manifest.json", "verification.json", "run_info.json", "outer_assignments.csv",
        "frozen_train_features.parquet", "frozen_test_features.parquet",
        "final_predictions.parquet", "final_predictions_receipt.json",
        "outer_predictions.parquet", "warning_events.parquet", "aggregate_metrics.csv")}
    for path in (source / "fit_cache").glob("*.json"):
        metadata = _read_json(path)
        candidate = metadata.get("candidate", {}).get("candidate_id")
        if candidate not in BASE_ARMS:
            continue
        fitting, validation = metadata["fit_engine_ids"], metadata["validation_engine_ids"]
        if (not fitting or not validation or fitting != sorted(set(fitting))
                or validation != sorted(set(validation)) or set(fitting) & set(validation)):
            raise ValueError("source HGB cache requires disjoint unique engine sets")
        key = (candidate, tuple(fitting), tuple(validation))
        if key in metadata_by_task:
            raise ValueError("duplicate source HGB fit tasks")
        prediction = path.with_suffix(".parquet")
        payload = {"source_identity": source_id, "source_hash": frame_digest,
                   "spec": candidates[candidate], "config": identity["config"],
                   "fit_engine_ids": fitting, "validation_engine_ids": validation}
        expected_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if (metadata.get("status") != "completed" or path.stem != expected_key
                or metadata.get("cache_key") != expected_key
                or metadata.get("source_identity") != source_id
                or metadata.get("source_hash") != frame_digest
                or metadata.get("candidate") != candidates[candidate]
                or metadata.get("config") != identity["config"]
                or metadata.get("feature_columns") != identity["feature_columns"]
                or metadata.get("prediction_sha256") != _sha(prediction)):
            raise ValueError("source HGB cache identity/checksum mismatch")
        fit_frame = train.loc[train.engine_id.isin(fitting)]
        validation_frame = train.loc[train.engine_id.isin(validation)]
        weights = metadata.get("weight_metadata", {})
        effective = metadata.get("effective_parameters", {})
        if (set(fit_frame.engine_id) != set(fitting)
                or set(validation_frame.engine_id) != set(validation)
                or metadata.get("n_rows") != len(validation_frame)
                or metadata.get("n_features") != 169
                or weights.get("scheme") != "engine_total_one"
                or weights.get("n_rows") != len(fit_frame)
                or weights.get("n_engines") != len(fitting)
                or effective.get("max_iter") != 200
                or effective.get("model_kind") != "hgb"
                or effective.get("target_cap") != candidates[candidate]["target_cap"]):
            raise ValueError("source HGB cache fitting/weight contract mismatch")
        metadata_by_task[key] = (prediction, path, metadata)
    records = []
    expected_tasks = set()
    for outer in range(0, 6):
        stage_name = f"outer_{outer}" if outer else "final_inner"
        outer_validation = set(map(int, assignments.loc[assignments.outer_fold.eq(outer), "engine_id"]))
        fitting_ids = tuple(engine for engine in all_ids if engine not in outer_validation)
        fitting_frame = train.loc[train.engine_id.isin(fitting_ids)].reset_index(drop=True)
        splitter = GroupKFold(n_splits=3)
        inner_rows = []
        cache_map = _read_json(source / "stages" / stage_name / "fit_cache_map.json")
        artifacts.update(source / "stages" / stage_name / name
                         for name in ("fit_cache_map.json", "inner_assignments.csv", "selection.json"))
        for inner, (fit_index, validation_index) in enumerate(
            splitter.split(fitting_frame, fitting_frame.rul, fitting_frame.engine_id), start=1,
        ):
            fit_ids = tuple(sorted(map(int, fitting_frame.iloc[fit_index].engine_id.unique())))
            validation_ids = tuple(sorted(map(int, fitting_frame.iloc[validation_index].engine_id.unique())))
            inner_rows.extend({"outer_fold": outer, "inner_fold": inner, "engine_id": engine}
                              for engine in validation_ids)
            for candidate in BASE_ARMS:
                task = (candidate, fit_ids, validation_ids)
                if task not in metadata_by_task:
                    raise ValueError("source HGB inner cache task coverage incomplete")
                prediction, path, _ = metadata_by_task[task]
                if _resolve(source, cache_map.get(f"{candidate}/fold_{inner}", "")) != prediction:
                    raise ValueError("source HGB stage map disagrees with exact engine task")
                expected_tasks.add(task)
                records.append({"stage": stage_name, "task_kind": "inner", "outer_fold": outer,
                                "fold": inner, "candidate_id": candidate,
                                "prediction_path": prediction.relative_to(root).as_posix(),
                                "metadata_path": path.relative_to(root).as_posix(),
                                "fit_engine_ids": list(fit_ids), "validation_engine_ids": list(validation_ids)})
                artifacts.update((prediction, path))
        expected_assignments = pd.DataFrame(inner_rows)
        actual_assignments = pd.read_csv(source / "stages" / stage_name / "inner_assignments.csv")
        try:
            pd.testing.assert_frame_equal(actual_assignments, expected_assignments, check_exact=True)
        except AssertionError as error:
            raise ValueError("source inner fold coverage/membership mismatch") from error
        if outer:
            for candidate in BASE_ARMS:
                task = (candidate, fitting_ids, tuple(sorted(outer_validation)))
                if task not in metadata_by_task:
                    raise ValueError("source HGB outer cache task coverage incomplete")
                prediction, path, _ = metadata_by_task[task]
                expected_tasks.add(task)
                records.append({"stage": "outer", "task_kind": "outer", "outer_fold": outer,
                                "fold": outer, "candidate_id": candidate,
                                "prediction_path": prediction.relative_to(root).as_posix(),
                                "metadata_path": path.relative_to(root).as_posix(),
                                "fit_engine_ids": list(fitting_ids),
                                "validation_engine_ids": sorted(outer_validation)})
                artifacts.update((prediction, path))
    if set(metadata_by_task) != expected_tasks or len(expected_tasks) != 46 or len(records) != 46:
        raise ValueError("source must have exactly 46 unique reused HGB CV fits")
    model_records = []
    for candidate in BASE_ARMS:
        folder = source / "models" / candidate
        model_path, metadata_path = folder / "model.joblib", folder / "model.json"
        info = _read_json(metadata_path)
        if (info.get("source_identity") != source_id or info.get("spec") != candidates[candidate]
                or info.get("model_sha256") != _sha(model_path)
                or info.get("fit_engine_ids") != list(all_ids)
                or info.get("fit_rows") != len(train)
                or info.get("feature_columns") != identity["feature_columns"]
                or info.get("saved_model_replay_max_abs_delta") != 0.0):
            raise ValueError("source final HGB model identity/checksum/coverage mismatch")
        model_records.append({"candidate_id": candidate,
                              "model_path": model_path.relative_to(root).as_posix(),
                              "metadata_path": metadata_path.relative_to(root).as_posix(),
                              "fit_engine_ids": list(all_ids)})
        artifacts.update((model_path, metadata_path))
    receipt = _read_json(source / "final_predictions_receipt.json")
    if (receipt.get("source_identity") != source_id or receipt.get("test_labels_read") is not False
            or receipt.get("saved_model_replay_exact") is not True
            or receipt.get("predictions_sha256") != _sha(source / "final_predictions.parquet")
            or any(receipt.get("models_sha256", {}).get(arm) != _sha(source / "models" / arm / "model.joblib")
                   for arm in BASE_ARMS)):
        raise ValueError("source final-prediction receipt checksum/order mismatch")
    # Snapshot provenance files only, not earlier selection or test scores.
    for field, snapshot in (("source_sha256", "source_snapshot"), ("inputs_sha256", "input_snapshot")):
        for name in identity[field]:
            artifacts.update((_resolve(root, name), source / snapshot / name))
    for name in identity["outer_fold_source_sha256"]:
        artifacts.add(_resolve(root, name))
    for name in ("uv.lock", "pyproject.toml"):
        artifacts.update((root / name, source / "source_snapshot" / name))
    artifacts.add(root / "scripts/verify_lightgbm_study.py")
    inventory = {"source_run_identity": source_id, "cv_records": records,
                 "final_model_records": model_records,
                 "counts": {"outer_inner": 30, "outer": 10, "final_inner": 6,
                            "final_models": 2, "reused_fits": 48, "fresh_fits": 0}}
    return inventory, artifacts


def _artifact_snapshot_paths(names: Iterable[str]) -> dict[str, str]:
    """Map artifact paths to short hashed file names for the snapshot folder."""
    paths = tuple(names)
    if len(set(paths)) != len(paths):
        raise ValueError("source artifact snapshot names must be unique")
    mapping = {name: hashlib.sha256(name.encode("utf-8")).hexdigest()[:24] + Path(name).suffix
               for name in paths}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("source artifact snapshot filename collision")
    return mapping


def _freeze(
    root: Path, source: Path, output: Path, source_manifest: dict[str, Any],
    inventory: dict[str, Any], artifacts: set[Path], config: StudyConfig,
    evaluate_test: bool, labels: Path,
) -> dict[str, Any]:
    """Create the output run and snapshot sources and used artifacts before selection."""
    if output.exists():
        raise ValueError("run output already exists; choose a fresh run ID (no unsafe resume)")
    allowed = root / "outputs/experiments/gated_blend_comparison"
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("derived output must be a run descendant of gated_blend_comparison")
    if config.smoke and "smoke" not in output.name.lower():
        raise ValueError("dry-run requires a distinct output name containing smoke")
    if not config.smoke and not config.scientific:
        raise ValueError("non-smoke run must retain the full predeclared scientific grid")
    source_names = _source_files(root)
    artifact_names = tuple(path.relative_to(root).as_posix() for path in sorted(artifacts))
    identity = audit._json_value({
        "config": asdict(config), "candidates": [{"candidate_id": item.candidate_id, **asdict(item)}
                                                    for item in specs(config)],
        "source_sha256": {name: _sha(root / name) for name in source_names},
        "source_artifacts_sha256": {path.relative_to(root).as_posix(): _sha(path)
                                     for path in sorted(artifacts)},
        "source_artifact_snapshot_paths": _artifact_snapshot_paths(artifact_names),
        "source_run_dir": source.relative_to(root).as_posix(),
        "source_run_identity": source_manifest["source_identity"],
        "dependency_lock_sha256": _sha(root / "uv.lock"),
        "project_sha256": _sha(root / "pyproject.toml"), "versions": _versions(),
        "feature_columns": source_manifest["identity"]["feature_columns"],
        "evaluate_test": evaluate_test, "test_labels_path": labels.relative_to(root).as_posix(),
    })
    output.mkdir(parents=True, exist_ok=False)
    for directory, mapping in (("source_snapshot", identity["source_sha256"]),
                               ("source_artifact_snapshot", identity["source_artifacts_sha256"])):
        for name, digest in mapping.items():
            snapshot_name = (identity["source_artifact_snapshot_paths"][name]
                             if directory == "source_artifact_snapshot" else name)
            destination = output / directory / snapshot_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination)
            if _sha(destination) != digest or _sha(root / name) != digest:
                raise ValueError("source artifact changed during provenance snapshot")
    for name in ("uv.lock", "pyproject.toml"):
        shutil.copy2(root / name, output / "source_snapshot" / name)
    manifest = {"schema_version": 1, "author": AUTHOR, "identity": identity,
                "source_identity": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "reuse_counts": inventory["counts"], "fresh_fits": 0,
                "selection_source": "inner grouped OOF only; freeze choice before outer prediction read",
                "target_scale": "raw RUL; prediction-only gate; no output clipping",
                "test_history": "historically inspected FD001 test; reporting only",
                "resume_policy": "reject any existing output directory; new run ID required"}
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "reuse_inventory.json", inventory)
    _assert_immutable(root, output, manifest)
    return manifest


def _assert_immutable(root: Path, output: Path, manifest: dict[str, Any]) -> None:
    """Check current files and their snapshots against the manifest checksums."""
    identity = manifest["identity"]
    artifact_paths = identity["source_artifact_snapshot_paths"]
    if artifact_paths != _artifact_snapshot_paths(identity["source_artifacts_sha256"]):
        raise ValueError("source artifact snapshot path mapping mismatch")
    for field, snapshot in (("source_sha256", "source_snapshot"),
                            ("source_artifacts_sha256", "source_artifact_snapshot")):
        for name, digest in identity[field].items():
            snapshot_name = artifact_paths[name] if field == "source_artifacts_sha256" else name
            if _sha(root / name) != digest or _sha(output / snapshot / snapshot_name) != digest:
                raise ValueError(f"immutable {field} checksum mismatch: {name}")
    for name, field in (("uv.lock", "dependency_lock_sha256"), ("pyproject.toml", "project_sha256")):
        if _sha(root / name) != identity[field] or _sha(output / "source_snapshot" / name) != identity[field]:
            raise ValueError("derived dependency/project snapshot checksum mismatch")


def _load_cached_prediction(root: Path, record: dict[str, Any], train: pd.DataFrame) -> pd.DataFrame:
    """Load one cached prediction and check its keys, truth and error columns."""
    prediction_path = _resolve(root, record["prediction_path"])
    metadata = _read_json(_resolve(root, record["metadata_path"]))
    if _sha(prediction_path) != metadata["prediction_sha256"]:
        raise ValueError("reused prediction checksum mismatch before reading")
    prediction = pd.read_parquet(prediction_path)
    validation = train.loc[train.engine_id.isin(record["validation_engine_ids"])].reset_index(drop=True)
    if (prediction.empty or not prediction.columns.is_unique or prediction.duplicated(list(KEY)).any()
            or not prediction.candidate_id.eq(record["candidate_id"]).all()
            or not np.isfinite(prediction[["y_true", "y_pred"]]).all().all()):
        raise ValueError("invalid reused raw prediction vector")
    try:
        pd.testing.assert_frame_equal(prediction[list(KEY)], validation[list(KEY)], check_exact=True)
        np.testing.assert_array_equal(prediction.y_true.to_numpy(dtype="float64"),
                                      validation.rul.to_numpy(dtype="float64"))
        rescored = add_prediction_errors(prediction)
        for column in ("signed_error", "absolute_error", "squared_error", "nasa_score_component"):
            if column in prediction:
                np.testing.assert_array_equal(prediction[column], rescored[column])
    except AssertionError as error:
        raise ValueError("reused predictions must match exact validation keys, raw truth and errors") from error
    return prediction


def _progress(output: Path, stage: str, **values: Any) -> None:
    payload = {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
               "fresh_fits": 0, **values}
    _write_json(output / "progress.json", payload)
    print(json.dumps(audit._json_value(payload), ensure_ascii=True), flush=True)


def _inner_stage(
    root: Path, source: Path, output: Path, manifest: dict[str, Any], inventory: dict[str, Any],
    train: pd.DataFrame, outer_fold: int, config: StudyConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Score and calibrate every gate on inner OOF rows, then save the selection."""
    _assert_immutable(root, output, manifest)
    stage_name = f"outer_{outer_fold}" if outer_fold else "final_inner"
    stage = output / "stages" / stage_name
    stage.mkdir(parents=True, exist_ok=False)
    assignments_path = source / "stages" / stage_name / "inner_assignments.csv"
    shutil.copy2(assignments_path, stage / "inner_assignments.csv")
    records = [item for item in inventory["cv_records"]
               if item["stage"] == stage_name and item["task_kind"] == "inner"]
    _write_json(stage / "source_fit_cache_map.json", {
        f"{item['candidate_id']}/fold_{item['fold']}": item["prediction_path"] for item in records})
    predictions = {}
    for arm in BASE_ARMS:
        parts = [_load_cached_prediction(root, item, train).assign(inner_fold=item["fold"])
                 for item in records if item["candidate_id"] == arm]
        prediction = pd.concat(parts, ignore_index=True).sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
        if prediction.duplicated(list(KEY)).any():
            raise ValueError("inner OOF duplicate engine-cycle coverage")
        expected_ids = set(pd.read_csv(assignments_path).engine_id)
        if set(prediction.engine_id) != expected_ids:
            raise ValueError("inner OOF engine coverage incomplete")
        predictions[arm] = prediction
    rows, policy_rows, policy_by_candidate = [], [], {}
    source_selection = _read_json(source / "stages" / stage_name / "selection.json")
    for arm in BASE_ARMS:
        policy, grid = calibrate_policy(predictions[arm], config)
        if asdict(policy) != source_selection["arms"][arm]["policy"]:
            raise ValueError("source HGB reference policy must reproduce exact inner calibration")
        policy_by_candidate[arm] = policy
        policy_rows.append(grid.assign(candidate_id=arm, procedure=arm, outer_fold=outer_fold))
        events = audit.warning_events(predictions[arm], policy, lead_requirement=config.lead_requirement)
        rows.append({"outer_fold": outer_fold, "candidate_id": arm, "family": arm,
                     **candidate_readout(predictions[arm], config), **audit._warning_readout(events),
                     "threshold": policy.threshold, "persistence": policy.persistence,
                     "policy_threshold": policy.threshold, "policy_persistence": policy.persistence,
                     "numeric_controls_met": True})
    for completed, spec in enumerate(specs(config), start=1):
        frame = blend_predictions(predictions["hgb_uncapped"], predictions["hgb_cap125"], spec)
        predictions[spec.candidate_id] = frame
        policy, grid = calibrate_policy(frame, config)
        policy_by_candidate[spec.candidate_id] = policy
        events = audit.warning_events(frame, policy, lead_requirement=config.lead_requirement)
        rows.append({"outer_fold": outer_fold, "candidate_id": spec.candidate_id,
                     "family": "gated_blend", **asdict(spec),
                     **candidate_readout(frame, config), **audit._warning_readout(events),
                     "threshold": policy.threshold, "persistence": policy.persistence,
                     "policy_threshold": policy.threshold, "policy_persistence": policy.persistence})
        policy_rows.append(grid.assign(candidate_id=spec.candidate_id, procedure="gated_blend",
                                        outer_fold=outer_fold))
        _progress(output, "inner_gate_evaluated", outer_fold=outer_fold,
                  completed_gates=completed, total_gates=len(specs(config)), candidate_id=spec.candidate_id)
    table = pd.DataFrame(rows)
    reference = table.loc[table.candidate_id.eq("hgb_uncapped")].iloc[0]
    chosen_gate = select_gate_candidate(table.loc[table.family.eq("gated_blend")], reference, config)
    gate_id = chosen_gate["candidate_id"]
    chosen = {}
    gate_specs = {spec.candidate_id: spec for spec in specs(config)}
    arm_rows = []
    for arm, candidate in (("hgb_uncapped", "hgb_uncapped"), ("hgb_cap125", "hgb_cap125"),
                            ("gated_blend", gate_id)):
        row = table.loc[table.candidate_id.eq(candidate)].iloc[0].dropna().to_dict()
        row.update(procedure=arm, numeric_controls_met=(bool(chosen_gate["numeric_controls_met"])
                                                      if arm == "gated_blend" else True))
        policy = policy_by_candidate[candidate]
        chosen[arm] = {"candidate_id": candidate, "policy": asdict(policy),
                       "numeric_controls_met": row["numeric_controls_met"], "inner_metrics": row,
                       "spec": asdict(gate_specs[candidate]) if arm == "gated_blend" else None}
        if arm == "gated_blend":
            chosen[arm].update({key: value for key, value in chosen_gate.items()
                               if key not in ("candidate_id", "spec", "policy", "inner_metrics")})
        arm_rows.append(row)
    procedure = choose_procedure(pd.DataFrame(arm_rows), config)
    table["selected_within_family"] = table.candidate_id.eq(gate_id) | table.candidate_id.isin(BASE_ARMS)
    table.to_csv(stage / "inner_candidate_metrics.csv", index=False)
    pd.DataFrame(arm_rows).to_csv(stage / "inner_arm_metrics.csv", index=False)
    pd.concat(policy_rows, ignore_index=True).to_csv(stage / "inner_policy_metrics.csv", index=False)
    selection = {"outer_fold": outer_fold, "arms": chosen, "procedure": procedure,
                 "gate_selection": chosen_gate, "source_identity": manifest["source_identity"],
                 "selection_source": "inner OOF fitting engines only; no outer predictions read",
                 "test_labels_read": False, "outer_prediction_values_read_before_selection": False,
                 "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(stage / "selection.json", selection)
    _assert_immutable(root, output, manifest)
    return chosen, selection


def _outer_fold(
    root: Path, output: Path, inventory: dict[str, Any], train: pd.DataFrame,
    choices: dict[str, dict[str, Any]], selection: dict[str, Any], fold: int, config: StudyConfig,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Score one outer fold; requires that fold's saved selection."""
    saved = _read_json(output / "stages" / f"outer_{fold}" / "selection.json")
    if saved != audit._json_value(selection):
        raise ValueError("outer scoring requires the previously frozen inner selection")
    records = [item for item in inventory["cv_records"]
               if item["outer_fold"] == fold and item["task_kind"] == "outer"]
    base = {item["candidate_id"]: _load_cached_prediction(root, item, train) for item in records}
    if set(base) != set(BASE_ARMS):
        raise ValueError("outer base prediction coverage incomplete")
    gate = GateSpec(**choices["gated_blend"]["spec"])
    base["gated_blend"] = blend_predictions(base["hgb_uncapped"], base["hgb_cap125"], gate)
    prediction_parts, event_parts = [], []
    for arm in ARMS:
        choice = choices[arm]
        policy = audit.WarningPolicy(**choice["policy"])
        prediction = base[arm].assign(procedure=arm, outer_fold=fold,
                                      candidate_id=choice["candidate_id"],
                                      threshold=policy.threshold, persistence=policy.persistence)
        events = audit.warning_events(prediction, policy, lead_requirement=config.lead_requirement).assign(
            procedure=arm, outer_fold=fold, candidate_id=choice["candidate_id"])
        prediction_parts.append(prediction)
        event_parts.append(events)
        if arm == selection["procedure"]["selected_procedure_id"]:
            prediction_parts.append(prediction.assign(procedure="selected_pipeline", selected_arm=arm))
            event_parts.append(events.assign(procedure="selected_pipeline", selected_arm=arm))
    _write_json(output / "stages" / f"outer_{fold}" / "outer_read_receipt.json", {
        "selection_sha256": _sha(output / "stages" / f"outer_{fold}" / "selection.json"),
        "outer_predictions_read_after_selection": True, "outer_fold": fold,
        "read_completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return prediction_parts, event_parts


def _save_final_inference(
    root: Path, source: Path, output: Path, manifest: dict[str, Any], test: pd.DataFrame,
    choices: dict[str, dict[str, Any]], selection: dict[str, Any], inventory: dict[str, Any],
) -> pd.DataFrame:
    """Copy final models, write inference.json and check predictions replay exactly."""
    base_info = {}
    for record in inventory["final_model_records"]:
        arm = record["candidate_id"]
        folder = output / "models" / arm
        folder.mkdir(parents=True, exist_ok=False)
        shutil.copy2(root / record["model_path"], folder / "model.joblib")
        shutil.copy2(root / record["metadata_path"], folder / "model.json")
        base_info[arm] = {"model_sha256": _sha(folder / "model.joblib"),
                          "metadata_sha256": _sha(folder / "model.json"),
                          "source_model_path": record["model_path"],
                          "source_metadata_path": record["metadata_path"]}
    gate = GateSpec(**choices["gated_blend"]["spec"])
    inference = {"schema_version": 1, "author": AUTHOR, "source_identity": manifest["source_identity"],
                 "source_run_identity": inventory["source_run_identity"], "base_models": base_info,
                 "feature_columns": manifest["identity"]["feature_columns"],
                 "gate_spec": asdict(gate), "gate_candidate_id": gate.candidate_id,
                 "selected_procedure_id": selection["procedure"]["selected_procedure_id"],
                 "policies": {arm: choices[arm]["policy"] for arm in ARMS},
                 "test_labels_read": False, "raw_prediction_units": "cycles",
                 "fresh_fits": 0, "selection_sha256": _sha(output / "selection.json")}
    _write_json(output / "inference.json", inference)
    restored, reloaded = load_inference(output), load_inference(output)
    final_test = test.groupby("engine_id", sort=True, as_index=False).tail(1).reset_index(drop=True)
    stored = pd.read_parquet(source / "final_predictions.parquet")
    if {"y_true", "rul"} & set(stored.columns):
        raise ValueError("source final prediction file must not contain test truth")
    parts = []
    for arm in ARMS:
        predicted = restored.predict(final_test, arm)
        np.testing.assert_array_equal(predicted, reloaded.predict(final_test, arm))
        frame = final_test[list(KEY)].copy()
        frame["y_pred"] = predicted
        frame["procedure"] = arm
        frame["candidate_id"] = choices[arm]["candidate_id"]
        if arm in BASE_ARMS:
            old = stored.loc[stored.procedure.eq(arm)].sort_values(list(KEY)).reset_index(drop=True)
            pd.testing.assert_frame_equal(old[list(KEY)], frame[list(KEY)], check_exact=True)
            np.testing.assert_array_equal(old.y_pred, frame.y_pred)
        else:
            frame["uncapped_prediction"] = restored.predict(final_test, "hgb_uncapped")
            frame["cap125_prediction"] = restored.predict(final_test, "hgb_cap125")
        parts.append(frame)
        if arm == inference["selected_procedure_id"]:
            parts.append(frame.assign(procedure="selected_pipeline", selected_arm=arm))
    final = pd.concat(parts, ignore_index=True)
    audit._write_parquet(final, output / "final_predictions.parquet")
    persisted = pd.read_parquet(output / "final_predictions.parquet")
    pd.testing.assert_frame_equal(final, persisted, check_exact=True)
    for arm, group in persisted.groupby("procedure", sort=True):
        np.testing.assert_array_equal(group.y_pred, load_inference(output).predict(final_test, arm))
    _write_json(output / "final_predictions_receipt.json", {
        "source_identity": manifest["source_identity"], "predictions_sha256": _sha(output / "final_predictions.parquet"),
        "models_sha256": {arm: info["model_sha256"] for arm, info in base_info.items()},
        "models_metadata_sha256": {arm: info["metadata_sha256"] for arm, info in base_info.items()},
        "inference_sha256": _sha(output / "inference.json"), "selection_sha256": _sha(output / "selection.json"),
        "outer_decision_sha256": _sha(output / "outer_decision_before_test.json"),
        "saved_model_replay_exact": True, "original_base_prediction_replay_exact": True,
        "all_final_procedures_replayed_exact": True, "test_labels_read": False,
        "fresh_fits": 0, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
    return final


def _evaluate_final_test(
    output: Path, predictions: pd.DataFrame, labels_path: Path,
) -> dict[str, Any]:
    """Score final predictions on test labels after re-checking every frozen artifact."""
    receipt = _read_json(output / "final_predictions_receipt.json")
    for filename, field in (("final_predictions.parquet", "predictions_sha256"),
                             ("selection.json", "selection_sha256"),
                             ("inference.json", "inference_sha256"),
                             ("outer_decision_before_test.json", "outer_decision_sha256")):
        if _sha(output / filename) != receipt[field]:
            raise ValueError("frozen inference/decision/prediction checksum mismatch before test labels")
    if (receipt.get("test_labels_read") is not False or receipt.get("saved_model_replay_exact") is not True
            or receipt.get("original_base_prediction_replay_exact") is not True
            or receipt.get("all_final_procedures_replayed_exact") is not True):
        raise ValueError("exact final replay receipt required before test labels")
    selection = _read_json(output / "selection.json")
    inference = _read_json(output / "inference.json")
    if (selection.get("test_labels_read_for_selection") is not False
            or selection.get("source_identity") != receipt.get("source_identity")
            or inference.get("test_labels_read") is not False
            or inference.get("source_identity") != receipt.get("source_identity")):
        raise ValueError("final selection/inference identity must precede test label access")
    for arm in BASE_ARMS:
        if (_sha(output / "models" / arm / "model.joblib") != receipt["models_sha256"][arm]
                or _sha(output / "models" / arm / "model.json") != receipt["models_metadata_sha256"][arm]):
            raise ValueError("final model checksum mismatch before test labels")
    pd.testing.assert_frame_equal(predictions, pd.read_parquet(output / "final_predictions.parquet"), check_exact=True)
    if set(predictions.procedure) != set(ARMS) | {"selected_pipeline"}:
        raise ValueError("all final procedures must exist before test labels")
    labels = read_test_labels(labels_path).rename(columns={"rul": "y_true"})
    evaluations, metrics = [], []
    for arm, frame in predictions.groupby("procedure", sort=True):
        joined = frame.merge(labels, on=["split", "engine_id"], how="outer",
                             validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError("test prediction and label engine coverage must match exactly")
        evaluation = add_prediction_errors(joined.drop(columns="_merge"))
        evaluations.append(evaluation)
        metrics.append({"procedure": arm, **summarize_prediction_errors(evaluation)})
    audit._write_parquet(pd.concat(evaluations, ignore_index=True), output / "final_evaluation.parquet")
    pd.DataFrame(metrics).to_csv(output / "test_metrics.csv", index=False)
    result = {"procedures": metrics, "test_labels_sha256": _sha(labels_path),
              "reporting_only": True, "historically_inspected_test": True,
              "selection_used_test_labels": False, "outer_gate_changed_by_test": False,
              "labels_read_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(output / "test_evaluation.json", result)
    return result


def run_gated_blend_study(
    *, repo_root: str | Path | None = None, source_run_dir: str | Path = DEFAULT_SOURCE,
    output_dir: str | Path = DEFAULT_OUTPUT, test_labels_path: str | Path = "data/raw/RUL_FD001.txt",
    config: StudyConfig | None = None, evaluate_test: bool = True,
) -> dict[str, Any]:
    """Run the full gated study from `source_run_dir` into a new `output_dir`.

    Reuses the 48 source fits. `output_dir` must not exist. Test labels are only
    used for reporting. Returns the run_info payload.
    """
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[4]
    source, output, labels = (_resolve(root, item) for item in (source_run_dir, output_dir, test_labels_path))
    if output.exists():
        raise ValueError("run output already exists; choose a fresh run ID (no unsafe resume)")
    config = config or StudyConfig()
    started = time.perf_counter()
    source_manifest = _guard_source(root, source)
    train, test, assignments, frame_digest = _validate_frozen_frames(source, source_manifest)
    inventory, artifacts = _inventory(root, source, source_manifest, train, assignments, frame_digest)
    manifest = _freeze(root, source, output, source_manifest, inventory, artifacts, config, evaluate_test, labels)
    shutil.copy2(source / "outer_assignments.csv", output / "outer_assignments.csv")
    predictions, events, selections = [], [], []
    for fold in range(1, 6):
        choices, selection = _inner_stage(root, source, output, manifest, inventory, train, fold, config)
        _assert_immutable(root, output, manifest)
        before = _sha(output / "stages" / f"outer_{fold}" / "selection.json")
        fold_predictions, fold_events = _outer_fold(root, output, inventory, train, choices, selection, fold, config)
        if _sha(output / "stages" / f"outer_{fold}" / "selection.json") != before:
            raise ValueError("inner choice changed during outer scoring")
        predictions.extend(fold_predictions)
        events.extend(fold_events)
        validation_ids = sorted(map(int, assignments.loc[assignments.outer_fold.eq(fold), "engine_id"]))
        selections.append({**selection, "outer_fit_engine_ids": sorted(set(train.engine_id) - set(validation_ids)),
                           "outer_validation_engine_ids": validation_ids})
        _write_json(output / "outer_selections.json", selections)
        _progress(output, "outer_completed", completed_outer_folds=fold, total_outer_folds=5,
                  selected_procedure=selection["procedure"]["selected_procedure_id"])
    outer = pd.concat(predictions, ignore_index=True).sort_values(["procedure", *KEY]).reset_index(drop=True)
    warnings = pd.concat(events, ignore_index=True).sort_values(["procedure", "engine_id"]).reset_index(drop=True)
    audit._write_parquet(outer, output / "outer_predictions.parquet")
    audit._write_parquet(warnings, output / "warning_events.parquet")
    endpoints = pd.concat([audit.build_endpoint_panel(group, config.endpoint_fractions)
                           for _, group in outer.groupby("procedure", sort=True)], ignore_index=True)
    audit._write_parquet(endpoints, output / "outer_endpoints.parquet")
    _assert_immutable(root, output, manifest)
    decision = write_readouts(outer, warnings, config, output)
    _write_json(output / "outer_decision_before_test.json", decision)
    choices, final_selection = _inner_stage(root, source, output, manifest, inventory, train, 0, config)
    final_selection.update(author=AUTHOR, config=asdict(config), outer_decision=decision,
                           test_labels_read_for_selection=False)
    _write_json(output / "selection.json", final_selection)
    for name in ("inner_candidate_metrics.csv", "inner_arm_metrics.csv", "inner_assignments.csv", "inner_policy_metrics.csv"):
        shutil.copy2(output / "stages/final_inner" / name, output / f"final_{name}")
        pd.concat([pd.read_csv(output / "stages" / f"outer_{fold}" / name, float_precision="round_trip")
                   for fold in range(1, 6)], ignore_index=True).to_csv(output / name, index=False)
    _assert_immutable(root, output, manifest)
    final = _save_final_inference(root, source, output, manifest, test, choices, final_selection, inventory)
    decision_hash = _sha(output / "outer_decision_before_test.json")
    _assert_immutable(root, output, manifest)
    report = _evaluate_final_test(output, final, labels) if evaluate_test else None
    if _sha(output / "outer_decision_before_test.json") != decision_hash:
        raise ValueError("outer decision changed after reporting-only test read")
    _assert_immutable(root, output, manifest)
    result = {"status": "completed", "author": AUTHOR, "source_identity": manifest["source_identity"],
              "source_run_identity": inventory["source_run_identity"], "reuse_counts": inventory["counts"],
              "output_dir": output.relative_to(root).as_posix(), "decision": decision,
              "selected_procedure": final_selection["procedure"]["selected_procedure_id"],
              "test_evaluated": report is not None, "test_reporting_only": True,
              "historical_baseline_overwritten": False, "fresh_fits": 0,
              "seconds": time.perf_counter() - started}
    _write_json(output / "run_info.json", result)
    _progress(output, "completed", **result)
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI entry point; `--dry-run` runs a small smoke grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--source-run-dir", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-labels", default="data/raw/RUL_FD001.txt")
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = StudyConfig()
    if args.dry_run:
        config = replace(config, smoke=True, bootstrap_resamples=20,
                         lower_ruls=(60.,), upper_ruls=(150.,), max_capped_weights=(.25, .5))
    run_gated_blend_study(repo_root=args.repo_root, source_run_dir=args.source_run_dir,
                         output_dir=args.output_dir, test_labels_path=args.test_labels,
                         config=config, evaluate_test=not args.skip_test_evaluation)


__all__ = ["run_gated_blend_study", "main"]
