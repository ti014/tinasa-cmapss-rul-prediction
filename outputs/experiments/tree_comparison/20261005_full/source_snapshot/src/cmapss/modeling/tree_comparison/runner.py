"""Run the frozen HGB/XGBoost nested comparison on shared causal features.

Four arms use one persisted 169-feature matrix. Inner folds tune XGBoost and
warning policies; each outer fold evaluates engines excluded from those
choices. Historically inspected test labels are read only after every final
choice, fitted model and test prediction has been persisted.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from cmapss.data.contracts import KEY
from cmapss.data.readers import read_test_labels
from cmapss.data.targets import validate_train_targets
from cmapss.modeling import audit_study as audit
from cmapss.modeling.metrics import (
    add_prediction_errors, bootstrap_metric_intervals, summarize_prediction_errors,
)
from cmapss.modeling.tree_comparison.analysis import (
    calibrate_policy, candidate_readout, choose_procedure, select_xgb_candidate,
    write_readouts,
)
from cmapss.modeling.tree_comparison.cache import (
    load_frozen_features, run_fit_task,
)
from cmapss.modeling.tree_comparison.contracts import (
    FEATURES, StudyConfig, TreeSpec, specs,
)
from cmapss.modeling.tree_comparison.estimators import fit_predictor

ARMS = ("hgb_uncapped", "hgb_cap125", "xgb_uncapped", "xgb_cap125")
AUTHOR = {"full_name": "Phan Nguyễn Mai Phương", "student_id": "25002711"}


def _sha(path: Path) -> str:
    """Hash an artifact without retaining the entire file in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Require a JSON object for immutable run receipts."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    """Atomically publish JSON with native scalar values and finite numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(audit._json_value(value), ensure_ascii=False,
                                    sort_keys=True, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _freeze_json(path: Path, value: Any) -> None:
    """Reject changing a previously published choice or provenance receipt."""
    normalized = audit._json_value(value)
    if path.exists():
        if _read_json(path) != normalized:
            raise ValueError(f"frozen receipt changed on resume: {path}")
    else:
        _write_json(path, normalized)


def _resolve(root: Path, value: str | Path) -> Path:
    """Require portable repository descendants for every artifact path."""
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path must remain inside repository: {path}")
    return path


def _source_files(root: Path) -> tuple[str, ...]:
    """Collect local import dependencies, package initializers and entrypoints."""
    folder = root / "src/cmapss/modeling/tree_comparison"
    pending = [path.relative_to(root).as_posix() for path in folder.glob("*.py")]
    pending.extend(["src/cmapss/modeling/audit_study.py",
                    "scripts/run_tree_comparison.py"])
    found: set[str] = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"comparison source missing: {name}")
        found.add(name)
        if not name.startswith("src/"):
            continue
        package = Path(name).parent.relative_to("src").as_posix().replace("/", ".")
        parents = Path(name).parent
        while parents != Path("src"):
            initializer = (parents / "__init__.py").as_posix()
            if (root / initializer).is_file():
                pending.append(initializer)
            parents = parents.parent
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules: list[str] = []
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


def _versions() -> dict[str, str]:
    """Record exact installed numeric and serialization package versions."""
    names = ("numpy", "pandas", "scikit-learn", "xgboost", "pyarrow", "joblib",
             "scipy", "threadpoolctl")
    return {"python": platform.python_version(),
            **{name: version(name) for name in names}}


def _assert_immutable(root: Path, output: Path, manifest: dict[str, Any]) -> None:
    """Stop if source, lock, inputs, folds or frozen feature bytes change."""
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


def _freeze_inputs(
    root: Path, output: Path, observations_path: Path, targets_path: Path,
    config: StudyConfig,
) -> tuple[dict[str, Any], str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Snapshot provenance and persist one exact feature matrix before fits.

    On compatible resume, use the stored matrix without recomputing rolling
    statistics. Inputs, local imports, dependency lock and configuration are
    independently checked against the original manifest.
    """
    allowed = root / "outputs/experiments/tree_comparison"
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("output must be a new run descendant of tree_comparison")
    source_names = _source_files(root)
    provenance = audit._json_value({
        "config": asdict(config), "candidates": [asdict(item) for item in specs(config)],
        "source_sha256": {name: _sha(root / name) for name in source_names},
        "inputs_sha256": {path.relative_to(root).as_posix(): _sha(path)
                          for path in (observations_path, targets_path)},
        "dependency_lock_sha256": _sha(root / "uv.lock"), "project_sha256": _sha(root / "pyproject.toml"),
        "versions": _versions(),
        "outer_fold_source_sha256": ({
            "outputs/modeling/tables/fold_assignments.csv": _sha(
                root / "outputs/modeling/tables/fold_assignments.csv")}
            if config.outer_splits == 5 else {}),
        "feature_columns": FEATURES, "target_column": "rul",
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
        feature_columns = list(dict.fromkeys((*KEY, *FEATURES)))
        features = audit.build_study_features(observations).loc[:, feature_columns]
        train = features[features.split.eq("train")].merge(
            targets, on=list(KEY), how="left", validate="one_to_one")
        train = train.sort_values(["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
        test = features[features.split.eq("test")].sort_values(
            ["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
        if train.rul.isna().any() or not np.isfinite(train[list(FEATURES)]).all().all():
            raise ValueError("missing train labels or nonfinite frozen features")
        if len(FEATURES) != 169 or not np.isfinite(test[list(FEATURES)]).all().all():
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
            "procedures": list(ARMS),
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
            "author": AUTHOR, "window_cycles": 30, "feature_count": len(FEATURES),
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


def _progress(output: Path, stage: str, **values: Any) -> None:
    """Publish compact checkpoints and an immediately visible console line."""
    payload = {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **values}
    _write_json(output / "progress.json", payload)
    # Windows pipes can use cp1252; persisted JSON remains UTF-8.
    print(json.dumps(audit._json_value(payload), ensure_ascii=True), flush=True)


def _active_specs(config: StudyConfig) -> tuple[TreeSpec, ...]:
    """Return the declared scientific grid, or a small explicit smoke grid."""
    declared = specs(config)
    if config.scientific:
        return declared
    keep = {"hgb_uncapped", "hgb_cap125",
            "xgb_uncapped_d3_lr0.1_l21", "xgb_uncapped_d5_lr0.1_l21",
            "xgb_cap125_d3_lr0.1_l21", "xgb_cap125_d5_lr0.1_l21"}
    selected = tuple(item for item in declared if item.candidate_id in keep)
    if len(selected) != len(keep):
        raise ValueError("smoke grid does not match declared candidate IDs")
    return selected


def _fit_batch(
    pool: ProcessPoolExecutor, tasks: list[tuple[TreeSpec, tuple[int, ...], tuple[int, ...], int]],
    root: Path, output: Path, manifest: dict[str, Any], identity: str,
    config: StudyConfig, stage: str,
) -> dict[tuple[str, int], str]:
    """Execute independent folds in retained workers using file paths only."""
    _assert_immutable(root, output, manifest)
    futures: dict[Future[str], tuple[str, int]] = {}
    for spec, fitting, validation, fold in tasks:
        future = pool.submit(run_fit_task, output / "frozen_train_features.parquet",
                             spec, fitting, validation, config, output / "fit_cache", identity)
        futures[future] = (spec.candidate_id, fold)
    paths: dict[tuple[str, int], str] = {}
    try:
        for completed, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            paths[key] = future.result()
            _assert_immutable(root, output, manifest)
            _progress(output, stage, completed_fits=completed, total_fits=len(tasks),
                      candidate_id=key[0], fold=key[1])
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    return paths


def _inner_stage(
    frame: pd.DataFrame, fold: int, pool: ProcessPoolExecutor, root: Path,
    output: Path, manifest: dict[str, Any], identity: str, config: StudyConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame], dict[str, Any]]:
    """Tune XGBoost, calibrate four policies and freeze an inner-only choice."""
    stage = output / "stages" / (f"outer_{fold}" if fold else "final_inner")
    stage.mkdir(parents=True, exist_ok=True)
    rows, tasks = [], []
    candidates = {item.candidate_id: item for item in _active_specs(config)}
    splitter = GroupKFold(n_splits=config.inner_splits)
    for inner_fold, (fit_index, validation_index) in enumerate(
        splitter.split(frame, frame.rul, frame.engine_id), start=1,
    ):
        fitting = tuple(sorted(map(int, frame.iloc[fit_index].engine_id.unique())))
        validation = tuple(sorted(map(int, frame.iloc[validation_index].engine_id.unique())))
        if set(fitting) & set(validation):
            raise ValueError("inner fitting and validation engines overlap")
        rows.extend({"outer_fold": fold, "inner_fold": inner_fold, "engine_id": engine}
                    for engine in validation)
        tasks.extend((spec, fitting, validation, inner_fold) for spec in candidates.values())
    assignments = pd.DataFrame(rows)
    assignments.to_csv(stage / "inner_assignments.csv", index=False)
    paths = _fit_batch(pool, tasks, root, output, manifest, identity, config,
                       f"outer_{fold}_inner" if fold else "final_inner")
    cache_map = {f"{candidate}/fold_{inner}": Path(path).relative_to(output).as_posix()
                 for (candidate, inner), path in sorted(paths.items())}
    _freeze_json(stage / "fit_cache_map.json", cache_map)
    predictions, metrics = {}, []
    for candidate, spec in candidates.items():
        parts = [pd.read_parquet(paths[candidate, inner]).assign(inner_fold=inner)
                 for inner in range(1, config.inner_splits + 1)]
        prediction = pd.concat(parts, ignore_index=True).sort_values(
            ["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
        predictions[candidate] = prediction
        metrics.append({"outer_fold": fold, **asdict(spec), **candidate_readout(prediction, config)})
    table = pd.DataFrame(metrics)
    chosen: dict[str, dict[str, Any]] = {}
    for arm in ARMS[:2]:
        spec = next(spec for spec in candidates.values() if spec.family == arm)
        chosen[arm] = {"candidate_id": spec.candidate_id, "numeric_controls_met": True,
                       "rule": "fixed HGB reference"}
    for arm, reference in (("xgb_uncapped", "hgb_uncapped"), ("xgb_cap125", "hgb_cap125")):
        reference_row = table.loc[table.candidate_id.eq(chosen[reference]["candidate_id"])].iloc[0]
        chosen[arm] = select_xgb_candidate(table, reference_row, arm, config)
    policy_tables, arm_metrics = [], []
    for arm in ARMS:
        candidate = chosen[arm]["candidate_id"]
        policy, grid = calibrate_policy(predictions[candidate], config)
        events = audit.warning_events(predictions[candidate], policy,
                                      lead_requirement=config.lead_requirement)
        readout = {"procedure": arm, "candidate_id": candidate,
                   **candidate_readout(predictions[candidate], config),
                   **audit._warning_readout(events),
                   "numeric_controls_met": bool(chosen[arm]["numeric_controls_met"])}
        arm_metrics.append(readout)
        chosen[arm].update(policy=asdict(policy), spec=asdict(candidates[candidate]),
                           inner_metrics=readout)
        policy_tables.append(grid.assign(procedure=arm, candidate_id=candidate, outer_fold=fold))
    procedure = choose_procedure(pd.DataFrame(arm_metrics), config)
    table["selected_within_family"] = table.candidate_id.isin(
        [choice["candidate_id"] for choice in chosen.values()])
    table.to_csv(stage / "inner_candidate_metrics.csv", index=False)
    pd.DataFrame(arm_metrics).to_csv(stage / "inner_arm_metrics.csv", index=False)
    pd.concat(policy_tables, ignore_index=True).to_csv(stage / "inner_policy_metrics.csv", index=False)
    selection = {"outer_fold": fold, "arms": chosen, "procedure": procedure,
                 "selection_source": "inner OOF from fitting engines only",
                 "source_identity": identity, "test_labels_read": False}
    _freeze_json(stage / "selection.json", selection)
    return chosen, predictions, selection


def _final_model_task(
    feature_path: Path, spec: TreeSpec, config: StudyConfig, folder: Path, identity: str,
) -> str:
    """Fit/save/reload an all-train model in one worker and verify exact replay."""
    folder.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path = folder / "model.joblib", folder / "model.json"
    train = load_frozen_features(feature_path, identity)
    if model_path.exists() and metadata_path.exists():
        metadata = _read_json(metadata_path)
        if (metadata["source_identity"] != identity
                or metadata["spec"] != audit._json_value(asdict(spec))
                or metadata["model_sha256"] != _sha(model_path)):
            raise ValueError("final model checksum or identity mismatch")
        fitted = joblib.load(model_path)
        if fitted.spec != spec:
            raise ValueError("final model object spec mismatch")
        return str(model_path)
    if model_path.exists() or metadata_path.exists():
        interrupted = folder / "interrupted_publications"
        interrupted.mkdir(exist_ok=True)
        for path in (model_path, metadata_path):
            if path.exists():
                path.replace(interrupted / f"{time.time_ns()}_{path.name}")
    started = time.perf_counter()
    fitted = fit_predictor(train, spec, config)
    temporary = model_path.with_suffix(".tmp.joblib")
    joblib.dump(fitted, temporary)
    temporary.replace(model_path)
    restored = joblib.load(model_path)
    with threadpool_limits(limits=config.fit_threads):
        expected, actual = fitted.predict(train), restored.predict(train)
    np.testing.assert_array_equal(expected, actual)
    _write_json(metadata_path, {
        "source_identity": identity, "spec": asdict(spec), "model_sha256": _sha(model_path),
        "effective_parameters": fitted.effective_parameters,
        "weight_metadata": fitted.weight_metadata, "feature_columns": fitted.feature_columns,
        "fit_engine_ids": sorted(map(int, train.engine_id.unique())),
        "fit_rows": len(train), "saved_model_replay_max_abs_delta": 0.0,
        "seconds": time.perf_counter() - started,
    })
    return str(model_path)


def _evaluate_final_test(
    predictions: pd.DataFrame, labels_path: Path, config: StudyConfig, output: Path,
) -> dict[str, Any]:
    """Read labels only after all four models, selections and predictions exist."""
    required = [output / "selection.json", output / "final_predictions.parquet",
                output / "final_predictions_receipt.json"]
    required.extend(output / "models" / arm / name
                    for arm in ARMS for name in ("model.json", "model.joblib"))
    if not all(path.is_file() for path in required):
        raise ValueError("all choices/models/predictions must precede test label access")
    receipt = _read_json(output / "final_predictions_receipt.json")
    selection = _read_json(output / "selection.json")
    if (selection.get("test_labels_read_for_selection") is not False
            or selection.get("source_identity") != receipt.get("source_identity")):
        raise ValueError("final selection must be frozen before test label access")
    if receipt.get("test_labels_read") is not False or receipt.get("saved_model_replay_exact") is not True:
        raise ValueError("final prediction receipt does not prove pre-label replay")
    if receipt.get("predictions_sha256") != _sha(output / "final_predictions.parquet"):
        raise ValueError("final prediction receipt checksum mismatch")
    try:
        pd.testing.assert_frame_equal(predictions, pd.read_parquet(output / "final_predictions.parquet"),
                                      check_exact=True)
    except AssertionError as error:
        raise ValueError("test reporting must use the exact persisted predictions") from error
    for arm in ARMS:
        model_path = output / "models" / arm / "model.joblib"
        if receipt.get("models_sha256", {}).get(arm) != _sha(model_path):
            raise ValueError(f"final model checksum mismatch before test read: {arm}")
    labels = read_test_labels(labels_path).rename(columns={"rul": "y_true"})
    evaluations, summaries = [], []
    for procedure, frame in predictions.groupby("procedure", sort=True):
        joined = frame.merge(labels, on=["split", "engine_id"], how="outer",
                             validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError("test prediction and label engines differ")
        evaluation = add_prediction_errors(joined.drop(columns="_merge"))
        evaluations.append(evaluation)
        metrics = {"procedure": procedure, **summarize_prediction_errors(evaluation),
                   "bootstrap_intervals": bootstrap_metric_intervals(
                       evaluation, n_resamples=config.bootstrap_resamples, seed=config.seed)}
        summaries.append(metrics)
        folder = output / "test" / procedure
        folder.mkdir(parents=True, exist_ok=True)
        audit._write_parquet(frame.rename(columns={"y_pred": "rul_pred"}), folder / "final_predictions.parquet")
        audit._write_parquet(evaluation, folder / "final_evaluation.parquet")
        _write_json(folder / "final_metrics.json", metrics)
    audit._write_parquet(pd.concat(evaluations, ignore_index=True), output / "final_evaluation.parquet")
    pd.DataFrame([{key: value for key, value in row.items() if key != "bootstrap_intervals"}
                  for row in summaries]).to_csv(output / "test_metrics.csv", index=False)
    result = {"test_labels_sha256": _sha(labels_path), "procedures": summaries,
              "selection_used_test_labels": False, "outer_gate_changed_by_test": False,
              "reporting_only": True, "historically_inspected_test": True,
              "interval_scope": "engine bootstrap conditional on fitted model; excludes retraining variability"}
    _write_json(output / "test_evaluation.json", result)
    return result


def run_tree_comparison(
    *, repo_root: str | Path | None = None,
    observations_path: str | Path = "outputs/etl/silver/observations.parquet",
    targets_path: str | Path = "outputs/etl/silver/train_targets.parquet",
    test_labels_path: str | Path = "data/raw/RUL_FD001.txt",
    output_dir: str | Path = "outputs/experiments/tree_comparison/20261005",
    config: StudyConfig | None = None, evaluate_test: bool = True,
) -> dict[str, Any]:
    """Execute four controlled arms and the inner-selected outer procedure.

    Args:
        repo_root: Repository with canonical inputs and published engine folds.
        observations_path: Canonical train/test observations without test labels.
        targets_path: Complete uncapped train labels.
        test_labels_path: Historically inspected reporting labels, read last.
        output_dir: Isolated fresh or compatible resumable study directory.
        config: Predeclared fitting, split, selection and warning configuration.
        evaluate_test: Report test metrics only after frozen final predictions.

    Returns:
        Completed run receipt with outer decision and reporting-only test status.
    """
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[4]
    config = config or StudyConfig()
    output = _resolve(root, output_dir)
    observations, targets, labels = (_resolve(root, path) for path in
                                     (observations_path, targets_path, test_labels_path))
    manifest, identity, train, test, assignments = _freeze_inputs(
        root, output, observations, targets, config)
    started = time.perf_counter()
    candidate_specs = {item.candidate_id: item for item in _active_specs(config)}
    predictions, events, selections, stage_tables = [], [], [], []
    with ProcessPoolExecutor(max_workers=config.max_workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for fold in range(1, config.outer_splits + 1):
            validation_ids = tuple(sorted(map(int, assignments.loc[
                assignments.outer_fold.eq(fold), "engine_id"])))
            fitting_ids = tuple(sorted(set(map(int, train.engine_id.unique())) - set(validation_ids)))
            fitting = train[train.engine_id.isin(fitting_ids)].reset_index(drop=True)
            choices, _, selection = _inner_stage(
                fitting, fold, pool, root, output, manifest, identity, config)
            selections.append({**selection, "outer_fit_engine_ids": fitting_ids,
                               "outer_validation_engine_ids": validation_ids})
            stage_tables.append(output / "stages" / f"outer_{fold}")
            tasks = [(candidate_specs[choices[arm]["candidate_id"]], fitting_ids,
                      validation_ids, fold) for arm in ARMS]
            paths = _fit_batch(pool, tasks, root, output, manifest, identity, config,
                               f"outer_{fold}_evaluation")
            for arm in ARMS:
                candidate = choices[arm]["candidate_id"]
                policy = audit.WarningPolicy(**choices[arm]["policy"])
                prediction = pd.read_parquet(paths[candidate, fold]).assign(
                    procedure=arm, outer_fold=fold, threshold=policy.threshold,
                    persistence=policy.persistence)
                warning = audit.warning_events(prediction, policy,
                                                lead_requirement=config.lead_requirement).assign(
                    procedure=arm, outer_fold=fold, candidate_id=candidate)
                predictions.append(prediction)
                events.append(warning)
                if arm == selection["procedure"]["selected_procedure_id"]:
                    predictions.append(prediction.assign(procedure="selected_pipeline", selected_arm=arm))
                    events.append(warning.assign(procedure="selected_pipeline", selected_arm=arm))
            audit._write_parquet(pd.concat(predictions, ignore_index=True), output / "outer_predictions.parquet")
            audit._write_parquet(pd.concat(events, ignore_index=True), output / "warning_events.parquet")
            _write_json(output / "outer_selections.json", selections)
            for name in ("inner_candidate_metrics.csv", "inner_assignments.csv", "inner_policy_metrics.csv"):
                pd.concat([pd.read_csv(folder / name) for folder in stage_tables], ignore_index=True).to_csv(
                    output / name, index=False)
            _progress(output, "outer_completed", completed_outer_folds=fold,
                      total_outer_folds=config.outer_splits)
        outer = pd.concat(predictions, ignore_index=True).sort_values(
            ["procedure", "engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
        warnings = pd.concat(events, ignore_index=True).sort_values(
            ["procedure", "engine_id"], kind="mergesort").reset_index(drop=True)
        audit._write_parquet(outer, output / "outer_predictions.parquet")
        audit._write_parquet(warnings, output / "warning_events.parquet")
        endpoints = pd.concat([audit.build_endpoint_panel(group, config.endpoint_fractions)
                               for _, group in outer.groupby("procedure", sort=True)], ignore_index=True)
        audit._write_parquet(endpoints, output / "outer_endpoints.parquet")
        _assert_immutable(root, output, manifest)
        decision = write_readouts(outer, warnings, config, output)
        _freeze_json(output / "outer_decision_before_test.json", decision)
        choices, _, final_selection = _inner_stage(train, 0, pool, root, output,
                                                    manifest, identity, config)
        for name in ("inner_candidate_metrics.csv", "inner_assignments.csv", "inner_policy_metrics.csv"):
            shutil.copy2(output / "stages/final_inner" / name, output / f"final_{name}")
        final_selection.update(config=asdict(config), outer_decision=decision,
                               author=AUTHOR, test_labels_read_for_selection=False)
        _freeze_json(output / "selection.json", final_selection)
        _assert_immutable(root, output, manifest)
        futures = {pool.submit(_final_model_task, output / "frozen_train_features.parquet",
                               candidate_specs[choices[arm]["candidate_id"]], config,
                               output / "models" / arm, identity): arm for arm in ARMS}
        model_paths = {}
        for completed, future in enumerate(as_completed(futures), start=1):
            arm = futures[future]
            model_paths[arm] = Path(future.result())
            _assert_immutable(root, output, manifest)
            _progress(output, "final_models", completed_fits=completed, total_fits=len(ARMS), procedure=arm)
    final_test = test.groupby("engine_id", sort=True, as_index=False).tail(1).reset_index(drop=True)
    if final_test.engine_id.nunique() != test.engine_id.nunique():
        raise ValueError("final test endpoints do not cover all engines")
    final_predictions = []
    for arm in ARMS:
        fitted = joblib.load(model_paths[arm])
        with threadpool_limits(limits=config.fit_threads):
            predicted = fitted.predict(final_test)
            replayed = joblib.load(model_paths[arm]).predict(final_test)
        np.testing.assert_array_equal(predicted, replayed)
        frame = final_test[list(KEY)].copy()
        frame["candidate_id"] = choices[arm]["candidate_id"]
        frame["procedure"] = arm
        frame["y_pred"] = predicted
        final_predictions.append(frame)
        if arm == final_selection["procedure"]["selected_procedure_id"]:
            final_predictions.append(frame.assign(procedure="selected_pipeline", selected_arm=arm))
    final = pd.concat(final_predictions, ignore_index=True)
    audit._write_parquet(final, output / "final_predictions.parquet")
    _freeze_json(output / "final_predictions_receipt.json", {
        "source_identity": identity, "predictions_sha256": _sha(output / "final_predictions.parquet"),
        "test_labels_read": False, "saved_model_replay_exact": True,
        "models_sha256": {arm: _sha(path) for arm, path in model_paths.items()},
    })
    _assert_immutable(root, output, manifest)
    test_report = _evaluate_final_test(final, labels, config, output) if evaluate_test else None
    if _read_json(output / "outer_decision_before_test.json") != audit._json_value(decision):
        raise ValueError("outer decision changed after test reporting")
    result = {"status": "completed", "author": AUTHOR, "source_identity": identity,
              "output_dir": output.relative_to(root).as_posix(), "decision": decision,
              "test_evaluated": evaluate_test, "test_reporting_only": True,
              "selected_procedure": final_selection["procedure"]["selected_procedure_id"],
              "seconds": time.perf_counter() - started,
              "historical_baseline_overwritten": False}
    _write_json(output / "run_info.json", result)
    _progress(output, "completed", **result)
    return result


def main(argv: list[str] | None = None) -> None:
    """Expose portable scientific and reduced smoke-run command line modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--observations", default="outputs/etl/silver/observations.parquet")
    parser.add_argument("--targets", default="outputs/etl/silver/train_targets.parquet")
    parser.add_argument("--test-labels", default="data/raw/RUL_FD001.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-threads", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = StudyConfig(fit_threads=args.fit_threads, max_workers=args.max_workers)
    if args.dry_run:
        config = replace(config, outer_splits=2, inner_splits=2, bootstrap_resamples=20,
                         hgb_max_iter=20, xgb_n_estimators=20)
    run_tree_comparison(repo_root=args.repo_root, observations_path=args.observations,
                        targets_path=args.targets, test_labels_path=args.test_labels,
                        output_dir=args.output_dir, config=config,
                        evaluate_test=not args.skip_test_evaluation)
