"""Huấn luyện theo fold, kiểm tra cache và lưu model."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any
import time

from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import joblib
import numpy as np
import pandas as pd

from cmapss.modeling import audit_study as audit
from cmapss.modeling.comparison_runner.artifact_io import (
    _freeze_json,
    _progress,
    _read_json,
    _sha,
    _write_json,
)
from cmapss.modeling.comparison_runner.contracts import (
    CandidateSpec,
    ComparisonBackend,
    ComparisonConfig,
)
from cmapss.modeling.comparison_runner.provenance import _assert_immutable


def fit_batch(
    pool: ProcessPoolExecutor, tasks: list[tuple[CandidateSpec, tuple[int, ...], tuple[int, ...], int]],
    root: Path, output: Path, manifest: dict[str, Any], identity: str,
    config: ComparisonConfig, stage: str, backend: ComparisonBackend,
) -> dict[tuple[str, int], str]:
    """Run fit tasks in the pool (workers get file paths, not frames)."""
    _assert_immutable(root, output, manifest)
    futures: dict[Future[str], tuple[str, int]] = {}
    for spec, fitting, validation, fold in tasks:
        future = pool.submit(backend.fit_task, output / "frozen_train_features.parquet",
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


def inner_stage(
    frame: pd.DataFrame, fold: int, pool: ProcessPoolExecutor, root: Path,
    output: Path, manifest: dict[str, Any], identity: str, config: ComparisonConfig, backend: ComparisonBackend,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame], dict[str, Any]]:
    """Fit and score candidates on inner folds, calibrate policies, save the selection."""
    stage = output / "stages" / (f"outer_{fold}" if fold else "final_inner")
    stage.mkdir(parents=True, exist_ok=True)
    rows, tasks = [], []
    candidates = {item.candidate_id: item for item in backend.active_specs(config)}
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
    paths = fit_batch(pool, tasks, root, output, manifest, identity, config,
                       f"outer_{fold}_inner" if fold else "final_inner", backend)
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
        metrics.append({"outer_fold": fold, **asdict(spec), **backend.candidate_readout(prediction, config)})
    table = pd.DataFrame(metrics)
    chosen: dict[str, dict[str, Any]] = {}
    for arm in backend.arms[:2]:
        spec = next(spec for spec in candidates.values() if spec.family == arm)
        chosen[arm] = {"candidate_id": spec.candidate_id, "numeric_controls_met": True,
                       "rule": "fixed HGB reference"}
    for arm, reference in zip(backend.arms[2:], backend.arms[:2], strict=True):
        reference_row = table.loc[table.candidate_id.eq(chosen[reference]["candidate_id"])].iloc[0]
        chosen[arm] = backend.select_candidate(table, reference_row, arm, config)
    policy_tables, arm_metrics = [], []
    for arm in backend.arms:
        candidate = chosen[arm]["candidate_id"]
        policy, grid = backend.calibrate_policy(predictions[candidate], config)
        events = audit.warning_events(predictions[candidate], policy,
                                      lead_requirement=config.lead_requirement)
        readout = {"procedure": arm, "candidate_id": candidate,
                   **backend.candidate_readout(predictions[candidate], config),
                   **audit._warning_readout(events),
                   "numeric_controls_met": bool(chosen[arm]["numeric_controls_met"])}
        arm_metrics.append(readout)
        chosen[arm].update(policy=asdict(policy), spec=asdict(candidates[candidate]),
                           inner_metrics=readout)
        policy_tables.append(grid.assign(procedure=arm, candidate_id=candidate, outer_fold=fold))
    procedure = backend.choose_procedure(pd.DataFrame(arm_metrics), config)
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


def final_model_task(
    feature_path: Path, spec: CandidateSpec, config: ComparisonConfig, folder: Path, identity: str,
    backend: ComparisonBackend,
) -> str:
    """Fit on all train rows, save, reload and check that predictions match exactly."""
    folder.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path = folder / "model.joblib", folder / "model.json"
    train = backend.load_frozen_features(feature_path, identity)
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
    fitted = backend.fit_predictor(train, spec, config)
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
