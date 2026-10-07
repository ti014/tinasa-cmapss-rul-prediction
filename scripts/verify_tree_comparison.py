"""Đối chiếu read-only bằng chứng của tree-comparison study.

Verifier này không fit model, không đọc nhãn test và không thay đổi artifact
đã có.  Nó kiểm tra provenance, phân tách engine, cache prediction, model
final, metric outer và các cờ reporting-only; kết quả duy nhất được ghi là
``verification.json`` trong run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss.data.contracts import KEY
from cmapss.modeling import audit_study as audit
from cmapss.modeling.metrics import add_prediction_errors
from cmapss.modeling.tree_comparison import analysis, runner
from cmapss.modeling.tree_comparison.contracts import FEATURES, StudyConfig, specs
from cmapss.modeling.tree_comparison.estimators import FittedTree


ARMS = ("hgb_uncapped", "hgb_cap125", "xgb_uncapped", "xgb_cap125")
PROCEDURES = frozenset((*ARMS, "selected_pipeline"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes repository: {resolved}")
    return resolved


def _same_frame(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str] | tuple[str, ...],
    *,
    exact: bool = False,
) -> bool:
    """Compare stable keyed tables without depending on parquet row order."""
    keys = list(keys)
    if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
        return False
    columns = list(expected.columns)
    try:
        left = actual.loc[:, columns].sort_values(keys, kind="mergesort").reset_index(drop=True)
        right = expected.loc[:, columns].sort_values(keys, kind="mergesort").reset_index(drop=True)
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=exact,
            atol=0.0 if exact else 1e-8,
            rtol=0.0 if exact else 1e-10,
        )
    except (AssertionError, KeyError, TypeError, ValueError):
        return False
    return True


def _record(checks: dict[str, bool], name: str, value: bool) -> None:
    checks[name] = bool(value)


def _config(identity: dict[str, Any]) -> StudyConfig:
    values = dict(identity["config"])
    for name in ("endpoint_fractions", "thresholds", "persistences"):
        values[name] = tuple(values[name])
    return StudyConfig(**values)


def _frozen_prediction_truth(
    frozen: pd.DataFrame, prediction: pd.DataFrame, validation_ids: set[int],
) -> bool:
    """Require exact validation keys and raw RUL, independent of predictions."""
    expected = frozen[frozen.engine_id.isin(validation_ids)].loc[
        :, [*KEY, "rul"]
    ].rename(columns={"rul": "y_true"}).reset_index(drop=True)
    actual = prediction.loc[:, [*KEY, "y_true"]].reset_index(drop=True)
    if not _same_frame(actual, expected, list(KEY), exact=True):
        return False
    return np.isfinite(prediction["y_pred"].to_numpy(dtype="float64")).all()


def _verify_cache(
    folder: Path,
    identity: dict[str, Any],
    frozen: pd.DataFrame,
    cfg: StudyConfig,
    checks: dict[str, bool],
) -> tuple[list[tuple[Path, dict[str, Any]]], dict[str, int]]:
    """Check every CV cache object and its engine-bound raw truth."""
    cache_folder = folder / "fit_cache"
    metadata_paths = sorted(cache_folder.glob("*.json"))
    records: list[tuple[Path, dict[str, Any]]] = []
    valid = True
    candidates = {item.candidate_id: item for item in specs(cfg)}
    feature_order = list(dict.fromkeys((*KEY, *FEATURES, "rul")))
    digest = hashlib.sha256()
    source_frame = frozen.loc[:, feature_order]
    digest.update(repr(tuple(source_frame.columns)).encode("utf-8"))
    digest.update(repr(tuple(str(dtype) for dtype in source_frame.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(source_frame, index=False).to_numpy().tobytes())
    source_hash = digest.hexdigest()
    for metadata_path in metadata_paths:
        metadata = _json(metadata_path)
        parquet = metadata_path.with_suffix(".parquet")
        records.append((parquet, metadata))
        valid = valid and parquet.is_file()
        if not parquet.is_file():
            continue
        valid = valid and metadata.get("source_identity") == identity["source_identity"]
        valid = valid and metadata.get("prediction_sha256") == _sha(parquet)
        candidate = metadata.get("candidate", {})
        candidate_id = candidate.get("candidate_id")
        valid = valid and candidate_id in candidates
        if candidate_id in candidates:
            valid = valid and candidate == asdict(candidates[candidate_id])
        valid = valid and metadata.get("config") == audit._json_value(asdict(cfg))
        valid = valid and metadata.get("source_hash") == source_hash
        valid = valid and metadata.get("n_features") == len(FEATURES)
        valid = valid and metadata.get("feature_columns") == list(FEATURES)
        fit_ids = set(map(int, metadata.get("fit_engine_ids", [])))
        validation_ids = set(map(int, metadata.get("validation_engine_ids", [])))
        valid = valid and bool(fit_ids) and bool(validation_ids) and not fit_ids.intersection(validation_ids)
        prediction = pd.read_parquet(parquet)
        required = {*KEY, "candidate_id", "y_true", "y_pred",
                    "absolute_error", "squared_error", "nasa_score_component"}
        valid = valid and required.issubset(prediction.columns)
        valid = valid and not prediction.duplicated(list(KEY)).any()
        if required.issubset(prediction.columns):
            valid = valid and prediction.candidate_id.eq(candidate_id).all()
            valid = valid and _frozen_prediction_truth(frozen, prediction, validation_ids)
            rescored = add_prediction_errors(prediction.loc[:, [*KEY, "candidate_id", "y_true", "y_pred"]])
            valid = valid and _same_frame(
                prediction.loc[:, list(rescored.columns)], rescored, list(KEY), exact=True,
            )
            valid = valid and set(map(int, prediction.engine_id.unique())) == validation_ids
    expected_inner = (cfg.outer_splits * cfg.inner_splits + cfg.inner_splits) * len(candidates)
    expected_outer = cfg.outer_splits * len(ARMS)
    expected_cv = expected_inner + expected_outer
    counts = {
        "cache_json": len(metadata_paths),
        "expected_inner": expected_inner,
        "expected_outer": expected_outer,
        "expected_cv": expected_cv,
        "expected_final_models": len(ARMS),
        "expected_total_fits": expected_cv + len(ARMS),
    }
    _record(checks, "cache_prediction_integrity", valid)
    _record(checks, "complete_cv_cache_count", len(metadata_paths) == expected_cv)
    _record(checks, "full_candidate_grid", len(candidates) == 26)
    return records, counts


def _cache_match(
    records: list[tuple[Path, dict[str, Any]]],
    candidate_id: str,
    fit_ids: set[int],
    validation_ids: set[int],
) -> list[tuple[Path, dict[str, Any]]]:
    return [
        (path, metadata) for path, metadata in records
        if metadata["candidate"]["candidate_id"] == candidate_id
        and set(map(int, metadata["fit_engine_ids"])) == fit_ids
        and set(map(int, metadata["validation_engine_ids"])) == validation_ids
    ]


def _verify_fit_coverage(
    folder: Path,
    frozen: pd.DataFrame,
    cfg: StudyConfig,
    records: list[tuple[Path, dict[str, Any]]],
    checks: dict[str, bool],
    counts: dict[str, int],
) -> None:
    """Match every declared inner/outer task to exactly one cache publication."""
    all_engines = set(map(int, frozen.engine_id.unique()))
    outer_assignments = pd.read_csv(folder / "outer_assignments.csv")
    valid_inner = True
    inner_paths: set[Path] = set()
    for outer_fold in range(cfg.outer_splits + 1):
        stage = folder / "stages" / (f"outer_{outer_fold}" if outer_fold else "final_inner")
        assignments = pd.read_csv(stage / "inner_assignments.csv")
        cohort = set(map(int, assignments.engine_id))
        expected_cohort = all_engines - set(map(int, outer_assignments.loc[
            outer_assignments.outer_fold.eq(outer_fold), "engine_id"
        ])) if outer_fold else all_engines
        valid_inner = valid_inner and cohort == expected_cohort
        path_map = _json(stage / "fit_cache_map.json")
        expected_names: set[str] = set()
        for spec in specs(cfg):
            for inner_fold in range(1, cfg.inner_splits + 1):
                validation = set(map(int, assignments.loc[
                    assignments.inner_fold.eq(inner_fold), "engine_id"
                ]))
                fitting = cohort - validation
                matches = _cache_match(records, spec.candidate_id, fitting, validation)
                label = f"{spec.candidate_id}/fold_{inner_fold}"
                expected_names.add(label)
                valid_inner = valid_inner and len(matches) == 1 and label in path_map
                if len(matches) == 1 and label in path_map:
                    path = matches[0][0]
                    inner_paths.add(path)
                    valid_inner = valid_inner and path == _resolve(folder, path_map[label])
        valid_inner = valid_inner and set(path_map) == expected_names
    outer_selections = json.loads((folder / "outer_selections.json").read_text(encoding="utf-8"))
    outer_prediction = pd.read_parquet(folder / "outer_predictions.parquet")
    valid_outer = True
    outer_paths: set[Path] = set()
    for case in outer_selections:
        fold = int(case["outer_fold"])
        validation = set(map(int, outer_assignments.loc[
            outer_assignments.outer_fold.eq(fold), "engine_id"
        ]))
        fitting = all_engines - validation
        valid_outer = valid_outer and set(case["outer_fit_engine_ids"]) == fitting
        valid_outer = valid_outer and set(case["outer_validation_engine_ids"]) == validation
        for arm in ARMS:
            candidate_id = case["arms"][arm]["candidate_id"]
            matches = _cache_match(records, candidate_id, fitting, validation)
            valid_outer = valid_outer and len(matches) == 1
            if len(matches) == 1:
                path = matches[0][0]
                outer_paths.add(path)
                prediction = pd.read_parquet(path)
                expected = outer_prediction[
                    outer_prediction.outer_fold.eq(fold) & outer_prediction.procedure.eq(arm)
                ]
                valid_outer = valid_outer and _same_frame(
                    expected.loc[:, list(prediction.columns)], prediction, list(KEY), exact=True,
                )
    counts["unique_inner_fits"] = len(inner_paths)
    counts["unique_outer_fits"] = len(outer_paths)
    counts["unique_final_fits"] = len(list((folder / "models").glob("*/model.joblib")))
    counts["unique_total_fits"] = len(inner_paths | outer_paths) + counts["unique_final_fits"]
    _record(checks, "complete_inner_task_coverage", valid_inner and len(inner_paths) == counts["expected_inner"])
    _record(checks, "complete_outer_task_coverage", valid_outer and len(outer_paths) == counts["expected_outer"])
    _record(checks, "unique_492_full_fit_coverage", counts["unique_total_fits"] == 492
            and counts["unique_final_fits"] == 4 and not (inner_paths & outer_paths))


def _verify_assignments(
    folder: Path,
    frozen: pd.DataFrame,
    cfg: StudyConfig,
    checks: dict[str, bool],
) -> None:
    outer = pd.read_csv(folder / "outer_assignments.csv")
    inner = pd.read_csv(folder / "inner_assignments.csv")
    final_inner = pd.read_csv(folder / "final_inner_assignments.csv")
    all_engines = set(map(int, frozen.engine_id.unique()))
    valid = set(outer.columns) >= {"engine_id", "outer_fold"}
    valid = valid and not outer.duplicated(["engine_id"]).any()
    valid = valid and set(map(int, outer.engine_id)) == all_engines
    valid = valid and set(map(int, outer.outer_fold)) == set(range(1, cfg.outer_splits + 1))
    valid = valid and set(inner.columns) >= {"outer_fold", "inner_fold", "engine_id"}
    valid = valid and set(map(int, inner.outer_fold)) == set(range(1, cfg.outer_splits + 1))
    for fold in range(1, cfg.outer_splits + 1):
        validation = set(map(int, outer.loc[outer.outer_fold.eq(fold), "engine_id"]))
        fitting = all_engines - validation
        block = inner[inner.outer_fold.eq(fold)]
        valid = valid and set(map(int, block.engine_id)) == fitting
        valid = valid and not block.duplicated(["inner_fold", "engine_id"]).any()
        valid = valid and set(map(int, block.inner_fold)) == set(range(1, cfg.inner_splits + 1))
        for inner_fold in range(1, cfg.inner_splits + 1):
            rows = block[block.inner_fold.eq(inner_fold)]
            valid = valid and not rows.duplicated(["engine_id"]).any()
    valid = valid and set(map(int, final_inner.outer_fold)) == {0}
    valid = valid and set(map(int, final_inner.engine_id)) == all_engines
    _record(checks, "engine_fold_assignments", valid)
    _record(checks, "inner_outer_engine_separation", valid)


def _verify_outer_aliases(
    folder: Path,
    frozen: pd.DataFrame,
    cfg: StudyConfig,
    checks: dict[str, bool],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = pd.read_parquet(folder / "outer_predictions.parquet")
    events = pd.read_parquet(folder / "warning_events.parquet")
    valid = set(outer.procedure) == PROCEDURES
    valid = valid and not outer.duplicated(["procedure", "outer_fold", *KEY]).any()
    valid = valid and not events.duplicated(["procedure", "outer_fold", "engine_id"]).any()
    truth = frozen.loc[:, [*KEY, "rul"]].rename(columns={"rul": "y_true"})
    for procedure in PROCEDURES:
        frame = outer[outer.procedure.eq(procedure)]
        joined = frame.merge(truth, on=list(KEY), how="outer", suffixes=("", "_frozen"), indicator=True)
        valid = valid and joined._merge.eq("both").all()
        if "y_true_frozen" in joined:
            valid = valid and np.array_equal(joined.y_true, joined.y_true_frozen)
    selections = json.loads((folder / "outer_selections.json").read_text(encoding="utf-8"))
    if len(selections) != cfg.outer_splits:
        valid = False
    for selection in selections:
        fold = int(selection["outer_fold"])
        selected_arm = selection["procedure"]["selected_procedure_id"]
        base = outer[(outer.outer_fold.eq(fold)) & (outer.procedure.eq(selected_arm))]
        alias = outer[(outer.outer_fold.eq(fold)) & (outer.procedure.eq("selected_pipeline"))]
        valid = valid and selected_arm in ARMS
        valid = valid and _same_frame(
            base.loc[:, [*KEY, "candidate_id", "y_true", "y_pred", "threshold", "persistence"]],
            alias.loc[:, [*KEY, "candidate_id", "y_true", "y_pred", "threshold", "persistence"]],
            list(KEY), exact=True,
        )
        valid = valid and alias.get("selected_arm", pd.Series(dtype=object)).eq(selected_arm).all()
    _record(checks, "outer_prediction_truth_and_keys", valid)
    _record(checks, "selected_pipeline_aliases_inner_choice", valid)
    return outer, events


def _verify_outer_readouts(
    folder: Path,
    outer: pd.DataFrame,
    events: pd.DataFrame,
    cfg: StudyConfig,
    checks: dict[str, bool],
) -> None:
    valid_events = True
    reproduced_events: list[pd.DataFrame] = []
    for (procedure, fold), frame in outer.groupby(["procedure", "outer_fold"], sort=True):
        if frame.threshold.nunique() != 1 or frame.persistence.nunique() != 1:
            valid_events = False
            continue
        policy = audit.WarningPolicy(int(frame.threshold.iloc[0]), int(frame.persistence.iloc[0]))
        recalculated = audit.warning_events(frame, policy, lead_requirement=cfg.lead_requirement)
        recalculated = recalculated.assign(
            procedure=procedure,
            outer_fold=fold,
            candidate_id=frame.candidate_id.iloc[0],
        )
        if "selected_arm" in frame:
            recalculated = recalculated.assign(selected_arm=frame.selected_arm.iloc[0])
        reproduced_events.append(recalculated)
    if reproduced_events:
        expected_events = pd.concat(reproduced_events, ignore_index=True)
        valid_events = valid_events and _same_frame(
            events, expected_events, ["procedure", "outer_fold", "engine_id"], exact=False,
        )
    else:
        valid_events = False
    _record(checks, "warning_events_recomputed", valid_events)
    aggregate_rows: list[dict[str, Any]] = []
    for procedure, frame in outer.groupby("procedure", sort=True):
        policy_events = events[events.procedure.eq(procedure)]
        endpoint_frame = audit.build_endpoint_panel(frame, cfg.endpoint_fractions)
        aggregate_rows.append({
            "procedure": procedure,
            **_independent_metrics(endpoint_frame, "endpoint"),
            **_independent_metrics(frame, "trajectory"),
            **audit._warning_readout(policy_events),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    stored = pd.read_csv(folder / "aggregate_metrics.csv")
    _record(checks, "outer_aggregate_metrics_recomputed", _same_frame(stored, aggregate, ["procedure"]))
    endpoints = pd.read_parquet(folder / "outer_endpoints.parquet")
    reproduced_endpoints = pd.concat(
        [audit.build_endpoint_panel(frame, cfg.endpoint_fractions)
         for _, frame in outer.groupby("procedure", sort=True)],
        ignore_index=True,
    )
    _record(
        checks,
        "outer_endpoint_panel_recomputed",
        _same_frame(endpoints, reproduced_endpoints, ["procedure", "engine_id", "endpoint_fraction"]),
    )
    paired = analysis.paired_comparisons(outer, events, cfg)
    stored_pairs = pd.read_csv(folder / "paired_comparison.csv", float_precision="round_trip")
    _record(checks, "paired_bootstrap_recomputed", _same_frame(
        stored_pairs, paired, ["reference", "procedure", "metric"], exact=True,
    ))
    decision = _json(folder / "outer_decision_before_test.json")
    selected = paired[paired.reference.eq("hgb_uncapped") & paired.procedure.eq("selected_pipeline")]
    controls = analysis._gate_controls(selected)
    gate_valid = decision["controls"] == controls
    gate_valid = gate_valid and decision["full_protocol_configuration"] == cfg.scientific
    gate_valid = gate_valid and decision["promotion_gate_met"] == bool(cfg.scientific and all(controls.values()))
    _record(checks, "outer_gate_recomputed_without_test", gate_valid)


def _independent_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    """Compute published errors directly from raw NumPy prediction arrays."""
    truth = frame.y_true.to_numpy(dtype="float64")
    predicted = frame.y_pred.to_numpy(dtype="float64")
    errors = predicted - truth
    absolute = np.abs(errors)
    squared = errors ** 2
    nasa = np.where(errors < 0.0, np.exp(-errors / 13.0) - 1.0, np.exp(errors / 10.0) - 1.0)
    engines = frame.engine_id.to_numpy()
    engine_mae: list[float] = []
    engine_nasa: list[float] = []
    for engine in np.unique(engines):
        mask = engines == engine
        engine_mae.append(float(absolute[mask].mean()))
        engine_nasa.append(float(nasa[mask].mean()))
    return {
        f"{prefix}_n_rows": len(frame),
        f"{prefix}_n_engines": len(engine_mae),
        f"{prefix}_mean_window_mae": float(absolute.mean()),
        f"{prefix}_mean_engine_mae": float(np.mean(engine_mae)),
        f"{prefix}_rmse": float(np.sqrt(squared.mean())),
        f"{prefix}_nasa_score": float(nasa.sum()),
        f"{prefix}_mean_engine_nasa": float(np.mean(engine_nasa)),
    }


def _verify_final_models(
    folder: Path,
    identity: dict[str, Any],
    frozen_test: pd.DataFrame,
    selection: dict[str, Any],
    checks: dict[str, bool],
) -> None:
    final = pd.read_parquet(folder / "final_predictions.parquet")
    valid = not ({"y_true", "rul"} & set(final.columns)) and set(final.procedure) == PROCEDURES
    test_endpoints = frozen_test.sort_values(["engine_id", "cycle"], kind="mergesort").groupby(
        "engine_id", sort=True, as_index=False,
    ).tail(1).reset_index(drop=True)
    for arm in ARMS:
        model_path = folder / "models" / arm / "model.joblib"
        metadata_path = folder / "models" / arm / "model.json"
        valid = valid and model_path.is_file() and metadata_path.is_file()
        if not model_path.is_file() or not metadata_path.is_file():
            continue
        metadata = _json(metadata_path)
        model = joblib.load(model_path)
        candidate_id = selection["arms"][arm]["candidate_id"]
        valid = valid and metadata.get("source_identity") == identity["source_identity"]
        valid = valid and metadata.get("model_sha256") == _sha(model_path)
        valid = valid and isinstance(model, FittedTree)
        valid = valid and model.spec.candidate_id == candidate_id
        valid = valid and metadata.get("spec") == asdict(model.spec)
        valid = valid and metadata.get("feature_columns") == list(FEATURES)
        valid = valid and model.feature_columns == FEATURES
        predicted = model.predict(test_endpoints)
        stored = final[final.procedure.eq(arm)].sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
        expected = test_endpoints.loc[:, list(KEY)].copy()
        expected["candidate_id"] = candidate_id
        expected["procedure"] = arm
        expected["y_pred"] = predicted
        valid = valid and _same_frame(stored.loc[:, list(expected.columns)], expected, list(KEY), exact=True)
    receipt = _json(folder / "final_predictions_receipt.json")
    valid = valid and receipt.get("source_identity") == identity["source_identity"]
    valid = valid and receipt.get("test_labels_read") is False
    valid = valid and receipt.get("saved_model_replay_exact") is True
    valid = valid and receipt.get("predictions_sha256") == _sha(folder / "final_predictions.parquet")
    valid = valid and receipt.get("models_sha256") == {
        arm: _sha(folder / "models" / arm / "model.joblib") for arm in ARMS
    }
    final_selected = selection["procedure"]["selected_procedure_id"]
    base = final[final.procedure.eq(final_selected)]
    alias = final[final.procedure.eq("selected_pipeline")]
    columns = [*KEY, "candidate_id", "y_pred"]
    valid = valid and _same_frame(base.loc[:, columns], alias.loc[:, columns], list(KEY), exact=True)
    valid = valid and alias.selected_arm.eq(final_selected).all()
    _record(checks, "final_models_checksum_and_prediction_replay", valid)


def verify_tree_comparison(
    output_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Verify a completed full study using read-only operations only."""
    root = Path(repo_root).resolve()
    folder = _resolve(root, output_dir)
    manifest = _json(folder / "manifest.json")
    identity = manifest["identity"]
    runtime_identity = {**identity, "source_identity": manifest["source_identity"]}
    cfg = _config(identity)
    checks: dict[str, bool] = {}
    # This call is the canonical source/input/lock/frozen-byte guard.
    runner._assert_immutable(root, folder, manifest)
    _record(checks, "runner_immutable_source_input_lock_guard", True)
    snapshot_valid = all(_sha(folder / "source_snapshot" / name) == digest
                         for name, digest in identity["source_sha256"].items())
    snapshot_valid = snapshot_valid and all(_sha(folder / "input_snapshot" / name) == digest
                         for name, digest in identity["inputs_sha256"].items())
    snapshot_valid = snapshot_valid and _sha(folder / "source_snapshot/uv.lock") == identity["dependency_lock_sha256"]
    snapshot_valid = snapshot_valid and _sha(folder / "source_snapshot/pyproject.toml") == identity["project_sha256"]
    _record(checks, "exact_source_and_input_snapshots", snapshot_valid)
    frozen = pd.read_parquet(folder / "frozen_train_features.parquet")
    frozen_test = pd.read_parquet(folder / "frozen_test_features.parquet")
    _record(checks, "frozen_feature_schema_169_unique", len(FEATURES) == 169
            and frozen.loc[:, list(FEATURES)].columns.is_unique
            and frozen_test.loc[:, list(FEATURES)].columns.is_unique)
    cache_records, counts = _verify_cache(folder, runtime_identity, frozen, cfg, checks)
    _verify_assignments(folder, frozen, cfg, checks)
    _verify_fit_coverage(folder, frozen, cfg, cache_records, checks, counts)
    outer, events = _verify_outer_aliases(folder, frozen, cfg, checks)
    _verify_outer_readouts(folder, outer, events, cfg, checks)
    selection = _json(folder / "selection.json")
    frozen_test = frozen_test.loc[:, list(dict.fromkeys((*KEY, *FEATURES)))]
    _verify_final_models(folder, runtime_identity, frozen_test, selection, checks)
    decision = _json(folder / "outer_decision_before_test.json")
    info = _json(folder / "run_info.json")
    test_report = _json(folder / "test_evaluation.json")
    flags_valid = (
        selection.get("test_labels_read_for_selection") is False
        and info.get("test_reporting_only") is True
        and test_report.get("selection_used_test_labels") is False
        and test_report.get("outer_gate_changed_by_test") is False
        and test_report.get("reporting_only") is True
        and decision == _json(folder / "decision.json")
        and decision == info.get("decision")
        and selection.get("source_identity") == manifest.get("source_identity") == info.get("source_identity")
        and selection.get("config") == identity.get("config")
        and not any(Path(name).name.startswith("RUL_") for name in identity["inputs_sha256"])
    )
    _record(checks, "test_reporting_only_and_gate_frozen_before_labels", flags_valid)
    _record(checks, "completed_full_run", info.get("status") == "completed" and cfg.scientific)
    checks["all_checks_passed"] = all(checks.values())
    result = {
        "schema_version": 1,
        "status": "passed" if checks["all_checks_passed"] else "failed",
        "author": manifest.get("author"),
        "study_dir": folder.relative_to(root).as_posix(),
        "source_identity": manifest.get("source_identity"),
        "source_script_sha256": _sha(Path(__file__).resolve()),
        "checks": checks,
        "counts": counts,
        "cache_metadata_records": len(cache_records),
        "raw_test_labels_read": False,
        "model_fit_performed": False,
        "artifacts_modified": False,
        "evaluation_scope": (
            "read-only verification of disjoint engine CV, retrospective outer panel, "
            "paired bootstrap and reporting-only test flags"
        ),
    }
    verification_path = folder / "verification.json"
    if verification_path.exists():
        raise FileExistsError(f"verification artifact already exists: {verification_path}")
    verification_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint requiring an explicit completed output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = verify_tree_comparison(args.output_dir, repo_root=args.repo_root)
    # Keep the console payload ASCII-safe for Windows cp1252 terminals; the
    # artifact itself remains UTF-8 and preserves the author's name.
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
