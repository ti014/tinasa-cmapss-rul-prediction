"""Audit a frozen HGB blend without fitting or reading raw test labels.

The verifier independently rebuilds prediction-only weights, endpoint panels,
warning policies, inner choices and paired engine bootstrap intervals. Its only
publication is a new verification.json; --dry-probe performs no writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from importlib.metadata import version
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss.data.contracts import KEY
from cmapss.modeling.lightgbm_comparison.contracts import FEATURES
from cmapss.modeling.lightgbm_comparison.estimators import FittedTree

BASES = ("hgb_uncapped", "hgb_cap125")
ENGINE_METRICS = (
    "endpoint_mean_engine_mae", "endpoint_mean_engine_nasa",
    "trajectory_mean_engine_mae", "trajectory_mean_engine_nasa",
    "on_time_rate", "early_alarm_burden", "mean_transitions", "mean_retractions",
)


def _sha(path: Path) -> str:
    """Compute a streaming checksum without changing file bytes."""
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
    result = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"path escapes repository: {result}")
    return result


def _snapshot_mapping_valid(artifacts: dict[str, str], mapping: dict[str, str]) -> bool:
    """Bind short snapshot names to complete original paths, including folders."""
    expected = {name: hashlib.sha256(name.encode()).hexdigest()[:24] + Path(name).suffix
                for name in artifacts}
    return (mapping == expected and len(set(mapping.values())) == len(mapping)
            and all(Path(name).name == name for name in mapping.values()))


def _native(value: Any) -> Any:
    """Convert NumPy readouts to strict JSON values."""
    if isinstance(value, dict):
        return {str(name): _native(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    if hasattr(value, "item"):
        return _native(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _same_frame(actual: pd.DataFrame, expected: pd.DataFrame,
                keys: list[str] | tuple[str, ...], *, exact: bool = False) -> bool:
    """Compare keyed artifacts without assuming their physical row order."""
    if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
        return False
    columns = list(expected.columns)
    try:
        left = actual.loc[:, columns].sort_values(list(keys), kind="mergesort").reset_index(drop=True)
        right = expected.loc[:, columns].sort_values(list(keys), kind="mergesort").reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=exact,
                                      atol=0.0 if exact else 1e-8,
                                      rtol=0.0 if exact else 1e-10)
    except (AssertionError, KeyError, TypeError, ValueError):
        return False
    return True


def _gate(old: np.ndarray, cap: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Independently compute smoothstep using prediction arrays only."""
    before = np.asarray(old, dtype="float64")
    capped = np.asarray(cap, dtype="float64")
    if (before.ndim != 1 or before.size == 0 or before.shape != capped.shape
            or not np.isfinite(before).all() or not np.isfinite(capped).all()):
        raise ValueError("paired gate predictions must be finite vectors of equal length")
    lower, upper, maximum = (float(spec[name]) for name in
                              ("lower_rul", "upper_rul", "max_capped_weight"))
    if not (np.isfinite([lower, upper, maximum]).all()
            and 0.0 <= lower < upper and 0.0 < maximum <= 1.0):
        raise ValueError("invalid smooth gate specification")
    ratio = (upper - np.clip(before, lower, upper)) / (upper - lower)
    weights = maximum * (ratio * ratio * (3.0 - 2.0 * ratio))
    blended = (1.0 - weights) * before + weights * capped
    return blended, weights


def _score(frame: pd.DataFrame) -> pd.DataFrame:
    """Score raw RUL directly without invoking study metric functions."""
    result = frame.copy()
    errors = result.y_pred.to_numpy(dtype="float64") - result.y_true.to_numpy(dtype="float64")
    if not np.isfinite(errors).all() or (result.y_true < 0).any():
        raise ValueError("errors must be finite and truth must be raw nonnegative RUL")
    result["absolute_error"] = np.abs(errors)
    if "signed_error" in result:
        result["signed_error"] = errors
    result["squared_error"] = errors ** 2
    result["nasa_score_component"] = np.where(
        errors < 0.0, np.expm1(-errors / 13.0), np.expm1(errors / 10.0))
    return result


def _endpoints(frame: pd.DataFrame, fractions: list[float] | tuple[float, ...]) -> pd.DataFrame:
    """Rebuild retrospective lifetime positions without altering inference."""
    rows: list[dict[str, Any]] = []
    for (_, engine), group in frame.groupby(["split", "engine_id"], sort=True):
        lifetime = group.cycle.to_numpy() + group.y_true.to_numpy()
        if not np.isfinite(lifetime).all() or not np.all(lifetime == lifetime[0]):
            raise ValueError(f"inconsistent raw lifetime for engine {engine}")
        for fraction in fractions:
            requested = int(np.floor(float(fraction) * lifetime[0]))
            origin = max(int(group.cycle.min()), requested)
            chosen = group[group.cycle.eq(origin)]
            if len(chosen) != 1:
                raise ValueError("endpoint key is missing or duplicated")
            row = chosen.iloc[0].to_dict()
            row.update(endpoint_fraction=float(fraction), requested_cycle=requested,
                       origin_clamped_to_available_prefix=origin != requested)
            rows.append(row)
    return pd.DataFrame(rows)


def _metrics(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    scored = _score(frame)
    engines = scored.groupby("engine_id", sort=True)
    return {
        f"{prefix}_n_rows": len(scored),
        f"{prefix}_n_engines": scored.engine_id.nunique(),
        f"{prefix}_mean_window_mae": float(scored.absolute_error.mean()),
        f"{prefix}_mean_engine_mae": float(engines.absolute_error.mean().mean()),
        f"{prefix}_rmse": float(np.sqrt(scored.squared_error.mean())),
        f"{prefix}_nasa_score": float(scored.nasa_score_component.sum()),
        f"{prefix}_mean_engine_nasa": float(engines.nasa_score_component.mean().mean()),
    }


def _model_metrics(frame: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    return {**_metrics(_endpoints(frame, cfg["endpoint_fractions"]), "endpoint"),
            **_metrics(frame, "trajectory")}


def _warning_events(frame: pd.DataFrame, policy: dict[str, int], lead_requirement: int) -> pd.DataFrame:
    """Derive persistent current/past alarms on contiguous complete runs."""
    threshold, persistence = int(policy["threshold"]), int(policy["persistence"])
    rows: list[dict[str, Any]] = []
    for engine, block in frame.groupby("engine_id", sort=True):
        group = block.sort_values("cycle", kind="mergesort")
        cycles, truth = group.cycle.to_numpy(dtype="int64"), group.y_true.to_numpy(dtype="float64")
        if (not group.split.eq("train").all() or not np.all(np.diff(cycles) == 1)
                or not np.array_equal(truth, cycles[-1] - cycles)):
            raise ValueError("warning evaluation requires contiguous complete train trajectories")
        below = group.y_pred.to_numpy() <= threshold
        alarm = np.zeros(len(group), dtype=bool)
        if persistence <= len(group):
            alarm[persistence - 1:] = np.lib.stride_tricks.sliding_window_view(
                below, persistence).all(axis=1)
        hits = np.flatnonzero(alarm)
        lead = float(truth[hits[0]]) if len(hits) else np.nan
        eligible = truth > lead_requirement
        transitions = np.diff(np.r_[False, alarm].astype("int64"))
        rows.append({
            "engine_id": int(engine), "threshold": threshold, "persistence": persistence,
            "first_warning_cycle": int(cycles[hits[0]]) if len(hits) else -1,
            "lead_time_cycles": lead, "missed_warning": not bool(len(hits)),
            "on_time": bool(len(hits) and lead >= lead_requirement),
            "early_alarm_burden": float(alarm[eligible].mean()) if eligible.any() else np.nan,
            "n_pre_danger_rows": int(eligible.sum()), "n_alarm_rows": int(alarm.sum()),
            "n_transitions": int(np.count_nonzero(transitions)),
            "n_retractions": int((transitions == -1).sum()),
            "n_alarm_episodes": int((transitions == 1).sum()), "n_rows": len(group),
            "lead_requirement": lead_requirement,
        })
    return pd.DataFrame(rows)


def _warnings(events: pd.DataFrame) -> dict[str, Any]:
    leads = events.lead_time_cycles.dropna()
    return {
        "on_time_rate": float(events.on_time.mean()), "on_time_count": int(events.on_time.sum()),
        "late_or_missed_rate": float((~events.on_time).mean()),
        "missed_warning_count": int(events.missed_warning.sum()),
        "early_alarm_burden": float(events.early_alarm_burden.mean()),
        "mean_transitions": float(events.n_transitions.mean()),
        "mean_retractions": float(events.n_retractions.mean()),
        "engines_with_retractions": int(events.n_retractions.gt(0).sum()),
        "median_lead_time": float(leads.median()) if len(leads) else None,
        "min_lead_time": float(leads.min()) if len(leads) else None,
        "max_lead_time": float(leads.max()) if len(leads) else None, "n_engines": len(events),
    }


def _policy(frame: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, int], pd.DataFrame]:
    rows = [{"threshold": int(threshold), "persistence": int(persistence),
             **_warnings(_warning_events(frame, {"threshold": threshold, "persistence": persistence},
                                         int(cfg["lead_requirement"])))}
            for threshold in cfg["thresholds"] for persistence in cfg["persistences"]]
    grid = pd.DataFrame(rows)
    valid = grid.dropna(subset=["on_time_rate", "early_alarm_burden"])
    feasible = valid[valid.on_time_rate.ge(cfg["target_on_time_rate"])]
    if feasible.empty:
        winner = valid.sort_values(["on_time_rate", "early_alarm_burden", "threshold", "persistence"],
                                  ascending=[False, True, True, True], kind="mergesort").iloc[0]
    else:
        winner = feasible.sort_values(["early_alarm_burden", "threshold", "persistence"],
                                     kind="mergesort").iloc[0]
    policy = {"threshold": int(winner.threshold), "persistence": int(winner.persistence)}
    grid["selected"] = grid.threshold.eq(policy["threshold"]) & grid.persistence.eq(policy["persistence"])
    grid["inner_target_met"] = grid.on_time_rate.ge(cfg["target_on_time_rate"])
    return policy, grid


def _engine_table(frame: pd.DataFrame, events: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    endpoint = _score(_endpoints(frame, cfg["endpoint_fractions"]))
    trajectory = _score(frame)
    table = pd.DataFrame({
        "endpoint_mean_engine_mae": endpoint.groupby("engine_id").absolute_error.mean(),
        "endpoint_mean_engine_nasa": endpoint.groupby("engine_id").nasa_score_component.mean(),
        "trajectory_mean_engine_mae": trajectory.groupby("engine_id").absolute_error.mean(),
        "trajectory_mean_engine_nasa": trajectory.groupby("engine_id").nasa_score_component.mean(),
        "low_rul_engine_mae": trajectory[trajectory.y_true.le(30)].groupby("engine_id").absolute_error.mean(),
        "trajectory_sse": trajectory.groupby("engine_id").squared_error.sum(),
        "trajectory_n_rows": trajectory.groupby("engine_id").squared_error.size(),
    }).sort_index()
    warning = events.set_index("engine_id").rename(columns={"on_time": "on_time_rate",
                                                           "n_transitions": "mean_transitions",
                                                           "n_retractions": "mean_retractions"})
    return table.join(warning.loc[:, ["on_time_rate", "early_alarm_burden", "mean_transitions", "mean_retractions"]])


def _paired(before: pd.DataFrame, after: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Preserve every sampled engine occurrence when bootstrapping pooled RMSE."""
    if not before.index.equals(after.index):
        raise ValueError("paired bootstrap requires identical ordered engine indices")
    size = len(before)
    indices = np.random.default_rng(cfg["seed"]).integers(0, size, size=(cfg["bootstrap_resamples"], size))
    rows = []
    for metric in ("trajectory_rmse", *ENGINE_METRICS):
        if metric == "trajectory_rmse":
            old_sse, new_sse = (table.trajectory_sse.to_numpy(dtype="float64") for table in (before, after))
            old_n, new_n = (table.trajectory_n_rows.to_numpy(dtype="float64") for table in (before, after))
            old, new = float(np.sqrt(old_sse.sum() / old_n.sum())), float(np.sqrt(new_sse.sum() / new_n.sum()))
            samples = (np.sqrt(new_sse[indices].sum(axis=1) / new_n[indices].sum(axis=1))
                       - np.sqrt(old_sse[indices].sum(axis=1) / old_n[indices].sum(axis=1)))
        else:
            old_values, new_values = (table[metric].to_numpy(dtype="float64") for table in (before, after))
            old, new = float(old_values.mean()), float(new_values.mean())
            samples = (new_values - old_values)[indices].mean(axis=1)
        rows.append({"metric": metric, "baseline": old, "candidate": new, "delta": new - old,
                     "ci_lower": float(np.quantile(samples, .025)), "ci_upper": float(np.quantile(samples, .975)),
                     "n_engines": size, "resamples": cfg["bootstrap_resamples"], "seed": cfg["seed"]})
    return pd.DataFrame(rows)


def _verify_provenance(root: Path, folder: Path, manifest: dict[str, Any],
                       checks: dict[str, bool]) -> Path:
    """Require every reused byte and current source/lock to match the freeze."""
    identity = manifest["identity"]
    source = _resolve(root, identity["source_run_dir"])
    old_manifest = _json(source / "manifest.json")
    old_identity = old_manifest["identity"]
    old_verification = _json(source / "verification.json")
    checks["verified_source_run_identity"] = (
        identity["source_run_identity"] == old_manifest["source_identity"]
        and old_manifest["source_identity"] == hashlib.sha256(json.dumps(old_identity, sort_keys=True).encode()).hexdigest()
        and old_verification.get("status") == "passed"
        and old_verification.get("source_identity") == old_manifest["source_identity"]
        and bool(old_verification.get("checks")) and all(value is True for value in old_verification["checks"].values())
        and old_verification.get("source_script_sha256") == _sha(root / "scripts/verify_lightgbm_study.py")
        and old_verification.get("raw_test_labels_read") is False and old_verification.get("model_fit_performed") is False)
    checks["new_source_identity_digest"] = manifest["source_identity"] == hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()
    actual_snapshot_paths = identity["source_artifact_snapshot_paths"]
    checks["flat_snapshot_mapping_binds_original_paths_without_collision"] = _snapshot_mapping_valid(
        identity["source_artifacts_sha256"], actual_snapshot_paths)
    valid = True
    for key, snapshot in (("source_sha256", "source_snapshot"),
                          ("source_artifacts_sha256", "source_artifact_snapshot")):
        for name, digest in identity[key].items():
            valid = valid and _sha(_resolve(root, name)) == digest
            saved_name = actual_snapshot_paths[name] if key == "source_artifacts_sha256" else name
            valid = valid and _sha(_resolve(folder / snapshot, saved_name)) == digest
    for name, digest in old_identity["source_sha256"].items():
        valid = valid and _sha(_resolve(root, name)) == digest
        valid = valid and _sha(_resolve(source / "source_snapshot", name)) == digest
    for key in ("inputs_sha256", "outer_fold_source_sha256"):
        for name, digest in old_identity[key].items():
            valid = valid and _sha(_resolve(root, name)) == digest
    for name, digest in old_identity["inputs_sha256"].items():
        valid = valid and _sha(_resolve(source / "input_snapshot", name)) == digest
        valid = valid and not Path(name).name.startswith("RUL_")
    for name, key in (("uv.lock", "dependency_lock_sha256"),
                      ("pyproject.toml", "project_sha256")):
        current = _sha(root / name)
        valid = valid and current == identity[key] == old_identity[key]
        valid = valid and _sha(folder / "source_snapshot" / name) == current
        valid = valid and _sha(source / "source_snapshot" / name) == current
    for name, digest in old_identity["frozen_features_sha256"].items():
        valid = valid and _sha(source / name) == digest
    valid = valid and _sha(source / "outer_assignments.csv") == old_identity["outer_assignments_sha256"]
    checks["current_guards_and_exact_source_artifact_snapshots"] = valid
    required = {source.relative_to(root).as_posix() + "/" + name for name in (
        "manifest.json", "verification.json", "frozen_train_features.parquet",
        "frozen_test_features.parquet", "final_predictions.parquet", "final_predictions_receipt.json",
        *(f"models/{arm}/{name}" for arm in BASES for name in ("model.joblib", "model.json")),
    )}
    checks["source_coverage_includes_base_models_features_and_predictions"] = required.issubset(
        identity["source_artifacts_sha256"])
    versions = {"python": platform.python_version(), **{name: version(name) for name in (
        "numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow", "joblib", "scipy", "threadpoolctl")}}
    checks["exact_numeric_and_serialization_runtime_identity"] = identity["versions"] == old_identity["versions"] == versions
    return source


def _verify_inventory(root: Path, folder: Path, source: Path, frozen: pd.DataFrame,
                      checks: dict[str, bool]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Verify 46 disjoint CV tasks plus two whole-training frozen models."""
    inventory = _json(folder / "reuse_inventory.json")
    records = inventory["cv_records"]
    counts = inventory["counts"]
    expected = {"outer_inner": 30, "outer": 10, "final_inner": 6,
                "final_models": 2, "reused_fits": 48, "fresh_fits": 0}
    checks["declared_reused_48_and_fresh_zero"] = all(counts.get(name) == value for name, value in expected.items())
    valid = len(records) == 46
    used: set[Path] = set()
    raw = frozen.loc[:, [*KEY, "rul"]].rename(columns={"rul": "y_true"})
    identity = _json(source / "manifest.json")["source_identity"]
    source_manifest_identity = _json(source / "manifest.json")["identity"]
    source_frame = frozen.loc[:, list(dict.fromkeys((*KEY, *FEATURES, "rul")))]
    digest = hashlib.sha256()
    digest.update(repr(tuple(source_frame.columns)).encode())
    digest.update(repr(tuple(str(dtype) for dtype in source_frame.dtypes)).encode())
    digest.update(pd.util.hash_pandas_object(source_frame, index=False).to_numpy().tobytes())
    frame_digest = digest.hexdigest()
    declared = {spec["candidate_id"]: spec for spec in source_manifest_identity["candidates"] if spec["candidate_id"] in BASES}
    for record in records:
        path, metadata_path = (_resolve(root, record[name]) for name in ("prediction_path", "metadata_path"))
        metadata = _json(metadata_path)
        pred = pd.read_parquet(path)
        fit_ids, held_ids = (set(map(int, record[name])) for name in ("fit_engine_ids", "validation_engine_ids"))
        valid = valid and path not in used and path.is_relative_to(source / "fit_cache")
        used.add(path)
        valid = valid and metadata["source_identity"] == identity
        valid = valid and metadata["source_hash"] == frame_digest and metadata["config"] == source_manifest_identity["config"]
        valid = valid and metadata["prediction_sha256"] == _sha(path)
        valid = valid and metadata["candidate"]["candidate_id"] == record["candidate_id"] in BASES
        valid = valid and metadata["candidate"] == declared[record["candidate_id"]]
        valid = valid and set(metadata["fit_engine_ids"]) == fit_ids
        valid = valid and set(metadata["validation_engine_ids"]) == held_ids
        valid = valid and bool(fit_ids) and bool(held_ids) and not fit_ids.intersection(held_ids)
        valid = valid and metadata["feature_columns"] == list(FEATURES) and metadata["n_features"] == 169
        fitting = frozen[frozen.engine_id.isin(fit_ids)]
        weights, effective = metadata["weight_metadata"], metadata["effective_parameters"]
        valid = valid and weights["scheme"] == "engine_total_one" and weights["n_engines"] == len(fit_ids)
        valid = valid and weights["n_rows"] == len(fitting) and np.isclose(weights["total_weight"], len(fit_ids))
        valid = valid and effective["model_kind"] == "hgb" and effective["max_iter"] == 200
        valid = valid and effective["target_cap"] == declared[record["candidate_id"]]["target_cap"]
        cache_payload = {"source_identity": identity, "source_hash": frame_digest, "spec": metadata["candidate"],
                         "config": source_manifest_identity["config"], "fit_engine_ids": sorted(fit_ids),
                         "validation_engine_ids": sorted(held_ids)}
        cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()
        valid = valid and path.stem == metadata["cache_key"] == cache_key and metadata["status"] == "completed"
        valid = valid and _same_frame(pred.loc[:, [*KEY, "y_true"]], raw[raw.engine_id.isin(held_ids)], KEY, exact=True)
        valid = valid and not pred.duplicated(list(KEY)).any()
        scored = _score(pred)
        valid = valid and _same_frame(pred, scored, KEY)
    checks["reused_cache_prediction_truth_integrity"] = valid
    outer = pd.read_csv(folder / "outer_assignments.csv")
    all_ids = set(map(int, frozen.engine_id.unique()))
    final_records = inventory["final_model_records"]
    final_coverage = len(final_records) == 2 and {item["candidate_id"] for item in final_records} == set(BASES)
    for record in final_records:
        metadata = _json(_resolve(root, record["metadata_path"]))
        final_coverage = final_coverage and set(record["fit_engine_ids"]) == set(metadata["fit_engine_ids"]) == all_ids
        final_coverage = final_coverage and metadata["fit_rows"] == len(frozen) and metadata["source_identity"] == identity
        final_coverage = final_coverage and metadata["model_sha256"] == _sha(_resolve(root, record["model_path"]))
    checks["two_original_final_models_cover_all_training_engines"] = final_coverage
    valid = (not outer.engine_id.duplicated().any() and set(outer.engine_id) == all_ids
             and set(outer.outer_fold) == {1, 2, 3, 4, 5})
    expected_tasks: set[tuple[str, str, int, str, tuple[int, ...], tuple[int, ...]]] = set()
    for fold in range(6):
        stage = f"outer_{fold}" if fold else "final_inner"
        assignment = pd.read_csv(folder / "stages" / stage / "inner_assignments.csv")
        held = set(map(int, outer.loc[outer.outer_fold.eq(fold), "engine_id"])) if fold else set()
        fitting = all_ids - held
        valid = valid and set(assignment.engine_id) == fitting and not assignment.engine_id.duplicated().any()
        valid = valid and set(assignment.inner_fold) == {1, 2, 3}
        for inner_fold in range(1, 4):
            validation = set(map(int, assignment.loc[assignment.inner_fold.eq(inner_fold), "engine_id"]))
            for base in BASES:
                expected_tasks.add((stage, "inner", inner_fold, base, tuple(sorted(fitting - validation)), tuple(sorted(validation))))
        if fold:
            for base in BASES:
                expected_tasks.add(("outer", "outer", fold, base, tuple(sorted(fitting)), tuple(sorted(held))))
    actual_tasks = {(record["stage"], record["task_kind"], int(record["fold"]), record["candidate_id"],
                     tuple(sorted(record["fit_engine_ids"])), tuple(sorted(record["validation_engine_ids"])))
                    for record in records}
    checks["complete_46_unique_cv_task_coverage_and_engine_separation"] = valid and actual_tasks == expected_tasks and len(actual_tasks) == 46
    return records, counts


def _blend_frame(old: pd.DataFrame, cap: pd.DataFrame, spec: dict[str, Any],
                  candidate_id: str) -> pd.DataFrame:
    """Join by exact observation keys and retain audit-visible components."""
    left = old.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
    right = cap.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
    preserved = [*KEY, *(name for name in ("y_true", "inner_fold", "outer_fold") if name in left)]
    if not _same_frame(left.loc[:, preserved], right.loc[:, preserved], KEY, exact=True):
        raise ValueError("gate component keys, truth or folds do not align exactly")
    result = left.loc[:, preserved].copy()
    result["uncapped_prediction"] = left.y_pred.to_numpy(dtype="float64")
    result["cap125_prediction"] = right.y_pred.to_numpy(dtype="float64")
    result["y_pred"], result["capped_weight"] = _gate(
        result.uncapped_prediction.to_numpy(), result.cap125_prediction.to_numpy(), spec)
    result["candidate_id"] = candidate_id
    return _score(result) if "y_true" in result else result


def _inner_controls(row: pd.Series, baseline: pd.Series) -> dict[str, bool]:
    """Reproduce zero-margin safeguards from inner metrics only."""
    endpoints = bool(row.endpoint_mean_engine_mae <= baseline.endpoint_mean_engine_mae
                     and row.endpoint_mean_engine_nasa <= baseline.endpoint_mean_engine_nasa)
    warnings = bool(row.on_time_rate >= baseline.on_time_rate
                    and row.early_alarm_burden <= baseline.early_alarm_burden)
    return {"numeric_controls_met": endpoints, "warning_controls_met": warnings,
            "eligible": endpoints and warnings}


def _winner(records: pd.DataFrame, baseline: pd.Series) -> tuple[str, str, dict[str, dict[str, bool]]]:
    controls = {str(row.candidate_id): _inner_controls(row, baseline) for _, row in records.iterrows()}
    feasible = records.loc[records.candidate_id.map(lambda name: controls[str(name)]["eligible"])]
    pool = feasible if not feasible.empty else records
    winner = pool.sort_values(["trajectory_rmse", "endpoint_mean_engine_mae", "candidate_id"],
                             kind="mergesort").iloc[0]
    diagnostic = str(winner.candidate_id)
    selected = ("gated_blend" if not feasible.empty and winner.trajectory_rmse < baseline.trajectory_rmse
                else "hgb_uncapped")
    return diagnostic, selected, controls


def _gate_controls(paired: pd.DataFrame) -> dict[str, bool]:
    table = paired.set_index("metric")
    return {
        "trajectory_rmse_improved": bool(table.loc["trajectory_rmse", "ci_upper"] < 0.0),
        "endpoint_mae_not_worse": bool(table.loc["endpoint_mean_engine_mae", "ci_upper"] <= 0.0),
        "endpoint_nasa_not_worse": bool(table.loc["endpoint_mean_engine_nasa", "ci_upper"] <= 0.0),
        "warning_on_time_not_worse": bool(table.loc["on_time_rate", "ci_lower"] >= 0.0),
        "warning_burden_not_worse": bool(table.loc["early_alarm_burden", "ci_upper"] <= 0.0),
    }


def _candidate_specs(identity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in identity["candidates"]:
        spec = {name: raw[name] for name in ("lower_rul", "upper_rul", "max_capped_weight")}
        candidate = (f"gate_lo{float(spec['lower_rul']):.15g}_hi{float(spec['upper_rul']):.15g}"
                     f"_w{float(spec['max_capped_weight']):.15g}")
        if "candidate_id" in raw and raw["candidate_id"] != candidate:
            raise ValueError("gate candidate identity does not match its parameter triple")
        if candidate in result:
            raise ValueError("duplicate declared gate")
        result[candidate] = spec
    return result


def _scientific_protocol(cfg: dict[str, Any]) -> bool:
    fixed = {
        "outer_splits": 5, "inner_splits": 3, "seed": 42, "fit_threads": 4, "hgb_max_iter": 200,
        "endpoint_fractions": [.5, .7, .9], "thresholds": [30, 35, 40, 45, 50], "persistences": [1, 3, 5],
        "lead_requirement": 30, "target_on_time_rate": .9, "endpoint_mae_margin": 0., "nasa_score_margin": 0.,
        "warning_rate_margin": 0., "early_alarm_burden_margin": 0.,
        "reference_policy_source": "source_stage_inner_calibration", "lower_ruls": [60., 90.],
        "upper_ruls": [125., 150.], "max_capped_weights": [.25, .5, .75, 1.], "smoke": False,
    }
    return all(cfg.get(name) == value for name, value in fixed.items()) and cfg.get("bootstrap_resamples", 0) >= 10_000


def _cached_inner(root: Path, records: list[dict[str, Any]], stage: str,
                   candidate: str) -> pd.DataFrame:
    pieces = []
    for record in records:
        if record["stage"] == stage and record["task_kind"] == "inner" and record["candidate_id"] == candidate:
            pieces.append(pd.read_parquet(_resolve(root, record["prediction_path"])).assign(inner_fold=int(record["fold"])))
    if len(pieces) != 3:
        raise ValueError("inner derivation requires all three original base cache tasks")
    return pd.concat(pieces, ignore_index=True).sort_values(list(KEY), kind="mergesort").reset_index(drop=True)


def _numeric_row_matches(actual: dict[str, Any] | pd.Series, expected: dict[str, Any]) -> bool:
    """Compare independently recomputed metrics, allowing floating algebra."""
    for name, value in expected.items():
        if name not in actual:
            return False
        observed = actual[name]
        if value is None:
            if observed is not None and not pd.isna(observed):
                return False
        elif isinstance(value, (float, int, np.number)) and not isinstance(value, bool):
            if not np.isclose(observed, value, rtol=1e-10, atol=1e-8, equal_nan=True):
                return False
        elif observed != value:
            return False
    return True


def _verify_inner(root: Path, folder: Path, source: Path, manifest: dict[str, Any],
                   records: list[dict[str, Any]], checks: dict[str, bool]) -> None:
    """Recover every gate/policy/choice solely from fitting-engine inner OOF."""
    cfg = manifest["identity"]["config"]
    gates = _candidate_specs(manifest["identity"])
    valid_metrics = valid_policies = valid_choices = valid_source_references = True
    for fold in range(6):
        stage_name = f"outer_{fold}" if fold else "final_inner"
        stage = folder / "stages" / stage_name
        stored = _json(stage / "selection.json")
        old_selection = _json(source / "stages" / stage_name / "selection.json")
        bases = {arm: _cached_inner(root, records, stage_name, arm) for arm in BASES}
        predictions = {**bases, **{candidate: _blend_frame(bases[BASES[0]], bases[BASES[1]], spec, candidate)
                                  for candidate, spec in gates.items()}}
        metrics: dict[str, dict[str, Any]] = {}
        policies: dict[str, dict[str, int]] = {}
        grids: dict[str, pd.DataFrame] = {}
        for candidate, frame in predictions.items():
            policy, grid = _policy(frame, cfg)
            if candidate in BASES:
                original_policy = old_selection["arms"][candidate]["policy"]
                valid_source_references = valid_source_references and policy == original_policy
                policy = original_policy
            policies[candidate], grids[candidate] = policy, grid
            metrics[candidate] = {"candidate_id": candidate, **_model_metrics(frame, cfg),
                                  **_warnings(_warning_events(frame, policy, int(cfg["lead_requirement"])))}
        baseline = pd.Series(metrics["hgb_uncapped"])
        gate_rows = pd.DataFrame([metrics[candidate] for candidate in gates])
        diagnostic, selected, controls = _winner(gate_rows, baseline)
        stored_metrics = pd.read_csv(stage / "inner_candidate_metrics.csv", float_precision="round_trip")
        valid_metrics = valid_metrics and not stored_metrics.candidate_id.duplicated().any()
        ids = set(stored_metrics.candidate_id)
        valid_metrics = valid_metrics and ids in (set(gates), set(predictions))
        for _, row in stored_metrics.iterrows():
            expected = metrics.get(str(row.candidate_id))
            valid_metrics = valid_metrics and expected is not None
            if expected:
                valid_metrics = valid_metrics and _numeric_row_matches(row, expected)
        expected_arm_candidates = {"hgb_uncapped": "hgb_uncapped", "hgb_cap125": "hgb_cap125",
                                   "gated_blend": diagnostic}
        arm_rows = pd.read_csv(stage / "inner_arm_metrics.csv", float_precision="round_trip")
        valid_metrics = valid_metrics and set(arm_rows.procedure) == set(expected_arm_candidates)
        for arm, candidate in expected_arm_candidates.items():
            choice = stored["arms"][arm]
            valid_choices = valid_choices and choice.get("candidate_id") == candidate
            valid_policies = valid_policies and choice.get("policy") == policies[candidate]
            rows = arm_rows[arm_rows.procedure.eq(arm)]
            valid_metrics = valid_metrics and len(rows) == 1
            if len(rows) == 1:
                valid_metrics = valid_metrics and _numeric_row_matches(rows.iloc[0], metrics[candidate])
            valid_metrics = valid_metrics and _numeric_row_matches(choice["inner_metrics"], metrics[candidate])
            if arm in BASES:
                valid_source_references = valid_source_references and _numeric_row_matches(
                    old_selection["arms"][arm]["inner_metrics"], metrics[candidate])
            else:
                valid_choices = valid_choices and choice["spec"] == gates[candidate]
                valid_choices = valid_choices and choice.get("numeric_controls_met") == any(
                    entry["eligible"] for entry in controls.values())
        grid = pd.read_csv(stage / "inner_policy_metrics.csv", float_precision="round_trip")
        policy_ids = set(grid.candidate_id)
        valid_policies = valid_policies and policy_ids in (set(gates), set(predictions))
        for candidate in policy_ids:
            actual = grid[grid.candidate_id.eq(candidate)]
            expected = grids[candidate]
            valid_policies = valid_policies and len(actual) == len(expected)
            for _, row in actual.iterrows():
                match = expected[expected.threshold.eq(row.threshold) & expected.persistence.eq(row.persistence)]
                valid_policies = valid_policies and len(match) == 1
                if len(match) == 1:
                    valid_policies = valid_policies and _numeric_row_matches(row, match.iloc[0].to_dict())
        valid_choices = valid_choices and stored["procedure"]["selected_procedure_id"] == selected
        valid_choices = valid_choices and stored.get("test_labels_read") is False
        valid_choices = valid_choices and stored["procedure"].get("fallback_used") == (selected == "hgb_uncapped")
        if "gate_selection" in stored:
            selection = stored["gate_selection"]
            feasible = sum(item["eligible"] for item in controls.values())
            improving = sum(controls[str(row.candidate_id)]["eligible"] and row.trajectory_rmse < baseline.trajectory_rmse
                            for _, row in gate_rows.iterrows())
            valid_choices = valid_choices and selection.get("diagnostic_candidate_id") == diagnostic
            valid_choices = valid_choices and selection.get("selected_candidate_id") == (
                diagnostic if selected == "gated_blend" else None)
            valid_choices = valid_choices and selection.get("n_feasible") == feasible
            valid_choices = valid_choices and selection.get("n_improving") == improving
            valid_choices = valid_choices and selection.get("n_evaluated") == len(gates)
    checks["inner_candidate_and_arm_metrics_independently_recomputed"] = valid_metrics
    checks["inner_gate_policy_grids_and_selected_policies_recomputed"] = valid_policies
    checks["inner_only_gate_feasibility_choice_and_fallback_recomputed"] = valid_choices
    checks["inner_hgb_reference_metrics_and_policies_match_original"] = valid_source_references


def _verify_outer(root: Path, folder: Path, source: Path, manifest: dict[str, Any],
                   records: list[dict[str, Any]], checks: dict[str, bool]) -> None:
    """Derive outer vectors from frozen inner choices then score independently."""
    cfg = manifest["identity"]["config"]
    outer = pd.read_parquet(folder / "outer_predictions.parquet")
    events = pd.read_parquet(folder / "warning_events.parquet")
    choices = json.loads((folder / "outer_selections.json").read_text(encoding="utf-8"))
    procedures = {*BASES, "gated_blend", "selected_pipeline"}
    valid_vectors = set(outer.procedure) == set(events.procedure) == procedures and len(choices) == 5
    valid_vectors = valid_vectors and not outer.duplicated(["procedure", *KEY]).any()
    valid_vectors = valid_vectors and not events.duplicated(["procedure", "engine_id"]).any()
    expected_events = []
    for case in choices:
        fold = int(case["outer_fold"])
        stage = _json(folder / "stages" / f"outer_{fold}" / "selection.json")
        valid_vectors = valid_vectors and case["arms"] == stage["arms"] and case["procedure"] == stage["procedure"]
        components = {}
        for base in BASES:
            matches = [record for record in records if record["task_kind"] == "outer" and record["fold"] == fold
                       and record["candidate_id"] == base]
            if len(matches) != 1:
                raise ValueError("outer derivation requires exactly one frozen cache per base and fold")
            components[base] = pd.read_parquet(_resolve(root, matches[0]["prediction_path"])).assign(outer_fold=fold)
        candidate = stage["arms"]["gated_blend"]["candidate_id"]
        gate = stage["arms"]["gated_blend"]["spec"]
        derived = _blend_frame(components[BASES[0]], components[BASES[1]], gate, candidate)
        expected = {**components, "gated_blend": derived}
        selected = stage["procedure"]["selected_procedure_id"]
        expected["selected_pipeline"] = expected[selected]
        for procedure, frame in expected.items():
            actual = outer[outer.outer_fold.eq(fold) & outer.procedure.eq(procedure)]
            columns = [*KEY, "y_true", "y_pred", "candidate_id"]
            if procedure == "gated_blend" or (procedure == "selected_pipeline" and selected == "gated_blend"):
                columns.extend(["uncapped_prediction", "cap125_prediction", "capped_weight"])
            valid_vectors = valid_vectors and _same_frame(actual.loc[:, columns], frame.loc[:, columns], KEY, exact=True)
            policy_arm = selected if procedure == "selected_pipeline" else procedure
            policy = stage["arms"][policy_arm]["policy"]
            valid_vectors = valid_vectors and actual.threshold.eq(policy["threshold"]).all()
            valid_vectors = valid_vectors and actual.persistence.eq(policy["persistence"]).all()
            if procedure == "selected_pipeline":
                valid_vectors = valid_vectors and actual.selected_arm.eq(selected).all()
            recalculated = _warning_events(actual, policy, int(cfg["lead_requirement"]))
            recalculated = recalculated.assign(procedure=procedure, outer_fold=fold,
                                               candidate_id=actual.candidate_id.iloc[0])
            if "selected_arm" in actual:
                recalculated["selected_arm"] = actual.selected_arm.iloc[0]
            expected_events.append(recalculated)
    checks["outer_predictions_exact_gate_weights_and_inner_choice_aliases"] = valid_vectors
    recomputed_events = pd.concat(expected_events, ignore_index=True)
    checks["outer_warning_events_independently_recomputed"] = _same_frame(
        events, recomputed_events, ["procedure", "outer_fold", "engine_id"])
    endpoints = pd.read_parquet(folder / "outer_endpoints.parquet")
    expected_endpoint = pd.concat([_endpoints(frame, cfg["endpoint_fractions"])
                                   for _, frame in outer.groupby("procedure", sort=True)], ignore_index=True)
    checks["outer_retrospective_endpoint_panel_recomputed"] = _same_frame(
        endpoints, expected_endpoint, ["procedure", "engine_id", "endpoint_fraction"])
    aggregate = pd.DataFrame([{"procedure": name, **_model_metrics(frame, cfg),
                               **_warnings(events[events.procedure.eq(name)])}
                              for name, frame in outer.groupby("procedure", sort=True)])
    stored_aggregate = pd.read_csv(folder / "aggregate_metrics.csv", float_precision="round_trip")
    checks["outer_aggregate_metrics_independently_recomputed"] = _same_frame(stored_aggregate, aggregate, ["procedure"])
    original = pd.read_csv(source / "aggregate_metrics.csv", float_precision="round_trip")
    checks["hgb_reference_outer_aggregate_matches_original"] = _same_frame(
        stored_aggregate[stored_aggregate.procedure.isin(BASES)], original[original.procedure.isin(BASES)],
        ["procedure"], exact=True)
    original_pred = pd.read_parquet(source / "outer_predictions.parquet")
    original_events = pd.read_parquet(source / "warning_events.parquet")
    reference_columns = [*KEY, "outer_fold", "candidate_id", "y_true", "y_pred", "threshold", "persistence", "procedure"]
    checks["hgb_reference_outer_predictions_match_original_exactly"] = _same_frame(
        outer[outer.procedure.isin(BASES)].loc[:, reference_columns],
        original_pred[original_pred.procedure.isin(BASES)].loc[:, reference_columns],
        ["procedure", *KEY], exact=True)
    event_columns = [name for name in original_events.columns if name != "selected_arm"]
    checks["hgb_reference_outer_warning_events_match_original_exactly"] = _same_frame(
        events[events.procedure.isin(BASES)].loc[:, event_columns],
        original_events[original_events.procedure.isin(BASES)].loc[:, event_columns],
        ["procedure", "engine_id"], exact=True)
    tables = {name: _engine_table(frame, events[events.procedure.eq(name)], cfg)
              for name, frame in outer.groupby("procedure", sort=True)}
    stored_engines = pd.read_csv(folder / "engine_metrics.csv", float_precision="round_trip")
    expected_engines = pd.concat([table.assign(procedure=name).reset_index() for name, table in tables.items()], ignore_index=True)
    checks["paired_engine_sufficient_statistics_recomputed"] = _same_frame(
        stored_engines, expected_engines, ["procedure", "engine_id"])
    expected_pairs = pd.concat([_paired(tables[reference], tables[candidate], cfg).assign(reference=reference, procedure=candidate)
                                for reference in BASES for candidate in ("gated_blend", "selected_pipeline")], ignore_index=True)
    stored_pairs = pd.read_csv(folder / "paired_comparison.csv", float_precision="round_trip")
    checks["paired_engine_bootstrap_pooled_rmse_and_multiplicity_recomputed"] = _same_frame(
        stored_pairs.loc[:, expected_pairs.columns], expected_pairs, ["reference", "procedure", "metric"])
    selected = expected_pairs[expected_pairs.reference.eq("hgb_uncapped") & expected_pairs.procedure.eq("selected_pipeline")]
    decision = _json(folder / "outer_decision_before_test.json")
    scientific = _scientific_protocol(cfg)
    controls = _gate_controls(selected)
    checks["outer_gate_independently_recomputed_without_test"] = (
        decision["controls"] == controls and decision["full_protocol_configuration"] == scientific
        and decision["promotion_gate_met"] == bool(scientific and all(controls.values()))
        and decision == _json(folder / "decision.json"))


def _verify_final(folder: Path, source: Path, manifest: dict[str, Any],
                   frozen_test: pd.DataFrame, checks: dict[str, bool]) -> None:
    """Replay two saved models and smooth gate before checking stored reports."""
    from threadpoolctl import threadpool_limits

    selection = _json(folder / "selection.json")
    inner_selection = _json(folder / "stages/final_inner/selection.json")
    inference = _json(folder / "inference.json")
    receipt = _json(folder / "final_predictions_receipt.json")
    info = _json(folder / "run_info.json")
    final = pd.read_parquet(folder / "final_predictions.parquet")
    original_final = pd.read_parquet(source / "final_predictions.parquet")
    valid = (set(final.procedure) == {*BASES, "gated_blend", "selected_pipeline"}
             and not {"y_true", "rul"}.intersection(final.columns)
             and not {"y_true", "rul"}.intersection(frozen_test.columns)
             and not final.duplicated(["procedure", *KEY]).any())
    final_test = frozen_test.sort_values(list(KEY), kind="mergesort").groupby("engine_id", sort=True, as_index=False).tail(1).reset_index(drop=True)
    predicted = {}
    for base, cap in (("hgb_uncapped", None), ("hgb_cap125", 125)):
        model_path, metadata_path = folder / "models" / base / "model.joblib", folder / "models" / base / "model.json"
        metadata, model = _json(metadata_path), joblib.load(model_path)
        valid = valid and isinstance(model, FittedTree) and tuple(model.feature_columns) == FEATURES
        valid = valid and model.spec.candidate_id == base and model.spec.target_cap == cap and model.spec.model_kind == "hgb"
        valid = valid and metadata["source_identity"] == manifest["identity"]["source_run_identity"]
        valid = valid and metadata["model_sha256"] == _sha(model_path)
        valid = valid and metadata["weight_metadata"]["scheme"] == "engine_total_one"
        valid = valid and metadata["effective_parameters"]["max_iter"] == 200
        valid = valid and _sha(model_path) == _sha(source / "models" / base / "model.joblib")
        valid = valid and _sha(metadata_path) == _sha(source / "models" / base / "model.json")
        with threadpool_limits(limits=4):
            predicted[base] = np.asarray(model.model.predict(final_test.loc[:, list(FEATURES)].to_numpy(dtype="float64")), dtype="float64")
        valid = valid and np.array_equal(predicted[base], model.predict(final_test))
        actual = final[final.procedure.eq(base)].sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
        original = original_final[original_final.procedure.eq(base)].sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
        valid = valid and _same_frame(actual.loc[:, [*KEY, "y_pred"]], original.loc[:, [*KEY, "y_pred"]], KEY, exact=True)
        valid = valid and _same_frame(actual.loc[:, list(KEY)], final_test.loc[:, list(KEY)], KEY, exact=True)
        valid = valid and np.array_equal(actual.y_pred.to_numpy(), predicted[base])
    gate = selection["arms"]["gated_blend"]
    predicted["gated_blend"], _ = _gate(predicted[BASES[0]], predicted[BASES[1]], gate["spec"])
    selected = selection["procedure"]["selected_procedure_id"]
    predicted["selected_pipeline"] = predicted[selected]
    for arm in ("gated_blend", "selected_pipeline"):
        actual = final[final.procedure.eq(arm)].sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
        valid = valid and _same_frame(actual.loc[:, list(KEY)], final_test.loc[:, list(KEY)], KEY, exact=True)
        valid = valid and np.array_equal(actual.y_pred.to_numpy(dtype="float64"), predicted[arm])
        expected_candidate = gate["candidate_id"] if arm == "gated_blend" or selected == "gated_blend" else "hgb_uncapped"
        valid = valid and actual.candidate_id.eq(expected_candidate).all()
        if arm == "selected_pipeline":
            valid = valid and actual.selected_arm.eq(selected).all()
        if arm == "gated_blend" or selected == "gated_blend":
            valid = valid and np.array_equal(actual.uncapped_prediction.to_numpy(), predicted[BASES[0]])
            valid = valid and np.array_equal(actual.cap125_prediction.to_numpy(), predicted[BASES[1]])
    checks["two_saved_original_base_models_and_blended_predictions_replay_exact"] = valid
    checks["final_inner_selection_gate_policy_and_inference_frozen"] = (
        selection["arms"] == inner_selection["arms"] and selection["procedure"] == inner_selection["procedure"]
        and inference["gate_spec"] == gate["spec"] and inference["gate_candidate_id"] == gate["candidate_id"]
        and inference["selected_procedure_id"] == selected
        and inference["policies"] == {name: value["policy"] for name, value in selection["arms"].items()}
        and inference["feature_columns"] == list(FEATURES)
        and inference["selection_sha256"] == _sha(folder / "selection.json")
        and inference["test_labels_read"] is False and inference["fresh_fits"] == 0)
    valid_receipt = (receipt["source_identity"] == manifest["source_identity"] == inference["source_identity"]
                     == selection["source_identity"] == info["source_identity"])
    for filename, field in (("final_predictions.parquet", "predictions_sha256"), ("selection.json", "selection_sha256"),
                             ("inference.json", "inference_sha256"), ("outer_decision_before_test.json", "outer_decision_sha256")):
        valid_receipt = valid_receipt and receipt[field] == _sha(folder / filename)
    valid_receipt = valid_receipt and all(receipt["models_sha256"][arm] == _sha(folder / "models" / arm / "model.joblib")
                                       and receipt["models_metadata_sha256"][arm] == _sha(folder / "models" / arm / "model.json")
                                       for arm in BASES)
    valid_receipt = valid_receipt and all(receipt.get(name) is True for name in (
        "saved_model_replay_exact", "original_base_prediction_replay_exact", "all_final_procedures_replayed_exact"))
    valid_receipt = valid_receipt and receipt["test_labels_read"] is False and receipt["fresh_fits"] == 0
    checks["final_unlabelled_model_inference_selection_prediction_receipt_hashes"] = valid_receipt
    report_path = folder / "test_evaluation.json"
    flags = (selection["test_labels_read_for_selection"] is False and info["test_reporting_only"] is True
             and selection["outer_decision"] == info["decision"] == _json(folder / "outer_decision_before_test.json"))
    if report_path.exists():
        report = _json(report_path)
        flags = flags and report["selection_used_test_labels"] is False and report["outer_gate_changed_by_test"] is False
        flags = flags and report["reporting_only"] is True
        flags = flags and datetime.fromisoformat(receipt["frozen_at_utc"]) < datetime.fromisoformat(report["labels_read_at_utc"])
        evaluation = pd.read_parquet(folder / "final_evaluation.parquet")
        test_metrics = pd.read_csv(folder / "test_metrics.csv", float_precision="round_trip")
        report_valid = set(evaluation.procedure) == set(final.procedure)
        report_valid = report_valid and not evaluation.duplicated(["procedure", *KEY]).any()
        reference_truth = evaluation[evaluation.procedure.eq("hgb_uncapped")].loc[:, [*KEY, "y_true"]]
        for arm, frame in evaluation.groupby("procedure", sort=True):
            predicted_frame = final[final.procedure.eq(arm)]
            report_valid = report_valid and _same_frame(frame.loc[:, [*KEY, "y_pred"]], predicted_frame.loc[:, [*KEY, "y_pred"]], KEY, exact=True)
            report_valid = report_valid and _same_frame(frame.loc[:, [*KEY, "y_true"]], reference_truth, KEY, exact=True)
            independently = _metrics(frame, "test")
            independently = {name.removeprefix("test_"): value for name, value in independently.items()
                             if name != "test_mean_engine_nasa"}
            observed = test_metrics[test_metrics.procedure.eq(arm)]
            report_valid = report_valid and len(observed) == 1 and _numeric_row_matches(observed.iloc[0], independently)
        checks["stored_test_reporting_metrics_recomputed_without_raw_label_file"] = report_valid
    else:
        flags = flags and info["test_evaluated"] is False and manifest["identity"]["evaluate_test"] is False
    for fold in range(1, 6):
        stage = folder / "stages" / f"outer_{fold}"
        read_receipt = _json(stage / "outer_read_receipt.json")
        choice = _json(stage / "selection.json")
        flags = flags and read_receipt["selection_sha256"] == _sha(stage / "selection.json")
        flags = flags and read_receipt["outer_predictions_read_after_selection"] is True
        flags = flags and choice["outer_prediction_values_read_before_selection"] is False
        flags = flags and datetime.fromisoformat(choice["frozen_at_utc"]) < datetime.fromisoformat(read_receipt["read_completed_at_utc"])
    checks["inner_before_outer_and_all_final_artifacts_before_reporting_labels"] = flags


def verify(output_dir: str | Path, *, root: str | Path = ROOT,
           strict_full: bool = True, write: bool = True) -> dict[str, Any]:
    """Verify a completed derived study; optionally write one fresh receipt."""
    root = Path(root).resolve()
    folder = _resolve(root, output_dir)
    namespace = root / "outputs/experiments/gated_blend_comparison"
    if folder == namespace or not folder.is_relative_to(namespace):
        raise ValueError("run must be an isolated gated_blend_comparison descendant")
    destination = folder / "verification.json"
    if write and destination.exists():
        raise FileExistsError(f"verification artifact already exists: {destination}")
    manifest = _json(folder / "manifest.json")
    checks: dict[str, bool] = {}
    source = _verify_provenance(root, folder, manifest, checks)
    frozen = pd.read_parquet(source / "frozen_train_features.parquet")
    test = pd.read_parquet(source / "frozen_test_features.parquet")
    checks["frozen_169_causal_features_unique_and_test_label_free"] = (
        len(FEATURES) == 169 and len(set(FEATURES)) == 169 and not {"y_true", "rul"}.intersection(test.columns)
        and not frozen.duplicated(list(KEY)).any() and not test.duplicated(list(KEY)).any()
        and np.isfinite(frozen.loc[:, list(FEATURES)]).all().all()
        and np.isfinite(test.loc[:, list(FEATURES)]).all().all())
    records, counts = _verify_inventory(root, folder, source, frozen, checks)
    _verify_inner(root, folder, source, manifest, records, checks)
    _verify_outer(root, folder, source, manifest, records, checks)
    _verify_final(folder, source, manifest, test, checks)
    cfg = manifest["identity"]["config"]
    full = _scientific_protocol(cfg) and len(_candidate_specs(manifest["identity"])) == 16
    checks["completed_requested_protocol_scope"] = _json(folder / "run_info.json")["status"] == "completed" and (full or not strict_full)
    checks["all_checks_passed"] = all(checks.values())
    result = {"schema_version": 1, "status": "passed" if checks["all_checks_passed"] else "failed",
              "author": manifest["author"], "study_dir": folder.relative_to(root).as_posix(),
              "source_identity": manifest["source_identity"], "source_run_identity": manifest["identity"]["source_run_identity"],
              "source_script_sha256": _sha(Path(__file__).resolve()), "checks": checks, "counts": counts,
              "scientific_protocol": full, "strict_full_requested": strict_full,
              "raw_test_labels_read": False, "model_fit_performed": False, "artifacts_modified": False,
              "numeric_comparison": {"gate_weights_prediction_alignment_and_model_replay": "exact equality",
                                     "independent_metrics_and_bootstrap_atol": 1e-8, "independent_metrics_and_bootstrap_rtol": 1e-10},
              "evaluation_scope": "independent derived-gate verification on historically inspected engine folds; fixed-prediction conditional uncertainty"}
    result = _native(result)
    if write:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    """Run full verification or a read-only engineering probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", "--output-dir", required=True, type=Path, dest="output_dir")
    parser.add_argument("--dry-probe", action="store_true", help="do not publish verification.json")
    parser.add_argument("--allow-smoke", action="store_true", help="verify engineering scope without claiming full protocol")
    args = parser.parse_args(argv)
    result = verify(args.output_dir, root=args.repo_root, strict_full=not args.allow_smoke, write=not args.dry_probe)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
