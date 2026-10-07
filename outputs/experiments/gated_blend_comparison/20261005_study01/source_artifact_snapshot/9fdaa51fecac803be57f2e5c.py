"""Isolated, nested FD001 study following the 2026-10-05 audit protocol.

The published model and historical experiments are immutable inputs. Model
selection uses retrospective 50/70/90-percent lifetime endpoints, not the full
trajectory score. Lifetime positions evaluation origins only: it is never an
input feature. Warning rules are selected on inner predictions and evaluated
on engines excluded from both model and policy selection. FD001 test labels
have already been examined historically; test results are reporting only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from cmapss.data.contracts import KEY, SENSORS
from cmapss.data.readers import read_test_labels
from cmapss.data.targets import validate_train_targets
from cmapss.features.contracts import ROLLING_FEATURE_VALUE_COLUMNS
from cmapss.features.rolling import build_rolling_features
from cmapss.modeling.metrics import (
    add_prediction_errors,
    bootstrap_metric_intervals,
    summarize_prediction_errors,
)

BASE_FEATURES = tuple(ROLLING_FEATURE_VALUE_COLUMNS)
HISTORY_FEATURES = ("cycle",) + tuple(
    f"{sensor}__from_initial_{stat}" for sensor in SENSORS for stat in ("last", "mean")
)
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
    "src/cmapss/modeling/audit_study.py",
)
BASELINE_FILES = (
    "outputs/modeling/models/selected_model.joblib",
    "outputs/modeling/predictions/oof_predictions.parquet",
    "outputs/modeling/predictions/final_evaluation.parquet",
    "outputs/modeling/tables/fold_assignments.csv",
    "outputs/modeling/tables/final_metrics.json",
)


@dataclass(frozen=True)
class CandidateSpec:
    """Describe one predeclared model ablation.

    Attributes:
        candidate_id: Stable name, including the declared interaction row.
        model_kind: HistGradientBoosting or weighted, standardized Ridge.
        history: Include causal age and first30-baseline-relative features.
        prune: Remove only features constant in this fitting fold.
        target_cap: Optional upper bound applied to fitting labels only.
        loss: HGB objective; ignored for Ridge.
        alpha: Ridge regularization under mean-one sample weights.
    """

    candidate_id: str
    model_kind: str = "hist_gradient_boosting"
    history: bool = False
    prune: bool = False
    target_cap: float | None = None
    loss: str = "squared_error"
    alpha: float = 1.0


def make_candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return the fixed six HGB rows and four fair Ridge rows.

    Returns:
        Immutable candidate specifications; cap+history is an interaction,
        not evidence attributable to either single change.
    """
    return (
        CandidateSpec("hgb_baseline"),
        CandidateSpec("hgb_pruned", prune=True),
        CandidateSpec("hgb_cap125", target_cap=125.0),
        CandidateSpec("hgb_history", history=True),
        CandidateSpec("hgb_absolute_error", loss="absolute_error"),
        CandidateSpec("hgb_cap125_history", history=True, target_cap=125.0),
        *(CandidateSpec(f"ridge_alpha_{alpha:g}", model_kind="ridge", prune=True, alpha=alpha)
          for alpha in (0.1, 1.0, 10.0, 100.0)),
    )


@dataclass(frozen=True)
class AuditStudyConfig:
    """Freeze split, feature, endpoint and illustrative warning decisions.

    Attributes:
        outer_splits: Outer engine folds, five in the scientific study.
        inner_splits: Inner engine folds, three in the scientific study.
        endpoint_fractions: Retrospective evaluation-origin fractions.
        candidate_ids: Predeclared candidates included in this run.
        thresholds: Predicted-RUL thresholds in the warning grid.
        persistences: Consecutive confirmations in the warning grid.
        lead_requirement: Illustrative benchmark lead, not a maintenance SLA.
        target_on_time_rate: Inner illustrative feasibility target.
        bootstrap_resamples: Paired engine resamples conditional on predictions.
        seed: Fixed deterministic model and bootstrap seed.
        fit_threads: Limit native fitting threads.
        hgb_max_iter: Two hundred scientifically; reduced only for dry-run.
    """

    outer_splits: int = 5
    inner_splits: int = 3
    endpoint_fractions: tuple[float, ...] = (0.5, 0.7, 0.9)
    candidate_ids: tuple[str, ...] = tuple(spec.candidate_id for spec in make_candidate_specs())
    thresholds: tuple[int, ...] = (30, 35, 40, 45, 50)
    persistences: tuple[int, ...] = (1, 3, 5)
    lead_requirement: int = 30
    target_on_time_rate: float = 0.9
    bootstrap_resamples: int = 10_000
    seed: int = 42
    fit_threads: int = 4
    hgb_max_iter: int = 200

    def __post_init__(self) -> None:
        """Reject invalid configurations before any run state exists."""
        if min(self.outer_splits, self.inner_splits) < 2:
            raise ValueError("at least two inner and outer folds are required")
        if min(self.bootstrap_resamples, self.fit_threads, self.hgb_max_iter) < 1:
            raise ValueError("resamples, threads and iterations must be positive")
        if not self.endpoint_fractions or tuple(sorted(set(self.endpoint_fractions))) != self.endpoint_fractions:
            raise ValueError("endpoint fractions must be unique and increasing")
        if any(not 0 < fraction < 1 for fraction in self.endpoint_fractions):
            raise ValueError("endpoint fractions must be strictly inside (0, 1)")
        available = {spec.candidate_id for spec in make_candidate_specs()}
        if (not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids)
                or not set(self.candidate_ids).issubset(available)
                or "hgb_baseline" not in self.candidate_ids):
            raise ValueError("candidate IDs must be unique, known and include hgb_baseline")
        if not 0 < self.target_on_time_rate <= 1 or self.lead_requirement < 1:
            raise ValueError("warning target must be in (0, 1] and lead positive")
        if (not self.thresholds or not self.persistences
                or any(value < 0 for value in self.thresholds)
                or any(value < 1 for value in self.persistences)):
            raise ValueError("warning grids must be nonempty and valid")
        if (tuple(sorted(set(self.thresholds))) != self.thresholds
                or tuple(sorted(set(self.persistences))) != self.persistences):
            raise ValueError("warning grids must be unique and increasing")
        for threshold in self.thresholds:
            for persistence in self.persistences:
                WarningPolicy(threshold, persistence)


@dataclass(frozen=True)
class WarningPolicy:
    """Store one threshold and consecutive-confirmation policy.

    Attributes:
        threshold: Predicted-RUL cutoff in cycles.
        persistence: Number of consecutive predictions at/below the cutoff.
    """

    threshold: int
    persistence: int

    def __post_init__(self) -> None:
        """Validate an integer, nonnegative cutoff and positive persistence."""
        if (isinstance(self.threshold, bool) or isinstance(self.persistence, bool)
                or not isinstance(self.threshold, (int, np.integer))
                or not isinstance(self.persistence, (int, np.integer))
                or self.threshold < 0 or self.persistence < 1):
            raise ValueError("invalid warning policy")


@dataclass
class FittedCandidate:
    """Keep all learned feature and scaling state needed for inference.

    Attributes:
        spec: Frozen candidate identity.
        feature_columns: Columns retained using fitting rows only.
        scaler: Weighted Ridge scaler, or None for unscaled HGB.
        model: Fitted sklearn regressor.
    """

    spec: CandidateSpec
    feature_columns: tuple[str, ...]
    scaler: StandardScaler | None
    model: Any

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict without fitting or consulting truth/validation statistics.

        Args:
            features: Causal features containing the retained column names.

        Returns:
            Finite predictions on the same raw-RUL reporting scale.

        Raises:
            ValueError: Inputs or predictions are nonfinite.
        """
        values = features.loc[:, list(self.feature_columns)]
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise ValueError("nonfinite inference features")
        transformed = self.scaler.transform(values) if self.scaler is not None else values
        prediction = np.asarray(self.model.predict(transformed), dtype="float64")
        if not np.isfinite(prediction).all():
            raise ValueError("nonfinite model predictions")
        return prediction


def build_study_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Build trailing30 features and causal first30-relative history features.

    No learned statistics cross engines or splits. For every retained cycle
    (>=30), the initial baseline is already observed. The full lifetime and
    labels are not read or included. Raw features remain available unchanged.

    Args:
        observations: Canonical full observations starting at cycle1.

    Returns:
        Keys, 126 original features, and 42 first30-relative features. Cycle
        becomes a model feature only for the declared history candidates.

    Raises:
        ValueError: A trajectory lacks exactly the first30 baseline samples.
    """
    features = build_rolling_features(observations, window_cycles=30)
    initial = observations.loc[observations.cycle.le(30)]
    counts = initial.groupby(["split", "engine_id"]).size()
    if not counts.eq(30).all() or len(counts) != observations.groupby(["split", "engine_id"]).ngroups:
        raise ValueError("causal history requires exactly cycles1..30 per engine/split")
    means = initial.groupby(["split", "engine_id"], sort=True)[list(SENSORS)].mean()
    joined = features.merge(means.add_suffix("__initial_mean"),
                            on=["split", "engine_id"], validate="many_to_one")
    history = {
        f"{sensor}__from_initial_{stat}":
            joined[f"{sensor}__{stat}"] - joined[f"{sensor}__initial_mean"]
        for sensor in SENSORS for stat in ("last", "mean")
    }
    return pd.concat([features.reset_index(drop=True), pd.DataFrame(history)], axis=1)


def engine_weights(frame: pd.DataFrame, *, mean_one: bool = False) -> np.ndarray:
    """Return engine-equal weights with an explicit global scale.

    Args:
        frame: Fitting rows containing engine_id.
        mean_one: Scale to mean1 for Ridge; otherwise engine total1 for HGB.

    Returns:
        Positive row weights with identical total weight per fitting engine.

    Raises:
        ValueError: Fitting rows are empty or engine IDs are missing.
    """
    if frame.empty or frame.engine_id.isna().any():
        raise ValueError("engine weights require nonempty, identified rows")
    counts = frame.groupby("engine_id").engine_id.transform("size").to_numpy(dtype="float64")
    weights = 1. / counts
    return weights / weights.mean() if mean_one else weights


def _make_model(spec: CandidateSpec, config: AuditStudyConfig) -> Any:
    """Construct a model using only declared candidate parameters."""
    if spec.model_kind == "ridge":
        return Ridge(alpha=spec.alpha, solver="svd")
    return HistGradientBoostingRegressor(
        loss=spec.loss, learning_rate=0.05, max_iter=config.hgb_max_iter,
        max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0,
        early_stopping=False, random_state=config.seed,
    )


def fit_candidate(train: pd.DataFrame, spec: CandidateSpec,
                  config: AuditStudyConfig) -> FittedCandidate:
    """Fit pruning, scaler and regressor strictly within fitting engines.

    Args:
        train: Train-only feature rows with raw rul labels.
        spec: Predeclared ablation.
        config: Frozen resource/model configuration.

    Returns:
        Predictor retaining its learned preprocessing state.

    Raises:
        ValueError: Rows are not train-only or no nonconstant columns remain.
    """
    if train.empty or not train["split"].eq("train").all() or "rul" not in train:
        raise ValueError("fitting requires train-only rows with raw rul")
    columns = BASE_FEATURES + (HISTORY_FEATURES if spec.history else ())
    if spec.prune:
        unique = train.loc[:, list(columns)].nunique(dropna=False)
        columns = tuple(column for column in columns if unique[column] > 1)
    if not columns:
        raise ValueError("no nonconstant fitting features remain")
    values = train.loc[:, list(columns)]
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ValueError("nonfinite fitting features")
    raw_y = train.rul.to_numpy(dtype="float64")
    if not np.isfinite(raw_y).all() or (raw_y < 0).any():
        raise ValueError("fitting labels must be finite raw RUL >=0")
    target = np.minimum(raw_y, spec.target_cap) if spec.target_cap is not None else raw_y
    weights = engine_weights(train, mean_one=spec.model_kind == "ridge")
    scaler = StandardScaler() if spec.model_kind == "ridge" else None
    model = _make_model(spec, config)
    with threadpool_limits(limits=config.fit_threads):
        transformed = scaler.fit_transform(values, sample_weight=weights) if scaler is not None else values
        model.fit(transformed, target, sample_weight=weights)
    return FittedCandidate(spec, columns, scaler, model)


def build_endpoint_panel(predictions: pd.DataFrame,
                         fractions: tuple[float, ...]) -> pd.DataFrame:
    """Position a fixed retrospective panel without changing model inputs.

    Args:
        predictions: Complete train trajectory predictions with raw y_true.
        fractions: Declared ordered lifetime fractions, identical for models.

    Returns:
        One row per engine/fraction, with requested and actual origins visible.

    Raises:
        ValueError: Lifetime is inconsistent or an endpoint key is absent.
    """
    if predictions.empty:
        raise ValueError("empty endpoint prediction input")
    records = []
    for (_, engine), group in predictions.groupby(["split", "engine_id"], sort=True):
        lifetimes = group.cycle.to_numpy(dtype="float64") + group.y_true.to_numpy(dtype="float64")
        if not np.isfinite(lifetimes).all() or not np.all(lifetimes == lifetimes[0]):
            raise ValueError(f"inconsistent lifetime for engine{engine}")
        lifetime = int(lifetimes[0])
        for fraction in fractions:
            requested = int(np.floor(fraction * lifetime))
            origin = max(int(group.cycle.min()), requested)
            row = group.loc[group.cycle.eq(origin)]
            if len(row) != 1:
                raise ValueError(f"missing or duplicate endpoint engine{engine}/cycle{origin}")
            record = row.iloc[0].to_dict()
            record.update(endpoint_fraction=fraction, requested_cycle=requested,
                          origin_clamped_to_available_prefix=origin != requested)
            records.append(record)
    return pd.DataFrame(records)


def _prediction_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    """Score equal-engine MAE plus separately named row RMSE/NASA totals."""
    metrics = summarize_prediction_errors(frame)
    scored = add_prediction_errors(frame)
    mean_nasa = scored.groupby("engine_id").nasa_score_component.mean().mean()
    return {**{f"{prefix}_{key}": value for key, value in metrics.items()},
            f"{prefix}_mean_engine_nasa": float(mean_nasa)}


def _model_readout(predictions: pd.DataFrame, config: AuditStudyConfig) -> dict[str, Any]:
    """Keep endpoint and full-trajectory estimands separate."""
    endpoint = build_endpoint_panel(predictions, config.endpoint_fractions)
    return {**_prediction_metrics(endpoint, "endpoint"),
            **_prediction_metrics(predictions, "trajectory")}


def select_candidate(records: pd.DataFrame) -> str:
    """Choose the declared inner endpoint-MAE winner and deterministic ties.

    Args:
        records: Inner candidate summaries, never outer/test metrics.

    Returns:
        Stable candidate ID minimizing endpoint MAE, then NASA/engine, then ID.

    Raises:
        ValueError: Selection metrics are empty or nonfinite.
    """
    columns = ["endpoint_mean_engine_mae", "endpoint_mean_engine_nasa"]
    if records.empty or not np.isfinite(records[columns].to_numpy()).all():
        raise ValueError("invalid inner model-selection metrics")
    return str(records.sort_values(columns + ["candidate_id"], kind="mergesort").iloc[0].candidate_id)


def _alarm_flags(group: pd.DataFrame, policy: WarningPolicy) -> np.ndarray:
    """Require current and preceding k-1 predictions to satisfy the cutoff."""
    below = group.y_pred.le(policy.threshold)
    return below.rolling(policy.persistence, min_periods=policy.persistence).sum().ge(policy.persistence).to_numpy()


def warning_events(predictions: pd.DataFrame, policy: WarningPolicy,
                   *, lead_requirement: int) -> pd.DataFrame:
    """Measure an illustrative policy's timeliness, burden and instability.

    Args:
        predictions: Complete ordered run-to-failure train predictions.
        policy: Threshold/persistence fixed before this engine is evaluated.
        lead_requirement: Illustrative minimum lead; not a real maintenance SLA.

    Returns:
        One row per engine. Early burden is time flagged when raw RUL exceeds
        the illustrative lead requirement; it is not a false-positive rate.

    Raises:
        ValueError: Data do not describe complete, contiguous train runs.
    """
    if predictions.empty or not predictions.split.eq("train").all():
        raise ValueError("warning evaluation requires complete train trajectories")
    rows = []
    for engine, unsorted in predictions.groupby("engine_id", sort=True):
        group = unsorted.sort_values("cycle", kind="mergesort")
        cycles = group.cycle.to_numpy(dtype="int64")
        actual = group.y_true.to_numpy(dtype="float64")
        if (len(np.unique(cycles)) != len(cycles) or np.any(np.diff(cycles) != 1)
                or not np.array_equal(actual, cycles[-1] - cycles)
                or not np.isfinite(group.y_pred.to_numpy()).all()):
            raise ValueError(f"warning trajectory engine{engine} is not complete/contiguous")
        alarm = _alarm_flags(group, policy)
        first = np.flatnonzero(alarm)
        lead = float(actual[first[0]]) if len(first) else np.nan
        eligible = actual > lead_requirement
        transitions = np.diff(np.r_[False, alarm].astype("int64"))
        rows.append({
            "engine_id": int(engine), "threshold": policy.threshold,
            "persistence": policy.persistence,
            "first_warning_cycle": int(cycles[first[0]]) if len(first) else -1,
            "lead_time_cycles": lead, "missed_warning": not bool(len(first)),
            "on_time": bool(len(first) and lead >= lead_requirement),
            "early_alarm_burden": float(alarm[eligible].mean()) if eligible.any() else np.nan,
            "n_pre_danger_rows": int(eligible.sum()), "n_alarm_rows": int(alarm.sum()),
            "n_transitions": int(np.count_nonzero(transitions)),
            "n_retractions": int(np.sum(transitions == -1)),
            "n_alarm_episodes": int(np.sum(transitions == 1)), "n_rows": int(len(group)),
            "lead_requirement": lead_requirement,
        })
    return pd.DataFrame(rows)


def _warning_readout(events: pd.DataFrame) -> dict[str, Any]:
    """Summarize engine-level policy observations without inventing negatives."""
    lead = events.lead_time_cycles.dropna()
    return {
        "on_time_rate": float(events.on_time.mean()),
        "on_time_count": int(events.on_time.sum()),
        "late_or_missed_rate": float((~events.on_time).mean()),
        "missed_warning_count": int(events.missed_warning.sum()),
        "early_alarm_burden": float(events.early_alarm_burden.mean()),
        "mean_transitions": float(events.n_transitions.mean()),
        "mean_retractions": float(events.n_retractions.mean()),
        "engines_with_retractions": int(events.n_retractions.gt(0).sum()),
        "median_lead_time": float(lead.median()) if len(lead) else None,
        "min_lead_time": float(lead.min()) if len(lead) else None,
        "max_lead_time": float(lead.max()) if len(lead) else None,
        "n_engines": int(len(events)),
    }


def choose_policy(records: pd.DataFrame, target_rate: float) -> WarningPolicy:
    """Choose feasible minimum burden, else maximum timeliness then burden.

    Args:
        records: Inner-grid summaries, not outer or test observations.
        target_rate: Declared inner illustrative on-time rate target.

    Returns:
        Policy with deterministic threshold/persistence tie breaks.

    Raises:
        ValueError: The policy grid has no finite burden/timeliness values.
    """
    valid = records.dropna(subset=["on_time_rate", "early_alarm_burden"])
    if valid.empty:
        raise ValueError("no evaluable warning policies")
    feasible = valid.loc[valid.on_time_rate.ge(target_rate)]
    if len(feasible):
        ordered = feasible.sort_values(["early_alarm_burden", "threshold", "persistence"])
    else:
        ordered = valid.sort_values(["on_time_rate", "early_alarm_burden", "threshold", "persistence"],
                                   ascending=[False, True, True, True])
    winner = ordered.iloc[0]
    return WarningPolicy(int(winner.threshold), int(winner.persistence))


def _policy_grid(predictions: pd.DataFrame, config: AuditStudyConfig) -> tuple[WarningPolicy, pd.DataFrame]:
    """Calibrate a declared grid exclusively on inner OOF predictions."""
    rows = []
    for threshold in config.thresholds:
        for persistence in config.persistences:
            policy = WarningPolicy(threshold, persistence)
            events = warning_events(predictions, policy, lead_requirement=config.lead_requirement)
            rows.append({**asdict(policy), **_warning_readout(events)})
    table = pd.DataFrame(rows)
    policy = choose_policy(table, config.target_on_time_rate)
    table["selected"] = table.threshold.eq(policy.threshold) & table.persistence.eq(policy.persistence)
    table["inner_target_met"] = table.on_time_rate.ge(config.target_on_time_rate)
    return policy, table


def _sha256(path: Path) -> str:
    """Hash exact bytes without changing the source artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value: Any) -> Any:
    """Use strict JSON primitives and explicit null for unavailable metrics."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    """Atomically publish strict UTF-8 JSON inside the isolated run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(_json_value(payload), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Atomically publish one prediction table in the isolated run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _relative(path: Path, repo_root: Path) -> str:
    """Require portable, repository-relative input and source identities."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("study inputs and output must be inside the repository") from error


def _initialize_run(output: Path, repo_root: Path, inputs: tuple[Path, ...],
                    config: AuditStudyConfig) -> tuple[dict[str, Any], str]:
    """Freeze current-used source/input bytes before any fit or test-label read."""
    allowed = repo_root / "outputs/experiments/audit_remediation"
    if output == allowed or allowed not in output.parents:
        raise ValueError("study output must be a new run under outputs/experiments/audit_remediation")
    sources = {name: _sha256(repo_root / name) for name in SOURCE_FILES}
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
            raise ValueError("resume mismatch: source, input, config, dependency or baseline changed")
        for name, digest in sources.items():
            if _sha256(output / "source_snapshot" / name) != digest:
                raise ValueError("source snapshot checksum mismatch")
        if _sha256(output / "source_snapshot/uv.lock") != identity["dependency_lock_sha256"]:
            raise ValueError("dependency snapshot checksum mismatch")
        manifest = saved
    else:
        if any(output.iterdir()):
            raise ValueError("run directory is nonempty without a manifest; choose a new output")
        for name in SOURCE_FILES + ("uv.lock",):
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


def _fit_predict(train: pd.DataFrame, validation: pd.DataFrame, spec: CandidateSpec,
                 config: AuditStudyConfig, cache_dir: Path,
                 source_identity: str) -> pd.DataFrame:
    """Fit/cache one disjoint-engine fold with exact data/source binding."""
    if set(train.engine_id) & set(validation.engine_id):
        raise ValueError("fitting and validation engines overlap")
    if not validation.split.eq("train").all():
        raise ValueError("CV validation must be train-only")
    identity = hashlib.sha256(source_identity.encode())
    identity.update(json.dumps(_json_value(asdict(spec)), sort_keys=True).encode())
    for frame in (train, validation):
        identity.update(str(tuple(frame.columns)).encode())
        identity.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    cache_key = identity.hexdigest()
    parquet = cache_dir / f"{cache_key}.parquet"
    metadata_path = cache_dir / f"{cache_key}.json"
    if parquet.exists() or metadata_path.exists():
        if not parquet.exists() or not metadata_path.exists():
            # Preserve an interrupted atomic-pair publication before retrying.
            orphan_dir = cache_dir / "interrupted_publications"
            orphan_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.time_ns()
            for orphan in (parquet, metadata_path):
                if orphan.exists():
                    orphan.replace(orphan_dir / f"{stamp}_{orphan.name}")
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (metadata["cache_key"] != cache_key
                    or metadata["source_identity"] != source_identity
                    or metadata["candidate"] != _json_value(asdict(spec))):
                raise ValueError("fit-cache metadata identity mismatch")
            if _sha256(parquet) != metadata["prediction_sha256"]:
                raise ValueError("fit-cache prediction checksum mismatch")
            cached = pd.read_parquet(parquet)
            pd.testing.assert_frame_equal(cached[list(KEY)].reset_index(drop=True),
                                          validation[list(KEY)].reset_index(drop=True))
            np.testing.assert_array_equal(cached.y_true, validation.rul)
            if not cached.candidate_id.eq(spec.candidate_id).all():
                raise ValueError("fit-cache candidate mismatch")
            return cached
    started = time.perf_counter()
    fitted = fit_candidate(train, spec, config)
    with threadpool_limits(limits=config.fit_threads):
        prediction = fitted.predict(validation)
    result = validation.loc[:, list(KEY)].copy().reset_index(drop=True)
    result["candidate_id"] = spec.candidate_id
    result["y_true"] = validation.rul.to_numpy(dtype="float64")
    result["y_pred"] = prediction
    result = add_prediction_errors(result)
    _write_parquet(result, parquet)
    _write_json(metadata_path, {
        "status": "completed", "cache_key": cache_key,
        "source_identity": source_identity, "candidate": asdict(spec),
        "prediction_sha256": _sha256(parquet), "seconds": time.perf_counter() - started,
        "fit_engine_ids": sorted(map(int, train.engine_id.unique())),
        "validation_engine_ids": sorted(map(int, validation.engine_id.unique())),
        "selected_feature_columns": list(fitted.feature_columns),
        "n_input_features": len(BASE_FEATURES) + (len(HISTORY_FEATURES) if spec.history else 0),
        "n_retained_features": len(fitted.feature_columns),
    })
    return result


def _inner_evaluation(frame: pd.DataFrame, config: AuditStudyConfig,
                      output: Path, identity: str,
                      outer_fold: int) -> tuple[str, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Evaluate every fixed candidate on identical inner engine folds."""
    splitter = GroupKFold(n_splits=config.inner_splits)
    folds = list(splitter.split(frame, frame.rul, frame.engine_id))
    assignment_rows = []
    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        train_ids = set(frame.iloc[fit_idx].engine_id)
        val_ids = set(frame.iloc[val_idx].engine_id)
        if train_ids & val_ids:
            raise ValueError("inner engine leakage")
        assignment_rows.extend({"outer_fold": outer_fold, "inner_fold": fold,
                                "engine_id": int(engine)} for engine in sorted(val_ids))
    rows, prediction_map = [], {}
    for spec in make_candidate_specs():
        if spec.candidate_id not in config.candidate_ids:
            continue
        print(f"outer{outer_fold}: candidate {spec.candidate_id}", flush=True)
        pieces = []
        for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
            prediction = _fit_predict(frame.iloc[fit_idx].reset_index(drop=True),
                                      frame.iloc[val_idx].reset_index(drop=True), spec,
                                      config, output / "fit_cache", identity)
            prediction = prediction.assign(inner_fold=fold)
            pieces.append(prediction)
        predictions = pd.concat(pieces, ignore_index=True).sort_values(["engine_id", "cycle"])
        prediction_map[spec.candidate_id] = predictions
        rows.append({"outer_fold": outer_fold, **asdict(spec), **_model_readout(predictions, config)})
    table = pd.DataFrame(rows)
    selected = select_candidate(table)
    table["selected"] = table.candidate_id.eq(selected)
    return selected, prediction_map, table, pd.DataFrame(assignment_rows)


def _outer_assignments(frame: pd.DataFrame, config: AuditStudyConfig,
                       repo_root: Path) -> pd.DataFrame:
    """Reuse published five folds, or deterministic GroupKFold for dry-run."""
    if config.outer_splits == 5:
        assignments = pd.read_csv(repo_root / "outputs/modeling/tables/fold_assignments.csv")
        assignments = assignments[["engine_id", "fold"]].rename(columns={"fold": "outer_fold"})
    else:
        splitter = GroupKFold(n_splits=config.outer_splits)
        rows = []
        for fold, (_, validation) in enumerate(splitter.split(frame, frame.rul, frame.engine_id), start=1):
            rows.extend({"engine_id": int(engine), "outer_fold": fold}
                        for engine in sorted(frame.iloc[validation].engine_id.unique()))
        assignments = pd.DataFrame(rows)
    if (assignments.engine_id.duplicated().any()
            or set(assignments.engine_id) != set(frame.engine_id)
            or set(assignments.outer_fold) != set(range(1, config.outer_splits + 1))):
        raise ValueError("outer assignments must cover all engines exactly once")
    return assignments.sort_values("engine_id").reset_index(drop=True)


def _engine_readout(predictions: pd.DataFrame, events: pd.DataFrame,
                     config: AuditStudyConfig) -> pd.DataFrame:
    """Build matched per-engine endpoint/trajectory/warning metrics."""
    endpoint = add_prediction_errors(build_endpoint_panel(predictions, config.endpoint_fractions))
    trajectory = add_prediction_errors(predictions)
    result = endpoint.groupby("engine_id").agg(
        endpoint_mean_engine_mae=("absolute_error", "mean"),
        endpoint_mean_engine_nasa=("nasa_score_component", "mean"),
    ).join(trajectory.groupby("engine_id").absolute_error.mean().rename("trajectory_mean_engine_mae"))
    result = result.join(trajectory[trajectory.y_true.le(30)].groupby("engine_id")
                         .absolute_error.mean().rename("low_rul_engine_mae"))
    warning = events.set_index("engine_id")[["on_time", "early_alarm_burden", "n_transitions", "n_retractions"]]
    warning = warning.rename(columns={"on_time": "on_time_rate", "n_transitions": "mean_transitions",
                                      "n_retractions": "mean_retractions"})
    return result.join(warning).astype("float64")


def _paired_comparison(baseline: pd.DataFrame, candidate: pd.DataFrame,
                       config: AuditStudyConfig) -> pd.DataFrame:
    """Resample paired engine differences while preserving multiplicity."""
    if not baseline.index.equals(candidate.index):
        raise ValueError("paired engine metrics need identical ordered IDs")
    rng = np.random.default_rng(config.seed)
    indices = rng.integers(0, len(baseline), size=(config.bootstrap_resamples, len(baseline)))
    rows = []
    for metric in baseline.columns:
        differences = (candidate[metric] - baseline[metric]).to_numpy(dtype="float64")
        valid = np.isfinite(differences)
        if not valid.any():
            rows.append({"metric": metric, "baseline": None, "candidate": None, "delta": None,
                         "ci_lower": None, "ci_upper": None, "n_engines": 0})
            continue
        with np.errstate(invalid="ignore"):
            sampled = differences[indices]
            counts = np.isfinite(sampled).sum(axis=1)
            draws = np.divide(np.nansum(sampled, axis=1), counts,
                              out=np.full(len(counts), np.nan), where=counts > 0)
        finite = draws[np.isfinite(draws)]
        rows.append({"metric": metric, "baseline": float(baseline[metric].mean()),
                     "candidate": float(candidate[metric].mean()),
                     "delta": float(differences[valid].mean()),
                     "ci_lower": float(np.quantile(finite, .025)),
                     "ci_upper": float(np.quantile(finite, .975)),
                     "n_engines": int(valid.sum()), "resamples": config.bootstrap_resamples,
                     "seed": config.seed,
                     "interval_scope": "paired engines conditional on stored outer predictions; excludes retraining variability"})
    return pd.DataFrame(rows)


def _write_comparisons(predictions: pd.DataFrame, events: pd.DataFrame,
                        config: AuditStudyConfig, output: Path) -> dict[str, Any]:
    """Publish all estimands, error slices, tradeoffs and the frozen gate."""
    summaries, engine_tables, region_rows = [], {}, []
    for procedure, frame in predictions.groupby("procedure", sort=True):
        policy_events = events[events.procedure.eq(procedure)]
        summaries.append({"procedure": procedure, **_model_readout(frame, config),
                          **_warning_readout(policy_events)})
        engine_tables[procedure] = _engine_readout(frame, policy_events, config)
        scored = add_prediction_errors(frame)
        scored["rul_region"] = pd.cut(scored.y_true, [-1, 30, 60, 100, 150, np.inf],
                                      labels=["0–30", "31–60", "61–100", "101–150", ">150"])
        for region, group in scored.groupby("rul_region", observed=False):
            if group.empty:
                continue
            region_rows.append({"procedure": procedure, "rul_region": str(region),
                                "n_rows": len(group), "n_engines": group.engine_id.nunique(),
                                "mean_window_mae": float(group.absolute_error.mean()),
                                "mean_engine_mae": float(group.groupby("engine_id").absolute_error.mean().mean()),
                                "signed_bias": float((group.y_pred - group.y_true).mean()),
                                "rmse": float(np.sqrt(group.squared_error.mean()))})
    pd.DataFrame(summaries).to_csv(output / "aggregate_metrics.csv", index=False)
    pd.concat([table.assign(procedure=name).reset_index() for name, table in engine_tables.items()],
              ignore_index=True).to_csv(output / "engine_metrics.csv", index=False)
    comparisons = []
    for reference, candidate in (("baseline_fixed", "baseline_calibrated"),
                                  ("baseline_fixed", "selected_calibrated"),
                                  ("baseline_calibrated", "selected_calibrated")):
        comparison = _paired_comparison(engine_tables[reference], engine_tables[candidate], config)
        comparisons.append(comparison.assign(reference=reference, procedure=candidate))
    table = pd.concat(comparisons, ignore_index=True)
    table.to_csv(output / "paired_comparison.csv", index=False)
    pd.DataFrame(region_rows).to_csv(output / "region_metrics.csv", index=False)
    gate = table[(table.reference == "baseline_fixed") & (table.procedure == "selected_calibrated")
                 & table.metric.eq("endpoint_mean_engine_mae")].iloc[0]
    standard = AuditStudyConfig()
    scientific = (config.outer_splits == 5 and config.inner_splits == 3
                  and config.hgb_max_iter == 200 and config.candidate_ids == standard.candidate_ids
                  and config.endpoint_fractions == standard.endpoint_fractions
                  and config.thresholds == standard.thresholds
                  and config.persistences == standard.persistences
                  and config.lead_requirement == standard.lead_requirement
                  and config.target_on_time_rate == standard.target_on_time_rate
                  and config.seed == standard.seed
                  and config.bootstrap_resamples >= 10000)
    decision = {"full_protocol_configuration": scientific,
                "outer_endpoint_mae_delta": float(gate.delta),
                "outer_endpoint_mae_ci_lower": float(gate.ci_lower),
                "outer_endpoint_mae_ci_upper": float(gate.ci_upper),
                "promotion_gate_met": bool(scientific and gate.ci_upper < 0),
                "historical_baseline_overwritten": False,
                "gate": "paired outer endpoint-MAE delta upper95% CI<0; no test gating",
                "tradeoff_note": "gate is only endpoint accuracy; inspect NASA, full trajectory and warning tradeoffs before describing the result",
                "evaluation_limit": "retrospective train endpoint panel; conditional engine bootstrap; not an untouched real-world test"}
    _write_json(output / "decision.json", decision)
    return decision


def _freeze_selection(path: Path, selection: dict[str, Any]) -> None:
    """Publish settings before test labels, rejecting changes on resume."""
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != _json_value(selection):
            raise ValueError("final selection changed on resume")
    else:
        _write_json(path, selection)


def _final_predictor(train: pd.DataFrame, spec: CandidateSpec,
                     config: AuditStudyConfig, output: Path,
                     identity: str) -> FittedCandidate:
    """Save/reload the all-train predictor with checksum-bound learned state."""
    model_path = output / "selected_model.joblib"
    metadata_path = output / "selected_model.json"
    if model_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata["source_identity"] != identity
                or metadata["candidate"] != _json_value(asdict(spec))
                or metadata["model_sha256"] != _sha256(model_path)):
            raise ValueError("final-model identity or checksum mismatch")
        fitted = joblib.load(model_path)
        if not isinstance(fitted, FittedCandidate) or fitted.spec != spec:
            raise ValueError("final-model object mismatch")
        return fitted
    if model_path.exists() or metadata_path.exists():
        interrupted = output / "interrupted_publications"
        interrupted.mkdir(exist_ok=True)
        for orphan in (model_path, metadata_path):
            if orphan.exists():
                orphan.replace(interrupted / f"{time.time_ns()}_{orphan.name}")
    started = time.perf_counter()
    fitted = fit_candidate(train, spec, config)
    temporary = model_path.with_suffix(".tmp.joblib")
    joblib.dump(fitted, temporary)
    temporary.replace(model_path)
    reloaded = joblib.load(model_path)
    with threadpool_limits(limits=config.fit_threads):
        replay_delta = float(np.max(np.abs(fitted.predict(train) - reloaded.predict(train))))
    if replay_delta != 0.0:
        raise ValueError("saved final model failed exact prediction replay")
    _write_json(metadata_path, {
        "source_identity": identity, "candidate": asdict(spec),
        "model_sha256": _sha256(model_path), "seconds": time.perf_counter() - started,
        "selected_feature_columns": list(fitted.feature_columns),
        "fit_engine_ids": sorted(map(int, train.engine_id.unique())),
        "fit_row_count": len(train), "saved_model_replay_max_abs_delta": replay_delta,
        "input_contract": "build_study_features canonical observations; never pass lifetime or RUL as model input",
    })
    return fitted


def _baseline_replay(predictions: pd.DataFrame, config: AuditStudyConfig,
                     repo_root: Path, output: Path) -> dict[str, Any]:
    """Check exact old HGB OOF keys/truth/predictions for the full protocol."""
    if config.outer_splits != 5 or config.hgb_max_iter != 200 or config.seed != 42:
        result = {"applicable": False, "reason": "reduced/custom dry-run configuration"}
    else:
        historical = pd.read_parquet(repo_root / BASELINE_FILES[1])
        historical = historical[historical.model_name.eq("hist_gradient_boosting")]
        current = predictions[predictions.procedure.eq("baseline_fixed")]
        columns = list(KEY)
        joined = historical[columns + ["y_true", "y_pred"]].merge(
            current[columns + ["y_true", "y_pred"]], on=columns,
            how="outer", validate="one_to_one", indicator=True,
            suffixes=("_old", "_new"),
        )
        keys_match = bool(joined._merge.eq("both").all())
        truth_match = bool(keys_match and np.array_equal(joined.y_true_old, joined.y_true_new))
        delta = float((joined.y_pred_old - joined.y_pred_new).abs().max()) if keys_match else None
        passed = bool(truth_match and delta is not None and delta <= 1e-8)
        result = {"applicable": True, "keys_match": keys_match, "raw_truth_match": truth_match,
                  "prediction_max_abs_delta": delta, "tolerance": 1e-8, "passed": passed,
                  "historical_model": "hist_gradient_boosting", "n_rows": len(joined)}
    _write_json(output / "baseline_replay.json", result)
    if result.get("applicable") and not result.get("passed"):
        raise ValueError("published baseline OOF replay failed; preserve outputs and inspect before test evaluation")
    return result


def _evaluate_test(predictions: pd.DataFrame, labels_path: Path,
                   config: AuditStudyConfig, output: Path,
                   repo_root: Path) -> dict[str, Any]:
    """Read historically inspected labels only after frozen settings/predictions."""
    if not (output / "selection.json").is_file() or not (output / "final_predictions.parquet").is_file():
        raise ValueError("test labels require saved selection and final predictions")
    labels = read_test_labels(labels_path)
    joined = predictions.merge(labels.rename(columns={"rul": "y_true"}),
                               on=["split", "engine_id"], how="outer",
                               validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("final-test prediction and label engines differ")
    evaluation = add_prediction_errors(joined.drop(columns="_merge")
                                       .rename(columns={"rul_pred": "y_pred"}))
    _write_parquet(evaluation, output / "final_evaluation.parquet")
    metrics = {**summarize_prediction_errors(evaluation),
               "bootstrap_intervals": bootstrap_metric_intervals(
                   evaluation, n_resamples=config.bootstrap_resamples, seed=config.seed),
               "test_labels_sha256": _sha256(labels_path),
               "test_history": "historically inspected FD001 test; reporting only, never used as selection or gate",
               "bootstrap_scope": "engine resampling conditional on fitted model; excludes retraining variability"}
    _write_json(output / "final_metrics.json", metrics)
    historical = pd.read_parquet(repo_root / BASELINE_FILES[2])
    columns = list(KEY)
    pair = historical[columns + ["y_true", "absolute_error", "nasa_score_component"]].merge(
        evaluation[columns + ["y_true", "absolute_error", "nasa_score_component"]],
        on=columns, how="outer", validate="one_to_one", indicator=True,
        suffixes=("_old", "_new"),
    )
    if not pair._merge.eq("both").all() or not np.array_equal(pair.y_true_old, pair.y_true_new):
        raise ValueError("test comparison baseline keys or raw truth differ")
    pair = pair.sort_values("engine_id").set_index("engine_id")
    baseline = pd.DataFrame({"test_endpoint_mae": pair.absolute_error_old,
                             "test_endpoint_nasa": pair.nasa_score_component_old})
    candidate = pd.DataFrame({"test_endpoint_mae": pair.absolute_error_new,
                              "test_endpoint_nasa": pair.nasa_score_component_new})
    _paired_comparison(baseline, candidate, config).assign(
        reporting_only=True, promotion_gate=False,
    ).to_csv(output / "paired_test_comparison.csv", index=False)
    return metrics


def run_audit_study(
    *,
    repo_root: str | Path | None = None,
    observations_path: str | Path = "outputs/etl/silver/observations.parquet",
    targets_path: str | Path = "outputs/etl/silver/train_targets.parquet",
    test_labels_path: str | Path = "data/raw/RUL_FD001.txt",
    output_dir: str | Path = "outputs/experiments/audit_remediation/20261005",
    config: AuditStudyConfig | None = None,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    """Execute the frozen nested study without mutating historical artifacts.

    Args:
        repo_root: Repository containing portable inputs and immutable baseline.
        observations_path: Canonical FD001 observations, separate from labels.
        targets_path: Complete raw train targets before window filtering.
        test_labels_path: Historical test labels, read only after frozen settings.
        output_dir: New isolated run descendant of audit_remediation.
        config: Frozen protocol; a reduced configuration cannot pass promotion.
        evaluate_test: Include reporting-only final-test metrics after prediction.

    Returns:
        Portable run summary, selection, gate and reporting-only test metrics.

    Raises:
        ValueError: Leakage, raw-target mismatch, cache/provenance mismatch, or
            baseline replay/invariance failure occurs.
    """
    started = time.perf_counter()
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[3]
    config = config or AuditStudyConfig()
    def resolve(value: str | Path) -> Path:
        """Resolve a repository-relative path without changing working directory."""
        path = Path(value)
        return (root / path).resolve() if not path.is_absolute() else path.resolve()
    observations_path = resolve(observations_path)
    targets_path = resolve(targets_path)
    labels_path = resolve(test_labels_path)
    output = resolve(output_dir)
    _relative(labels_path, root)  # Check portability without reading label bytes.
    manifest, identity = _initialize_run(output, root, (observations_path, targets_path), config)
    observations = pd.read_parquet(observations_path)
    targets = pd.read_parquet(targets_path)
    train_observations = observations[observations.split.eq("train")].copy()
    test_observations = observations[observations.split.eq("test")].copy()
    validate_train_targets(train_observations, targets)
    if set(observations.split) != {"train", "test"}:
        raise ValueError("study observations must have exactly train and test splits")
    features = build_study_features(observations)
    train = features[features.split.eq("train")].merge(
        targets, on=list(KEY), how="left", validate="one_to_one",
    ).sort_values(["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    if train.rul.isna().any():
        raise ValueError("window features lack raw train labels")
    test_features = features[features.split.eq("test")]
    final_test = test_features.sort_values(["engine_id", "cycle"]).groupby(
        "engine_id", sort=True, as_index=False).tail(1).reset_index(drop=True)
    if final_test.engine_id.nunique() != test_observations.engine_id.nunique():
        raise ValueError("test engine lacks the required30-cycle prefix")
    expected_train = len(train_observations) - 29 * train_observations.engine_id.nunique()
    if len(train) != expected_train:
        raise ValueError("train window count differs from complete window30 boundary")
    boundaries = {
        "raw_observations": len(observations), "raw_train_rows": len(train_observations),
        "raw_test_rows": len(test_observations), "invalid_raw_rows_removed": 0,
        "invalid_raw_policy": "fail fast, never silently drop observations",
        "train_engines": int(train.engine_id.nunique()), "test_engines": int(final_test.engine_id.nunique()),
        "window_cycles": 30, "feature_rows": len(features), "train_window_rows": len(train),
        "test_window_rows": len(test_features), "train_prefix_rows_excluded": len(train_observations) - len(train),
        "test_prefix_rows_excluded": len(test_observations) - len(test_features),
        "base_feature_count": len(BASE_FEATURES), "history_feature_count": len(HISTORY_FEATURES),
        "fold_local_pruning": "learned on fitting engines only; explicit selected columns in each fit cache",
        "endpoint_fractions": config.endpoint_fractions,
        "outer_endpoint_rows": int(train.engine_id.nunique() * len(config.endpoint_fractions)),
        "test_endpoint_rows": len(final_test), "target_scale": "raw RUL for every evaluation",
    }
    _write_json(output / "data_boundaries.json", boundaries)
    assignments = _outer_assignments(train, config, root)
    assignments.to_csv(output / "outer_assignments.csv", index=False)
    specs = {spec.candidate_id: spec for spec in make_candidate_specs()}
    all_predictions, all_events, inner_tables, inner_assignments, policy_tables, selections = [], [], [], [], [], []
    for fold in range(1, config.outer_splits + 1):
        val_ids = set(assignments.loc[assignments.outer_fold.eq(fold), "engine_id"])
        outer_train = train[~train.engine_id.isin(val_ids)].reset_index(drop=True)
        outer_validation = train[train.engine_id.isin(val_ids)].reset_index(drop=True)
        selected, inner_predictions, table, inner_assignment = _inner_evaluation(
            outer_train, config, output, identity, fold)
        inner_tables.append(table)
        inner_assignments.append(inner_assignment)
        baseline_policy, baseline_grid = _policy_grid(inner_predictions["hgb_baseline"], config)
        selected_policy, selected_grid = _policy_grid(inner_predictions[selected], config)
        policy_tables.extend([baseline_grid.assign(outer_fold=fold, role="baseline", candidate_id="hgb_baseline"),
                              selected_grid.assign(outer_fold=fold, role="selected", candidate_id=selected)])
        selections.append({"outer_fold": fold, "selected_candidate_id": selected,
                           "baseline_calibrated_policy": asdict(baseline_policy),
                           "selected_calibrated_policy": asdict(selected_policy),
                           "outer_validation_engine_ids": sorted(map(int, val_ids)),
                           "outer_train_engine_ids": sorted(map(int, outer_train.engine_id.unique()))})
        baseline_predictions = _fit_predict(outer_train, outer_validation, specs["hgb_baseline"],
                                            config, output / "fit_cache", identity)
        selected_predictions = _fit_predict(outer_train, outer_validation, specs[selected],
                                            config, output / "fit_cache", identity)
        procedures = (("baseline_fixed", baseline_predictions, WarningPolicy(30, 1)),
                      ("baseline_calibrated", baseline_predictions, baseline_policy),
                      ("selected_calibrated", selected_predictions, selected_policy))
        for procedure, predictions, policy in procedures:
            all_predictions.append(predictions.assign(procedure=procedure, outer_fold=fold,
                                                       threshold=policy.threshold, persistence=policy.persistence))
            events = warning_events(predictions, policy, lead_requirement=config.lead_requirement)
            all_events.append(events.assign(procedure=procedure, outer_fold=fold,
                                             candidate_id=predictions.candidate_id.iloc[0]))
        # Checkpoints are deterministic and regenerated from valid fit caches.
        _write_parquet(pd.concat(all_predictions, ignore_index=True), output / "outer_predictions.parquet")
        _write_parquet(pd.concat(all_events, ignore_index=True), output / "warning_events.parquet")
        pd.concat(inner_tables, ignore_index=True).to_csv(output / "inner_candidate_metrics.csv", index=False)
        pd.concat(inner_assignments, ignore_index=True).to_csv(output / "inner_assignments.csv", index=False)
        pd.concat(policy_tables, ignore_index=True).to_csv(output / "inner_policy_metrics.csv", index=False)
        _write_json(output / "outer_selections.json", selections)
        _write_json(output / "progress.json", {"completed_outer_folds": fold, "total_outer_folds": config.outer_splits})
    outer_predictions = pd.concat(all_predictions, ignore_index=True).sort_values(
        ["procedure", "engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    events = pd.concat(all_events, ignore_index=True).sort_values(
        ["procedure", "engine_id"], kind="mergesort").reset_index(drop=True)
    _write_parquet(outer_predictions, output / "outer_predictions.parquet")
    _write_parquet(events, output / "warning_events.parquet")
    endpoints = pd.concat([build_endpoint_panel(frame, config.endpoint_fractions)
                           for _, frame in outer_predictions.groupby("procedure")], ignore_index=True)
    _write_parquet(endpoints, output / "outer_endpoints.parquet")
    decision = _write_comparisons(outer_predictions, events, config, output)
    replay = _baseline_replay(outer_predictions, config, root, output)
    selected, final_inner, final_table, final_assignment = _inner_evaluation(train, config, output, identity, 0)
    final_baseline_policy, final_baseline_grid = _policy_grid(final_inner["hgb_baseline"], config)
    final_policy, final_grid = _policy_grid(final_inner[selected], config)
    final_table.to_csv(output / "final_inner_candidate_metrics.csv", index=False)
    final_assignment.to_csv(output / "final_inner_assignments.csv", index=False)
    pd.concat([final_baseline_grid.assign(role="baseline", candidate_id="hgb_baseline"),
               final_grid.assign(role="selected", candidate_id=selected)], ignore_index=True).to_csv(
        output / "final_inner_policy_metrics.csv", index=False)
    chosen_grid = final_grid.loc[final_grid.selected].iloc[0]
    selection = {
        "candidate": asdict(specs[selected]), "config": asdict(config),
        "selected_calibrated_policy": asdict(final_policy),
        "baseline_calibrated_policy": asdict(final_baseline_policy),
        "inner_policy_target_met": bool(chosen_grid.inner_target_met),
        "inner_policy_on_time_rate": float(chosen_grid.on_time_rate),
        "promotion_gate_met": decision["promotion_gate_met"],
        "selection_source": "train only: all-train grouped inner CV; outer gate evaluated separately",
        "test_labels_read_for_selection": False, "source_identity": identity,
    }
    _freeze_selection(output / "selection.json", selection)
    fitted = _final_predictor(train, specs[selected], config, output, identity)
    with threadpool_limits(limits=config.fit_threads):
        prediction = fitted.predict(final_test)
        restored_prediction = joblib.load(output / "selected_model.joblib").predict(final_test)
    np.testing.assert_array_equal(prediction, restored_prediction)
    final_predictions = final_test.loc[:, list(KEY)].copy()
    final_predictions["model_name"] = selected
    final_predictions["rul_pred"] = prediction
    _write_parquet(final_predictions, output / "final_predictions.parquet")
    metrics = _evaluate_test(final_predictions, labels_path, config, output, root) if evaluate_test else None
    current_baseline = {name: _sha256(root / name) for name in BASELINE_FILES}
    if current_baseline != manifest["identity"]["baseline_sha256"]:
        raise ValueError("historical baseline changed during study")
    current_sources = {name: _sha256(root / name) for name in SOURCE_FILES}
    fit_metadata = [json.loads(path.read_text(encoding="utf-8")) for path in (output / "fit_cache").glob("*.json")]
    summary = {
        "status": "completed", "output_dir": _relative(output, root),
        "manifest": _relative(output / "manifest.json", root),
        "selection": _relative(output / "selection.json", root),
        "selected_model": _relative(output / "selected_model.joblib", root),
        "final_predictions": _relative(output / "final_predictions.parquet", root),
        "source_identity": identity, "decision": decision, "selected_candidate_id": selected,
        "warning_policy": asdict(final_policy), "baseline_replay": replay,
        "test_evaluation_completed": evaluate_test, "final_metrics": metrics,
        "baseline_bytes_unchanged": True,
        "current_used_source_equivalent_to_snapshot": current_sources == manifest["identity"]["source_sha256"],
        "data_boundaries": boundaries, "elapsed_seconds": time.perf_counter() - started,
        "unique_cached_cv_fits": len(fit_metadata),
        "sum_cached_cv_fit_seconds": sum(item["seconds"] for item in fit_metadata),
        "fit_upper_bound": len(config.candidate_ids) * config.inner_splits * (config.outer_splits + 1)
                           + 2 * config.outer_splits + 1,
        "bootstrap_scope": "paired engines conditional on predictions, not full procedure retraining uncertainty",
    }
    _write_json(output / "run_info.json", summary)
    _write_json(output / "progress.json", {"status": "completed", "completed_outer_folds": config.outer_splits})
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run the isolated module CLI; reduced dry-runs never satisfy promotion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", default="outputs/etl/silver/observations.parquet")
    parser.add_argument("--targets", default="outputs/etl/silver/train_targets.parquet")
    parser.add_argument("--test-labels", default="data/raw/RUL_FD001.txt")
    parser.add_argument("--output-dir", default="outputs/experiments/audit_remediation/20261005")
    parser.add_argument("--fit-threads", type=int, default=4)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="2x2 folds, two candidates,20 rounds,20 resamples, no test labels")
    args = parser.parse_args(argv)
    if args.dry_run:
        if args.output_dir == parser.get_default("output_dir"):
            parser.error("--dry-run requires a separate explicit --output-dir")
        config = AuditStudyConfig(outer_splits=2, inner_splits=2,
                                  candidate_ids=("hgb_baseline", "hgb_history"),
                                  hgb_max_iter=20, bootstrap_resamples=20,
                                  fit_threads=args.fit_threads)
    else:
        config = AuditStudyConfig(fit_threads=args.fit_threads)
    summary = run_audit_study(observations_path=args.observations, targets_path=args.targets,
                              test_labels_path=args.test_labels, output_dir=args.output_dir,
                              config=config, evaluate_test=not (args.dry_run or args.skip_test_evaluation))
    print(json.dumps(_json_value({key: summary[key] for key in (
        "status", "output_dir", "selected_candidate_id", "warning_policy", "decision",
        "unique_cached_cv_fits", "elapsed_seconds")}), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    # Keep pickle references importable when invoked with python -m.
    from cmapss.modeling.audit_study import main as canonical_main

    canonical_main()
