"""Arm selection and paired outer readouts for the multi-scale HGB study.

Works on raw-RUL predictions and warning events from the nested runner.
Test labels are never read here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from cmapss.modeling import audit_study as audit
from cmapss.modeling.lightgbm_comparison import analysis as tree_analysis

if TYPE_CHECKING:
    from .config import StudyConfig


ARMS = (
    "hgb_baseline169",
    "hgb_multiscale10_30",
    "hgb_initial_profile42",
)
BASELINE_ARM = ARMS[0]
READOUT_ARMS = ARMS + ("selected_pipeline",)
NUMERIC_METRICS = (
    "trajectory_rmse",
    "endpoint_mean_engine_mae",
    "endpoint_mean_engine_nasa",
)
WARNING_METRICS = ("on_time_rate", "early_alarm_burden")
PAIRED_ENGINE_METRICS = (
    "endpoint_mean_engine_mae",
    "endpoint_mean_engine_nasa",
    "trajectory_mean_engine_mae",
    "trajectory_mean_engine_nasa",
    "on_time_rate",
    "early_alarm_burden",
    "mean_transitions",
    "mean_retractions",
)
INTERVAL_SCOPE = (
    "paired engines conditional on fixed outer predictions; duplicate sampled "
    "engines retain multiplicity; excludes refitting variability"
)


class _ConfigAdapter:
    """Expose the legacy ``audit_config`` hook for the shared readout code."""

    def __init__(self, config: StudyConfig) -> None:
        self._config = config

    def audit_config(self) -> audit.AuditStudyConfig:
        """Build the shared audit metric configuration."""
        return audit.AuditStudyConfig(
            outer_splits=self._config.outer_splits,
            inner_splits=self._config.inner_splits,
            endpoint_fractions=self._config.endpoint_fractions,
            thresholds=self._config.thresholds,
            persistences=self._config.persistences,
            lead_requirement=self._config.lead_requirement,
            target_on_time_rate=self._config.target_on_time_rate,
            bootstrap_resamples=self._config.bootstrap_resamples,
            seed=self._config.seed,
            fit_threads=self._config.fit_threads,
            hgb_max_iter=self._config.hgb_max_iter,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._config, name)


def _tree_config(config: StudyConfig) -> StudyConfig | _ConfigAdapter:
    """Return config in the interface expected by audited tree readouts."""
    return config if callable(getattr(config, "audit_config", None)) else _ConfigAdapter(config)


def _require_finite(frame: pd.DataFrame | pd.Series,
                    columns: tuple[str, ...]) -> None:
    """Require named readout fields to be present and finite."""
    available = set(frame.index if isinstance(frame, pd.Series) else frame.columns)
    missing = set(columns) - available
    if missing:
        raise ValueError(f"missing required metrics: {sorted(missing)}")
    values = frame.loc[list(columns)] if isinstance(frame, pd.Series) else frame.loc[:, list(columns)]
    try:
        finite = np.isfinite(values.to_numpy(dtype="float64")).all()
    except (TypeError, ValueError) as error:
        raise ValueError("readout metrics must be numeric") from error
    if not finite:
        raise ValueError("readout metrics must be finite")


def _validate_arm_records(records: pd.DataFrame) -> pd.DataFrame:
    """Validate one inner arm table and return it indexed by procedure."""
    if "procedure" not in records:
        raise ValueError("procedure column is required")
    if records.procedure.isna().any() or records.procedure.duplicated().any():
        raise ValueError("procedure identities must be unique and non-null")
    if set(records.procedure) != set(ARMS):
        raise ValueError("procedure table must contain exactly the three declared arms")
    _require_finite(records, NUMERIC_METRICS + WARNING_METRICS)
    return records.set_index("procedure", drop=False)


def candidate_readout(predictions: pd.DataFrame, cfg: StudyConfig) -> dict[str, Any]:
    """Compute audited trajectory and retrospective endpoint metrics.

    The prediction vector must contain raw ``y_true`` and use train engines;
    endpoint positions are evaluated from lifetime inferred from those labels.
    """
    return tree_analysis.candidate_readout(predictions, _tree_config(cfg))


def calibrate_policy(
    predictions: pd.DataFrame, cfg: StudyConfig,
) -> tuple[audit.WarningPolicy, pd.DataFrame]:
    """Calibrate the fixed warning grid using inner OOF predictions only."""
    return tree_analysis.calibrate_policy(predictions, _tree_config(cfg))


def choose_procedure(arm_records: pd.DataFrame, cfg: StudyConfig) -> dict[str, Any]:
    """Pick an arm from inner OOF metrics, falling back to the baseline.

    A candidate must strictly improve trajectory RMSE and be no worse than the
    baseline (within the configured margins) on endpoint MAE, endpoint NASA,
    on-time rate and early-alarm burden. Outer metrics are not used.
    """
    records = _validate_arm_records(arm_records)
    baseline = records.loc[BASELINE_ARM]
    checks: dict[str, dict[str, bool]] = {}
    for procedure in ARMS:
        row = records.loc[procedure]
        endpoint = bool(
            row.endpoint_mean_engine_mae <= baseline.endpoint_mean_engine_mae + cfg.endpoint_mae_margin
            and row.endpoint_mean_engine_nasa <= baseline.endpoint_mean_engine_nasa + cfg.nasa_score_margin
        )
        warning = bool(
            row.on_time_rate >= baseline.on_time_rate - cfg.warning_rate_margin
            and row.early_alarm_burden <= baseline.early_alarm_burden + cfg.early_alarm_burden_margin
        )
        improves = bool(
            procedure != BASELINE_ARM
            and row.trajectory_rmse < baseline.trajectory_rmse
        )
        checks[procedure] = {
            "endpoint_controls_met": endpoint,
            "warning_controls_met": warning,
            "trajectory_improves": improves,
            "eligible": bool(procedure != BASELINE_ARM and endpoint and warning and improves),
        }

    eligible = [procedure for procedure in ARMS[1:] if checks[procedure]["eligible"]]
    if eligible:
        selected = str(
            records.loc[eligible]
            .reset_index(drop=True)
            .sort_values(
                ["trajectory_rmse", "endpoint_mean_engine_mae", "procedure"],
                kind="mergesort",
            )
            .iloc[0]
            .procedure
        )
    else:
        selected = BASELINE_ARM
    return {
        "procedure_id": selected,
        "selected_procedure_id": selected,
        "baseline_procedure_id": BASELINE_ARM,
        "fallback_used": selected == BASELINE_ARM,
        "arm_controls": checks,
        "rule": (
            "inner grouped OOF only; candidate requires strict trajectory RMSE "
            "improvement and endpoint MAE/NASA, calibrated on-time and early "
            "alarm burden no worse than hgb_baseline169; ties by trajectory "
            "RMSE, endpoint MAE, then arm ID; otherwise baseline fallback"
        ),
    }


def paired_comparisons(
    predictions: pd.DataFrame, events: pd.DataFrame, cfg: StudyConfig,
) -> pd.DataFrame:
    """Return paired engine bootstrap comparisons against the fixed baseline."""
    tables = _engine_tables(predictions, events, cfg)
    if BASELINE_ARM not in tables:
        raise ValueError("baseline predictions are required for paired comparison")
    pairs = [
        (BASELINE_ARM, procedure)
        for procedure in READOUT_ARMS
        if procedure != BASELINE_ARM and procedure in tables
    ]
    return pd.concat(
        [
            tree_analysis._paired_table(tables[reference], tables[procedure], _tree_config(cfg))
            .assign(reference=reference, procedure=procedure)
            for reference, procedure in pairs
        ],
        ignore_index=True,
    )


def _engine_tables(
    predictions: pd.DataFrame, events: pd.DataFrame, cfg: StudyConfig,
) -> dict[str, pd.DataFrame]:
    """Per-engine statistics for each procedure, checked for identical keys and truth."""
    if "procedure" not in predictions or "procedure" not in events:
        raise ValueError("predictions and events require procedure identities")
    if predictions.procedure.isna().any() or events.procedure.isna().any():
        raise ValueError("procedure identities cannot be missing")
    if set(predictions.procedure) != set(events.procedure):
        raise ValueError("prediction and warning procedure identities differ")
    if BASELINE_ARM not in set(predictions.procedure):
        raise ValueError(f"{BASELINE_ARM} reference predictions are required")
    tables: dict[str, pd.DataFrame] = {}
    reference: pd.DataFrame | None = None
    audited_cfg = _tree_config(cfg).audit_config()
    for procedure, frame in predictions.groupby("procedure", sort=True):
        scored = tree_analysis._validate_predictions(frame)
        keys_truth = scored.loc[:, [*audit.KEY, "y_true"]]
        if reference is None:
            reference = keys_truth
        else:
            try:
                pd.testing.assert_frame_equal(keys_truth, reference, check_exact=True)
            except AssertionError as error:
                raise ValueError("paired procedures require identical keys and raw truth") from error
        policy_events = events.loc[events.procedure.eq(procedure)].copy()
        if (policy_events.empty or "engine_id" not in policy_events
                or policy_events.engine_id.isna().any()
                or policy_events.engine_id.duplicated().any()
                or set(policy_events.engine_id) != set(scored.engine_id)):
            raise ValueError("warning events must match each prediction engine exactly once")
        _require_finite(policy_events, ("on_time", "early_alarm_burden",
                                        "n_transitions", "n_retractions"))
        if (not policy_events.on_time.isin((False, True)).all()
                or not policy_events.early_alarm_burden.between(0.0, 1.0).all()
                or policy_events.n_transitions.lt(0).any()
                or policy_events.n_retractions.lt(0).any()):
            raise ValueError("invalid warning rate, burden or transition counts")
        table = audit._engine_readout(scored, policy_events, audited_cfg).sort_index()
        sufficient = scored.groupby("engine_id", sort=True).agg(
            trajectory_sse=("squared_error", "sum"),
            trajectory_n_rows=("squared_error", "size"),
            trajectory_mean_engine_nasa=("nasa_score_component", "mean"),
        )
        table = table.join(sufficient)
        _require_finite(table, PAIRED_ENGINE_METRICS + ("trajectory_sse", "trajectory_n_rows"))
        if table.trajectory_n_rows.le(0).any():
            raise ValueError("each engine must contribute trajectory rows")
        tables[str(procedure)] = table
    return tables


def _gate_controls(comparison: pd.DataFrame) -> dict[str, bool]:
    """Check the CI bounds of the five promotion conditions."""
    if comparison.empty:
        raise ValueError("selected paired comparison is empty")
    rows = comparison.set_index("metric")
    required = ("trajectory_rmse", "endpoint_mean_engine_mae",
                "endpoint_mean_engine_nasa", "on_time_rate",
                "early_alarm_burden")
    if not set(required).issubset(rows.index):
        raise ValueError("selected comparison misses promotion metrics")
    return {
        "trajectory_rmse_improved": bool(rows.loc["trajectory_rmse", "ci_upper"] < 0.0),
        "endpoint_mae_not_worse": bool(rows.loc["endpoint_mean_engine_mae", "ci_upper"] <= 0.0),
        "endpoint_nasa_not_worse": bool(rows.loc["endpoint_mean_engine_nasa", "ci_upper"] <= 0.0),
        "warning_on_time_not_worse": bool(rows.loc["on_time_rate", "ci_lower"] >= 0.0),
        "warning_burden_not_worse": bool(rows.loc["early_alarm_burden", "ci_upper"] <= 0.0),
    }


def _region_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Error metrics per raw-RUL region."""
    rows: list[dict[str, Any]] = []
    for procedure, frame in predictions.groupby("procedure", sort=True):
        scored = tree_analysis._validate_predictions(frame)
        scored["rul_region"] = pd.cut(
            scored.y_true, [-1, 30, 60, 100, 150, np.inf],
            labels=["0-30", "31-60", "61-100", "101-150", ">150"],
        )
        for region, group in scored.groupby("rul_region", observed=False):
            if group.empty:
                continue
            rows.append({
                "procedure": procedure,
                "rul_region": str(region),
                "n_rows": int(len(group)),
                "n_engines": int(group.engine_id.nunique()),
                "mean_window_mae": float(group.absolute_error.mean()),
                "mean_engine_mae": float(group.groupby("engine_id").absolute_error.mean().mean()),
                "signed_bias": float((group.y_pred - group.y_true).mean()),
                "rmse": float(np.sqrt(group.squared_error.mean())),
            })
    return rows


def write_readouts(
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    cfg: StudyConfig,
    output: str | Path,
) -> dict[str, Any]:
    """Write outer readout tables and the paired promotion decision."""
    if "procedure" not in predictions or set(predictions.procedure) != set(READOUT_ARMS):
        raise ValueError("readouts require the three arms and selected_pipeline")
    if "procedure" not in events or set(events.procedure) != set(READOUT_ARMS):
        raise ValueError("warning events require the three arms and selected_pipeline")
    tables = _engine_tables(predictions, events, cfg)
    summaries: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    for procedure, frame in predictions.groupby("procedure", sort=True):
        policy_events = events.loc[events.procedure.eq(procedure)]
        summaries.append({
            "procedure": procedure,
            **candidate_readout(frame, cfg),
            **audit._warning_readout(policy_events),
        })
        warning_rows.append({"procedure": procedure, **audit._warning_readout(policy_events)})

    comparisons = paired_comparisons(predictions, events, cfg)
    selected = comparisons.loc[
        comparisons.reference.eq(BASELINE_ARM)
        & comparisons.procedure.eq("selected_pipeline")
    ]
    controls = _gate_controls(selected)
    trajectory_row = selected.loc[selected.metric.eq("trajectory_rmse")]
    if len(trajectory_row) != 1:
        raise ValueError("selected comparison must contain one trajectory RMSE row")
    trajectory_row = trajectory_row.iloc[0]
    decision = {
        "full_protocol_configuration": bool(cfg.scientific),
        "promotion_gate_met": bool(cfg.scientific and all(controls.values())),
        "controls": controls,
        "outer_trajectory_rmse_delta": float(trajectory_row.delta),
        "outer_trajectory_rmse_ci_lower": float(trajectory_row.ci_lower),
        "outer_trajectory_rmse_ci_upper": float(trajectory_row.ci_upper),
        "selected_reference": BASELINE_ARM,
        "selection_source": "inner grouped OOF only; no procedure selection from outer readouts",
        "gate": (
            "selected_pipeline vs hgb_baseline169: paired trajectory RMSE upper "
            "95% CI<0; endpoint MAE/NASA and early alarm burden upper 95% CI<=0; "
            "on-time rate lower 95% CI>=0; no test gating"
        ),
        "bootstrap_scope": INTERVAL_SCOPE,
        "historical_baseline_overwritten": False,
        "evaluation_limit": (
            "follow-up study on historically inspected FD001 engine folds; "
            "conditional fixed-prediction bootstrap excludes refitting variability"
        ),
    }
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(destination / "aggregate_metrics.csv", index=False)
    pd.concat(
        [table.assign(procedure=name).reset_index() for name, table in tables.items()],
        ignore_index=True,
    ).to_csv(destination / "engine_metrics.csv", index=False)
    comparisons.to_csv(destination / "paired_comparison.csv", index=False)
    pd.DataFrame(_region_rows(predictions)).to_csv(destination / "region_metrics.csv", index=False)
    pd.DataFrame(warning_rows).to_csv(destination / "warning_metrics.csv", index=False)
    audit._write_json(destination / "decision.json", decision)
    return decision


__all__ = [
    "ARMS", "BASELINE_ARM", "READOUT_ARMS", "candidate_readout",
    "calibrate_policy", "choose_procedure", "paired_comparisons",
    "write_readouts",
]
