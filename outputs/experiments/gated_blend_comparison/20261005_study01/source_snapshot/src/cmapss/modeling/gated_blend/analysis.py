"""Inner-only selection and paired readouts for the gated HGB study.

The prediction gate itself lives in :mod:`gating`.  This module owns the
selection contract: every gate is scored on the same inner OOF rows as the two
HGB references, warning policies are calibrated from those rows only, and the
outer readout never changes a frozen choice.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from cmapss.modeling import audit_study as audit
from cmapss.modeling.lightgbm_comparison import analysis as tree_analysis

if TYPE_CHECKING:
    from cmapss.modeling.tree_comparison.contracts import StudyConfig

GATE_ARMS = ("hgb_uncapped", "hgb_cap125", "gated_blend")
READOUT_ARMS = GATE_ARMS + ("selected_pipeline",)
NUMERIC_METRICS = (
    "trajectory_rmse", "endpoint_mean_engine_mae", "endpoint_mean_engine_nasa",
)
WARNING_METRICS = ("on_time_rate", "early_alarm_burden")
INTERVAL_SCOPE = (
    "paired engines conditional on fixed outer predictions; duplicate sampled engines "
    "retain multiplicity; excludes refitting variability"
)


def _require_finite(frame: pd.DataFrame | pd.Series, columns: tuple[str, ...]) -> None:
    """Require named metrics to exist and contain finite numeric values."""
    missing = set(columns) - set(frame.index if isinstance(frame, pd.Series) else frame.columns)
    if missing:
        raise ValueError(f"missing required metrics: {sorted(missing)}")
    values = frame.loc[list(columns)] if isinstance(frame, pd.Series) else frame.loc[:, list(columns)]
    try:
        finite = np.isfinite(values.to_numpy(dtype="float64")).all()
    except (TypeError, ValueError) as error:
        raise ValueError("selection metrics must be numeric") from error
    if not finite:
        raise ValueError("selection metrics must be finite")


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Use the audited tree validator for raw train trajectory vectors."""
    return tree_analysis._validate_predictions(predictions)


def candidate_readout(predictions: pd.DataFrame, cfg: StudyConfig) -> dict[str, Any]:
    """Return audited endpoint-panel and full-trajectory metrics."""
    return tree_analysis.candidate_readout(predictions, cfg)


def calibrate_policy(
    predictions: pd.DataFrame, cfg: StudyConfig,
) -> tuple[audit.WarningPolicy, pd.DataFrame]:
    """Calibrate the declared 15-policy warning grid on inner OOF rows."""
    return tree_analysis.calibrate_policy(predictions, cfg)


def _control_mask(records: pd.DataFrame, reference: pd.Series, cfg: StudyConfig) -> pd.Series:
    """Apply the point-inner endpoint and warning safeguards."""
    return (
        records.endpoint_mean_engine_mae.le(
            float(reference.endpoint_mean_engine_mae) + cfg.endpoint_mae_margin
        )
        & records.endpoint_mean_engine_nasa.le(
            float(reference.endpoint_mean_engine_nasa) + cfg.nasa_score_margin
        )
        & records.on_time_rate.ge(
            float(reference.on_time_rate) - cfg.warning_rate_margin
        )
        & records.early_alarm_burden.le(
            float(reference.early_alarm_burden) + cfg.early_alarm_burden_margin
        )
    )


def select_gate_candidate(
    records: pd.DataFrame, reference: pd.Series, cfg: StudyConfig,
) -> dict[str, Any]:
    """Select a gate using inner point controls and a strict RMSE improvement.

    ``diagnostic_candidate_id`` is retained separately from the selected
    candidate.  It is the minimum-RMSE gate satisfying endpoint and warning
    controls even when it does not improve trajectory RMSE.  If no gate is
    feasible, the unconstrained RMSE winner is reported only as a diagnostic;
    ``diagnostic_unconstrained`` is then true and it cannot be promoted.
    """
    required = ("candidate_id",) + NUMERIC_METRICS + WARNING_METRICS
    if any(column not in records for column in required):
        raise ValueError("gate records miss candidate or required readout columns")
    if "family" in records:
        gates = records.loc[records.family.eq("gated_blend")].copy()
    else:
        gates = records.loc[records.candidate_id.astype(str).str.startswith("gate_")].copy()
    if gates.empty or gates.candidate_id.isna().any() or gates.candidate_id.duplicated().any():
        raise ValueError("gate candidates must be nonempty and uniquely identified")
    _require_finite(gates, NUMERIC_METRICS + WARNING_METRICS)
    _require_finite(reference, NUMERIC_METRICS + WARNING_METRICS)
    controls = _control_mask(gates, reference, cfg)
    feasible = gates.loc[controls]
    diagnostic_pool = feasible if not feasible.empty else gates
    diagnostic = diagnostic_pool.sort_values(
        ["trajectory_rmse", "endpoint_mean_engine_mae", "candidate_id"],
        kind="mergesort",
    ).iloc[0]
    improving = feasible.loc[feasible.trajectory_rmse < float(reference.trajectory_rmse)]
    selected = improving.sort_values(
        ["trajectory_rmse", "endpoint_mean_engine_mae", "candidate_id"],
        kind="mergesort",
    ).iloc[0] if not improving.empty else None
    return {
        "candidate_id": str(diagnostic.candidate_id),
        "diagnostic_candidate_id": str(diagnostic.candidate_id),
        "selected_candidate_id": None if selected is None else str(selected.candidate_id),
        "numeric_controls_met": bool(not feasible.empty),
        "diagnostic_numeric_controls_met": bool(not feasible.empty),
        "diagnostic_unconstrained": bool(feasible.empty),
        "trajectory_improves": bool(selected is not None),
        "selection_eligible": bool(selected is not None),
        "n_evaluated": int(len(gates)), "n_feasible": int(len(feasible)),
        "n_improving": int(len(improving)),
        "rule": (
            "inner point controls: endpoint MAE/NASA <= hgb_uncapped and calibrated "
            "on-time >= hgb_uncapped, burden <= hgb_uncapped; select minimum "
            "trajectory RMSE only among strict improvements; otherwise use fixed "
            "hgb_uncapped fallback; retain minimum feasible RMSE as diagnostic"
        ),
    }


def choose_procedure(arm_records: pd.DataFrame, cfg: StudyConfig) -> dict[str, Any]:
    """Choose gated blend or fall back to the fixed uncapped HGB reference.

    ``hgb_cap125`` is accepted for reporting but is never a promotion option.
    The final gate row must already represent the diagnostic gate selected from
    inner OOF records; no outer metric is consulted here.
    """
    if "procedure" not in arm_records:
        raise ValueError("procedure column is required")
    allowed = set(GATE_ARMS)
    if set(arm_records.procedure) != allowed or arm_records.procedure.duplicated().any():
        raise ValueError("procedure selection requires hgb_uncapped, hgb_cap125 and gated_blend")
    required = NUMERIC_METRICS + WARNING_METRICS
    _require_finite(arm_records, required)
    if "numeric_controls_met" not in arm_records:
        raise ValueError("numeric_controls_met is required for gated blend")
    if not arm_records.numeric_controls_met.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("numeric_controls_met must be boolean")
    records = arm_records.set_index("procedure", drop=False)
    reference = records.loc["hgb_uncapped"]
    checks: dict[str, dict[str, bool]] = {}
    for procedure in GATE_ARMS:
        row = records.loc[procedure]
        endpoint = bool(
            row.endpoint_mean_engine_mae <= reference.endpoint_mean_engine_mae + cfg.endpoint_mae_margin
            and row.endpoint_mean_engine_nasa <= reference.endpoint_mean_engine_nasa + cfg.nasa_score_margin
        )
        warning = bool(
            row.on_time_rate >= reference.on_time_rate - cfg.warning_rate_margin
            and row.early_alarm_burden <= reference.early_alarm_burden + cfg.early_alarm_burden_margin
        )
        numeric = bool(row.numeric_controls_met)
        improves = bool(row.trajectory_rmse < reference.trajectory_rmse) if procedure == "gated_blend" else False
        checks[procedure] = {
            "endpoint_controls_met": endpoint,
            "warning_controls_met": warning,
            "numeric_controls_met": numeric,
            "trajectory_improves": improves,
            "eligible": bool(procedure == "gated_blend" and endpoint and warning and numeric and improves),
        }
    selected = "gated_blend" if checks["gated_blend"]["eligible"] else "hgb_uncapped"
    # The accepted protocol normally passes the HGB reference policy through
    # the row (for example an outer fold may use 40/3).  Keep 40/1 only as a
    # compatibility default for synthetic callers that do not carry policy
    # columns; the runner must never invent a policy after inner selection.
    fallback_policy = {
        "threshold": int(records.loc["hgb_uncapped"].get("policy_threshold", 40)),
        "persistence": int(records.loc["hgb_uncapped"].get("policy_persistence", 1)),
    }
    return {
        "procedure_id": selected,
        "selected_procedure_id": selected,
        "fallback_used": selected == "hgb_uncapped",
        "fixed_fallback_policy": fallback_policy,
        "arm_controls": checks,
        **checks[selected],
        "rule": (
            "gated blend requires strict inner trajectory RMSE improvement plus "
            "endpoint and warning controls; otherwise fallback to the frozen "
            "hgb_uncapped reference policy; hgb_cap125 is reporting-only"
        ),
    }


def paired_comparisons(
    predictions: pd.DataFrame, events: pd.DataFrame, cfg: StudyConfig,
) -> pd.DataFrame:
    """Return paired engine bootstrap comparisons with duplicate multiplicity."""
    tables = tree_analysis._engine_tables(predictions, events, cfg)
    pairs = [("hgb_uncapped", "gated_blend"), ("hgb_uncapped", "selected_pipeline")]
    if "hgb_cap125" in tables:
        pairs.extend([("hgb_cap125", "gated_blend"), ("hgb_cap125", "selected_pipeline")])
    return pd.concat([
        tree_analysis._paired_table(tables[reference], tables[candidate], cfg)
        .assign(reference=reference, procedure=candidate)
        for reference, candidate in pairs
        if candidate in tables
    ], ignore_index=True)


# Short alias for callers that describe the operation as gate selection.
select_gate = select_gate_candidate


def _gate_controls(comparison: pd.DataFrame) -> dict[str, bool]:
    """Apply the fixed confidence-bound signs used by the HGB study."""
    rows = comparison.set_index("metric")
    return {
        "trajectory_rmse_improved": bool(rows.loc["trajectory_rmse", "ci_upper"] < 0),
        "endpoint_mae_not_worse": bool(rows.loc["endpoint_mean_engine_mae", "ci_upper"] <= 0),
        "endpoint_nasa_not_worse": bool(rows.loc["endpoint_mean_engine_nasa", "ci_upper"] <= 0),
        "warning_on_time_not_worse": bool(rows.loc["on_time_rate", "ci_lower"] >= 0),
        "warning_burden_not_worse": bool(rows.loc["early_alarm_burden", "ci_upper"] <= 0),
    }


def write_readouts(
    predictions: pd.DataFrame, events: pd.DataFrame, cfg: StudyConfig, output: str | Path,
) -> dict[str, Any]:
    """Write aggregate, engine, region and paired readouts for outer predictions."""
    expected = set(READOUT_ARMS)
    if "procedure" not in predictions or set(predictions.procedure) != expected:
        raise ValueError("outer readouts require hgb references, gated_blend and selected_pipeline")
    tables = tree_analysis._engine_tables(predictions, events, cfg)
    summaries: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    for procedure, frame in predictions.groupby("procedure", sort=True):
        scored = _validate_predictions(frame)
        policy_events = events.loc[events.procedure.eq(procedure)]
        summaries.append({"procedure": procedure, **audit._model_readout(scored, cfg.audit_config()),
                          **audit._warning_readout(policy_events)})
        scored["rul_region"] = pd.cut(scored.y_true, [-1, 30, 60, 100, 150, np.inf],
                                       labels=["0-30", "31-60", "61-100", "101-150", ">150"])
        for region, group in scored.groupby("rul_region", observed=False):
            if group.empty:
                continue
            regions.append({"procedure": procedure, "rul_region": str(region), "n_rows": len(group),
                            "n_engines": group.engine_id.nunique(),
                            "mean_window_mae": float(group.absolute_error.mean()),
                            "mean_engine_mae": float(group.groupby("engine_id").absolute_error.mean().mean()),
                            "signed_bias": float((group.y_pred - group.y_true).mean()),
                            "rmse": float(np.sqrt(group.squared_error.mean()))})
    comparisons = paired_comparisons(predictions, events, cfg)
    selected = comparisons.loc[(comparisons.reference == "hgb_uncapped")
                               & comparisons.procedure.eq("selected_pipeline")]
    controls = _gate_controls(selected)
    decision = {
        "full_protocol_configuration": bool(cfg.scientific),
        "promotion_gate_met": bool(cfg.scientific and all(controls.values())),
        "controls": controls,
        "selection_source": "inner grouped OOF only; no outer procedure selection",
        "capped_reference_reporting_only": True,
        "gate": "selected_pipeline vs hgb_uncapped: paired RMSE upper CI<0; endpoint MAE/NASA and burden upper CI<=0; on-time lower CI>=0",
        "bootstrap_scope": INTERVAL_SCOPE,
        "historical_baseline_overwritten": False,
    }
    pd.DataFrame(summaries).to_csv(destination / "aggregate_metrics.csv", index=False)
    pd.concat([table.assign(procedure=name).reset_index() for name, table in tables.items()], ignore_index=True).to_csv(
        destination / "engine_metrics.csv", index=False)
    pd.DataFrame([
        {"procedure": procedure, **audit._warning_readout(events.loc[events.procedure.eq(procedure)])}
        for procedure in sorted(expected)
    ]).to_csv(destination / "warning_metrics.csv", index=False)
    comparisons.to_csv(destination / "paired_comparison.csv", index=False)
    pd.DataFrame(regions).to_csv(destination / "region_metrics.csv", index=False)
    audit._write_json(destination / "decision.json", decision)
    return decision


__all__ = [
    "GATE_ARMS", "READOUT_ARMS", "candidate_readout", "calibrate_policy",
    "select_gate_candidate", "select_gate", "choose_procedure", "paired_comparisons", "write_readouts",
]
