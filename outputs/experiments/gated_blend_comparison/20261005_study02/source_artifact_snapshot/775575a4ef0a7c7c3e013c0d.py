"""Frozen trajectory-first selection and conditional paired engine inference.

The endpoint panel and warning measurements reuse the audited implementation.
Outer predictions are readouts only: the selected pipeline must already have
been chosen inside each fitting fold. Bootstrap uncertainty is conditional on
these fixed predictions and does not describe model refitting variability.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.modeling import audit_study as audit
from cmapss.modeling.metrics import add_prediction_errors

if TYPE_CHECKING:
    from cmapss.modeling.lightgbm_comparison.contracts import StudyConfig

ARMS = ("hgb_uncapped", "hgb_cap125", "lgbm_uncapped", "lgbm_cap125")
NUMERIC_METRICS = ("trajectory_rmse", "endpoint_mean_engine_mae", "endpoint_mean_engine_nasa")
PAIRED_ENGINE_METRICS = (
    "endpoint_mean_engine_mae", "endpoint_mean_engine_nasa",
    "trajectory_mean_engine_mae", "trajectory_mean_engine_nasa",
    "on_time_rate", "early_alarm_burden", "mean_transitions", "mean_retractions",
)
INTERVAL_SCOPE = (
    "paired engines conditional on fixed outer predictions; duplicate sampled engines "
    "retain multiplicity; excludes refitting variability"
)


def _require_finite(frame: pd.DataFrame | pd.Series, columns: tuple[str, ...]) -> None:
    """Reject missing or nonfinite numerical inputs before selection/inference."""
    missing = set(columns) - set(frame.index if isinstance(frame, pd.Series) else frame.columns)
    if missing:
        raise ValueError(f"missing required metrics: {sorted(missing)}")
    try:
        values = frame.loc[list(columns)] if isinstance(frame, pd.Series) else frame.loc[:, list(columns)]
        valid = np.isfinite(values.to_numpy(dtype="float64")).all()
    except (TypeError, ValueError) as error:
        raise ValueError("selection and inference metrics must be numeric") from error
    if not valid:
        raise ValueError("selection and inference metrics must be finite")


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate one train trajectory vector and recompute its error columns."""
    required = set(KEY) | {"y_true", "y_pred"}
    if predictions.empty or not required.issubset(predictions.columns):
        raise ValueError("predictions require nonempty keys, raw y_true and y_pred")
    if predictions[list(KEY)].isna().any().any() or predictions.duplicated(list(KEY)).any():
        raise ValueError("prediction keys must be identified and unique")
    if not predictions.split.eq("train").all():
        raise ValueError("trajectory comparison uses train engines only")
    _require_finite(predictions, ("engine_id", "cycle", "y_true", "y_pred"))
    if (predictions.y_true.lt(0).any() or predictions.cycle.lt(1).any()
            or not np.equal(predictions.cycle, np.floor(predictions.cycle)).all()
            or not np.equal(predictions.engine_id, np.floor(predictions.engine_id)).all()):
        raise ValueError("invalid integer keys or negative raw RUL")
    result = add_prediction_errors(predictions)
    _require_finite(result, ("absolute_error", "squared_error", "nasa_score_component"))
    return result.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)


def candidate_readout(predictions: pd.DataFrame, cfg: StudyConfig) -> dict[str, Any]:
    """Score raw-RUL endpoints and full trajectories without changing selection.

    Args:
        predictions: One candidate's held-out train-engine predictions.
        cfg: Frozen tree study configuration.

    Returns:
        Audited endpoint and trajectory metrics, with pooled trajectory RMSE.
    """
    return audit._model_readout(_validate_predictions(predictions), cfg.audit_config())


def select_lgbm_candidate(
    records: pd.DataFrame, reference: pd.Series, family: str, cfg: StudyConfig,
) -> dict[str, Any]:
    """Minimize inner trajectory RMSE subject to corresponding HGB endpoints.

    If no member satisfies both endpoint controls, retain the unconstrained
    winner for measurement while marking it ineligible for pipeline selection.

    Args:
        records: Inner summaries containing candidate_id and optional family.
        reference: Corresponding capped or uncapped fixed HGB summary.
        family: lgbm_uncapped or lgbm_cap125.
        cfg: Frozen zero-margin study rules.

    Returns:
        Candidate identity, feasibility flag and declared selection rule.

    Raises:
        ValueError: Candidate identities or required metrics are invalid.
    """
    if family not in ("lgbm_uncapped", "lgbm_cap125") or "candidate_id" not in records:
        raise ValueError("unknown LightGBM family or missing candidate identities")
    mask = (records.family.eq(family) if "family" in records
            else records.candidate_id.astype(str).str.startswith(family + "_"))
    candidates = records.loc[mask].copy()
    if (candidates.empty or candidates.candidate_id.isna().any()
            or candidates.candidate_id.duplicated().any()):
        raise ValueError("family candidates must be nonempty and uniquely identified")
    _require_finite(candidates, NUMERIC_METRICS)
    _require_finite(reference, NUMERIC_METRICS[1:])
    controls = (
        candidates.endpoint_mean_engine_mae.le(reference.endpoint_mean_engine_mae + cfg.endpoint_mae_margin)
        & candidates.endpoint_mean_engine_nasa.le(reference.endpoint_mean_engine_nasa + cfg.nasa_score_margin)
    )
    feasible = candidates.loc[controls]
    pool = feasible if not feasible.empty else candidates
    winner = pool.sort_values(
        ["trajectory_rmse", "endpoint_mean_engine_mae", "candidate_id"], kind="mergesort",
    ).iloc[0]
    return {
        "candidate_id": str(winner.candidate_id), "family": family,
        "numeric_controls_met": bool(not feasible.empty),
        "n_evaluated": len(candidates), "n_feasible": len(feasible),
        "rule": "inner pooled trajectory RMSE, then endpoint MAE, then candidate ID; "
                "endpoint engine MAE and NASA <= corresponding HGB; "
                "unconstrained diagnostic winner if none is feasible",
    }


def calibrate_policy(
    predictions: pd.DataFrame, cfg: StudyConfig,
) -> tuple[audit.WarningPolicy, pd.DataFrame]:
    """Calibrate the declared warning grid on one selected arm's inner vector.

    Call once per selected arm, after numeric family selection. The returned
    grid contains the same inner target and feasibility readouts as the audit.
    """
    return audit._policy_grid(_validate_predictions(predictions), cfg.audit_config())


def choose_procedure(arm_records: pd.DataFrame, cfg: StudyConfig) -> dict[str, Any]:
    """Choose an inner arm with zero-margin endpoint and warning safeguards.

    Args:
        arm_records: Four selected arm summaries, including numeric_controls_met
            from LightGBM family selection and calibrated warning summaries.
        cfg: Frozen study margins.

    Returns:
        Selected procedure ID, eligibility checks and frozen rule description.

    Raises:
        ValueError: An arm is missing, duplicated, or has invalid metrics.
    """
    if ("procedure" not in arm_records or arm_records.procedure.duplicated().any()
            or set(arm_records.procedure) != set(ARMS)):
        raise ValueError("inner procedure selection requires exactly four unique arms")
    required = NUMERIC_METRICS + ("on_time_rate", "early_alarm_burden")
    _require_finite(arm_records, required)
    if "numeric_controls_met" not in arm_records:
        raise ValueError("numeric family feasibility flags are required")
    if not arm_records.numeric_controls_met.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("numeric family feasibility flags must be boolean")
    records = arm_records.set_index("procedure", drop=False)
    reference = records.loc["hgb_uncapped"]
    checks: dict[str, dict[str, bool]] = {}
    for procedure, row in records.iterrows():
        endpoint = bool(
            row.endpoint_mean_engine_mae <= reference.endpoint_mean_engine_mae + cfg.endpoint_mae_margin
            and row.endpoint_mean_engine_nasa <= reference.endpoint_mean_engine_nasa + cfg.nasa_score_margin
        )
        warnings = bool(
            row.on_time_rate >= reference.on_time_rate - cfg.warning_rate_margin
            and row.early_alarm_burden <= reference.early_alarm_burden
            + getattr(cfg, "early_alarm_burden_margin", 0.0)
        )
        family_feasible = bool(row.numeric_controls_met) if str(procedure).startswith("lgbm_") else True
        checks[str(procedure)] = {
            "numeric_controls_met": endpoint and family_feasible,
            "warning_controls_met": warnings,
            "eligible": endpoint and family_feasible and warnings,
        }
    eligible = [procedure for procedure in ARMS if checks[procedure]["eligible"]]
    winner = records.loc[eligible].reset_index(drop=True).sort_values(
        ["trajectory_rmse", "endpoint_mean_engine_mae", "procedure"], kind="mergesort",
    ).iloc[0]
    procedure_id = str(winner.procedure)
    return {
        "procedure_id": procedure_id, "selected_procedure_id": procedure_id,
        **checks[procedure_id], "arm_controls": checks,
        "rule": "inner pooled trajectory RMSE, then endpoint MAE, then arm ID; "
                "endpoint MAE/NASA and calibrated warning controls versus hgb_uncapped; "
                "LightGBM also requires corresponding-HGB family feasibility",
    }


def _engine_tables(
    predictions: pd.DataFrame, events: pd.DataFrame, cfg: StudyConfig,
) -> dict[str, pd.DataFrame]:
    """Require matched keys/truth/events and retain sufficient pooled statistics."""
    if "procedure" not in predictions or "procedure" not in events:
        raise ValueError("predictions and events require procedure identities")
    if predictions.procedure.isna().any() or events.procedure.isna().any():
        raise ValueError("procedure identities cannot be missing")
    if set(predictions.procedure) != set(events.procedure):
        raise ValueError("prediction and warning procedure identities differ")
    if "hgb_uncapped" not in set(predictions.procedure):
        raise ValueError("hgb_uncapped reference predictions are required")
    tables: dict[str, pd.DataFrame] = {}
    reference: pd.DataFrame | None = None
    for procedure, frame in predictions.groupby("procedure", sort=True):
        scored = _validate_predictions(frame)
        keys_truth = scored.loc[:, list(KEY) + ["y_true"]]
        if reference is None:
            reference = keys_truth
        else:
            try:
                pd.testing.assert_frame_equal(keys_truth, reference, check_exact=True)
            except AssertionError as error:
                raise ValueError("paired procedures require identical keys and raw truth") from error
        policy_events = events.loc[events.procedure.eq(procedure)].copy()
        if (policy_events.empty or "engine_id" not in policy_events
                or policy_events.engine_id.isna().any() or policy_events.engine_id.duplicated().any()
                or set(policy_events.engine_id) != set(scored.engine_id)):
            raise ValueError("warning events must match each prediction engine exactly once")
        _require_finite(policy_events, ("on_time", "early_alarm_burden", "n_transitions", "n_retractions"))
        if (not policy_events.on_time.isin((False, True)).all()
                or not policy_events.early_alarm_burden.between(0., 1.).all()
                or policy_events.n_transitions.lt(0).any() or policy_events.n_retractions.lt(0).any()):
            raise ValueError("invalid warning rate, burden or transition counts")
        table = audit._engine_readout(scored, policy_events, cfg.audit_config()).sort_index()
        sufficient = scored.groupby("engine_id", sort=True).agg(
            trajectory_sse=("squared_error", "sum"), trajectory_n_rows=("squared_error", "size"),
            trajectory_mean_engine_nasa=("nasa_score_component", "mean"),
        )
        table = table.join(sufficient)
        _require_finite(table, PAIRED_ENGINE_METRICS + ("trajectory_sse", "trajectory_n_rows"))
        if table.trajectory_n_rows.le(0).any():
            raise ValueError("each engine must contribute trajectory rows")
        tables[str(procedure)] = table
    return tables


def _paired_table(
    baseline: pd.DataFrame, candidate: pd.DataFrame, cfg: StudyConfig,
) -> pd.DataFrame:
    """Bootstrap paired pooled RMSE and equal-engine summaries with multiplicity."""
    if not baseline.index.equals(candidate.index):
        raise ValueError("paired engine metrics require identical ordered engine IDs")
    n_engines = len(baseline)
    indices = np.random.default_rng(cfg.seed).integers(
        0, n_engines, size=(cfg.bootstrap_resamples, n_engines),
    )
    rows: list[dict[str, Any]] = []
    for metric in ("trajectory_rmse",) + PAIRED_ENGINE_METRICS:
        if metric == "trajectory_rmse":
            old_sse = baseline.trajectory_sse.to_numpy(dtype="float64")
            new_sse = candidate.trajectory_sse.to_numpy(dtype="float64")
            old_n = baseline.trajectory_n_rows.to_numpy(dtype="float64")
            new_n = candidate.trajectory_n_rows.to_numpy(dtype="float64")
            before = float(np.sqrt(old_sse.sum() / old_n.sum()))
            after = float(np.sqrt(new_sse.sum() / new_n.sum()))
            draws = (np.sqrt(new_sse[indices].sum(axis=1) / new_n[indices].sum(axis=1))
                     - np.sqrt(old_sse[indices].sum(axis=1) / old_n[indices].sum(axis=1)))
        else:
            old = baseline[metric].to_numpy(dtype="float64")
            new = candidate[metric].to_numpy(dtype="float64")
            before, after = float(old.mean()), float(new.mean())
            draws = (new - old)[indices].mean(axis=1)
        rows.append({
            "metric": metric, "baseline": before, "candidate": after, "delta": after - before,
            "ci_lower": float(np.quantile(draws, .025)),
            "ci_upper": float(np.quantile(draws, .975)),
            "n_engines": n_engines, "resamples": cfg.bootstrap_resamples,
            "seed": cfg.seed, "interval_scope": INTERVAL_SCOPE,
        })
    return pd.DataFrame(rows)


def _comparisons_from_tables(
    tables: dict[str, pd.DataFrame], cfg: StudyConfig,
) -> pd.DataFrame:
    """Publish fixed arm comparisons and the already selected pipeline."""
    pairs = [("hgb_uncapped", arm) for arm in ARMS
             if arm in tables and arm != "hgb_uncapped"]
    if "hgb_cap125" in tables and "lgbm_cap125" in tables:
        pairs.append(("hgb_cap125", "lgbm_cap125"))
    if "selected_pipeline" in tables:
        pairs.extend((reference, "selected_pipeline")
                     for reference in ("hgb_uncapped", "hgb_cap125") if reference in tables)
    return pd.concat([
        _paired_table(tables[reference], tables[procedure], cfg)
        .assign(reference=reference, procedure=procedure)
        for reference, procedure in pairs
    ], ignore_index=True)


def paired_comparisons(
    predictions: pd.DataFrame, events: pd.DataFrame, cfg: StudyConfig,
) -> pd.DataFrame:
    """Compare matched procedures using fixed, conditional engine bootstrap.

    Full study inputs contain four arms and selected_pipeline. A subset with
    hgb_uncapped is accepted for independent checks of the inference math.
    """
    return _comparisons_from_tables(_engine_tables(predictions, events, cfg), cfg)


def _gate_controls(comparison: pd.DataFrame) -> dict[str, bool]:
    """Apply predeclared confidence-bound signs for the five joint safeguards."""
    rows = comparison.set_index("metric")
    return {
        "trajectory_rmse_improved": bool(rows.loc["trajectory_rmse", "ci_upper"] < 0.),
        "endpoint_mae_not_worse": bool(rows.loc["endpoint_mean_engine_mae", "ci_upper"] <= 0.),
        "endpoint_nasa_not_worse": bool(rows.loc["endpoint_mean_engine_nasa", "ci_upper"] <= 0.),
        "warning_on_time_not_worse": bool(rows.loc["on_time_rate", "ci_lower"] >= 0.),
        "warning_burden_not_worse": bool(rows.loc["early_alarm_burden", "ci_upper"] <= 0.),
    }


def write_readouts(
    predictions: pd.DataFrame, events: pd.DataFrame, cfg: StudyConfig, output: str | Path,
) -> dict[str, Any]:
    """Write outer readouts and gate without selecting from outer observations.

    Args:
        predictions: Matched outer vectors for four arms and selected_pipeline.
        events: Warning events obtained using each fold's inner-selected policy.
        cfg: Frozen study configuration; smoke configurations cannot pass.
        output: Isolated output directory owned by the tree study runner.

    Returns:
        Joint conditional gate decision and separate capped-reference readout.
    """
    expected = set(ARMS) | {"selected_pipeline"}
    if "procedure" not in predictions or set(predictions.procedure) != expected:
        raise ValueError("outer readouts require four arms and inner-selected pipeline")
    tables = _engine_tables(predictions, events, cfg)
    summaries: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for procedure, frame in predictions.groupby("procedure", sort=True):
        scored = _validate_predictions(frame)
        policy_events = events.loc[events.procedure.eq(procedure)]
        summaries.append({"procedure": procedure,
                          **audit._model_readout(scored, cfg.audit_config()),
                          **audit._warning_readout(policy_events)})
        scored["rul_region"] = pd.cut(scored.y_true, [-1, 30, 60, 100, 150, np.inf],
                                       labels=["0-30", "31-60", "61-100", "101-150", ">150"])
        for region, group in scored.groupby("rul_region", observed=False):
            if group.empty:
                continue
            region_rows.append({
                "procedure": procedure, "rul_region": str(region), "n_rows": len(group),
                "n_engines": group.engine_id.nunique(),
                "mean_window_mae": float(group.absolute_error.mean()),
                "mean_engine_mae": float(group.groupby("engine_id").absolute_error.mean().mean()),
                "signed_bias": float((group.y_pred - group.y_true).mean()),
                "rmse": float(np.sqrt(group.squared_error.mean())),
            })
    comparisons = _comparisons_from_tables(tables, cfg)
    selected = comparisons.loc[
        comparisons.reference.eq("hgb_uncapped") & comparisons.procedure.eq("selected_pipeline")
    ]
    controls = _gate_controls(selected)
    capped = comparisons.loc[
        comparisons.reference.eq("hgb_cap125") & comparisons.procedure.eq("selected_pipeline")
    ].set_index("metric")
    capped_controls = {
        "endpoint_mae_not_worse": bool(capped.loc["endpoint_mean_engine_mae", "ci_upper"] <= 0.),
        "endpoint_nasa_not_worse": bool(capped.loc["endpoint_mean_engine_nasa", "ci_upper"] <= 0.),
    }
    scientific = bool(cfg.scientific)
    gate_rmse = selected.loc[selected.metric.eq("trajectory_rmse")].iloc[0]
    decision = {
        "full_protocol_configuration": scientific,
        "promotion_gate_met": bool(scientific and all(controls.values())),
        "controls": controls,
        "outer_trajectory_rmse_delta": float(gate_rmse.delta),
        "outer_trajectory_rmse_ci_lower": float(gate_rmse.ci_lower),
        "outer_trajectory_rmse_ci_upper": float(gate_rmse.ci_upper),
        "capped_reference_endpoint_controls": capped_controls,
        "capped_reference_endpoint_controls_met": bool(all(capped_controls.values())),
        "capped_reference_reporting_only": True,
        "gate": "selected_pipeline vs hgb_uncapped: paired trajectory RMSE upper95% CI<0; "
                "endpoint engine MAE/NASA and early burden upper95% CI<=0; "
                "on-time rate lower95% CI>=0; no test gating",
        "selection_source": "inner grouped CV only; no procedure selection from outer readouts",
        "historical_baseline_overwritten": False,
        "evaluation_limit": "exploratory follow-up using historically inspected engine folds; "
                            "retrospective endpoints; fixed-prediction conditional bootstrap",
        "bootstrap_scope": INTERVAL_SCOPE,
    }
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(destination / "aggregate_metrics.csv", index=False)
    pd.concat([table.assign(procedure=name).reset_index() for name, table in tables.items()],
              ignore_index=True).to_csv(destination / "engine_metrics.csv", index=False)
    comparisons.to_csv(destination / "paired_comparison.csv", index=False)
    pd.DataFrame(region_rows).to_csv(destination / "region_metrics.csv", index=False)
    audit._write_json(destination / "decision.json", decision)
    return decision


