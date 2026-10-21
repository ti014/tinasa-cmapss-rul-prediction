"""Selection and paired-readout contracts for the multi-scale HGB study."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cmapss.modeling import audit_study as audit
from cmapss.modeling.multiscale_comparison import analysis
from cmapss.modeling.multiscale_comparison.config import StudyConfig


def _config(resamples: int = 101) -> StudyConfig:
    """Use a small bootstrap only for fast engineering tests."""
    return replace(StudyConfig(), outer_splits=2, inner_splits=2,
                   bootstrap_resamples=resamples)


def _records(**overrides: float) -> pd.DataFrame:
    """Create three finite inner arm summaries with explicit safeguards."""
    rows = [
        {"procedure": "hgb_baseline169", "trajectory_rmse": 20.0,
         "endpoint_mean_engine_mae": 8.0, "endpoint_mean_engine_nasa": 10.0,
         "on_time_rate": .90, "early_alarm_burden": .10},
        {"procedure": "hgb_multiscale10_30", "trajectory_rmse": 19.0,
         "endpoint_mean_engine_mae": 8.0, "endpoint_mean_engine_nasa": 10.0,
         "on_time_rate": .90, "early_alarm_burden": .10},
        {"procedure": "hgb_initial_profile42", "trajectory_rmse": 18.0,
         "endpoint_mean_engine_mae": 8.1, "endpoint_mean_engine_nasa": 10.0,
         "on_time_rate": .90, "early_alarm_burden": .10},
    ]
    result = pd.DataFrame(rows)
    for column, value in overrides.items():
        result.loc[result.procedure.eq("hgb_multiscale10_30"), column] = value
    return result


def _trajectories(procedure: str, errors: tuple[float, float]) -> pd.DataFrame:
    """Build complete, unequal-length train trajectories."""
    rows = []
    for engine, lifetime in ((1, 70), (2, 110)):
        for cycle in range(30, lifetime + 1):
            truth = float(lifetime - cycle)
            rows.append({
                "split": "train", "engine_id": engine, "cycle": cycle,
                "y_true": truth, "y_pred": truth + errors[engine - 1],
                "procedure": procedure,
            })
    return pd.DataFrame(rows)


def _events(predictions: pd.DataFrame) -> pd.DataFrame:
    """Apply one fixed warning rule to each procedure's complete trajectories."""
    return pd.concat([
        audit.warning_events(frame, audit.WarningPolicy(30, 1), lead_requirement=30)
        .assign(procedure=procedure)
        for procedure, frame in predictions.groupby("procedure", sort=True)
    ], ignore_index=True)


def test_choose_procedure_requires_strict_rmse_and_all_controls() -> None:
    """A point endpoint improvement alone cannot promote an arm."""
    chosen = analysis.choose_procedure(_records(), _config())
    assert chosen["selected_procedure_id"] == "hgb_multiscale10_30"
    assert chosen["fallback_used"] is False
    assert chosen["arm_controls"]["hgb_initial_profile42"]["eligible"] is False
    assert chosen["arm_controls"]["hgb_initial_profile42"]["endpoint_controls_met"] is False

    rejected = analysis.choose_procedure(
        _records(trajectory_rmse=19.0, on_time_rate=.89), _config()
    )
    assert rejected["selected_procedure_id"] == "hgb_baseline169"
    assert rejected["fallback_used"] is True


def test_choose_procedure_validates_exact_arm_panel() -> None:
    """Missing or duplicated procedure rows are rejected before selection."""
    with pytest.raises(ValueError, match="đúng ba"):
        analysis.choose_procedure(_records().iloc[:2], _config())
    duplicate = pd.concat([_records(), _records().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duy nhất"):
        analysis.choose_procedure(duplicate, _config())


def test_candidate_readout_and_policy_use_raw_rul() -> None:
    """The shared audit readout infers lifetime from raw labels and calibrates policy."""
    predictions = _trajectories("hgb_baseline169", (0.0, 2.0))
    readout = analysis.candidate_readout(predictions, _config())
    assert readout["trajectory_rmse"] > 0.0
    policy, grid = analysis.calibrate_policy(predictions, _config())
    assert isinstance(policy, audit.WarningPolicy)
    assert len(grid) == 15
    assert grid.loc[grid.selected].shape[0] == 1


def test_paired_comparison_uses_baseline_and_preserves_duplicate_bootstrap_multiplicity() -> None:
    """The paired table contains each declared arm and pooled RMSE statistics."""
    procedures = analysis.READOUT_ARMS
    errors = {
        "hgb_baseline169": (0.0, 4.0),
        "hgb_multiscale10_30": (3.0, 1.0),
        "hgb_initial_profile42": (2.0, 2.0),
        "selected_pipeline": (3.0, 1.0),
    }
    predictions = pd.concat([
        _trajectories(procedure, errors[procedure]) for procedure in procedures
    ], ignore_index=True)
    result = analysis.paired_comparisons(predictions, _events(predictions), _config())
    assert set(result.reference) == {"hgb_baseline169"}
    assert set(result.procedure) == set(procedures[1:])
    row = result.query(
        "procedure == 'hgb_multiscale10_30' and metric == 'trajectory_rmse'"
    ).iloc[0]
    reference = predictions[predictions.procedure.eq("hgb_baseline169")]
    candidate = predictions[predictions.procedure.eq("hgb_multiscale10_30")]
    expected = np.sqrt(np.mean((candidate.y_pred - candidate.y_true) ** 2)) - np.sqrt(
        np.mean((reference.y_pred - reference.y_true) ** 2)
    )
    assert row.delta == pytest.approx(expected)
    assert row.n_engines == 2 and row.resamples == 101


def test_paired_comparison_rejects_truth_misalignment() -> None:
    """A candidate with a changed raw truth vector cannot be compared."""
    baseline = _trajectories("hgb_baseline169", (0.0, 1.0))
    candidate = _trajectories("hgb_multiscale10_30", (0.0, 1.0))
    predictions = pd.concat([
        baseline,
        candidate,
        _trajectories("hgb_initial_profile42", (0.0, 1.0)),
        _trajectories("selected_pipeline", (0.0, 1.0)),
    ], ignore_index=True)
    events = _events(predictions)
    predictions.loc[
        predictions.procedure.eq("hgb_multiscale10_30") & (predictions.index == 122),
        "y_true",
    ] += 1.0
    with pytest.raises(ValueError, match="nhãn"):
        analysis.paired_comparisons(predictions, events, _config())


def test_write_readouts_writes_decision_and_region_panels(tmp_path) -> None:
    """Outer readouts include every arm, the selected alias and promotion controls."""
    errors = {procedure: (0.0, 1.0) for procedure in analysis.READOUT_ARMS}
    predictions = pd.concat([
        _trajectories(procedure, errors[procedure])
        for procedure in analysis.READOUT_ARMS
    ], ignore_index=True)
    decision = analysis.write_readouts(predictions, _events(predictions), _config(), tmp_path)
    assert decision["promotion_gate_met"] is False
    assert decision["selected_reference"] == "hgb_baseline169"
    for name in (
        "aggregate_metrics.csv", "engine_metrics.csv", "paired_comparison.csv",
        "region_metrics.csv", "warning_metrics.csv", "decision.json",
    ):
        assert (tmp_path / name).is_file()
    aggregate = pd.read_csv(tmp_path / "aggregate_metrics.csv")
    assert set(aggregate.procedure) == set(analysis.READOUT_ARMS)
