"""Behavioral checks for constrained tree selection and paired engine inference."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cmapss.modeling import audit_study as audit
from cmapss.modeling.tree_comparison import analysis


def _config(resamples: int = 200) -> SimpleNamespace:
    """Build a small inference configuration without fitting an estimator."""
    frozen = audit.AuditStudyConfig(bootstrap_resamples=resamples)
    return SimpleNamespace(
        audit_config=lambda: frozen, bootstrap_resamples=resamples, seed=42,
        endpoint_mae_margin=0.0, nasa_score_margin=0.0,
        warning_rate_margin=0.0, scientific=False,
    )


def _trajectories(procedure: str, errors: tuple[float, float]) -> pd.DataFrame:
    """Use unequal engine lengths to distinguish pooled and equal-engine RMSE."""
    return pd.DataFrame([
        {"split": "train", "engine_id": engine, "cycle": cycle,
         "y_true": float(lifetime - cycle),
         "y_pred": float(lifetime - cycle + errors[engine - 1]),
         "procedure": procedure}
        for engine, lifetime in ((1, 70), (2, 110))
        for cycle in range(30, lifetime + 1)
    ])


def _events(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calibrate no policy: use one explicitly fixed illustrative warning rule."""
    return pd.concat([
        audit.warning_events(frame, audit.WarningPolicy(30, 1), lead_requirement=30)
        .assign(procedure=procedure)
        for procedure, frame in predictions.groupby("procedure")
    ], ignore_index=True)


def test_xgb_selection_uses_trajectory_rmse_subject_to_endpoint_controls() -> None:
    """A better endpoint alone cannot displace a lower feasible trajectory RMSE."""
    rows = pd.DataFrame([
        {"candidate_id": "xgb_uncapped_endpoint", "family": "xgb_uncapped",
         "trajectory_rmse": 20., "endpoint_mean_engine_mae": 4.,
         "endpoint_mean_engine_nasa": 5.},
        {"candidate_id": "xgb_uncapped_trajectory", "family": "xgb_uncapped",
         "trajectory_rmse": 10., "endpoint_mean_engine_mae": 5.,
         "endpoint_mean_engine_nasa": 6.},
        {"candidate_id": "xgb_uncapped_unsafe", "family": "xgb_uncapped",
         "trajectory_rmse": 1., "endpoint_mean_engine_mae": 6.1,
         "endpoint_mean_engine_nasa": 6.},
    ])
    reference = pd.Series({"endpoint_mean_engine_mae": 6.,
                           "endpoint_mean_engine_nasa": 7.})
    result = analysis.select_xgb_candidate(rows, reference, "xgb_uncapped", _config())
    assert result["candidate_id"] == "xgb_uncapped_trajectory"
    assert result["numeric_controls_met"] is True
    reference["endpoint_mean_engine_mae"] = 1.
    fallback = analysis.select_xgb_candidate(rows, reference, "xgb_uncapped", _config())
    assert fallback["candidate_id"] == "xgb_uncapped_unsafe"
    assert fallback["numeric_controls_met"] is False


def test_procedure_falls_back_when_warning_or_family_control_fails() -> None:
    """Selection cannot promote a fast candidate with worse warning timeliness."""
    rows = pd.DataFrame([
        {"procedure": "hgb_uncapped", "trajectory_rmse": 20.,
         "endpoint_mean_engine_mae": 8., "endpoint_mean_engine_nasa": 10.,
         "on_time_rate": .9, "early_alarm_burden": .1, "numeric_controls_met": True},
        {"procedure": "hgb_cap125", "trajectory_rmse": 30.,
         "endpoint_mean_engine_mae": 6., "endpoint_mean_engine_nasa": 8.,
         "on_time_rate": .9, "early_alarm_burden": .08, "numeric_controls_met": True},
        {"procedure": "xgb_uncapped", "trajectory_rmse": 5.,
         "endpoint_mean_engine_mae": 7., "endpoint_mean_engine_nasa": 9.,
         "on_time_rate": .8, "early_alarm_burden": .05, "numeric_controls_met": True},
        {"procedure": "xgb_cap125", "trajectory_rmse": 6.,
         "endpoint_mean_engine_mae": 7., "endpoint_mean_engine_nasa": 9.,
         "on_time_rate": .9, "early_alarm_burden": .05, "numeric_controls_met": False},
    ])
    assert analysis.choose_procedure(rows, _config())["procedure_id"] == "hgb_uncapped"
    rows.loc[rows.procedure.eq("xgb_uncapped"), "on_time_rate"] = .9
    assert analysis.choose_procedure(rows, _config())["procedure_id"] == "xgb_uncapped"


def test_paired_rmse_bootstrap_preserves_duplicate_engine_multiplicity() -> None:
    """Compare against explicit row concatenation for every sampled engine draw."""
    reference = _trajectories("hgb_uncapped", (0., 4.))
    candidate = _trajectories("xgb_uncapped", (3., 1.))
    predictions = pd.concat([reference, candidate], ignore_index=True)
    cfg = _config(101)
    results = analysis.paired_comparisons(predictions, _events(predictions), cfg)
    row = results.query("procedure == 'xgb_uncapped' and metric == 'trajectory_rmse'").iloc[0]
    indices = np.random.default_rng(42).integers(0, 2, size=(101, 2))
    draws = []
    for sampled in indices:
        before = pd.concat([reference[reference.engine_id.eq(i + 1)] for i in sampled])
        after = pd.concat([candidate[candidate.engine_id.eq(i + 1)] for i in sampled])
        before_rmse = np.sqrt(np.mean((before.y_pred - before.y_true) ** 2))
        after_rmse = np.sqrt(np.mean((after.y_pred - after.y_true) ** 2))
        draws.append(after_rmse - before_rmse)
    expected_delta = (np.sqrt(np.mean((candidate.y_pred - candidate.y_true) ** 2))
                      - np.sqrt(np.mean((reference.y_pred - reference.y_true) ** 2)))
    assert row.delta == pytest.approx(expected_delta)
    assert row.ci_lower == pytest.approx(np.quantile(draws, .025))
    assert row.ci_upper == pytest.approx(np.quantile(draws, .975))


@pytest.mark.parametrize("damage", ["engine_id", "truth", "nonfinite", "duplicate"])
def test_paired_comparison_rejects_unmatched_or_invalid_vectors(damage: str) -> None:
    """Every procedure must describe the same finite raw-truth observations."""
    reference = _trajectories("hgb_uncapped", (0., 4.))
    candidate = _trajectories("xgb_uncapped", (3., 1.))
    if damage == "engine_id":
        candidate.loc[candidate.engine_id.eq(2), "engine_id"] = 3
    elif damage == "truth":
        candidate.loc[0, "y_true"] += 1.
    elif damage == "nonfinite":
        candidate.loc[0, "y_pred"] = np.inf
    else:
        candidate = pd.concat([candidate, candidate.iloc[[0]]], ignore_index=True)
    predictions = pd.concat([reference, candidate], ignore_index=True)
    valid_events = _events(pd.concat([reference, _trajectories("xgb_uncapped", (3., 1.))]))
    with pytest.raises(ValueError):
        analysis.paired_comparisons(predictions, valid_events, _config())


def test_candidate_selection_rejects_nonfinite_metrics() -> None:
    """An invalid screening result cannot silently win or disappear."""
    rows = pd.DataFrame([{"candidate_id": "xgb_uncapped_1", "family": "xgb_uncapped",
                          "trajectory_rmse": np.nan, "endpoint_mean_engine_mae": 3.,
                          "endpoint_mean_engine_nasa": 4.}])
    reference = pd.Series({"endpoint_mean_engine_mae": 6.,
                           "endpoint_mean_engine_nasa": 7.})
    with pytest.raises(ValueError):
        analysis.select_xgb_candidate(rows, reference, "xgb_uncapped", _config())
