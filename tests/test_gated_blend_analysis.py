"""Selection and paired-inference checks for the gated HGB study."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cmapss.modeling import audit_study as audit
from cmapss.modeling.gated_blend import analysis


def _config(resamples: int = 101) -> SimpleNamespace:
    frozen = audit.AuditStudyConfig(bootstrap_resamples=resamples)
    return SimpleNamespace(
        audit_config=lambda: frozen, bootstrap_resamples=resamples, seed=42,
        endpoint_mae_margin=0.0, nasa_score_margin=0.0,
        warning_rate_margin=0.0, early_alarm_burden_margin=0.0,
        scientific=False,
    )


def _reference() -> pd.Series:
    return pd.Series({
        "trajectory_rmse": 10.0,
        "endpoint_mean_engine_mae": 5.0,
        "endpoint_mean_engine_nasa": 6.0,
        "on_time_rate": 0.9,
        "early_alarm_burden": 0.1,
    })


def _gate_rows(*, rmse: float = 9.0, endpoint_mae: float = 4.0,
               on_time: float = 0.9, burden: float = 0.1) -> pd.DataFrame:
    return pd.DataFrame([{
        "candidate_id": "gate_lo60_hi125_w0.5", "trajectory_rmse": rmse,
        "endpoint_mean_engine_mae": endpoint_mae, "endpoint_mean_engine_nasa": 5.0,
        "on_time_rate": on_time, "early_alarm_burden": burden,
    }])


def test_gate_selection_requires_point_controls_and_strict_improvement():
    gate_rows = _gate_rows()
    gate_rows["family"] = "gated_blend"
    refs = pd.DataFrame([{
        "candidate_id": "hgb_uncapped", "family": "hgb_uncapped",
        **_reference().to_dict(),
    }])
    selected = analysis.select_gate_candidate(pd.concat([refs, gate_rows], ignore_index=True), _reference(), _config())
    assert selected["candidate_id"] == "gate_lo60_hi125_w0.5"
    assert selected["selected_candidate_id"] == "gate_lo60_hi125_w0.5"
    assert selected["diagnostic_numeric_controls_met"] is True
    assert selected["trajectory_improves"] is True

    no_improvement = analysis.select_gate_candidate(_gate_rows(rmse=10.0), _reference(), _config())
    assert no_improvement["selected_candidate_id"] is None
    assert no_improvement["diagnostic_candidate_id"] == "gate_lo60_hi125_w0.5"


def test_gate_selection_falls_back_and_marks_unconstrained_diagnostic():
    rows = _gate_rows(endpoint_mae=5.1)
    result = analysis.select_gate_candidate(rows, _reference(), _config())
    assert result["selected_candidate_id"] is None
    assert result["numeric_controls_met"] is False
    assert result["diagnostic_unconstrained"] is True


def test_choose_procedure_falls_back_to_frozen_reference_policy():
    records = pd.DataFrame([
        {"procedure": "hgb_uncapped", **_reference().to_dict(),
         "numeric_controls_met": True, "policy_threshold": 40, "policy_persistence": 3},
        {"procedure": "hgb_cap125", "trajectory_rmse": 11.0,
         "endpoint_mean_engine_mae": 4.0, "endpoint_mean_engine_nasa": 5.0,
         "on_time_rate": .9, "early_alarm_burden": .1, "numeric_controls_met": True},
        {"procedure": "gated_blend", "trajectory_rmse": 9.0,
         "endpoint_mean_engine_mae": 4.0, "endpoint_mean_engine_nasa": 5.0,
         "on_time_rate": .8, "early_alarm_burden": .1, "numeric_controls_met": True},
    ])
    result = analysis.choose_procedure(records, _config())
    assert result["selected_procedure_id"] == "hgb_uncapped"
    assert result["fallback_used"] is True
    assert result["fixed_fallback_policy"] == {"threshold": 40, "persistence": 3}


def _trajectories(procedure: str, errors: tuple[float, float]) -> pd.DataFrame:
    return pd.DataFrame([
        {"split": "train", "engine_id": engine, "cycle": cycle,
         "y_true": float(lifetime - cycle),
         "y_pred": float(lifetime - cycle + errors[engine - 1]),
         "procedure": procedure}
        for engine, lifetime in ((1, 70), (2, 110))
        for cycle in range(30, lifetime + 1)
    ])


def _events(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        audit.warning_events(frame, audit.WarningPolicy(40, 1), lead_requirement=30)
        .assign(procedure=procedure)
        for procedure, frame in predictions.groupby("procedure")
    ], ignore_index=True)


def test_paired_bootstrap_preserves_duplicate_engine_multiplicity():
    procedures = ["hgb_uncapped", "hgb_cap125", "gated_blend", "selected_pipeline"]
    errors = {"hgb_uncapped": (0., 4.), "hgb_cap125": (1., 3.),
              "gated_blend": (3., 1.), "selected_pipeline": (3., 1.)}
    predictions = pd.concat([_trajectories(name, errors[name]) for name in procedures], ignore_index=True)
    result = analysis.paired_comparisons(predictions, _events(predictions), _config(101))
    row = result.query("reference == 'hgb_uncapped' and procedure == 'gated_blend' and metric == 'trajectory_rmse'").iloc[0]
    indices = np.random.default_rng(42).integers(0, 2, size=(101, 2))
    reference = _trajectories("hgb_uncapped", errors["hgb_uncapped"])
    candidate = _trajectories("gated_blend", errors["gated_blend"])
    draws = []
    for sampled in indices:
        before = pd.concat([reference[reference.engine_id.eq(i + 1)] for i in sampled])
        after = pd.concat([candidate[candidate.engine_id.eq(i + 1)] for i in sampled])
        draws.append(float(np.sqrt(np.mean((after.y_pred - after.y_true) ** 2))
                          - np.sqrt(np.mean((before.y_pred - before.y_true) ** 2))))
    assert row.delta == pytest.approx(draws[0] * 0 + (np.sqrt(np.mean((candidate.y_pred - candidate.y_true) ** 2))
                                                     - np.sqrt(np.mean((reference.y_pred - reference.y_true) ** 2))))
    assert row.ci_lower == pytest.approx(np.quantile(draws, .025))
    assert row.ci_upper == pytest.approx(np.quantile(draws, .975))


def test_paired_comparison_rejects_raw_truth_mismatch():
    reference = _trajectories("hgb_uncapped", (0., 4.))
    candidate = _trajectories("gated_blend", (3., 1.))
    candidate.loc[candidate.engine_id.eq(1), "y_true"] += 1.0
    predictions = pd.concat([
        reference, candidate, _trajectories("hgb_cap125", (1., 3.)),
        _trajectories("selected_pipeline", (3., 1.)),
    ], ignore_index=True)
    events = _events(pd.concat([
        reference, _trajectories("gated_blend", (3., 1.)),
        _trajectories("hgb_cap125", (1., 3.)),
        _trajectories("selected_pipeline", (3., 1.)),
    ], ignore_index=True))
    with pytest.raises(ValueError, match="truth|paired|keys"):
        analysis.paired_comparisons(predictions, events, _config())
