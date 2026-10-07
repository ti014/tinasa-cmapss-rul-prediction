"""Independent audit calculator checks for the frozen smooth HGB blend."""

from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_gated_blend_study.py"
    spec = importlib.util.spec_from_file_location("gated_blend_independent_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFY = _module()


def test_prediction_only_smooth_gate_extremes_and_midpoint():
    old = np.array([-1e300, 60., 105., 150., 1e300])
    cap = np.array([50., 50., 85., 120., 100.])
    predicted, weights = VERIFY._gate(old, cap, {
        "lower_rul": 60., "upper_rul": 150., "max_capped_weight": .5,
    })
    np.testing.assert_array_equal(weights, [.5, .5, .25, 0., 0.])
    np.testing.assert_array_equal(predicted[1:4], [55., 100., 150.])
    assert np.isfinite(predicted).all()
    # No labels are accepted by the calculator; changing truth cannot enter it.
    assert predicted[-1] == old[-1]


def test_warning_policy_uses_complete_past_and_counts_retractions():
    frame = pd.DataFrame({"split": "train", "engine_id": 1,
                          "cycle": np.arange(30, 40), "y_true": np.arange(9, -1, -1),
                          "y_pred": [4, 4, 4, 6, 4, 4, 4, 4, 6, 4]})
    events = VERIFY._warning_events(frame, {"threshold": 5, "persistence": 3}, 5)
    row = events.iloc[0]
    assert row.first_warning_cycle == 32
    assert row.lead_time_cycles == 7
    assert row.on_time
    assert row.n_alarm_rows == 3
    assert row.n_transitions == 4
    assert row.n_retractions == 2
    assert row.early_alarm_burden == .25
    with pytest.raises(ValueError, match="contiguous"):
        VERIFY._warning_events(frame.drop(index=3), {"threshold": 5, "persistence": 3}, 5)


def test_endpoint_panel_clamps_available_prefix_and_uses_raw_lifetime():
    frame = pd.DataFrame({"split": "train", "engine_id": 1,
                          "cycle": np.arange(30, 51), "y_true": np.arange(20, -1, -1),
                          "y_pred": 10.})
    panel = VERIFY._endpoints(frame, [.5, .7, .9])
    assert panel.cycle.tolist() == [30, 35, 45]
    assert panel.requested_cycle.tolist() == [25, 35, 45]
    assert panel.origin_clamped_to_available_prefix.tolist() == [True, False, False]
    frame.loc[0, "y_true"] = 125
    with pytest.raises(ValueError, match="lifetime"):
        VERIFY._endpoints(frame, [.5])


def test_bootstrap_pools_rows_and_preserves_duplicate_engine_multiplicity():
    before = pd.DataFrame(index=[1, 2, 3])
    after = pd.DataFrame(index=[1, 2, 3])
    for metric in VERIFY.ENGINE_METRICS:
        before[metric] = [1., 2., 3.]
        after[metric] = [2., 2., 2.]
    before["trajectory_sse"], after["trajectory_sse"] = [4., 90., 200.], [1., 100., 180.]
    before["trajectory_n_rows"] = after["trajectory_n_rows"] = [1, 10, 100]
    cfg = {"seed": 42, "bootstrap_resamples": 1000}
    result = VERIFY._paired(before, after, cfg).set_index("metric")
    indices = np.random.default_rng(42).integers(0, 3, size=(1000, 3))
    draws = []
    for sampled in indices:
        old, new = before.iloc[sampled], after.iloc[sampled]
        draws.append(np.sqrt(new.trajectory_sse.sum() / new.trajectory_n_rows.sum())
                     - np.sqrt(old.trajectory_sse.sum() / old.trajectory_n_rows.sum()))
    row = result.loc["trajectory_rmse"]
    assert row.baseline == pytest.approx(np.sqrt(294. / 111.))
    assert row.ci_lower == pytest.approx(np.quantile(draws, .025))
    assert row.ci_upper == pytest.approx(np.quantile(draws, .975))
    wrong = after.iloc[::-1]
    with pytest.raises(ValueError, match="indices"):
        VERIFY._paired(before, wrong, cfg)


def test_repository_resolution_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        VERIFY._resolve(tmp_path, "../elsewhere")


def test_blend_truth_is_passthrough_and_alignment_is_exact():
    old = pd.DataFrame({"split": ["train", "train"], "engine_id": [1, 2],
                        "cycle": [30, 30], "y_true": [50., 70.], "y_pred": [60., 105.]})
    cap = old.assign(y_pred=[50., 85.]).iloc[::-1]
    spec = {"lower_rul": 60., "upper_rul": 150., "max_capped_weight": .5}
    first = VERIFY._blend_frame(old, cap, spec, "candidate")
    second = VERIFY._blend_frame(old.assign(y_true=[200., 10.]),
                                 cap.assign(y_true=[10., 200.]), spec, "candidate")
    np.testing.assert_array_equal(first.y_pred, second.y_pred)
    np.testing.assert_array_equal(first.capped_weight, second.capped_weight)
    with pytest.raises(ValueError, match="align"):
        VERIFY._blend_frame(old, cap.iloc[:1], spec, "candidate")


def test_inner_choice_respects_warning_feasibility_and_strict_improvement():
    baseline = pd.Series({"trajectory_rmse": 10., "endpoint_mean_engine_mae": 5.,
                          "endpoint_mean_engine_nasa": 2., "on_time_rate": .9,
                          "early_alarm_burden": .1})
    rows = pd.DataFrame([
        {"candidate_id": "bad_warning", **baseline.to_dict(), "trajectory_rmse": 1., "on_time_rate": .8},
        {"candidate_id": "feasible", **baseline.to_dict(), "trajectory_rmse": 9.},
    ])
    gate, selected, controls = VERIFY._winner(rows, baseline)
    assert gate == "feasible" and selected == "gated_blend"
    assert not controls["bad_warning"]["eligible"]
    rows.loc[1, "trajectory_rmse"] = 10.
    gate, selected, _ = VERIFY._winner(rows, baseline)
    assert gate == "feasible" and selected == "hgb_uncapped"
    rows.loc[1, "endpoint_mean_engine_nasa"] = 3.
    gate, selected, _ = VERIFY._winner(rows, baseline)
    assert gate == "bad_warning" and selected == "hgb_uncapped"


def test_verification_never_overwrites_an_existing_receipt(tmp_path):
    run = tmp_path / "outputs/experiments/gated_blend_comparison/example"
    run.mkdir(parents=True)
    receipt = run / "verification.json"
    receipt.write_text('{"preserved": true}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        VERIFY.verify(run, root=tmp_path)
    assert receipt.read_text(encoding="utf-8") == '{"preserved": true}\n'


def test_flat_snapshots_bind_full_original_paths_and_reject_swaps():
    names = ["outputs/first/source_snapshot/a/model.json", "outputs/second/source_snapshot/a/model.json"]
    artifacts = dict.fromkeys(names, "same-content-digest")
    mapped = {name: hashlib.sha256(name.encode()).hexdigest()[:24] + ".json" for name in names}
    assert VERIFY._snapshot_mapping_valid(artifacts, mapped)
    swapped = dict(zip(names, reversed(list(mapped.values()))))
    assert not VERIFY._snapshot_mapping_valid(artifacts, swapped)
    assert not VERIFY._snapshot_mapping_valid(artifacts, {names[0]: mapped[names[0]]})


def test_error_recomputation_preserves_source_optional_column_schema():
    frame = pd.DataFrame({"split": ["train"], "engine_id": [1], "cycle": [30],
                          "y_true": [20.], "y_pred": [17.], "absolute_error": [3.],
                          "squared_error": [9.], "nasa_score_component": [np.expm1(3. / 13.)]})
    scored = VERIFY._score(frame)
    assert scored.columns.tolist() == frame.columns.tolist()
    assert VERIFY._same_frame(frame, scored, ["split", "engine_id", "cycle"])
    optional = VERIFY._score(frame.assign(signed_error=999.))
    assert optional.signed_error.tolist() == [-3.]
