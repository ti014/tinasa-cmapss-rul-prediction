"""Causal gates, exact engine-cycle pairing and preserved prediction scale."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from cmapss.modeling.gated_blend import (
    GateSpec, blend_predictions, blend_vectors, build_specs, capped_weights,
)


@pytest.fixture
def paired_frames():
    uncapped = pd.DataFrame({
        "split": ["train"] * 4, "engine_id": [1, 1, 2, 2],
        "cycle": [30, 31, 30, 31], "inner_fold": [1, 1, 2, 2],
        "y_true": [160.0, 159.0, 90.0, 89.0],
        "y_pred": [180.0, 150.0, 105.0, 60.0],
        "squared_error": [999.0] * 4,
    })
    capped = uncapped.drop(columns="squared_error").assign(y_pred=[120., 125., 95., 50.])
    return uncapped, capped


def test_gate_boundaries_weight_monotonicity_and_convex_bounds():
    spec = GateSpec(60, 150, 0.5)
    uncapped = np.array([-1e308, -10, 60, 105, 150, 200, 1e308])
    weights = capped_weights(uncapped, spec)
    np.testing.assert_array_equal(weights, [0.5, 0.5, 0.5, 0.25, 0., 0., 0.])
    assert (np.diff(weights) <= 0).all()
    capped = np.array([100., -20., 50., 150., 125., 250., 1.])
    predicted = blend_vectors(uncapped, capped, spec)
    assert np.isfinite(predicted).all()
    assert (predicted >= np.minimum(uncapped, capped)).all()
    assert (predicted <= np.maximum(uncapped, capped)).all()
    # Negative and high predictions remain on their original scale.
    assert predicted[1] == -15.
    assert predicted[-1] == 1e308


def test_gate_full_weight_can_use_capped_and_preserves_equal_predictions():
    spec = GateSpec(30, 150, 1.)
    np.testing.assert_array_equal(blend_vectors([20., 150.], [10., 125.], spec), [10., 150.])
    values = np.arange(-10, 210, dtype=float)
    np.testing.assert_allclose(blend_vectors(values, values, spec), values, rtol=1e-15, atol=1e-14)


def test_current_row_is_independent_of_future_rows_and_truth(paired_frames):
    spec = GateSpec(60, 150, 0.5)
    uncapped, capped = paired_frames
    full = blend_predictions(uncapped, capped, spec)
    prefix = blend_predictions(uncapped.iloc[:2], capped.iloc[:2], spec)
    pd.testing.assert_frame_equal(prefix, full.iloc[:2].reset_index(drop=True), check_exact=True)
    altered_truth = blend_predictions(
        uncapped.assign(y_true=uncapped.y_true + 1000),
        capped.assign(y_true=capped.y_true + 1000), spec,
    )
    np.testing.assert_array_equal(altered_truth.y_pred, full.y_pred)
    np.testing.assert_array_equal(altered_truth.capped_weight, full.capped_weight)
    inference = blend_predictions(uncapped.drop(columns="y_true"), capped.drop(columns="y_true"), spec)
    assert "y_true" not in inference
    np.testing.assert_array_equal(inference.y_pred, full.y_pred)


def test_pairing_uses_exact_keys_and_drops_stale_errors(paired_frames):
    uncapped, capped = paired_frames
    before = uncapped.copy(deep=True)
    spec = GateSpec(60, 150, .5)
    expected = blend_predictions(uncapped, capped, spec)
    actual = blend_predictions(uncapped.iloc[::-1], capped.sample(frac=1, random_state=42), spec)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert "squared_error" not in actual
    np.testing.assert_array_equal(actual.y_pred, [180., 150., 102.5, 55.])
    pd.testing.assert_frame_equal(uncapped, before, check_exact=True)


@pytest.mark.parametrize("change", ["missing", "duplicate", "key", "truth", "fold", "missing_fold"])
def test_mismatched_pairs_are_rejected(paired_frames, change):
    uncapped, capped = paired_frames
    if change == "missing":
        capped = capped.iloc[1:]
    elif change == "duplicate":
        capped = pd.concat([capped, capped.iloc[:1]], ignore_index=True)
    elif change == "key":
        capped.loc[0, "cycle"] += 1
    elif change == "truth":
        capped.loc[0, "y_true"] += .01
    elif change == "fold":
        capped.loc[0, "inner_fold"] = 3
    else:
        capped = capped.drop(columns="inner_fold")
    with pytest.raises(ValueError, match="exact|unique|both"):
        blend_predictions(uncapped, capped, GateSpec(60, 150, .5))


@pytest.mark.parametrize("values", [[], [[1., 2.]], [True], ["1"], [np.nan], [np.inf], [1 + 2j]])
def test_malformed_prediction_vectors_are_rejected(values):
    with pytest.raises(ValueError, match="vector|finite"):
        capped_weights(values, GateSpec(30, 150, .5))


def test_shape_mismatch_and_wrong_spec_are_rejected():
    with pytest.raises(ValueError, match="shapes"):
        blend_vectors([1., 2.], [1.], GateSpec(30, 150, .5))
    with pytest.raises(TypeError, match="GateSpec"):
        capped_weights([1.], None)


@pytest.mark.parametrize("parameters", [
    (-1, 150, .5), (150, 150, .5), (160, 150, .5),
    (30, 150, 0), (30, 150, 1.1), (30, np.inf, .5),
    (30, 150, np.nan), (True, 150, .5), (30, 150, True),
])
def test_invalid_gate_specs_are_rejected(parameters):
    with pytest.raises(ValueError):
        GateSpec(*parameters)


def test_explicit_grid_is_frozen_and_rejects_duplicate_budget():
    specs = build_specs(lower_ruls=(30, 60), upper_ruls=(150,), max_capped_weights=(.25, .5))
    assert len(specs) == len({spec.candidate_id for spec in specs}) == 4
    assert specs[0].candidate_id == "gate_lo30_hi150_w0.25"
    with pytest.raises(FrozenInstanceError):
        specs[0].upper_rul = 200
    with pytest.raises(ValueError, match="duplicate"):
        build_specs(lower_ruls=(30, 30), upper_ruls=(150,), max_capped_weights=(.5,))
    with pytest.raises(ValueError, match="nonempty"):
        build_specs(lower_ruls=(), upper_ruls=(150,), max_capped_weights=(.5,))

