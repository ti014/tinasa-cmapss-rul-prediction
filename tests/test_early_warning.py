"""Boundary proofs for the fixed-threshold, complete-train warning protocol."""

import numpy as np
import pandas as pd
import pytest

from cmapss.modeling.early_warning import build_warning_events


def trajectory(predictions, engine=1):
    return pd.DataFrame({
        "split": "train", "engine_id": engine, "model_name": "model",
        "cycle": np.arange(1, 6), "y_true": [4., 3., 2., 1., 0.],
        "y_pred": np.asarray(predictions, dtype=float),
    })


@pytest.mark.parametrize("predicted,timing,lead,delay", [
    ([3, 9, 9, 9, 9], "before_threshold", 4, -1),
    ([9, 3, 9, 9, 9], "at_threshold", 3, 0),
    ([9, 9, 9, 3, 9], "after_threshold", 1, 2),
    ([9, 9, 9, 9, 3], "at_failure", 0, 3),
    ([9, 9, 9, 9, 9], "no_warning", None, None),
])
def test_warning_timing_boundaries(predicted, timing, lead, delay):
    events, summary = build_warning_events(trajectory(predicted), model_name="model", rul_threshold=3)
    event = events.iloc[0]
    assert event["warning_timing"] == timing
    if lead is None:
        assert pd.isna(event["lead_time_to_failure_cycles"])
        assert pd.isna(event["threshold_delay_cycles"])
        assert summary["lead_time_distribution"]["median"] is None
    else:
        assert event["lead_time_to_failure_cycles"] == lead
        assert event["threshold_delay_cycles"] == delay
    assert summary["timing_counts"][timing] == 1
    assert summary["false_alarm_rate"] is None
    assert summary["n_negative_engines"] == 0


def test_engine_summary_preserves_equal_weight_and_row_order_independence():
    frame = pd.concat([trajectory([3, 9, 9, 9, 9]), trajectory([9, 9, 9, 3, 9], 2)])
    events, summary = build_warning_events(frame.sample(frac=1, random_state=42), model_name="model", rul_threshold=3)
    assert events.engine_id.tolist() == [1, 2]
    assert summary["mean_lead_time_cycles"] == 2.5
    assert summary["lead_time_distribution"] == {"n": 2, "min": 1., "q25": 1.75, "median": 2.5, "q75": 3.25, "max": 4.}


@pytest.mark.parametrize("violation", ["duplicate", "nan", "infinite", "incomplete", "wrong_rul", "fractional_rul", "gap", "test"])
def test_warning_rejects_invalid_trajectory(violation):
    frame = trajectory([4, 3, 2, 1, 0])
    if violation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif violation in {"nan", "infinite"}:
        frame.loc[0, "y_pred"] = np.nan if violation == "nan" else np.inf
    elif violation == "incomplete":
        frame = frame.iloc[:-1]
    elif violation == "wrong_rul":
        frame.loc[0, "y_true"] = 99
    elif violation == "fractional_rul":
        frame.loc[0, "y_true"] = 4.5
    elif violation == "gap":
        frame = frame.drop(index=2)
    else:
        frame["split"] = "test"
    with pytest.raises(ValueError, match="WARNING_INPUT|WARNING_TRAJECTORY"):
        build_warning_events(frame, model_name="model", rul_threshold=3)


@pytest.mark.parametrize("threshold", [-1, np.nan, np.inf, 1.5])
def test_warning_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="WARNING_CONFIG"):
        build_warning_events(trajectory([4, 3, 2, 1, 0]), model_name="model", rul_threshold=threshold)


def test_zero_threshold_keeps_failure_warning_separate():
    events, _ = build_warning_events(trajectory([4, 3, 2, 1, 0]), model_name="model", rul_threshold=0)
    assert events.iloc[0]["warning_timing"] == "at_failure"
    assert events.iloc[0]["lead_time_to_failure_cycles"] == 0


def test_missing_engine_id_does_not_silently_drop_rows():
    frame = trajectory([4, 3, 2, 1, 0])
    frame["engine_id"] = np.nan
    with pytest.raises(ValueError, match="WARNING_INPUT"):
        build_warning_events(frame, model_name="model", rul_threshold=3)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="WARNING_INPUT"):
        build_warning_events(trajectory([4, 3, 2, 1, 0]), model_name="unknown", rul_threshold=3)
