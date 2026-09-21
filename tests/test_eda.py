"""Ví dụ tính tay kiểm tra thống kê EDA, ties và đơn vị engine."""

import numpy as np
import pandas as pd
import pytest

from cmapss.eda import describe_measurements, select_example_engines, sensor_rul_correlations


def test_summary_matches_hand_calculation() -> None:
    data = pd.DataFrame({"split": ["train"] * 4, "varying": [1, 2, 3, 4], "constant": [7] * 4})
    summary = describe_measurements(data, ["varying", "constant"]).set_index("measurement")
    assert summary.loc["varying", "mean"] == 2.5
    assert summary.loc["varying", "q25"] == 1.75
    assert summary.loc["varying", "q75"] == 3.25
    assert summary.loc["varying", "std"] == pytest.approx(np.sqrt(5 / 3))
    assert summary.loc["constant", "is_constant"]
    assert summary.loc["constant", "range"] == 0


def test_examples_use_lifetime_distance_then_smallest_id() -> None:
    lifetimes = pd.DataFrame({"engine_id": [8, 3, 2, 1], "last_cycle": [20, 10, 30, 20]})
    examples = select_example_engines(lifetimes)
    assert examples["target_lifetime"].tolist() == [17.5, 20.0, 22.5]
    assert examples["engine_id"].tolist() == [1, 1, 1]
    pd.testing.assert_frame_equal(examples, select_example_engines(lifetimes.iloc[::-1]))


def test_correlations_keep_engines_separate_and_constants_undefined() -> None:
    data = pd.DataFrame({
        "split": ["train"] * 7,
        "engine_id": [1] * 3 + [2] * 4,
        "rul": [2, 1, 0, 3, 2, 1, 0],
        "sensor": [1, 2, 3, 4, 3, 2, 1],
        "constant": [7] * 7,
    })
    results = sensor_rul_correlations(data, ["sensor", "constant"])
    varying = results[results.sensor.eq("sensor")]
    assert varying["rho"].tolist() == pytest.approx([-1, 1])
    assert varying["n_observations"].tolist() == [3, 4]
    constant = results[results.sensor.eq("constant")]
    assert constant["rho"].isna().all()
    assert not constant["defined"].any()


def test_spearman_uses_average_ranks_for_ties() -> None:
    data = pd.DataFrame({
        "split": ["train"] * 4, "engine_id": [1] * 4,
        "rul": [3, 2, 1, 0], "sensor": [1, 1, 2, 3],
    })
    # Sensor ranks [1.5, 1.5, 3, 4], RUL ranks [4, 3, 2, 1].
    rho = sensor_rul_correlations(data, ["sensor"])["rho"].iloc[0]
    assert rho == pytest.approx(-4.5 / np.sqrt(4.5 * 5))


@pytest.mark.parametrize("split", ["test", None])
def test_analysis_rejects_non_train(split: str | None) -> None:
    data = pd.DataFrame({"split": [split], "engine_id": [1], "rul": [0], "sensor": [1]})
    with pytest.raises(ValueError, match="chỉ nhận dữ liệu train"):
        describe_measurements(data, ["sensor"])
    with pytest.raises(ValueError, match="chỉ nhận dữ liệu train"):
        sensor_rul_correlations(data, ["sensor"])
