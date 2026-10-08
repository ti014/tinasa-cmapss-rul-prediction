"""Kiểm thử rolling features."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import MEASUREMENTS, OBSERVATION_COLUMNS, ContractError
from cmapss.etl.pandas_pipeline import run_pandas_etl
from cmapss.features.contracts import ROLLING_FEATURE_COLUMNS
from cmapss.features.rolling import (
    RollingFeatureResult,
    build_rolling_features,
    validate_rolling_features,
    write_rolling_feature_artifacts,
)


def _observations_from_specs(specs: list[tuple[str, int, int, float]]) -> pd.DataFrame:
    rows = []
    for split, engine_id, n_cycles, offset in specs:
        for cycle in range(1, n_cycles + 1):
            values = {measurement: float(cycle + offset) for measurement in MEASUREMENTS}
            values["sensor_2"] = 5.0
            rows.append([split, engine_id, cycle, *[values[column] for column in MEASUREMENTS]])
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).astype(
        {
            "split": "string",
            "engine_id": "int64",
            "cycle": "int64",
            **dict.fromkeys(MEASUREMENTS, "float64"),
        }
    )


def test_rolling_features_match_hand_calculated_window_values() -> None:
    observations = _observations_from_specs([("train", 1, 31, 0.0)])
    features = build_rolling_features(observations)

    assert tuple(features.columns) == ROLLING_FEATURE_COLUMNS
    assert features[["split", "engine_id", "cycle"]].to_dict("records") == [
        {"split": "train", "engine_id": 1, "cycle": 30},
        {"split": "train", "engine_id": 1, "cycle": 31},
    ]
    first = features.iloc[0]
    assert first["sensor_1__last"] == 30.0
    assert first["sensor_1__mean"] == 15.5
    assert first["sensor_1__std"] == pytest.approx(np.std(np.arange(1.0, 31.0), ddof=1))
    assert first["sensor_1__min"] == 1.0
    assert first["sensor_1__max"] == 30.0
    assert first["sensor_1__slope"] == pytest.approx(1.0)
    assert first["sensor_2__std"] == 0.0
    assert first["sensor_2__slope"] == 0.0


@pytest.mark.parametrize("window", [10, 20, 30])
def test_rolling_features_do_not_use_future_observations(window) -> None:
    observations = _observations_from_specs([("train", 1, 31, 0.0)])
    baseline = build_rolling_features(observations, window_cycles=window)

    changed = observations.copy()
    changed.loc[changed["cycle"].eq(31), "sensor_1"] = 10_000.0
    candidate = build_rolling_features(changed, window_cycles=window)

    row_30_columns = ["sensor_1__last", "sensor_1__mean", "sensor_1__std", "sensor_1__slope"]
    pd.testing.assert_series_equal(
        baseline.loc[baseline["cycle"].eq(30), row_30_columns].iloc[0],
        candidate.loc[candidate["cycle"].eq(30), row_30_columns].iloc[0],
        check_names=False,
    )
    assert candidate.loc[candidate["cycle"].eq(31), "sensor_1__last"].iloc[0] == 10_000.0


@pytest.mark.parametrize("window", [10, 20, 30])
def test_rolling_features_do_not_cross_engine_boundaries(window) -> None:
    observations = _observations_from_specs([
        ("train", 1, 30, 0.0),
        ("train", 2, 30, 1_000.0),
    ])
    features = build_rolling_features(observations, window_cycles=window)
    features = features.loc[features.cycle.eq(30)]

    assert features[["engine_id", "cycle"]].to_dict("records") == [
        {"engine_id": 1, "cycle": 30},
        {"engine_id": 2, "cycle": 30},
    ]
    expected = (31-window+30)/2
    assert features.loc[features["engine_id"].eq(1), "sensor_1__mean"].iloc[0] == expected
    assert features.loc[features["engine_id"].eq(2), "sensor_1__mean"].iloc[0] == 1000+expected


@pytest.mark.parametrize("window", [10, 20, 30])
def test_rolling_features_reject_short_engine(window) -> None:
    observations = _observations_from_specs([("test", 1, window-1, 0.0)])

    with pytest.raises(ContractError, match="ROLLING_WINDOW"):
        build_rolling_features(observations, window_cycles=window)


def test_fd001_rolling_features_write_round_trip(raw_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    etl_result = run_pandas_etl(raw_dir)
    features = build_rolling_features(etl_result.observations)

    assert len(features) == 27_927
    assert features["split"].eq("train").sum() == 17_731
    assert features["split"].eq("test").sum() == 10_196
    assert features.shape[1] == 129
    validate_rolling_features(features, observations=etl_result.observations)

    output_dir = tmp_path / "features"
    payload = write_rolling_feature_artifacts(
        RollingFeatureResult(features=features, input_sha256={"observations.parquet": "test"}),
        output_dir,
    )
    restored = pd.read_parquet(output_dir / "gold" / "rolling_features.parquet")
    assert len(restored) == 27_927
    assert payload["feature_contract"]["std_ddof"] == 1
    assert payload["feature_contract"]["joins_test_labels"] is False
    assert payload["modeling"]["status"] == "not_started"
