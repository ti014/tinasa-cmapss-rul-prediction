"""Kiểm thử contract và protocol PCA--ARIMA train-only."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import MEASUREMENTS, OBSERVATION_COLUMNS
from cmapss.timeseries.arima import evaluate_arima_forecast, split_forecast_series
from cmapss.timeseries.contracts import (
    FORECAST_COLUMNS,
    HEALTH_INDEX_COLUMNS,
    SELECTED_ENGINE_COLUMNS,
)
from cmapss.timeseries.health_index import (
    fit_health_index,
    select_validation_engines,
    validate_health_index,
)


def _observations(n_engines: int = 6, length: int = 20) -> pd.DataFrame:
    rows: list[list[object]] = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, length + 1):
            values = [float(engine_id + cycle * (index + 1) / 10) for index in range(24)]
            rows.append(["train", engine_id, cycle, *values])
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).astype(
        {
            "split": "string",
            "engine_id": "int64",
            "cycle": "int64",
            **dict.fromkeys(MEASUREMENTS, "float64"),
        }
    )


def _folds(n_engines: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": pd.Series(["train"] * n_engines, dtype="string"),
            "engine_id": np.arange(1, n_engines + 1, dtype="int64"),
            "fold": np.array([1, 1, 2, 2, 3, 3][:n_engines], dtype="int64"),
        }
    )


def test_health_index_fit_is_train_only_and_validates_contract() -> None:
    observations = pd.concat(
        [
            _observations(),
            _observations(n_engines=1, length=3).assign(split="test", engine_id=99),
        ],
        ignore_index=True,
    )

    result = fit_health_index(observations)

    assert tuple(result.health_index.columns) == HEALTH_INDEX_COLUMNS
    assert result.health_index["split"].eq("train").all()
    assert result.health_index["engine_id"].nunique() == 6
    assert result.orientation_sign in {-1, 1}
    assert result.health_index["health_index"].mean() == pytest.approx(0.0, abs=1e-10)
    assert result.health_index["health_index"].std(ddof=0) == pytest.approx(1.0)
    validate_health_index(result.health_index)


def test_validation_engine_selection_is_deterministic_and_forecast_independent() -> None:
    observations = _observations()
    folds = _folds()

    first = select_validation_engines(observations, folds, n_engines=4)
    second = select_validation_engines(observations, folds, n_engines=4)

    assert tuple(first.columns) == SELECTED_ENGINE_COLUMNS
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 4
    assert first["engine_id"].is_unique
    assert first["fold"].value_counts().to_dict() == {1: 2, 2: 1, 3: 1}


def test_split_forecast_series_keeps_time_order_without_overlap() -> None:
    cycles = pd.Series(np.arange(1, 11), dtype="int64")
    values = pd.Series(np.arange(10, 20, dtype="float64"))

    train, holdout = split_forecast_series(
        cycles,
        values,
        train_fraction=0.7,
        minimum_train_size=5,
    )

    assert len(train) == 7
    assert len(holdout) == 3
    assert train.index.max() < holdout.index.min()
    assert train.iloc[-1] == 16.0
    assert holdout.iloc[0] == 17.0


def test_forecast_metrics_calculate_mae_rmse_and_coverage() -> None:
    actual = np.array([1.0, 2.0, 4.0])
    forecast = np.array([2.0, 2.0, 2.0])
    lower = np.array([0.0, 1.0, 3.0])
    upper = np.array([3.0, 3.0, 5.0])

    metrics = evaluate_arima_forecast(actual, forecast, lower, upper)

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5.0 / 3.0))
    assert metrics["coverage_95"] == pytest.approx(1.0)


def test_short_series_is_rejected_before_forecast() -> None:
    with pytest.raises(ValueError, match="FORECAST_SPLIT"):
        split_forecast_series(
            pd.Series([1, 2, 3]),
            pd.Series([1.0, 2.0, 3.0]),
            train_fraction=0.7,
            minimum_train_size=10,
        )
