"""Kiểm thử contract của Health Index fit trên train."""

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import MEASUREMENTS, OBSERVATION_COLUMNS
from cmapss.timeseries.contracts import (
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


def test_health_index_fit_can_exclude_temporal_holdout_engines() -> None:
    observations = _observations(n_engines=3, length=20)
    shifted = observations.loc[observations["engine_id"].eq(3)].copy()
    shifted.loc[:, list(MEASUREMENTS)] += 10_000.0
    observations = pd.concat(
        [observations.loc[~observations["engine_id"].eq(3)], shifted],
        ignore_index=True,
    )

    result = fit_health_index(observations, fit_engine_ids=[1, 2])

    assert result.fit_engine_ids == (1, 2)
    fit_scores = result.health_index.loc[result.health_index["engine_id"].isin([1, 2]), "health_index"]
    assert fit_scores.mean() == pytest.approx(0.0, abs=1e-10)
    assert fit_scores.std(ddof=0) == pytest.approx(1.0)
    assert result.health_index["engine_id"].nunique() == 3


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
