"""Kiểm thử contract của Health Index fit trên train."""

import pandas as pd
import pytest

from cmapss.data.contracts import MEASUREMENTS, OBSERVATION_COLUMNS
from cmapss.timeseries.contracts import HEALTH_INDEX_COLUMNS
from cmapss.timeseries.health_index import fit_health_index, validate_health_index


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
