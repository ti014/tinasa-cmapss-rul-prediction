"""Focused proofs for the bounded similarity pilot."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import KEY, MEASUREMENTS, OBSERVATION_COLUMNS, SENSORS, TRAIN_TARGET_COLUMNS
from cmapss.modeling.similarity import fit_similarity


def _observations(engines: list[tuple[str, int, int, float]]) -> pd.DataFrame:
    rows = []
    for split, engine, length, offset in engines:
        for cycle in range(1, length + 1):
            rows.append([split, engine, cycle, *([0.0, 1.0, 2.0] + [offset + cycle + sensor for sensor in range(21)])])
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).astype({"split": "string", "engine_id": "int64", "cycle": "int64", **dict.fromkeys(MEASUREMENTS, "float64")})


def _targets(observations: pd.DataFrame) -> pd.DataFrame:
    train = observations.loc[observations.split.eq("train"), list(KEY)].copy()
    train["rul"] = train.groupby("engine_id")["cycle"].transform("max") - train["cycle"]
    return train.loc[:, list(TRAIN_TARGET_COLUMNS)]


def test_known_match_and_distinct_donors() -> None:
    train = _observations([("train", 1, 32, 0.0), ("train", 2, 32, 100.0), ("train", 3, 32, 200.0)])
    model = fit_similarity(train, _targets(train))
    query = _observations([("test", 9, 30, 0.0)])
    keys, predictions, matches = model.predict_with_matches(query, neighbors=(1, 2))
    assert len(keys) == 1 and keys.iloc[0].cycle == 30
    assert predictions[1][0] == pytest.approx(2.0)
    assert predictions[2][0] == pytest.approx(2.0)
    assert matches.groupby("query_index").size().iloc[0] == 3
    assert matches.donor_engine_id.nunique() == 3


def test_query_prefix_and_truncation_parity() -> None:
    train = _observations([("train", 1, 40, 0.0), ("train", 2, 40, 10.0)])
    model = fit_similarity(train, _targets(train))
    full = _observations([("test", 7, 35, 0.0)])
    prefix = full.loc[full.cycle.le(30)].copy()
    np.testing.assert_allclose(model.predict(full, neighbors=(5,))[5][:1], model.predict(prefix, neighbors=(5,))[5])


def test_scaler_weights_each_engine_equally() -> None:
    train = _observations([("train", 1, 30, 0.0), ("train", 2, 60, 100.0)])
    model = fit_similarity(train, _targets(train))
    raw = train.loc[:, list(SENSORS)].to_numpy()
    expected = np.average(raw, axis=0, weights=np.r_[np.full(30, 1 / 30), np.full(60, 1 / 60)])
    np.testing.assert_allclose(model.scaler.mean_, expected)


def test_constant_sensor_is_removed_and_block_order_is_preserved() -> None:
    train = _observations([("train", 1, 30, 0.0), ("train", 2, 30, 10.0)])
    train["sensor_21"] = 7.0
    model = fit_similarity(train, _targets(train))
    assert "sensor_21" not in model.sensors
    assert model.library_matrix.shape[1] == 6 * 20
    # For sensor_1, the first and last block means are 3 and 28 respectively.
    first = model.scaler.transform(np.array([[3.0] + [0.0] * 20]))[0, 0]
    last = model.scaler.transform(np.array([[28.0] + [0.0] * 20]))[0, 0]
    assert model.library_matrix[0, 0] == pytest.approx(first)
    assert model.library_matrix[0, 5 * 20] == pytest.approx(last)


def test_fit_rejects_labels_and_query_overlap_or_incomplete_prefix() -> None:
    train = _observations([("train", 1, 30, 0.0)])
    model = fit_similarity(train, _targets(train))
    with pytest.raises(ValueError, match="SCHEMA"):
        fit_similarity(train.assign(rul=0), _targets(train))
    with pytest.raises(ValueError, match="overlap"):
        model.predict(train, neighbors=(5,))
    incomplete = _observations([("test", 4, 30, 0.0)]).loc[lambda frame: frame.cycle.ne(2)]
    with pytest.raises(ValueError):
        model.predict(incomplete, neighbors=(5,))


def test_target_contract_rejects_wrong_rul() -> None:
    train = _observations([("train", 1, 30, 0.0)])
    targets = _targets(train)
    targets.loc[0, "rul"] = 999
    with pytest.raises(ValueError, match="TRAIN_RUL"):
        fit_similarity(train, targets)


def test_constant_sensor_keeps_block_column_alignment() -> None:
    train = _observations([("train", 1, 32, 0.0), ("train", 2, 32, 10.0)])
    train["sensor_21"] = 7.0
    model = fit_similarity(train, _targets(train))
    assert "sensor_21" not in model.sensors
    query = _observations([("test", 9, 30, 0.0)])
    query["sensor_21"] = 7.0
    predictions = model.predict(query, neighbors=(1,))[1]
    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()
