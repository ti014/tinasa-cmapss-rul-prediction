"""Focused contract tests for the non-promotional TCN pilot."""

import numpy as np
import pandas as pd

from cmapss.modeling.tcn_comparison.contracts import TCNConfig, specs
from cmapss.modeling.tcn_comparison.data import build_sequences, fit_scaler


def _observations() -> pd.DataFrame:
    rows = []
    for engine in (1, 2):
        for cycle in range(1, 7):
            rows.append({"split": "train", "engine_id": engine, "cycle": cycle,
                         **{f"sensor_{i}": float(engine + cycle + i) for i in range(1, 22)}})
    return pd.DataFrame(rows)


def test_sequence_windows_do_not_cross_engines_and_keep_raw_target() -> None:
    observations = _observations()
    targets = observations[["split", "engine_id", "cycle"]].copy()
    targets["rul"] = targets.groupby("engine_id")["cycle"].transform("max") - targets["cycle"]
    result = build_sequences(observations, targets, split="train", sequence_length=3)
    assert result.values.shape == (8, 21, 3)
    assert result.metadata.iloc[0].to_dict() == {"split": "train", "engine_id": 1, "cycle": 3, "rul": 3.0}
    assert result.metadata.engine_id.tolist() == [1] * 4 + [2] * 4


def test_scaler_uses_only_fitting_engines() -> None:
    observations = _observations()
    targets = observations[["split", "engine_id", "cycle"]].assign(rul=0.0)
    result = build_sequences(observations, targets, split="train", sequence_length=3)
    scaler = fit_scaler(result.values, result.metadata, {1})
    expected = result.values[result.metadata.engine_id.eq(1).to_numpy()].mean(axis=(0, 2))
    assert np.allclose(scaler.mean, expected)
    assert not np.allclose(scaler.mean, result.values.mean(axis=(0, 2)))


def test_contract_marks_default_as_scientific_only_when_full_protocol() -> None:
    assert TCNConfig().scientific is False
    assert TCNConfig(epochs=8).scientific is False
    assert len(specs()) == 2
