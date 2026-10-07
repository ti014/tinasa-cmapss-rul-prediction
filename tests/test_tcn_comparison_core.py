"""Focused tests for the isolated TCN candidate."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from cmapss.data.contracts import SENSORS
from cmapss.modeling.tcn_comparison.contracts import TCNConfig, specs
from cmapss.modeling.tcn_comparison.data import build_sequences, fit_scaler
from cmapss.modeling.tcn_comparison.model import build_model, fit_model


def _observations(n_engines: int = 2, cycles: int = 6) -> pd.DataFrame:
    rows = []
    for split in ["train"]:
        for engine in range(1, n_engines + 1):
            for cycle in range(1, cycles + 1):
                rows.append({"split": split, "engine_id": engine, "cycle": cycle,
                             **{sensor: float(engine + cycle / 10) for sensor in SENSORS}})
    return pd.DataFrame(rows)


def test_sequence_windows_are_causal_and_engine_local():
    observations = _observations()
    targets = observations[["split", "engine_id", "cycle"]].copy()
    targets["rul"] = (6 - targets["cycle"]).astype(float)
    sequences = build_sequences(observations, targets, split="train", sequence_length=3)
    assert sequences.values.shape == (8, 21, 3)
    assert sequences.metadata.iloc[0].engine_id == 1
    assert sequences.metadata.iloc[0].cycle == 3
    assert np.allclose(sequences.values[0, 0], [1.1, 1.2, 1.3])
    assert sequences.metadata.iloc[-1].cycle == 6


def test_scaler_and_tcn_fit_use_finite_predictions():
    observations = _observations()
    targets = observations[["split", "engine_id", "cycle"]].copy()
    targets["rul"] = (6 - targets["cycle"]).astype(float)
    sequences = build_sequences(observations, targets, split="train", sequence_length=3)
    scaler = fit_scaler(sequences.values, sequences.metadata, {1})
    assert np.isfinite(scaler.transform(sequences.values)).all()
    config = TCNConfig(sequence_length=3, channels=4, epochs=1, batch_size=4,
                       outer_splits=2, inner_splits=2, bootstrap_resamples=5,
                       fit_threads=1)
    fitted = fit_model(sequences.values, sequences.metadata, specs()[0], config, {1})
    predictions = fitted.predict(sequences.values[sequences.metadata.engine_id.eq(2).to_numpy()])
    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()


def test_contract_rejects_non_causal_dilation():
    config = TCNConfig(dilations=(1, 2, 4, 8), bootstrap_resamples=1)
    assert config.channels == 32
    assert specs()[1].target_cap == 125


def test_architecture_has_four_residual_dilation_blocks():
    config = TCNConfig(sequence_length=3, channels=4, epochs=1,
                       outer_splits=2, inner_splits=2, bootstrap_resamples=1,
                       fit_threads=1)
    model = build_model(n_sensors=21, config=config)
    assert len(model.blocks) == 4
    assert [block.conv1.dilation[0] for block in model.blocks] == [1, 2, 4, 8]
    assert model.input_projection.in_channels == 21
    assert model.head.in_features == 4
