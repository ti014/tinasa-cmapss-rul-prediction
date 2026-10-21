"""Kiểm tra tạo và validate cặp sensor đầu/cuối đời cho FD001."""

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import ContractError, MEASUREMENTS, OBSERVATION_COLUMNS, SENSORS
from cmapss.data.readers import read_observations
from cmapss.data.validation import validate_fd001_profile
from cmapss.statistics import build_engine_sensor_pairs, validate_engine_sensor_pairs


def _train_observations(
    engines: tuple[tuple[int, int, float], ...],
) -> pd.DataFrame:
    """Tạo observations train liên tục; mọi sensor bằng cycle cộng offset."""
    rows = [
        ["train", engine_id, cycle, *[float(cycle + offset) for _ in MEASUREMENTS]]
        for engine_id, length, offset in engines
        for cycle in range(1, length + 1)
    ]
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).astype(
        {
            "split": "string",
            "engine_id": "int64",
            "cycle": "int64",
            **dict.fromkeys(MEASUREMENTS, "float64"),
        }
    )


def test_pairs_use_exact_boundaries_and_keep_engines_separate() -> None:
    data = _train_observations(((1, 60, 0.0), (2, 200, 1_000.0)))

    pairs = build_engine_sensor_pairs(data).set_index(["engine_id", "sensor"])

    first = pairs.loc[(1, "sensor_1")]
    assert first[["early_start", "early_end", "late_start", "late_end"]].tolist() == [1, 30, 31, 60]
    assert first[["n_cycles", "n_early", "n_late"]].tolist() == [60, 30, 30]
    assert first["early_mean"] == 15.5
    assert first["late_mean"] == 45.5
    assert first["difference"] == 30.0

    long_engine = pairs.loc[(2, "sensor_1")]
    assert long_engine[["early_start", "early_end", "late_start", "late_end"]].tolist() == [1, 30, 171, 200]
    assert long_engine["early_mean"] == 1_015.5
    assert long_engine["late_mean"] == 1_185.5
    assert long_engine["difference"] == 170.0


def test_pairs_are_canonical_and_independent_of_input_row_order() -> None:
    data = _train_observations(((2, 60, 1_000.0), (1, 60, 0.0)))

    canonical = build_engine_sensor_pairs(data)
    shuffled = build_engine_sensor_pairs(data.sample(frac=1.0, random_state=42))

    assert len(canonical) == 2 * len(SENSORS)
    assert canonical["engine_id"].tolist() == [1] * len(SENSORS) + [2] * len(SENSORS)
    assert canonical["sensor"].tolist() == list(SENSORS) * 2
    pd.testing.assert_frame_equal(canonical, shuffled)
    validate_engine_sensor_pairs(canonical)


@pytest.mark.parametrize(
    ("data", "rule", "item"),
    [
        (_train_observations(((7, 59, 0.0),)), "PAIR_WINDOW", "engine_id=7"),
        (_train_observations(((8, 60, 0.0),)).assign(cycle=lambda frame: frame["cycle"] + 1), "CYCLE_START", "engine_id=8"),
        (_train_observations(((9, 60, 0.0),)).assign(split="test"), "TRAIN_ONLY", "phân tích sensor ghép cặp"),
        (
            pd.concat(
                [
                    _train_observations(((10, 60, 0.0),)),
                    _train_observations(((11, 60, 0.0),)).assign(split="test"),
                ],
                ignore_index=True,
            ),
            "TRAIN_ONLY",
            "phân tích sensor ghép cặp",
        ),
    ],
)
def test_pairs_reject_inputs_outside_the_accepted_scope(
    data: pd.DataFrame,
    rule: str,
    item: str,
) -> None:
    with pytest.raises(ContractError, match=rf"\[{rule}\].*{item}") as error:
        build_engine_sensor_pairs(data)
    assert f"data/CONTRACTS.md#{rule.lower().replace('_', '-')}" in str(error.value)


def test_pairs_preserve_contract_failures_from_observations() -> None:
    data = _train_observations(((1, 60, 0.0),)).drop(index=30)

    with pytest.raises(ContractError, match=r"\[CYCLE_CONTINUITY\]"):
        build_engine_sensor_pairs(data)


def test_pairs_reject_nonfinite_sensor_values_from_observations() -> None:
    data = _train_observations(((1, 60, 0.0),))
    data.loc[0, "sensor_1"] = np.nan

    with pytest.raises(
        ContractError,
        match=r"\[NUMERIC\].*paired sensor observations\.sensor_1",
    ):
        build_engine_sensor_pairs(data)


@pytest.mark.parametrize(
    ("mutation", "rule"),
    [
        (lambda pairs: pairs.assign(pair_status="invalid"), "PAIR_STATUS"),
        (lambda pairs: pairs.assign(difference=pairs["difference"] + 1.0), "PAIR_DIFFERENCE"),
        (lambda pairs: pd.concat([pairs, pairs.iloc[[0]]], ignore_index=True), "PAIR_KEY"),
        (lambda pairs: pairs.drop(index=0).reset_index(drop=True), "PAIR_SENSOR"),
        (lambda pairs: pairs.iloc[::-1].reset_index(drop=True), "PAIR_ORDER"),
        (lambda pairs: pairs.assign(early_mean=np.nan), "PAIR_NUMERIC"),
        (lambda pairs: pairs.assign(early_mean=pairs["early_mean"].astype("float32")), "PAIR_NUMERIC"),
    ],
)
def test_pair_validator_rejects_invalid_artifacts(mutation, rule: str) -> None:
    pairs = build_engine_sensor_pairs(_train_observations(((1, 60, 0.0),)))

    with pytest.raises(ContractError, match=rf"\[{rule}\]"):
        validate_engine_sensor_pairs(mutation(pairs))


def test_pair_validator_rejects_difference_mismatch_at_large_scale() -> None:
    pairs = build_engine_sensor_pairs(_train_observations(((1, 60, 0.0),)))
    pairs.loc[0, ["early_mean", "late_mean", "difference"]] = [1e12, 2e12, 1e12 + 0.5]

    with pytest.raises(ContractError, match=r"\[PAIR_DIFFERENCE\]"):
        validate_engine_sensor_pairs(pairs)


def test_real_fd001_builds_the_expected_complete_pair_table(raw_dir) -> None:
    observations = read_observations(raw_dir / "train_FD001.txt", split="train")
    validate_fd001_profile(observations, split="train")

    pairs = build_engine_sensor_pairs(observations)

    validate_engine_sensor_pairs(pairs)
    assert len(pairs) == 100 * len(SENSORS)
    assert pairs.groupby("engine_id")["sensor"].nunique().eq(len(SENSORS)).all()
    assert pairs.groupby("sensor").size().eq(100).all()
    assert pairs["n_cycles"].ge(60).all()
    assert np.isfinite(pairs[["early_mean", "late_mean", "difference"]].to_numpy()).all()
