"""Kiểm tra RUL bằng giá trị tính tay và các lỗi có chủ đích."""

import pandas as pd
import pytest

from cmapss.data.contracts import ContractError
from cmapss.data.targets import make_train_targets, validate_train_targets


def test_train_rul_by_engine_and_key(observations: pd.DataFrame) -> None:
    before = observations.copy(deep=True)
    targets = make_train_targets(observations)
    assert targets["rul"].tolist() == [3, 2, 1, 0, 1, 0]
    assert len(targets) == len(observations)
    shuffled = targets.sample(frac=1, random_state=42).reset_index(drop=True)
    validate_train_targets(observations, shuffled)
    pd.testing.assert_frame_equal(observations, before)


def test_targets_preserve_unsorted_input(observations: pd.DataFrame) -> None:
    shuffled = observations.iloc[[3, 0, 5, 1, 4, 2]]
    targets = make_train_targets(shuffled)
    assert targets["rul"].tolist() == [0, 3, 0, 2, 1, 1]
    assert targets.index.equals(shuffled.index)


def test_primary_rul_is_not_capped_at_125(observations: pd.DataFrame) -> None:
    long_engine = pd.concat([observations.iloc[[0]]] * 130, ignore_index=True)
    long_engine["cycle"] = range(1, 131)
    targets = make_train_targets(long_engine)
    assert targets["rul"].iloc[:5].tolist() == [129, 128, 127, 126, 125]
    assert targets["rul"].iloc[-1] == 0
    validate_train_targets(long_engine, targets)


@pytest.mark.parametrize("mixed", [False, True])
def test_test_observations_cannot_generate_train_rul(observations: pd.DataFrame, mixed: bool) -> None:
    test = observations.assign(split="test")
    frame = pd.concat([observations, test], ignore_index=True) if mixed else test
    with pytest.raises(ContractError, match="TRAIN_ONLY"):
        make_train_targets(frame)


@pytest.mark.parametrize("mutation,rule", [
    (lambda df: df.assign(rul=df.rul + 1), "TRAIN_RUL"),
    (lambda df: df.iloc[:-1], "TARGET_KEYS"),
    (lambda df: df.assign(engine_id=df.engine_id + 10), "TARGET_KEYS"),
    (lambda df: pd.concat([df, df.iloc[[0]]]), "UNIQUE_KEY"),
    (lambda df: df.assign(split="test"), "TRAIN_ONLY"),
    (lambda df: df.assign(rul=df.rul.astype(float)), "NUMERIC"),
])
def test_incorrect_targets(observations: pd.DataFrame, mutation, rule: str) -> None:
    targets = make_train_targets(observations)
    with pytest.raises(ContractError, match=rule):
        validate_train_targets(observations, mutation(targets))
