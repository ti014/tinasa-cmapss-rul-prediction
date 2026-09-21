"""Positive/negative proof cho khóa, cycle, schema và profile FD001."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import ContractError
from cmapss.data.readers import read_observations, read_test_labels
from cmapss.data.validation import (
    validate_fd001_profile, validate_observations, validate_test_labels,
)


def test_same_engine_id_across_splits_is_allowed(observations: pd.DataFrame) -> None:
    test = observations.assign(split="test")
    combined = pd.concat([observations, test], ignore_index=True).sample(frac=1, random_state=42)
    before = combined.copy(deep=True)
    validate_observations(combined)
    pd.testing.assert_frame_equal(combined, before)


@pytest.mark.parametrize("value", [-2.5, 0.0])
def test_finite_sensor_values_have_no_invented_physical_threshold(
    observations: pd.DataFrame, value: float,
) -> None:
    validate_observations(observations.assign(sensor_1=value))


@pytest.mark.parametrize("mutation,rule", [
    (lambda df: df.drop(columns="sensor_21"), "SCHEMA"),
    (lambda df: df.assign(rul=0), "SCHEMA"),
    (lambda df: df.iloc[:0], "NONEMPTY"),
    (lambda df: df.assign(split="validation"), "SPLIT"),
    (lambda df: df.assign(split=pd.NA), "SPLIT"),
    (lambda df: df.assign(cycle=df.cycle.astype(float)), "NUMERIC"),
    (lambda df: df.assign(engine_id=True), "NUMERIC"),
    (lambda df: df.assign(sensor_1=np.nan), "NUMERIC"),
    (lambda df: df.assign(sensor_1=np.inf), "NUMERIC"),
    (lambda df: df.assign(sensor_1="bad"), "NUMERIC"),
    (lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True), "UNIQUE_KEY"),
    (lambda df: df.drop(index=1), "CYCLE_CONTINUITY"),
])
def test_observation_violations(observations: pd.DataFrame, mutation, rule: str) -> None:
    with pytest.raises(ContractError, match=rf"\[{rule}\]"):
        validate_observations(mutation(observations))


def test_small_fixture_is_valid_but_not_full_fd001(observations: pd.DataFrame) -> None:
    validate_observations(observations)
    with pytest.raises(ContractError, match="FD001_PROFILE"):
        validate_fd001_profile(observations, split="train")


def test_real_fd001_profiles(raw_dir: Path) -> None:
    for split in ("train", "test"):
        frame = read_observations(raw_dir / f"{split}_FD001.txt", split=split)
        validate_fd001_profile(frame, split=split)
    labels = read_test_labels(raw_dir / "RUL_FD001.txt")
    validate_test_labels(labels, fd001=True)


@pytest.mark.parametrize("mutation,rule", [
    (lambda df: df.assign(engine_id=df.engine_id + 1), "FD001_PROFILE"),
    (lambda df: df.assign(cycle=df.cycle + 1), "FD001_PROFILE"),
    (lambda df: df.assign(split="test"), "SPLIT"),
])
def test_fd001_profile_rejects_wrong_identity_and_origin(raw_dir: Path, mutation, rule: str) -> None:
    frame = read_observations(raw_dir / "train_FD001.txt", split="train")
    with pytest.raises(ContractError, match=rule):
        validate_fd001_profile(mutation(frame), split="train")


@pytest.mark.parametrize("mutation,rule", [
    (lambda df: df.assign(split="train"), "SPLIT"),
    (lambda df: df.assign(split=pd.NA), "SPLIT"),
    (lambda df: df.assign(engine_id=1), "UNIQUE_KEY"),
    (lambda df: df.assign(engine_id=[1, 3]), "LABEL_MAPPING"),
    (lambda df: df.assign(rul=-1), "NUMERIC"),
])
def test_label_contract_violations(mutation, rule: str) -> None:
    labels = pd.DataFrame({"split": ["test", "test"], "engine_id": [1, 2], "rul": [10, 20]})
    with pytest.raises(ContractError, match=rule):
        validate_test_labels(mutation(labels))


def test_label_count_must_match_fd001() -> None:
    labels = pd.DataFrame({"split": ["test"], "engine_id": [1], "rul": [10]})
    validate_test_labels(labels)
    with pytest.raises(ContractError, match="FD001_PROFILE"):
        validate_test_labels(labels, fd001=True)
