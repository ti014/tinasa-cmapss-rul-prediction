"""Focused leakage and key-panel proofs for fixed multi-scale feature arms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import KEY, MEASUREMENTS, OBSERVATION_COLUMNS, SENSORS
from cmapss.modeling.audit_study import build_study_features
from cmapss.modeling.multiscale_comparison.features import (
    build_arm_features,
    build_feature_arms,
    feature_columns,
)


ARMS = {
    "hgb_baseline169": 169,
    "hgb_multiscale10_30": 295,
    "hgb_initial_profile42": 211,
}


def _observations(specs: list[tuple[str, int, int, float]]) -> pd.DataFrame:
    rows = []
    for split, engine_id, length, offset in specs:
        for cycle in range(1, length + 1):
            measurements = {name: float(cycle + offset) for name in MEASUREMENTS}
            measurements["sensor_2"] = 5.0 + offset
            rows.append([split, engine_id, cycle, *[measurements[name] for name in MEASUREMENTS]])
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).astype({
        "split": "string", "engine_id": "int64", "cycle": "int64",
        **dict.fromkeys(MEASUREMENTS, "float64"),
    })


def test_feature_counts_order_and_shared_baseline_rows() -> None:
    observations = _observations([("train", 1, 34, 0.0), ("test", 1, 31, 100.0)])
    arms = build_feature_arms(observations)
    expected_keys = observations.loc[observations.cycle.ge(30), list(KEY)].sort_values(
        list(KEY), kind="mergesort"
    ).reset_index(drop=True)
    assert tuple(arms) == tuple(ARMS)
    for arm, count in ARMS.items():
        columns = feature_columns(arm)
        assert len(columns) == len(set(columns)) == count
        assert columns.count("cycle") == 1
        assert not {"rul", "y_true", "lifetime", "engine_id", "split"} & set(columns)
        assert len(arms[arm].columns) == count + 2
        assert not arms[arm].columns.duplicated().any()
        pd.testing.assert_frame_equal(arms[arm].loc[:, list(KEY)], expected_keys, check_exact=True)


def test_baseline_values_and_dtypes_match_existing_audit_exactly() -> None:
    observations = _observations([("train", 1, 41, 0.0), ("test", 7, 35, 1000.0)])
    observations = observations.sample(frac=1, random_state=13)
    baseline = build_feature_arms(observations)["hgb_baseline169"]
    stored_columns = list(dict.fromkeys((*KEY, *feature_columns("hgb_baseline169"))))
    reference = build_study_features(observations).loc[:, stored_columns]
    reference = reference.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline, reference, check_exact=True)


def test_hand_calculated_trailing10_and_initial30_values() -> None:
    observations = _observations([("train", 1, 31, 0.0)])
    arms = build_feature_arms(observations)
    at30 = arms["hgb_multiscale10_30"].iloc[0]
    assert at30["sensor_1__w10__last"] == 30.0
    assert at30["sensor_1__w10__mean"] == 25.5
    assert at30["sensor_1__w10__std"] == pytest.approx(np.std(np.arange(21.0, 31.0), ddof=1))
    assert at30["sensor_1__w10__min"] == 21.0
    assert at30["sensor_1__w10__max"] == 30.0
    assert at30["sensor_1__w10__slope"] == pytest.approx(1.0)
    assert at30["sensor_2__w10__std"] == 0.0
    profile = arms["hgb_initial_profile42"]
    assert profile["sensor_1__initial30_mean"].tolist() == [15.5, 15.5]
    np.testing.assert_allclose(profile["sensor_1__initial30_std"],
                               np.std(np.arange(1.0, 31.0), ddof=1))
    assert profile["sensor_2__initial30_std"].eq(0.0).all()


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_future_changes_do_not_affect_any_retained_prefix_feature(arm: str) -> None:
    observations = _observations([("train", 1, 37, 0.0), ("train", 2, 36, 10.0)])
    baseline = build_feature_arms(observations)[arm]
    changed = observations.copy()
    changed.loc[changed.cycle.ge(33), list(SENSORS)] = 10000.0
    candidate = build_feature_arms(changed)[arm]
    earlier = baseline.cycle.le(32)
    pd.testing.assert_frame_equal(baseline.loc[earlier].reset_index(drop=True),
                                  candidate.loc[earlier].reset_index(drop=True), check_exact=True)


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_batch_features_equal_features_available_at_each_observed_prefix(arm: str) -> None:
    observations = _observations([("test", 3, 38, 0.0)])
    batch = build_feature_arms(observations)[arm]
    prefix = build_feature_arms(observations.loc[observations.cycle.le(33)])[arm]
    pd.testing.assert_frame_equal(prefix, batch.loc[batch.cycle.le(33)].reset_index(drop=True),
                                  check_exact=True)


def test_engine_and_split_boundaries_are_isolated() -> None:
    observations = _observations([("train", 1, 33, 0.0), ("train", 2, 33, 1000.0),
                                   ("test", 1, 33, 2000.0)])
    arms = build_feature_arms(observations)
    for split, engine_id, offset in (("train", 1, 0.0), ("train", 2, 1000.0), ("test", 1, 2000.0)):
        multi = arms["hgb_multiscale10_30"]
        row = multi.loc[multi.split.eq(split) & multi.engine_id.eq(engine_id) & multi.cycle.eq(30)].iloc[0]
        assert row["sensor_1__w10__mean"] == 25.5 + offset
        profile = arms["hgb_initial_profile42"]
        row = profile.loc[profile.split.eq(split) & profile.engine_id.eq(engine_id) & profile.cycle.eq(30)].iloc[0]
        assert row["sensor_1__initial30_mean"] == 15.5 + offset
    altered = observations.copy()
    altered.loc[altered.split.eq("train") & altered.engine_id.eq(2), list(SENSORS)] = 777.0
    changed = build_feature_arms(altered)
    for arm in ARMS:
        unaffected = ~(arms[arm].split.eq("train") & arms[arm].engine_id.eq(2))
        pd.testing.assert_frame_equal(arms[arm].loc[unaffected].reset_index(drop=True),
                                      changed[arm].loc[unaffected].reset_index(drop=True), check_exact=True)


def test_input_order_is_canonicalized_and_observations_are_not_mutated() -> None:
    observations = _observations([("train", 1, 33, 0.0), ("test", 1, 31, 100.0)])
    shuffled = observations.sample(frac=1, random_state=71)
    snapshot = shuffled.copy(deep=True)
    expected = build_feature_arms(observations)
    actual = build_feature_arms(shuffled)
    for arm in ARMS:
        pd.testing.assert_frame_equal(expected[arm], actual[arm], check_exact=True)
    pd.testing.assert_frame_equal(shuffled, snapshot, check_exact=True)


@pytest.mark.parametrize("arm", tuple(ARMS))
def test_single_arm_api_matches_batch(arm: str) -> None:
    observations = _observations([("train", 1, 31, 0.0)])
    pd.testing.assert_frame_equal(build_arm_features(observations, arm),
                                  build_feature_arms(observations)[arm], check_exact=True)


def test_no_added_feature_uses_operating_settings() -> None:
    observations = _observations([("test", 1, 34, 0.0)])
    expected = build_feature_arms(observations)
    changed = observations.copy()
    changed.loc[:, ["setting_1", "setting_2", "setting_3"]] = -123.0
    actual = build_feature_arms(changed)
    for arm in ARMS:
        pd.testing.assert_frame_equal(expected[arm], actual[arm], check_exact=True)


@pytest.mark.parametrize("column", ("rul", "y_true", "lifetime"))
def test_rejects_labels_or_lifetime_in_observation_schema(column: str) -> None:
    observations = _observations([("train", 1, 31, 0.0)])
    observations[column] = 1.0
    with pytest.raises(ValueError, match="SCHEMA"):
        build_feature_arms(observations)


def test_rejects_incomplete_initial_prefix() -> None:
    observations = _observations([("train", 1, 35, 0.0)])
    observations = observations.loc[observations.cycle.ge(2)]
    with pytest.raises(ValueError, match="exactly cycles1..30"):
        build_feature_arms(observations)


def test_rejects_short_engine_without_silently_dropping_it() -> None:
    observations = _observations([("train", 1, 35, 0.0), ("train", 2, 29, 0.0)])
    with pytest.raises(ValueError, match="ROLLING_WINDOW"):
        build_feature_arms(observations)


def test_rejects_unknown_arm_before_feature_generation() -> None:
    with pytest.raises(ValueError, match="unknown multiscale feature arm"):
        build_arm_features(pd.DataFrame(), "anything")
