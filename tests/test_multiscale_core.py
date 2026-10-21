"""Explicit feature arms and frozen scientific config contracts."""

from dataclasses import replace

import pytest

from cmapss.modeling.multiscale_comparison.config import StudyConfig
from cmapss.modeling.multiscale_comparison.contracts import (
    BASELINE_FEATURES, FEATURE_COLUMNS_BY_ARM, INITIAL_PROFILE_FEATURES,
    WINDOW10_FEATURES, feature_spec, feature_specs,
)


def test_exact_three_arms_and_feature_sets():
    specs = feature_specs()
    assert tuple(item.arm_id for item in specs) == (
        "hgb_baseline169", "hgb_multiscale10_30", "hgb_initial_profile42",
    )
    assert [item.feature_count for item in specs] == [169, 295, 211]
    assert specs[0].feature_columns == BASELINE_FEATURES
    assert specs[1].feature_columns == BASELINE_FEATURES + WINDOW10_FEATURES
    assert specs[2].feature_columns == BASELINE_FEATURES + INITIAL_PROFILE_FEATURES
    assert specs[0].causal_windows == (30,)
    for spec in specs:
        assert spec.feature_columns == FEATURE_COLUMNS_BY_ARM[spec.arm_id]
        assert len(spec.feature_columns) == len(set(spec.feature_columns))
        assert not {"rul", "y_true", "lifetime", "engine_id", "split"} & set(spec.feature_columns)


def test_unknown_arm_and_feature_mutation_rejected():
    with pytest.raises(ValueError, match="chưa biết"):
        feature_spec("hgb_unplanned")
    with pytest.raises(ValueError, match="không khớp"):
        replace(feature_specs()[0], feature_columns=BASELINE_FEATURES + ("rul",))
    with pytest.raises(ValueError, match="feature_count"):
        replace(feature_specs()[0], feature_count=170)


def test_default_config_and_resource_only_changes_are_scientific():
    config = StudyConfig()
    assert config.scientific
    assert replace(config, max_workers=1, fit_threads=2).scientific
    audit = config.audit_config()
    assert audit.endpoint_fractions == (0.5, 0.7, 0.9)
    assert audit.outer_splits == 5 and audit.inner_splits == 3
    assert audit.bootstrap_resamples == 10_000


@pytest.mark.parametrize("override", [
    {"outer_splits": 2}, {"inner_splits": 2}, {"seed": 43},
    {"hgb_max_iter": 20}, {"bootstrap_resamples": 20},
    {"hgb_learning_rate": 0.1}, {"hgb_max_leaf_nodes": 15},
    {"hgb_min_samples_leaf": 10}, {"endpoint_mae_margin": 0.1},
    {"nasa_score_margin": 0.1}, {"warning_rate_margin": 0.1},
    {"early_alarm_burden_margin": 0.1},
])
def test_changed_scientific_choices_are_non_scientific(override):
    assert not replace(StudyConfig(), **override).scientific


@pytest.mark.parametrize("override", [
    {"outer_splits": 0}, {"fit_threads": 0}, {"hgb_learning_rate": 0},
    {"thresholds": (30, 30)}, {"persistences": (0,)},
    {"endpoint_fractions": (0.9, 0.5)}, {"endpoint_fractions": (1.0,)},
    {"endpoint_mae_margin": -0.1}, {"seed": -1},
])
def test_malformed_config_rejected(override):
    with pytest.raises(ValueError):
        replace(StudyConfig(), **override)
