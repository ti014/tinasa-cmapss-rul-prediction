"""HGB multi-scale feature ablation study."""

from .config import StudyConfig
from .contracts import (
    BASELINE_FEATURES,
    BASE_ROLLING_FEATURES,
    FEATURE_COLUMNS_BY_ARM,
    HISTORY_FEATURES,
    INITIAL_PROFILE_FEATURES,
    WINDOW10_FEATURES,
    FeatureSpec,
    feature_spec,
    feature_specs,
)

__all__ = [
    "StudyConfig", "FeatureSpec", "feature_spec", "feature_specs",
    "BASE_ROLLING_FEATURES", "BASELINE_FEATURES", "HISTORY_FEATURES",
    "WINDOW10_FEATURES", "INITIAL_PROFILE_FEATURES", "FEATURE_COLUMNS_BY_ARM",
]
