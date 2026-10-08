"""Build the three causal HGB feature arms on a shared key panel.

The baseline reuses the audit study features. The other arms add rolling-10
statistics or a first-30-cycle sensor profile. Nothing is fitted across engines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY, SENSORS
from cmapss.data.validation import validate_observations
from cmapss.features.rolling import build_rolling_features
from cmapss.modeling.audit_study import build_study_features
from cmapss.modeling.multiscale_comparison.contracts import (
    BASELINE_FEATURES,
    FEATURE_COLUMNS_BY_ARM,
    INITIAL_PROFILE_FEATURES,
    WINDOW10_FEATURES,
)

_BASELINE_ARM = "hgb_baseline169"
_MULTISCALE_ARM = "hgb_multiscale10_30"
_INITIAL_ARM = "hgb_initial_profile42"
_GROUP_KEY = ("split", "engine_id")


def feature_columns(arm: str) -> tuple[str, ...]:
    """Return the model feature columns of an arm, in fitting order.

    ``cycle`` is both a key and a feature; it appears once.
    """
    try:
        return tuple(FEATURE_COLUMNS_BY_ARM[arm])
    except KeyError as error:
        raise ValueError(f"unknown multiscale feature arm: {arm}") from error


def _stored_columns(arm: str) -> list[str]:
    """Keys first, then model columns, each once."""
    return list(dict.fromkeys((*KEY, *feature_columns(arm))))


def _initial_profile(observations: pd.DataFrame) -> pd.DataFrame:
    """Per-engine sensor mean and sample std (ddof=1) over cycles 1..30."""
    initial = observations.loc[observations.cycle.le(30)]
    grouped = initial.groupby(list(_GROUP_KEY), sort=True)[list(SENSORS)]
    means = grouped.mean()
    deviations = grouped.std(ddof=1)
    values = {
        f"{sensor}__initial30_{stat}": table[sensor]
        for sensor in SENSORS
        for stat, table in (("mean", means), ("std", deviations))
    }
    profile = pd.DataFrame(values).reset_index()
    return profile.loc[:, [*_GROUP_KEY, *INITIAL_PROFILE_FEATURES]]


def _validate_arm(frame: pd.DataFrame, arm: str, keys: pd.DataFrame) -> None:
    """Check column order, keys and finiteness of one arm."""
    if list(frame.columns) != _stored_columns(arm):
        raise ValueError(f"feature schema mismatch for {arm}")
    actual_keys = frame.loc[:, list(KEY)].reset_index(drop=True)
    if not actual_keys.equals(keys):
        raise ValueError(f"feature key panel mismatch for {arm}")
    if frame.empty or not np.isfinite(
        frame.loc[:, list(feature_columns(arm))].to_numpy(dtype="float64")
    ).all():
        raise ValueError(f"feature values must be nonempty and finite for {arm}")


def build_feature_arms(observations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the three arms on all rows with cycle >= 30.

    - ``hgb_baseline169``: identical to the audit study features (169).
    - ``hgb_multiscale10_30``: baseline + 126 trailing-10 sensor stats (295).
    - ``hgb_initial_profile42``: baseline + 42 first-30-cycle sensor mean/std (211).

    All arms share the same keys, so the shorter window does not change the
    evaluated rows. Each engine needs at least 30 contiguous cycles from cycle 1.
    Example: if sensor1 equals cycle, the cycle-30 row has a trailing-10 mean
    of 25.5 and an initial-30 mean of 15.5.
    """
    validate_observations(observations, source="multiscale feature observations")
    ordered = observations.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
    baseline = build_study_features(ordered).loc[
        :, list(dict.fromkeys((*KEY, *BASELINE_FEATURES)))
    ].sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
    keys = baseline.loc[:, list(KEY)].copy()

    rolling10 = build_rolling_features(ordered, window_cycles=10)
    renames = {
        name: name.replace("__", "__w10__", 1)
        for name in rolling10.columns if name not in KEY
    }
    rolling10 = rolling10.rename(columns=renames)
    multi = baseline.merge(
        rolling10.loc[:, [*KEY, *WINDOW10_FEATURES]],
        on=list(KEY), how="left", validate="one_to_one", sort=False,
    ).loc[:, _stored_columns(_MULTISCALE_ARM)]

    profile = _initial_profile(ordered)
    initial = baseline.merge(
        profile, on=list(_GROUP_KEY), how="left", validate="many_to_one", sort=False,
    ).loc[:, _stored_columns(_INITIAL_ARM)]

    arms = {_BASELINE_ARM: baseline, _MULTISCALE_ARM: multi, _INITIAL_ARM: initial}
    for arm, frame in arms.items():
        frame.reset_index(drop=True, inplace=True)
        _validate_arm(frame, arm, keys)
    return arms


def build_arm_features(observations: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Build a single arm (all arms are built so the shared-key checks still run)."""
    feature_columns(arm)
    return build_feature_arms(observations)[arm]


__all__ = ["build_feature_arms", "build_arm_features", "feature_columns"]
