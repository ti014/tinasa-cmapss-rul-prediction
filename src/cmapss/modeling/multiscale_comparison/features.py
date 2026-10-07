"""Build fixed causal HGB feature arms on one common observation key panel.

The baseline is the unchanged audit representation. Window10 statistics and
the first30 sensor profile are separate additions to that baseline. No
parameter is fitted across engines, and targets or future observations are
never accepted by the observation contract.
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
    """Return the frozen model-column order for one predeclared arm.

    ``cycle`` is both a key and a model feature; it is stored once in the
    DataFrame and occurs exactly once in the returned feature order.

    Args:
        arm: One of the three fixed HGB arm identifiers.

    Returns:
        Unique, label-free model feature names in fitting order.

    Raises:
        ValueError: The arm name is not predeclared.
    """
    try:
        return tuple(FEATURE_COLUMNS_BY_ARM[arm])
    except KeyError as error:
        raise ValueError(f"unknown multiscale feature arm: {arm}") from error


def _stored_columns(arm: str) -> list[str]:
    """Keep every key and model column once, with keys stored first."""
    return list(dict.fromkeys((*KEY, *feature_columns(arm))))


def _initial_profile(observations: pd.DataFrame) -> pd.DataFrame:
    """Compute each engine's first30 sensor mean and sample standard deviation."""
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
    """Reject a missing, reordered or nonfinite feature matrix before fitting."""
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
    """Construct three causal representations on all baseline cycle>=30 rows.

    The 169-feature baseline has the exact values and column dtypes of the
    existing audit representation on canonical observations. The multi-scale arm appends
    126 trailing10 sensor statistics. The initial-profile arm instead appends
    42 sensor means/sample standard deviations from cycles1..30. Every arm
    has exactly the same keys; adding a shorter window never adds or drops
    evaluation rows.

    Args:
        observations: Label-free observations with a complete continuous
            prefix from cycle1 and at least30 cycles for each engine/split.

    Returns:
        ``hgb_baseline169`` (169 model features),
        ``hgb_multiscale10_30`` (295), and ``hgb_initial_profile42`` (211).
        Keys are stored first and ``cycle`` is not duplicated.

    Raises:
        ContractError: Observation schema, keys or numeric values are invalid.
        ValueError: A first30 prefix is incomplete, keys fail alignment, or a
            feature is not finite.

    Example:
        If sensor1 equals cycle, the cycle30 row has a trailing10 mean of25.5
        and an initial30 mean of15.5. Changing cycle31 leaves that row intact.
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
    """Build one predeclared arm while applying the same common-panel checks."""
    feature_columns(arm)
    return build_feature_arms(observations)[arm]


__all__ = ["build_feature_arms", "build_arm_features", "feature_columns"]
