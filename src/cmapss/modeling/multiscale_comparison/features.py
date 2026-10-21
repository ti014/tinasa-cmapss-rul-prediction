"""
Tạo ba nhánh HGB cùng khóa; dùng lại baseline audit, thêm thống kê window 10 hoặc
profile 30 cycle đầu, không fit xuyên động cơ.
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
    """
    Trả cột features mô hình theo thứ tự fit; cycle là khóa và feature nhưng chỉ xuất
    hiện một lần.
    """
    try:
        return tuple(FEATURE_COLUMNS_BY_ARM[arm])
    except KeyError as error:
        raise ValueError(f"Nhánh features đa thang chưa biết: {arm}") from error


def _stored_columns(arm: str) -> list[str]:
    """Các khóa đứng trước rồi đến cột mô hình, mỗi cột đúng một lần."""
    return list(dict.fromkeys((*KEY, *feature_columns(arm))))


def _initial_profile(observations: pd.DataFrame) -> pd.DataFrame:
    """Mean và sample std (ddof=1) sensor của từng động cơ trên cycle 1–30."""
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
    """Kiểm tra thứ tự cột, khóa và tính hữu hạn của một nhánh."""
    if list(frame.columns) != _stored_columns(arm):
        raise ValueError(f"Schema features không khớp cho {arm}")
    actual_keys = frame.loc[:, list(KEY)].reset_index(drop=True)
    if not actual_keys.equals(keys):
        raise ValueError(f"Bảng khóa features không khớp cho {arm}")
    if frame.empty or not np.isfinite(
        frame.loc[:, list(feature_columns(arm))].to_numpy(dtype="float64")
    ).all():
        raise ValueError(f"Giá trị features phải không rỗng và hữu hạn cho {arm}")


def build_feature_arms(observations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Tạo ba nhánh cùng khóa từ cycle 30: baseline169, multiscale10_30 (295),
    initial_profile42 (211); mỗi động cơ cần ít nhất 30 cycle liên tục từ 1.
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
    """Dựng một nhánh qua bộ dựng cả ba nhánh để vẫn kiểm tra khóa chung."""
    feature_columns(arm)
    return build_feature_arms(observations)[arm]


__all__ = ["build_feature_arms", "build_arm_features", "feature_columns"]
