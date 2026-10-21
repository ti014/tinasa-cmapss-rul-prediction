"""Danh sách feature của ba arm trong study HGB multi-scale.

* ``hgb_baseline169``: rolling 30 cycles, ``cycle`` và history 30 cycles; *
``hgb_multiscale10_30``: baseline và thêm rolling 10 cycles; *
``hgb_initial_profile42``: baseline và thêm mean/std của 30 cycles đầu. Tất cả feature
đều causal. ``rul`` và các cột nhãn không thuộc bất kỳ arm nào.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cmapss.data.contracts import KEY, SENSORS
from cmapss.features.contracts import ROLLING_FEATURE_STATS, ROLLING_FEATURE_VALUE_COLUMNS


BASE_ROLLING_FEATURES: tuple[str, ...] = tuple(ROLLING_FEATURE_VALUE_COLUMNS)
"""126 rolling features ở cửa sổ 30 cycles."""

HISTORY_FEATURES: tuple[str, ...] = tuple(
    f"{sensor}__from_initial_{stat}"
    for sensor in SENSORS
    for stat in ("last", "mean")
)
"""42 causal features tương đối với 30 cycles đầu."""

BASELINE_FEATURES: tuple[str, ...] = BASE_ROLLING_FEATURES + ("cycle",) + HISTORY_FEATURES

WINDOW10_FEATURES: tuple[str, ...] = tuple(
    f"{sensor}__w10__{stat}"
    for sensor in SENSORS
    for stat in ROLLING_FEATURE_STATS
)
"""126 rolling features bổ sung ở cửa sổ 10 cycles."""

INITIAL_PROFILE_FEATURES: tuple[str, ...] = tuple(
    f"{sensor}__initial30_{stat}"
    for sensor in SENSORS
    for stat in ("mean", "std")
)
"""42 thống kê sensor của prefix 30 cycles đầu, dùng sample std (ddof=1)."""

ArmId = Literal[
    "hgb_baseline169",
    "hgb_multiscale10_30",
    "hgb_initial_profile42",
]

FEATURE_COLUMNS_BY_ARM: dict[ArmId, tuple[str, ...]] = {
    "hgb_baseline169": BASELINE_FEATURES,
    "hgb_multiscale10_30": BASELINE_FEATURES + WINDOW10_FEATURES,
    "hgb_initial_profile42": BASELINE_FEATURES + INITIAL_PROFILE_FEATURES,
}


@dataclass(frozen=True)
class FeatureSpec:
    """Danh sách feature và metadata của một arm."""

    arm_id: ArmId
    feature_columns: tuple[str, ...]
    feature_count: int
    added_feature_group: str
    causal_windows: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = FEATURE_COLUMNS_BY_ARM.get(self.arm_id)
        if expected is None or self.feature_columns != expected:
            raise ValueError(f"Các cột features không khớp nhánh {self.arm_id}")
        if self.feature_count != len(self.feature_columns):
            raise ValueError("feature_count không khớp feature_columns")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("Các cột features phải duy nhất")
        if any(column in {"rul", "y_true", "target", "lifetime"} for column in self.feature_columns):
            raise ValueError("Cột nhãn hoặc tuổi thọ không được làm feature")
        if not self.causal_windows or any(window < 1 for window in self.causal_windows):
            raise ValueError("Window chỉ dùng quá khứ phải dương")


def feature_specs() -> tuple[FeatureSpec, ...]:
    """Ba arm, baseline đứng đầu."""
    return (
        FeatureSpec("hgb_baseline169", BASELINE_FEATURES, 169, "baseline169", (30,)),
        FeatureSpec("hgb_multiscale10_30", BASELINE_FEATURES + WINDOW10_FEATURES, 295,
                    "rolling_window10", (10, 30)),
        FeatureSpec("hgb_initial_profile42", BASELINE_FEATURES + INITIAL_PROFILE_FEATURES, 211,
                    "initial_profile42", (30,)),
    )


def feature_spec(arm_id: str) -> FeatureSpec:
    """Lấy arm theo ID; ID không có báo ``ValueError``."""
    for spec in feature_specs():
        if spec.arm_id == arm_id:
            return spec
    raise ValueError(f"Nhánh features chưa biết: {arm_id}")


__all__ = [
    "KEY", "BASE_ROLLING_FEATURES", "HISTORY_FEATURES", "BASELINE_FEATURES",
    "WINDOW10_FEATURES", "INITIAL_PROFILE_FEATURES", "ArmId", "FeatureSpec",
    "FEATURE_COLUMNS_BY_ARM", "feature_spec", "feature_specs",
]
