"""Feature, cấu hình và grid candidate cho so sánh HGB/LightGBM trên 169 feature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from cmapss.modeling.audit_study.config import AuditStudyConfig

from cmapss.data.contracts import KEY, SENSORS
from cmapss.features.contracts import ROLLING_FEATURE_VALUE_COLUMNS

BASE_FEATURES: tuple[str, ...] = tuple(ROLLING_FEATURE_VALUE_COLUMNS)
HISTORY_FEATURES: tuple[str, ...] = ("cycle",) + tuple(
    f"{sensor}__from_initial_{stat}"
    for sensor in SENSORS
    for stat in ("last", "mean")
)
FEATURES: tuple[str, ...] = BASE_FEATURES + HISTORY_FEATURES

Family = Literal["hgb_uncapped", "hgb_cap125", "lgbm_uncapped", "lgbm_cap125"]
ModelKind = Literal["hgb", "lightgbm"]


@dataclass(frozen=True)
class StudyConfig:
    """Cấu hình split, tài nguyên và tiêu chí chọn mô hình.

    Giá trị mặc định là cấu hình đầy đủ. ``max_workers`` chỉ ảnh hưởng việc lập lịch
    process, không đổi kết quả.
    """

    outer_splits: int = 5
    inner_splits: int = 3
    seed: int = 42
    fit_threads: int = 4
    max_workers: int = 2
    bootstrap_resamples: int = 10_000
    hgb_max_iter: int = 200
    lgbm_n_estimators: int = 200
    endpoint_fractions: tuple[float, ...] = (0.5, 0.7, 0.9)
    thresholds: tuple[int, ...] = (30, 35, 40, 45, 50)
    persistences: tuple[int, ...] = (1, 3, 5)
    lead_requirement: int = 30
    target_on_time_rate: float = 0.9
    endpoint_mae_margin: float = 0.0
    nasa_score_margin: float = 0.0
    warning_rate_margin: float = 0.0
    early_alarm_burden_margin: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "outer_splits": self.outer_splits,
            "inner_splits": self.inner_splits,
            "fit_threads": self.fit_threads,
            "max_workers": self.max_workers,
            "bootstrap_resamples": self.bootstrap_resamples,
            "hgb_max_iter": self.hgb_max_iter,
            "lgbm_n_estimators": self.lgbm_n_estimators,
        }
        if any(isinstance(value, bool) or value < 1 for value in positive.values()):
            raise ValueError("Giá trị chia tập, tài nguyên và vòng lặp phải dương")
        if len(self.endpoint_fractions) == 0 or tuple(sorted(set(self.endpoint_fractions))) != self.endpoint_fractions:
            raise ValueError("Các tỷ lệ endpoint phải duy nhất và tăng dần")
        if any(not 0.0 < value < 1.0 for value in self.endpoint_fractions):
            raise ValueError("Các tỷ lệ endpoint phải nằm hẳn trong (0, 1)")
        for name, values in (("thresholds", self.thresholds), ("persistences", self.persistences)):
            if not values or tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} phải không rỗng, duy nhất và tăng dần")
        if any(value < 0 for value in self.thresholds) or any(value < 1 for value in self.persistences):
            raise ValueError("Ngưỡng/số cycle liên tiếp của cảnh báo không hợp lệ")
        if self.lead_requirement < 1 or not 0.0 < self.target_on_time_rate <= 1.0:
            raise ValueError("lead_requirement và target_on_time_rate không hợp lệ")
        margins = (self.endpoint_mae_margin, self.nasa_score_margin,
                   self.warning_rate_margin, self.early_alarm_burden_margin)
        if any(value < 0 for value in margins):
            raise ValueError("Biên đánh giá không được âm")

    @property
    def scientific(self) -> bool:
        """True nếu config trùng với cấu hình đầy đủ mặc định."""
        return (
            self.outer_splits == 5 and self.inner_splits == 3 and self.seed == 42
            and self.bootstrap_resamples >= 10_000 and self.hgb_max_iter == 200
            and self.lgbm_n_estimators == 200
            and self.endpoint_fractions == (0.5, 0.7, 0.9)
            and self.thresholds == (30, 35, 40, 45, 50)
            and self.persistences == (1, 3, 5) and self.lead_requirement == 30
            and self.target_on_time_rate == 0.9
            and self.endpoint_mae_margin == 0.0 and self.nasa_score_margin == 0.0
            and self.warning_rate_margin == 0.0
            and self.early_alarm_burden_margin == 0.0
        )

    def audit_config(self) -> AuditStudyConfig:
        """Chuyển các trường dùng chung sang ``AuditStudyConfig``."""
        from cmapss.modeling.audit_study import AuditStudyConfig

        return AuditStudyConfig(
            outer_splits=self.outer_splits,
            inner_splits=self.inner_splits,
            endpoint_fractions=self.endpoint_fractions,
            thresholds=self.thresholds,
            persistences=self.persistences,
            lead_requirement=self.lead_requirement,
            target_on_time_rate=self.target_on_time_rate,
            bootstrap_resamples=self.bootstrap_resamples,
            seed=self.seed,
            fit_threads=self.fit_threads,
            hgb_max_iter=self.hgb_max_iter,
        )


@dataclass(frozen=True)
class TreeSpec:
    """Tham số của một candidate (chưa fit)."""

    candidate_id: str
    family: Family
    model_kind: ModelKind
    target_cap: int | None
    # HGB giữ các giá trị mặc định bên dưới; LightGBM dùng max_depth=-1,
    # num_leaves=31 và reg_lambda 0/10.
    max_depth: int | None = None
    learning_rate: float = 0.05
    reg_lambda_nominal: float = 0.0
    min_child_weight_nominal: float = 0.001
    num_leaves: int = 31

    def __post_init__(self) -> None:
        if self.family not in {"hgb_uncapped", "hgb_cap125", "lgbm_uncapped", "lgbm_cap125"}:
            raise ValueError(f"Họ mô hình cây chưa biết: {self.family}")
        expected_kind = "hgb" if self.family.startswith("hgb_") else "lightgbm"
        if self.model_kind != expected_kind:
            raise ValueError("family/model_kind không khớp")
        expected_cap = 125 if self.family.endswith("cap125") else None
        if self.target_cap != expected_cap:
            raise ValueError("family/target_cap không khớp")
        if self.min_child_weight_nominal != 0.001:
            raise ValueError("min_child_weight không được hỗ trợ")
        if self.model_kind == "lightgbm":
            if self.max_depth != -1 or self.learning_rate != 0.05 or self.num_leaves != 31:
                raise ValueError("Siêu tham số LightGBM không được hỗ trợ")
            if self.reg_lambda_nominal not in (0.0, 10.0):
                raise ValueError("Regularization LightGBM không được hỗ trợ")
        elif (self.max_depth is not None or self.learning_rate != 0.05
              or self.reg_lambda_nominal != 0.0 or self.num_leaves != 31):
            raise ValueError("HGB phải giữ siêu tham số audit đã cố định")


def specs(config: StudyConfig | None = None) -> tuple[TreeSpec, ...]:
    """Trả 2 candidate HGB và 4 candidate LightGBM.

    ``config`` không được dùng, chỉ để khớp interface mà runner gọi.
    """
    del config
    hgb = (
        TreeSpec("hgb_uncapped", "hgb_uncapped", "hgb", None),
        TreeSpec("hgb_cap125", "hgb_cap125", "hgb", 125),
    )
    lgbm: list[TreeSpec] = []
    for cap_family, cap in (("lgbm_uncapped", None), ("lgbm_cap125", 125)):
        for reg_lambda in (0.0, 10.0):
            label_l2 = f"{reg_lambda:g}"
            candidate_id = f"{cap_family}_leaves31_lr0.05_l2{label_l2}"
            lgbm.append(TreeSpec(
                candidate_id, cap_family, "lightgbm", cap, -1,
                0.05, reg_lambda, 0.001, 31,
            ))
    return hgb + tuple(lgbm)


__all__ = [
    "KEY", "BASE_FEATURES", "HISTORY_FEATURES", "FEATURES", "StudyConfig",
    "TreeSpec", "specs",
]
