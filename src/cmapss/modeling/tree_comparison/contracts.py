"""Feature, cấu hình và grid candidate cho so sánh HGB/XGBoost trên 169 feature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cmapss.data.contracts import KEY, SENSORS
from cmapss.features.contracts import ROLLING_FEATURE_VALUE_COLUMNS

BASE_FEATURES: tuple[str, ...] = tuple(ROLLING_FEATURE_VALUE_COLUMNS)
HISTORY_FEATURES: tuple[str, ...] = ("cycle",) + tuple(
    f"{sensor}__from_initial_{stat}"
    for sensor in SENSORS
    for stat in ("last", "mean")
)
FEATURES: tuple[str, ...] = BASE_FEATURES + HISTORY_FEATURES

Family = Literal["hgb_uncapped", "hgb_cap125", "xgb_uncapped", "xgb_cap125"]
ModelKind = Literal["hgb", "xgboost"]


@dataclass(frozen=True)
class StudyConfig:
    """Cấu hình split, tài nguyên và tiêu chí chọn mô hình.

    Giá trị mặc định là cấu hình đầy đủ. ``max_workers`` chỉ ảnh hưởng việc
    lập lịch process, không đổi kết quả.
    """

    outer_splits: int = 5
    inner_splits: int = 3
    seed: int = 42
    fit_threads: int = 4
    max_workers: int = 2
    bootstrap_resamples: int = 10_000
    hgb_max_iter: int = 200
    xgb_n_estimators: int = 300
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
            "xgb_n_estimators": self.xgb_n_estimators,
        }
        if any(isinstance(value, bool) or value < 1 for value in positive.values()):
            raise ValueError("split, resource and iteration values must be positive")
        if len(self.endpoint_fractions) == 0 or tuple(sorted(set(self.endpoint_fractions))) != self.endpoint_fractions:
            raise ValueError("endpoint fractions must be unique and increasing")
        if any(not 0.0 < value < 1.0 for value in self.endpoint_fractions):
            raise ValueError("endpoint fractions must be strictly inside (0, 1)")
        for name, values in (("thresholds", self.thresholds), ("persistences", self.persistences)):
            if not values or tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be nonempty, unique and increasing")
        if any(value < 0 for value in self.thresholds) or any(value < 1 for value in self.persistences):
            raise ValueError("warning thresholds/persistences are invalid")
        if self.lead_requirement < 1 or not 0.0 < self.target_on_time_rate <= 1.0:
            raise ValueError("lead_requirement and target_on_time_rate are invalid")
        margins = (self.endpoint_mae_margin, self.nasa_score_margin,
                   self.warning_rate_margin, self.early_alarm_burden_margin)
        if any(value < 0 for value in margins):
            raise ValueError("evaluation margins cannot be negative")

    @property
    def scientific(self) -> bool:
        """True nếu config trùng với cấu hình đầy đủ mặc định."""
        return (
            self.outer_splits == 5 and self.inner_splits == 3 and self.seed == 42
            and self.bootstrap_resamples >= 10_000 and self.hgb_max_iter == 200
            and self.xgb_n_estimators == 300
            and self.endpoint_fractions == (0.5, 0.7, 0.9)
            and self.thresholds == (30, 35, 40, 45, 50)
            and self.persistences == (1, 3, 5) and self.lead_requirement == 30
            and self.target_on_time_rate == 0.9
            and self.endpoint_mae_margin == 0.0 and self.nasa_score_margin == 0.0
            and self.warning_rate_margin == 0.0
            and self.early_alarm_burden_margin == 0.0
        )

    def audit_config(self):
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
    # HGB giữ các giá trị mặc định bên dưới; XGBoost lấy max_depth 3/5/7,
    # learning_rate 0.03/0.1 và reg_lambda 1/10.
    max_depth: int | None = None
    learning_rate: float = 0.05
    reg_lambda_nominal: float = 0.0
    min_child_weight_nominal: float = 20.0

    def __post_init__(self) -> None:
        if self.family not in {"hgb_uncapped", "hgb_cap125", "xgb_uncapped", "xgb_cap125"}:
            raise ValueError(f"unknown tree family: {self.family}")
        expected_kind = "hgb" if self.family.startswith("hgb_") else "xgboost"
        if self.model_kind != expected_kind:
            raise ValueError("family/model_kind mismatch")
        expected_cap = 125 if self.family.endswith("cap125") else None
        if self.target_cap != expected_cap:
            raise ValueError("family/target_cap mismatch")
        if self.min_child_weight_nominal != 20.0:
            raise ValueError("unsupported min_child_weight")
        if self.model_kind == "xgboost":
            if self.max_depth not in (3, 5, 7) or self.learning_rate not in (0.03, 0.1):
                raise ValueError("unsupported XGBoost hyperparameter")
            if self.reg_lambda_nominal not in (1.0, 10.0):
                raise ValueError("unsupported XGBoost regularization")
        elif (self.max_depth is not None or self.learning_rate != 0.05
              or self.reg_lambda_nominal != 0.0):
            raise ValueError("HGB must retain the frozen audit hyperparameters")


def specs(config: StudyConfig | None = None) -> tuple[TreeSpec, ...]:
    """Trả 2 candidate HGB và 24 candidate XGBoost.

    ``config`` không được dùng, chỉ để khớp interface mà runner gọi.
    """
    del config
    hgb = (
        TreeSpec("hgb_uncapped", "hgb_uncapped", "hgb", None),
        TreeSpec("hgb_cap125", "hgb_cap125", "hgb", 125),
    )
    xgb: list[TreeSpec] = []
    for cap_family, cap in (("xgb_uncapped", None), ("xgb_cap125", 125)):
        for depth in (3, 5, 7):
            for learning_rate in (0.03, 0.1):
                for reg_lambda in (1.0, 10.0):
                    label_lr = f"{learning_rate:g}"
                    label_l2 = f"{reg_lambda:g}"
                    candidate_id = f"{cap_family}_d{depth}_lr{label_lr}_l2{label_l2}"
                    xgb.append(TreeSpec(
                        candidate_id, cap_family, "xgboost", cap, depth,
                        learning_rate, reg_lambda,
                    ))
    return hgb + tuple(xgb)


__all__ = [
    "KEY", "BASE_FEATURES", "HISTORY_FEATURES", "FEATURES", "StudyConfig",
    "TreeSpec", "specs",
]
