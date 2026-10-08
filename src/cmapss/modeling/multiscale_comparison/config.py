"""Cấu hình cho nested HGB multi-scale study."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class StudyConfig:
    """Tham số split, fit HGB, warning policy và điều kiện promotion.

    ``max_workers`` chỉ ảnh hưởng cách chạy process, không đổi kết quả.
    """

    outer_splits: int = 5
    inner_splits: int = 3
    seed: int = 42
    fit_threads: int = 4
    max_workers: int = 2
    bootstrap_resamples: int = 10_000
    hgb_max_iter: int = 200
    hgb_learning_rate: float = 0.05
    hgb_max_leaf_nodes: int = 31
    hgb_min_samples_leaf: int = 20
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
            "hgb_max_leaf_nodes": self.hgb_max_leaf_nodes,
            "hgb_min_samples_leaf": self.hgb_min_samples_leaf,
        }
        if any(isinstance(value, bool) or value < 1 for value in positive.values()):
            raise ValueError("split, resource and HGB values must be positive")
        if self.seed < 0 or isinstance(self.seed, bool):
            raise ValueError("seed must be a nonnegative integer")
        if self.hgb_learning_rate <= 0:
            raise ValueError("hgb_learning_rate must be positive")
        if not self.endpoint_fractions or tuple(sorted(set(self.endpoint_fractions))) != self.endpoint_fractions:
            raise ValueError("endpoint fractions must be unique and increasing")
        if any(not 0 < value < 1 for value in self.endpoint_fractions):
            raise ValueError("endpoint fractions must be strictly inside (0, 1)")
        for name, values in (("thresholds", self.thresholds), ("persistences", self.persistences)):
            if not values or tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be nonempty, unique and increasing")
        if any(value < 0 for value in self.thresholds) or any(value < 1 for value in self.persistences):
            raise ValueError("warning thresholds/persistences are invalid")
        if self.lead_requirement < 1 or not 0 < self.target_on_time_rate <= 1:
            raise ValueError("lead requirement and target on-time rate are invalid")
        margins = (self.endpoint_mae_margin, self.nasa_score_margin,
                   self.warning_rate_margin, self.early_alarm_burden_margin)
        if any(value < 0 for value in margins):
            raise ValueError("evaluation margins cannot be negative")

    @property
    def scientific(self) -> bool:
        """True khi mọi tham số khớp cấu hình đầy đủ của protocol."""
        return (
            self.outer_splits == 5 and self.inner_splits == 3 and self.seed == 42
            and self.bootstrap_resamples >= 10_000 and self.hgb_max_iter == 200
            and self.hgb_learning_rate == 0.05 and self.hgb_max_leaf_nodes == 31
            and self.hgb_min_samples_leaf == 20
            and self.endpoint_fractions == (0.5, 0.7, 0.9)
            and self.thresholds == (30, 35, 40, 45, 50)
            and self.persistences == (1, 3, 5) and self.lead_requirement == 30
            and self.target_on_time_rate == 0.9
            and self.endpoint_mae_margin == 0.0 and self.nasa_score_margin == 0.0
            and self.warning_rate_margin == 0.0 and self.early_alarm_burden_margin == 0.0
        )

    def audit_config(self):
        """Tạo ``AuditStudyConfig`` tương ứng cho phần đánh giá dùng chung."""
        from cmapss.modeling.audit_study import AuditStudyConfig

        return AuditStudyConfig(
            outer_splits=self.outer_splits, inner_splits=self.inner_splits,
            endpoint_fractions=self.endpoint_fractions, thresholds=self.thresholds,
            persistences=self.persistences, lead_requirement=self.lead_requirement,
            target_on_time_rate=self.target_on_time_rate,
            bootstrap_resamples=self.bootstrap_resamples, seed=self.seed,
            fit_threads=self.fit_threads, hgb_max_iter=self.hgb_max_iter,
        )


__all__ = ["StudyConfig"]
