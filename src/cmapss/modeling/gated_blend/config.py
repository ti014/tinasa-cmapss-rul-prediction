"""
Cấu hình HGB có gate, dùng lại mô hình 200 vòng và chia 5×3 theo động cơ; smoke chỉ giảm
lưới gate và số lượt bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass

from cmapss.modeling.audit_study import AuditStudyConfig

from .contracts import GateSpec, build_specs


@dataclass(frozen=True)
class StudyConfig:
    """Thiết lập cố định; smoke=True không tạo kết quả khoa học."""

    outer_splits: int = 5
    inner_splits: int = 3
    seed: int = 42
    fit_threads: int = 4
    hgb_max_iter: int = 200
    bootstrap_resamples: int = 10_000
    endpoint_fractions: tuple[float, ...] = (0.5, 0.7, 0.9)
    thresholds: tuple[int, ...] = (30, 35, 40, 45, 50)
    persistences: tuple[int, ...] = (1, 3, 5)
    lead_requirement: int = 30
    target_on_time_rate: float = 0.9
    endpoint_mae_margin: float = 0.0
    nasa_score_margin: float = 0.0
    warning_rate_margin: float = 0.0
    early_alarm_burden_margin: float = 0.0
    reference_policy_source: str = "source_stage_inner_calibration"
    lower_ruls: tuple[float, ...] = (60.0, 90.0)
    upper_ruls: tuple[float, ...] = (125.0, 150.0)
    max_capped_weights: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    smoke: bool = False

    def __post_init__(self) -> None:
        if (self.outer_splits != 5 or self.inner_splits != 3 or self.seed != 42
                or self.fit_threads != 4 or self.hgb_max_iter != 200):
            raise ValueError("Nghiên cứu dẫn xuất cần giữ protocol gốc 5x3/seed42/HGB200/threads4")
        if (isinstance(self.bootstrap_resamples, bool)
                or not isinstance(self.bootstrap_resamples, int) or self.bootstrap_resamples < 1):
            raise ValueError("bootstrap_resamples phải là số nguyên dương")
        if not isinstance(self.smoke, bool):
            raise ValueError("smoke phải là boolean")
        if (self.endpoint_fractions != (0.5, 0.7, 0.9)
                or self.thresholds != (30, 35, 40, 45, 50)
                or self.persistences != (1, 3, 5) or self.lead_requirement != 30
                or self.target_on_time_rate != 0.9
                or self.reference_policy_source != "source_stage_inner_calibration"
                or any(value != 0.0 for value in (
                    self.endpoint_mae_margin, self.nasa_score_margin,
                    self.warning_rate_margin, self.early_alarm_burden_margin))):
            raise ValueError("Nghiên cứu dẫn xuất phải giữ kiểm soát endpoint và cảnh báo gốc")
        build_specs(lower_ruls=self.lower_ruls, upper_ruls=self.upper_ruls,
                    max_capped_weights=self.max_capped_weights)

    @property
    def scientific(self) -> bool:
        """Chỉ True với đủ 16 gate và ít nhất 10.000 lượt bootstrap."""
        return (not self.smoke and self.bootstrap_resamples >= 10_000
                and self.lower_ruls == (60., 90.) and self.upper_ruls == (125., 150.)
                and self.max_capped_weights == (.25, .5, .75, 1.))

    def audit_config(self) -> AuditStudyConfig:
        """Cấu hình chung đo endpoint và cảnh báo."""
        return AuditStudyConfig(
            outer_splits=self.outer_splits, inner_splits=self.inner_splits,
            seed=self.seed, fit_threads=self.fit_threads, hgb_max_iter=self.hgb_max_iter,
            bootstrap_resamples=self.bootstrap_resamples,
            endpoint_fractions=self.endpoint_fractions, thresholds=self.thresholds,
            persistences=self.persistences, lead_requirement=self.lead_requirement,
            target_on_time_rate=self.target_on_time_rate,
        )


def specs(config: StudyConfig) -> tuple[GateSpec, ...]:
    """Lưới gate theo tích Descartes của cấu hình."""
    return build_specs(lower_ruls=config.lower_ruls, upper_ruls=config.upper_ruls,
                       max_capped_weights=config.max_capped_weights)


__all__ = ["StudyConfig", "specs"]
