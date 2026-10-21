"""Cấu hình và candidate đã khai báo cho study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cmapss.data.contracts import SENSORS
from cmapss.features.contracts import ROLLING_FEATURE_VALUE_COLUMNS


BASE_FEATURES = tuple(ROLLING_FEATURE_VALUE_COLUMNS)


HISTORY_FEATURES = ("cycle",) + tuple(
    f"{sensor}__from_initial_{stat}" for sensor in SENSORS for stat in ("last", "mean")
)


@dataclass(frozen=True)
class CandidateSpec:
    """
    Biến thể mô hình: history thêm cycle và độ lệch từ 30 cycle đầu, prune bỏ cột hằng
    trong fold, target_cap chỉ cắt nhãn train; loss chỉ dùng cho HGB.
    """

    candidate_id: str
    model_kind: str = "hist_gradient_boosting"
    history: bool = False
    prune: bool = False
    target_cap: float | None = None
    loss: str = "squared_error"
    alpha: float = 1.0


def make_candidate_specs() -> tuple[CandidateSpec, ...]:
    """Sáu biến thể HGB và bốn giá trị alpha của Ridge."""
    return (
        CandidateSpec("hgb_baseline"),
        CandidateSpec("hgb_pruned", prune=True),
        CandidateSpec("hgb_cap125", target_cap=125.0),
        CandidateSpec("hgb_history", history=True),
        CandidateSpec("hgb_absolute_error", loss="absolute_error"),
        CandidateSpec("hgb_cap125_history", history=True, target_cap=125.0),
        *(CandidateSpec(f"ridge_alpha_{alpha:g}", model_kind="ridge", prune=True, alpha=alpha)
          for alpha in (0.1, 1.0, 10.0, 100.0)),
    )


@dataclass(frozen=True)
class AuditStudyConfig:
    """
    Cấu hình nghiên cứu: lưới thresholds/persistences, lead_requirement tối thiểu và
    target_on_time_rate mục tiêu.
    """

    outer_splits: int = 5
    inner_splits: int = 3
    endpoint_fractions: tuple[float, ...] = (0.5, 0.7, 0.9)
    candidate_ids: tuple[str, ...] = tuple(spec.candidate_id for spec in make_candidate_specs())
    thresholds: tuple[int, ...] = (30, 35, 40, 45, 50)
    persistences: tuple[int, ...] = (1, 3, 5)
    lead_requirement: int = 30
    target_on_time_rate: float = 0.9
    bootstrap_resamples: int = 10_000
    seed: int = 42
    fit_threads: int = 4
    hgb_max_iter: int = 200

    def __post_init__(self) -> None:
        if min(self.outer_splits, self.inner_splits) < 2:
            raise ValueError("Cần ít nhất hai inner fold và hai outer fold")
        if min(self.bootstrap_resamples, self.fit_threads, self.hgb_max_iter) < 1:
            raise ValueError("Số lượt lấy mẫu, luồng và vòng lặp phải dương")
        if not self.endpoint_fractions or tuple(sorted(set(self.endpoint_fractions))) != self.endpoint_fractions:
            raise ValueError("Các tỷ lệ endpoint phải duy nhất và tăng dần")
        if any(not 0 < fraction < 1 for fraction in self.endpoint_fractions):
            raise ValueError("Các tỷ lệ endpoint phải nằm hẳn trong (0, 1)")
        available = {spec.candidate_id for spec in make_candidate_specs()}
        if (not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids)
                or not set(self.candidate_ids).issubset(available)
                or "hgb_baseline" not in self.candidate_ids):
            raise ValueError("ID ứng viên phải duy nhất, đã biết và gồm hgb_baseline")
        if not 0 < self.target_on_time_rate <= 1 or self.lead_requirement < 1:
            raise ValueError("Mục tiêu cảnh báo phải thuộc (0, 1] và lead phải dương")
        if (not self.thresholds or not self.persistences
                or any(value < 0 for value in self.thresholds)
                or any(value < 1 for value in self.persistences)):
            raise ValueError("Lưới cảnh báo phải không rỗng và hợp lệ")
        if (tuple(sorted(set(self.thresholds))) != self.thresholds
                or tuple(sorted(set(self.persistences))) != self.persistences):
            raise ValueError("Lưới cảnh báo phải duy nhất và tăng dần")
        for threshold in self.thresholds:
            for persistence in self.persistences:
                WarningPolicy(threshold, persistence)


@dataclass(frozen=True)
class WarningPolicy:
    """Cảnh báo sau persistence dự đoán liên tiếp có RUL không vượt threshold."""

    threshold: int
    persistence: int

    def __post_init__(self) -> None:
        if (isinstance(self.threshold, bool) or isinstance(self.persistence, bool)
                or not isinstance(self.threshold, (int, np.integer))
                or not isinstance(self.persistence, (int, np.integer))
                or self.threshold < 0 or self.persistence < 1):
            raise ValueError("Chính sách cảnh báo không hợp lệ")
