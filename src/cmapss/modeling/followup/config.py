"""Cấu hình và candidate đã khai báo cho study."""

from __future__ import annotations

from dataclasses import dataclass

from cmapss.features.enhanced import FeatureRecipe


@dataclass(frozen=True)
class StudyConfig:
    """Cấu hình nested cross-validation được ghi trong manifest nghiên cứu."""

    outer_splits: int = 5
    inner_splits: int = 3
    seed: int = 42
    bootstrap_resamples: int = 10_000
    fit_threads: int = 4
    max_iter: int = 200

    def __post_init__(self) -> None:
        if min(self.outer_splits, self.inner_splits) < 2:
            raise ValueError("Cần ít nhất hai inner fold và hai outer fold")
        if min(self.bootstrap_resamples, self.fit_threads, self.max_iter) < 1:
            raise ValueError("Số lượt lấy mẫu, luồng và vòng lặp phải dương")


def study_candidates(dataset_id: str) -> tuple[FeatureRecipe, ...]:
    """Danh sách ứng viên cố định của một dataset."""
    if dataset_id == "FD001":
        return tuple(
            FeatureRecipe(
                name=("baseline" if cap is None and not history else f"cap_{cap or 'none'}_history_{int(history)}"),
                target_cap=cap,
                history=history,
            )
            for history in (False, True)
            for cap in (None, 110, 125, 140)
        )
    if dataset_id not in {"FD002", "FD004"}:
        raise ValueError("Nghiên cứu đã chốt chỉ gồm FD001, FD002 và FD004")
    return (
        FeatureRecipe("baseline", window=10),
        FeatureRecipe("raw_settings", window=10, settings=True),
        FeatureRecipe("condition_normalized", window=10, settings=True, normalize_conditions=True),
    )
