"""Cấu hình và candidate đã khai báo cho study."""

from __future__ import annotations

from dataclasses import dataclass

from cmapss.features.enhanced import FeatureRecipe


@dataclass(frozen=True)
class StudyConfig:
    """Nested CV settings, recorded in the study manifest."""

    outer_splits: int = 5
    inner_splits: int = 3
    seed: int = 42
    bootstrap_resamples: int = 10_000
    fit_threads: int = 4
    max_iter: int = 200

    def __post_init__(self) -> None:
        if min(self.outer_splits, self.inner_splits) < 2:
            raise ValueError("At least two inner and outer folds are required")
        if min(self.bootstrap_resamples, self.fit_threads, self.max_iter) < 1:
            raise ValueError("Resamples, threads and iterations must be positive")


def study_candidates(dataset_id: str) -> tuple[FeatureRecipe, ...]:
    """Fixed candidate list for a dataset."""
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
        raise ValueError("The predeclared study covers FD001, FD002 and FD004")
    return (
        FeatureRecipe("baseline", window=10),
        FeatureRecipe("raw_settings", window=10, settings=True),
        FeatureRecipe("condition_normalized", window=10, settings=True, normalize_conditions=True),
    )
