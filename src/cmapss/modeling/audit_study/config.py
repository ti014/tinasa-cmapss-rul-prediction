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
    """One model variant.

    ``history`` adds cycle and sensor values relative to the first 30 cycles.
    ``prune`` drops features that are constant in the fitting fold.
    ``target_cap`` clips training labels only. ``loss`` applies to HGB only.
    """

    candidate_id: str
    model_kind: str = "hist_gradient_boosting"
    history: bool = False
    prune: bool = False
    target_cap: float | None = None
    loss: str = "squared_error"
    alpha: float = 1.0


def make_candidate_specs() -> tuple[CandidateSpec, ...]:
    """Six HGB variants and four Ridge alphas."""
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
    """Study settings.

    ``thresholds`` and ``persistences`` form the warning-policy grid.
    ``lead_requirement`` is the minimum lead (cycles) for a warning to count as
    on time; ``target_on_time_rate`` is the on-time rate a policy should reach.
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
            raise ValueError("at least two inner and outer folds are required")
        if min(self.bootstrap_resamples, self.fit_threads, self.hgb_max_iter) < 1:
            raise ValueError("resamples, threads and iterations must be positive")
        if not self.endpoint_fractions or tuple(sorted(set(self.endpoint_fractions))) != self.endpoint_fractions:
            raise ValueError("endpoint fractions must be unique and increasing")
        if any(not 0 < fraction < 1 for fraction in self.endpoint_fractions):
            raise ValueError("endpoint fractions must be strictly inside (0, 1)")
        available = {spec.candidate_id for spec in make_candidate_specs()}
        if (not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids)
                or not set(self.candidate_ids).issubset(available)
                or "hgb_baseline" not in self.candidate_ids):
            raise ValueError("candidate IDs must be unique, known and include hgb_baseline")
        if not 0 < self.target_on_time_rate <= 1 or self.lead_requirement < 1:
            raise ValueError("warning target must be in (0, 1] and lead positive")
        if (not self.thresholds or not self.persistences
                or any(value < 0 for value in self.thresholds)
                or any(value < 1 for value in self.persistences)):
            raise ValueError("warning grids must be nonempty and valid")
        if (tuple(sorted(set(self.thresholds))) != self.thresholds
                or tuple(sorted(set(self.persistences))) != self.persistences):
            raise ValueError("warning grids must be unique and increasing")
        for threshold in self.thresholds:
            for persistence in self.persistences:
                WarningPolicy(threshold, persistence)


@dataclass(frozen=True)
class WarningPolicy:
    """Warn after ``persistence`` consecutive predictions with RUL <= ``threshold``."""

    threshold: int
    persistence: int

    def __post_init__(self) -> None:
        if (isinstance(self.threshold, bool) or isinstance(self.persistence, bool)
                or not isinstance(self.threshold, (int, np.integer))
                or not isinstance(self.persistence, (int, np.integer))
                or self.threshold < 0 or self.persistence < 1):
            raise ValueError("invalid warning policy")
