"""Cấu hình và candidate đã khai báo cho study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


WINDOWS = (10, 20, 30)


LOW_RUL_THRESHOLD = 30


BASELINE_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}


WEIGHT_SCHEMES = {
    "uniform": (1.0, 1.0, 1.0),
    "moderate_low_rul": (1.0, 1.5, 2.0),
    "strong_low_rul": (1.0, 2.0, 3.0),
}


@dataclass(frozen=True)
class ImprovementConfig:
    """Settings for one improvement run."""

    seed: int = 42
    outer_splits: int = 5
    inner_splits: int = 3
    low_rul_threshold: int = LOW_RUL_THRESHOLD
    min_cycle: int = 30
    n_hgb_candidates: int = 24
    bootstrap_resamples: int = 10_000
    fit_threads: int = 4

    def __post_init__(self) -> None:
        if not 1 <= self.n_hgb_candidates <= 243:
            raise ValueError("n_hgb_candidates must be between 1 and 243")
        if self.min_cycle != 30 or self.low_rul_threshold != 30:
            raise ValueError("the accepted protocol fixes min_cycle and threshold at 30")
        if min(self.inner_splits, self.outer_splits) < 2:
            raise ValueError("at least two folds are required")
        if self.bootstrap_resamples < 1 or self.fit_threads < 1:
            raise ValueError("resamples and fit_threads must be positive")


def make_hgb_candidates(config: ImprovementConfig) -> list[dict[str, Any]]:
    """Baseline params plus random unique HGB configs (seeded)."""
    candidates: list[dict[str, Any]] = [{**BASELINE_PARAMS}]
    rng = np.random.default_rng(config.seed)
    choices = {
        "learning_rate": (0.03, 0.05, 0.1),
        "max_iter": (100, 200, 400),
        "max_leaf_nodes": (15, 31, 63),
        "min_samples_leaf": (10, 20, 40),
        "l2_regularization": (0.0, 0.01, 0.1),
    }
    seen = {tuple(BASELINE_PARAMS.items())}
    while len(candidates) < config.n_hgb_candidates:
        candidate = {
            name: values[int(rng.integers(0, len(values)))]
            for name, values in choices.items()
        }
        key = tuple(candidate.items())
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates
