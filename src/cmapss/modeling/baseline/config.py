"""Cấu hình huấn luyện và kết quả modeling trong bộ nhớ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from cmapss.modeling.contracts import (
    MODELING_BOOTSTRAP_RESAMPLES,
    MODELING_N_SPLITS,
    MODELING_SEED,
)


@dataclass(frozen=True)
class ModelingConfig:
    """Cấu hình CV và hyperparameter cho các model RUL ứng viên."""

    n_splits: int = MODELING_N_SPLITS
    seed: int = MODELING_SEED
    candidate_models: tuple[str, ...] = (
        "median_baseline",
        "ridge_alpha_1",
        "hist_gradient_boosting",
    )
    ridge_alpha: float = 1.0
    hgb_learning_rate: float = 0.05
    hgb_max_iter: int = 200
    hgb_max_leaf_nodes: int = 31
    hgb_min_samples_leaf: int = 20
    bootstrap_resamples: int = MODELING_BOOTSTRAP_RESAMPLES


@dataclass(frozen=True)
class ModelingResult:
    """Artifact trong bộ nhớ trước khi ghi ra `outputs/modeling`."""

    config: ModelingConfig
    selected_model: str
    selected_estimator: Any
    input_sha256: dict[str, str]
    fold_assignments: pd.DataFrame
    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    model_metrics: pd.DataFrame
    final_predictions: pd.DataFrame
