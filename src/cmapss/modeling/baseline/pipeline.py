"""Điều phối huấn luyện, chọn model bằng GroupKFold và dự đoán RUL FD001.

Test labels chỉ được đọc trong bước evaluation sau khi predictions đã ghi ra đĩa.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmapss.modeling.baseline.artifact_io import _sha256
from cmapss.modeling.baseline.config import ModelingConfig, ModelingResult
from cmapss.modeling.baseline.dataset import (
    build_group_folds,
    engine_equal_sample_weights,
    prepare_modeling_dataset,
)
from cmapss.modeling.baseline.training import (
    _cross_validate_candidates,
    _fit_estimator,
    _make_estimator,
    _select_final_test_windows,
)
from cmapss.modeling.contracts import MODELING_FEATURE_COLUMNS


def run_modeling_pipeline(
    rolling_features_path: str | Path,
    train_targets_path: str | Path,
    *,
    config: ModelingConfig | None = None,
) -> ModelingResult:
    """Chạy CV, chọn model và tạo final prediction cho từng test engine."""
    run_config = config or ModelingConfig()
    feature_path = Path(rolling_features_path)
    target_path = Path(train_targets_path)
    features = pd.read_parquet(feature_path)
    train_targets = pd.read_parquet(target_path)
    dataset = prepare_modeling_dataset(features, train_targets)
    fold_assignments = build_group_folds(dataset, n_splits=run_config.n_splits)
    oof_predictions, fold_metrics, model_metrics, selected_model = _cross_validate_candidates(
        dataset,
        config=run_config,
    )
    estimator = _make_estimator(selected_model, run_config)
    selected_estimator = _fit_estimator(
        estimator,
        dataset.loc[:, list(MODELING_FEATURE_COLUMNS)],
        dataset["rul"],
        engine_equal_sample_weights(dataset),
    )
    final_predictions = _select_final_test_windows(features, selected_model, selected_estimator)
    return ModelingResult(
        config=run_config,
        selected_model=selected_model,
        selected_estimator=selected_estimator,
        input_sha256={feature_path.name: _sha256(feature_path), target_path.name: _sha256(target_path)},
        fold_assignments=fold_assignments,
        oof_predictions=oof_predictions,
        fold_metrics=fold_metrics,
        model_metrics=model_metrics,
        final_predictions=final_predictions,
    )
