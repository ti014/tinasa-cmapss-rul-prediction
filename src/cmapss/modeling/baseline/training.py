"""Estimator RUL, chọn model bằng GroupKFold và dự đoán test windows cuối."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY, ContractError
from cmapss.modeling.baseline.config import ModelingConfig
from cmapss.modeling.baseline.dataset import engine_equal_sample_weights
from cmapss.modeling.contracts import (
    FINAL_PREDICTION_COLUMNS,
    FOLD_METRIC_COLUMNS,
    MODELING_FEATURE_COLUMNS,
    MODEL_METRIC_COLUMNS,
    OOF_PREDICTION_COLUMNS,
)
from cmapss.modeling.metrics import add_prediction_errors, summarize_prediction_errors


class WeightedMedianRegressor(BaseEstimator, RegressorMixin):
    """Baseline hằng bằng weighted median RUL của training fold."""

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "WeightedMedianRegressor":
        del X
        self.constant_ = _weighted_median(np.asarray(y, dtype="float64"), sample_weight)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.full(len(X), self.constant_, dtype="float64")


def _weighted_median(values: np.ndarray, sample_weight: np.ndarray | None) -> float:
    if sample_weight is None:
        return float(np.median(values))
    weights = np.asarray(sample_weight, dtype="float64")
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * ordered_weights.sum()
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left")])


def _make_estimator(name: str, config: ModelingConfig) -> Any:
    if name == "median_baseline":
        return WeightedMedianRegressor()
    if name == "ridge_alpha_1":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=config.ridge_alpha)),
            ]
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=config.hgb_learning_rate,
            max_iter=config.hgb_max_iter,
            max_leaf_nodes=config.hgb_max_leaf_nodes,
            min_samples_leaf=config.hgb_min_samples_leaf,
            early_stopping=False,
            random_state=config.seed,
        )
    raise ValueError(f"Unknown candidate model: {name}")


def _fit_estimator(estimator: Any, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray) -> Any:
    if isinstance(estimator, Pipeline):
        return estimator.fit(X, y, model__sample_weight=sample_weight)
    return estimator.fit(X, y, sample_weight=sample_weight)


def _prediction_frame(
    rows: pd.DataFrame,
    *,
    fold: int,
    model_name: str,
    predictions: np.ndarray,
) -> pd.DataFrame:
    frame = rows.loc[:, list(KEY)].copy()
    frame["fold"] = fold
    frame["model_name"] = model_name
    frame["y_true"] = rows["rul"].to_numpy(dtype="float64")
    frame["y_pred"] = predictions.astype("float64")
    frame = add_prediction_errors(frame)
    return frame.loc[:, list(OOF_PREDICTION_COLUMNS)]


def _cross_validate_candidates(
    dataset: pd.DataFrame,
    *,
    config: ModelingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    X = dataset.loc[:, list(MODELING_FEATURE_COLUMNS)]
    y = dataset["rul"]
    groups = dataset["engine_id"].to_numpy()
    splitter = GroupKFold(n_splits=config.n_splits)
    oof_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []

    for model_name in config.candidate_models:
        for fold, (train_index, validation_index) in enumerate(splitter.split(X, y, groups), start=1):
            train_rows = dataset.iloc[train_index]
            validation_rows = dataset.iloc[validation_index]
            estimator = _make_estimator(model_name, config)
            weights = engine_equal_sample_weights(train_rows)
            fitted = _fit_estimator(
                estimator,
                train_rows.loc[:, list(MODELING_FEATURE_COLUMNS)],
                train_rows["rul"],
                weights,
            )
            predictions = fitted.predict(validation_rows.loc[:, list(MODELING_FEATURE_COLUMNS)])
            prediction_frame = _prediction_frame(
                validation_rows,
                fold=fold,
                model_name=model_name,
                predictions=np.asarray(predictions, dtype="float64"),
            )
            oof_frames.append(prediction_frame)
            metric_records.append(
                {
                    "model_name": model_name,
                    "fold": fold,
                    **summarize_prediction_errors(prediction_frame),
                }
            )

    oof = pd.concat(oof_frames, ignore_index=True).loc[:, list(OOF_PREDICTION_COLUMNS)]
    fold_metrics = pd.DataFrame(metric_records, columns=FOLD_METRIC_COLUMNS)
    model_metrics = (
        fold_metrics.groupby("model_name", sort=True)
        .agg(
            n_folds=("fold", "nunique"),
            mean_engine_mae=("mean_engine_mae", "mean"),
            std_engine_mae=("mean_engine_mae", "std"),
            mean_rmse=("rmse", "mean"),
            mean_nasa_score=("nasa_score", "mean"),
        )
        .reset_index()
    )
    model_metrics["std_engine_mae"] = model_metrics["std_engine_mae"].fillna(0.0)
    selected_model = str(
        model_metrics.sort_values(
            ["mean_engine_mae", "mean_rmse", "mean_nasa_score", "model_name"],
            kind="mergesort",
        )
        .iloc[0]["model_name"]
    )
    model_metrics["selected"] = model_metrics["model_name"].eq(selected_model)
    model_metrics = model_metrics.loc[:, list(MODEL_METRIC_COLUMNS)]
    return oof, fold_metrics, model_metrics, selected_model


def _select_final_test_windows(features: pd.DataFrame, model_name: str, estimator: Any) -> pd.DataFrame:
    test_features = (
        features.loc[features["split"].eq("test")]
        .sort_values(list(KEY), kind="mergesort")
        .groupby("engine_id", as_index=False, sort=True)
        .tail(1)
        .reset_index(drop=True)
    )
    predictions = estimator.predict(test_features.loc[:, list(MODELING_FEATURE_COLUMNS)])
    final = test_features.loc[:, list(KEY)].copy()
    final["model_name"] = model_name
    final["rul_pred"] = np.asarray(predictions, dtype="float64")
    if not np.isfinite(final["rul_pred"].to_numpy(dtype="float64")).all():
        raise ContractError("MODELING_PREDICTION", "final predictions", "mọi prediction phải hữu hạn")
    return final.loc[:, list(FINAL_PREDICTION_COLUMNS)].reset_index(drop=True)
