"""Fit estimator và inference trên features đã chọn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd

from cmapss.modeling.audit_study.config import (
    AuditStudyConfig,
    BASE_FEATURES,
    CandidateSpec,
    HISTORY_FEATURES,
)


@dataclass
class FittedCandidate:
    """Fitted model with its feature columns and scaler (None for HGB)."""

    spec: CandidateSpec
    feature_columns: tuple[str, ...]
    scaler: StandardScaler | None
    model: Any

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict RUL; raises ValueError on non-finite inputs or outputs."""
        values = features.loc[:, list(self.feature_columns)]
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise ValueError("nonfinite inference features")
        transformed = self.scaler.transform(values) if self.scaler is not None else values
        prediction = np.asarray(self.model.predict(transformed), dtype="float64")
        if not np.isfinite(prediction).all():
            raise ValueError("nonfinite model predictions")
        return prediction


def engine_weights(frame: pd.DataFrame, *, mean_one: bool = False) -> np.ndarray:
    """Row weights giving each engine the same total weight.

    Each engine sums to one, or the weights are rescaled to mean one if
    ``mean_one`` is set (used for Ridge).
    """
    if frame.empty or frame.engine_id.isna().any():
        raise ValueError("engine weights require nonempty, identified rows")
    counts = frame.groupby("engine_id").engine_id.transform("size").to_numpy(dtype="float64")
    weights = 1. / counts
    return weights / weights.mean() if mean_one else weights


def _make_model(spec: CandidateSpec, config: AuditStudyConfig) -> Any:
    if spec.model_kind == "ridge":
        return Ridge(alpha=spec.alpha, solver="svd")
    return HistGradientBoostingRegressor(
        loss=spec.loss, learning_rate=0.05, max_iter=config.hgb_max_iter,
        max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0,
        early_stopping=False, random_state=config.seed,
    )


def fit_candidate(train: pd.DataFrame, spec: CandidateSpec,
                  config: AuditStudyConfig) -> FittedCandidate:
    """Fit feature pruning, scaler and regressor on the given train rows only."""
    if train.empty or not train["split"].eq("train").all() or "rul" not in train:
        raise ValueError("fitting requires train-only rows with raw rul")
    columns = BASE_FEATURES + (HISTORY_FEATURES if spec.history else ())
    if spec.prune:
        unique = train.loc[:, list(columns)].nunique(dropna=False)
        columns = tuple(column for column in columns if unique[column] > 1)
    if not columns:
        raise ValueError("no nonconstant fitting features remain")
    values = train.loc[:, list(columns)]
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ValueError("nonfinite fitting features")
    raw_y = train.rul.to_numpy(dtype="float64")
    if not np.isfinite(raw_y).all() or (raw_y < 0).any():
        raise ValueError("fitting labels must be finite raw RUL >=0")
    target = np.minimum(raw_y, spec.target_cap) if spec.target_cap is not None else raw_y
    weights = engine_weights(train, mean_one=spec.model_kind == "ridge")
    scaler = StandardScaler() if spec.model_kind == "ridge" else None
    model = _make_model(spec, config)
    with threadpool_limits(limits=config.fit_threads):
        transformed = scaler.fit_transform(values, sample_weight=weights) if scaler is not None else values
        model.fit(transformed, target, sample_weight=weights)
    return FittedCandidate(spec, columns, scaler, model)
