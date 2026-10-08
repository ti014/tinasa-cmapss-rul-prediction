"""HGB regressor fit and predict for the three feature arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .config import StudyConfig
from .contracts import feature_spec
from cmapss.modeling.lightgbm_comparison.estimators import engine_total_one_weights


@dataclass
class FittedHGB:
    """Fitted HGB model with its feature order and fit metadata."""

    arm_id: str
    feature_columns: tuple[str, ...]
    model: Any
    effective_parameters: dict[str, Any]
    weight_metadata: dict[str, Any]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict raw RUL from the arm's feature columns."""
        values = frame.loc[:, list(self.feature_columns)].to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError("nonfinite inference features")
        with threadpool_limits(limits=self.weight_metadata["fit_threads"]):
            prediction = np.asarray(self.model.predict(values), dtype="float64")
        if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
            raise ValueError("malformed or nonfinite predictions")
        return prediction


def fit_predictor(train: pd.DataFrame, arm_id: str, config: StudyConfig) -> FittedHGB:
    """Fit raw RUL with one total unit of sample weight per engine.

    For example, engines with 100 and 200 rows receive row weights 1/100 and
    1/200 respectively.
    """
    spec = feature_spec(arm_id)
    if train.empty or not train.split.eq("train").all() or "rul" not in train:
        raise ValueError("fitting requires nonempty train-only raw targets")
    values = train.loc[:, list(spec.feature_columns)].to_numpy(dtype="float64")
    target = train.rul.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("fitting features and raw targets must be finite")
    weights = engine_total_one_weights(train)
    parameters = dict(
        loss="squared_error", learning_rate=config.hgb_learning_rate,
        max_iter=config.hgb_max_iter, max_leaf_nodes=config.hgb_max_leaf_nodes,
        max_depth=None, min_samples_leaf=config.hgb_min_samples_leaf,
        l2_regularization=0.0, early_stopping=False, random_state=config.seed,
    )
    model = HistGradientBoostingRegressor(**parameters)
    with threadpool_limits(limits=config.fit_threads):
        model.fit(values, target, sample_weight=weights)
    return FittedHGB(arm_id, spec.feature_columns, model, parameters, {
        "scheme": "engine_total_one", "total_weight": float(weights.sum()),
        "n_rows": len(train), "n_engines": int(train.engine_id.nunique()),
        "min_row_weight": float(weights.min()), "max_row_weight": float(weights.max()),
        "fit_threads": config.fit_threads, "target_cap": None,
    })


__all__ = ["FittedHGB", "fit_predictor", "engine_total_one_weights"]
