"""Fitting state cho HGB và LightGBM tree comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .contracts import FEATURES, StudyConfig, TreeSpec


def engine_total_one_weights(frame: pd.DataFrame) -> np.ndarray:
    """Tạo sample weights sao cho tổng trọng số mỗi engine bằng một.

    Trọng số này được dùng nguyên vẹn cho cả HGB và LightGBM.  Với một engine
    có 100 dòng, mỗi dòng nhận trọng số ``1/100``.
    """
    if frame.empty or "engine_id" not in frame or frame["engine_id"].isna().any():
        raise ValueError("engine weights require nonempty rows with engine_id")
    counts = frame.groupby("engine_id", sort=False)["engine_id"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype="float64")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("engine weights are not finite and positive")
    return weights


@dataclass
class FittedTree:
    """Model cùng trạng thái feature cần thiết để tái sử dụng an toàn."""

    spec: TreeSpec
    feature_columns: tuple[str, ...]
    model: Any
    effective_parameters: dict[str, float | int | str | None]
    weight_metadata: dict[str, float | int | str]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Dự đoán từ feature columns đã đóng băng, không đọc nhãn."""
        missing = [column for column in self.feature_columns if column not in frame]
        if missing:
            raise ValueError(f"missing inference features: {missing[:3]}")
        values = frame.loc[:, list(self.feature_columns)].to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError("nonfinite inference features")
        with threadpool_limits(limits=int(self.weight_metadata.get("fit_threads", 1))):
            prediction = np.asarray(self.model.predict(values), dtype="float64")
        if prediction.ndim != 1 or len(prediction) != len(frame) or not np.isfinite(prediction).all():
            raise ValueError("nonfinite or malformed model predictions")
        return prediction


def _build_lgbm(spec: TreeSpec, config: StudyConfig, effective: dict[str, float | int | str | None]) -> Any:
    """Construct LGBMRegressor lazily so HGB-only imports remain usable."""
    try:
        from lightgbm import LGBMRegressor
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "LightGBM is required for lgbm candidates; install the pinned dependency"
        ) from error
    return LGBMRegressor(
        n_estimators=config.lgbm_n_estimators,
        num_leaves=spec.num_leaves,
        max_depth=-1,
        learning_rate=spec.learning_rate,
        min_child_weight=effective["min_child_weight"],
        min_child_samples=20,
        reg_lambda=effective["reg_lambda"],
        reg_alpha=0.0,
        min_split_gain=0.0,
        subsample=1.0,
        colsample_bytree=1.0,
        max_bin=255,
        objective="regression",
        n_jobs=config.fit_threads,
        random_state=config.seed,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def fit_predictor(train: pd.DataFrame, spec: TreeSpec, config: StudyConfig) -> FittedTree:
    """Fit one tree candidate on disjoint-engine training rows.

    Target caps are applied to fitting labels only.  The caller retains raw
    ``rul`` in validation rows, so endpoint and trajectory metrics remain on
    the same reporting scale for all candidates.
    """
    required = {"engine_id", "rul", *FEATURES}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"fitting frame misses required columns: {missing[:5]}")
    if train.empty or ("split" in train and not train["split"].eq("train").all()):
        raise ValueError("tree fitting requires nonempty train-only rows")
    values = train.loc[:, list(FEATURES)]
    matrix = values.to_numpy(dtype="float64")
    raw_y = train["rul"].to_numpy(dtype="float64")
    if not np.isfinite(matrix).all() or not np.isfinite(raw_y).all() or (raw_y < 0).any():
        raise ValueError("fitting features and raw RUL must be finite, with RUL >= 0")
    target = np.minimum(raw_y, spec.target_cap) if spec.target_cap is not None else raw_y
    weights = engine_total_one_weights(train)
    n_rows = len(train)
    n_engines = int(train["engine_id"].nunique())
    effective: dict[str, float | int | str | None] = {
        "model_kind": spec.model_kind,
        "n_rows": n_rows,
        "n_engines": n_engines,
        "target_cap": None if spec.target_cap is None else int(spec.target_cap),
    }
    if spec.model_kind == "lightgbm":
        scale = n_engines / n_rows
        effective.update({
            "min_child_weight_nominal": spec.min_child_weight_nominal,
            "reg_lambda_nominal": spec.reg_lambda_nominal,
            "min_child_weight": spec.min_child_weight_nominal * scale,
            "reg_lambda": spec.reg_lambda_nominal * scale,
            "min_child_samples": 20,
            "num_leaves": spec.num_leaves,
            "max_bin": 255,
            "deterministic": True,
            "force_col_wise": True,
        })
        model = _build_lgbm(spec, config, effective)
    else:
        effective.update({
            "max_iter": config.hgb_max_iter,
            "max_depth": "none",
            "learning_rate": 0.05,
            "reg_lambda": 0.0,
        })
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=config.hgb_max_iter,
            max_leaf_nodes=31,
            max_depth=None,
            min_samples_leaf=20,
            l2_regularization=0.0,
            early_stopping=False,
            random_state=config.seed,
        )
    with threadpool_limits(limits=config.fit_threads):
        model.fit(matrix, target, sample_weight=weights)
    weight_metadata: dict[str, float | int | str | None] = {
        "scheme": "engine_total_one",
        "n_rows": n_rows,
        "n_engines": n_engines,
        "total_weight": float(weights.sum()),
        "min_row_weight": float(weights.min()),
        "max_row_weight": float(weights.max()),
        "fit_threads": config.fit_threads,
    }
    return FittedTree(spec, FEATURES, model, effective, weight_metadata)


__all__ = ["FittedTree", "engine_total_one_weights", "fit_predictor"]
