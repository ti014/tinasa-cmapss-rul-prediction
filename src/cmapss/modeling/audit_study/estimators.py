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
    """Mô hình đã fit, thứ tự features và scaler; HGB không dùng scaler."""

    spec: CandidateSpec
    feature_columns: tuple[str, ...]
    scaler: StandardScaler | None
    model: Any

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Dự đoán RUL; báo ValueError khi đầu vào hoặc đầu ra không hữu hạn."""
        values = features.loc[:, list(self.feature_columns)]
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise ValueError("Features suy luận không hữu hạn")
        transformed = self.scaler.transform(values) if self.scaler is not None else values
        prediction = np.asarray(self.model.predict(transformed), dtype="float64")
        if not np.isfinite(prediction).all():
            raise ValueError("Dự đoán mô hình không hữu hạn")
        return prediction


def engine_weights(frame: pd.DataFrame, *, mean_one: bool = False) -> np.ndarray:
    """Trọng số có tổng bằng một mỗi động cơ; mean_one chuẩn hóa trung bình bằng một cho Ridge."""
    if frame.empty or frame.engine_id.isna().any():
        raise ValueError("Trọng số động cơ cần các dòng không rỗng và có định danh")
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
    """Fit loại feature hằng, scaler và hồi quy chỉ trên các dòng train được chỉ định."""
    if train.empty or not train["split"].eq("train").all() or "rul" not in train:
        raise ValueError("Fit yêu cầu chỉ các dòng train với rul gốc")
    columns = BASE_FEATURES + (HISTORY_FEATURES if spec.history else ())
    if spec.prune:
        unique = train.loc[:, list(columns)].nunique(dropna=False)
        columns = tuple(column for column in columns if unique[column] > 1)
    if not columns:
        raise ValueError("Không còn feature khác hằng để fit")
    values = train.loc[:, list(columns)]
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ValueError("Features fit không hữu hạn")
    raw_y = train.rul.to_numpy(dtype="float64")
    if not np.isfinite(raw_y).all() or (raw_y < 0).any():
        raise ValueError("Nhãn fit phải là RUL gốc hữu hạn >=0")
    target = np.minimum(raw_y, spec.target_cap) if spec.target_cap is not None else raw_y
    weights = engine_weights(train, mean_one=spec.model_kind == "ridge")
    scaler = StandardScaler() if spec.model_kind == "ridge" else None
    model = _make_model(spec, config)
    with threadpool_limits(limits=config.fit_threads):
        transformed = scaler.fit_transform(values, sample_weight=weights) if scaler is not None else values
        model.fit(transformed, target, sample_weight=weights)
    return FittedCandidate(spec, columns, scaler, model)
