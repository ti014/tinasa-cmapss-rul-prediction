"""Train-only Health Index từ StandardScaler và PCA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cmapss.data.contracts import KEY, SENSORS, ContractError
from cmapss.data.validation import validate_columns, validate_observations
from cmapss.timeseries.contracts import HEALTH_INDEX_COLUMNS


@dataclass(frozen=True)
class HealthIndexResult:
    """Health Index và transformer đã fit chỉ trên train observations."""

    health_index: pd.DataFrame
    scaler: StandardScaler
    pca: PCA
    orientation_sign: int
    normalization_mean: float
    normalization_scale: float


def _orient_scores(scores: np.ndarray, cycles: np.ndarray) -> tuple[np.ndarray, int]:
    """Chọn hướng để Health Index có xu hướng giảm theo cycle."""
    correlation = float(np.corrcoef(scores, cycles)[0, 1])
    if not np.isfinite(correlation) or correlation <= 0.0:
        return scores, 1
    return -scores, -1


def fit_health_index(observations: pd.DataFrame) -> HealthIndexResult:
    """Fit scaler/PCA trên train engines và trả Health Index (PC1 chuẩn hóa).

    Test observations bị loại trước khi fit, nên transformer chỉ học từ train.
    """
    validate_observations(observations, source="timeseries observations")
    train = (
        observations.loc[observations["split"].eq("train")]
        .sort_values(list(KEY), kind="mergesort")
        .reset_index(drop=True)
    )
    if train.empty:
        raise ContractError("TRAIN_ONLY", "Health Index", "cần train observations để fit PCA")

    fit_values = train.loc[:, list(SENSORS)].to_numpy(dtype="float64")
    scaler = StandardScaler()
    scaler.fit(fit_values)
    pca = PCA(n_components=1, svd_solver="full")
    fit_scaled = scaler.transform(fit_values)
    raw_fit_scores = pca.fit_transform(fit_scaled).ravel()
    oriented_scores, orientation_sign = _orient_scores(
        raw_fit_scores,
        train["cycle"].to_numpy(dtype="float64"),
    )
    scale = float(np.std(oriented_scores, ddof=0))
    if not np.isfinite(scale) or scale == 0.0:
        raise ContractError("HEALTH_INDEX_SCALE", "Health Index", "PC1 không có độ biến thiên hữu hạn")
    center = float(np.mean(oriented_scores))
    normalized = (oriented_scores - center) / scale
    result = train.loc[:, ["split", "engine_id", "cycle"]].copy()
    result["health_index"] = normalized.astype("float64")
    validate_health_index(result)
    return HealthIndexResult(
        health_index=result.loc[:, list(HEALTH_INDEX_COLUMNS)],
        scaler=scaler,
        pca=pca,
        orientation_sign=orientation_sign,
        normalization_mean=center,
        normalization_scale=scale,
    )


def validate_health_index(frame: pd.DataFrame) -> None:
    """Kiểm tra schema, train-only, khóa và giá trị Health Index."""
    validate_columns(frame, HEALTH_INDEX_COLUMNS, "health index")
    if not frame["split"].eq("train").all():
        raise ContractError("TRAIN_ONLY", "Health Index", "Health Index phải chỉ thuộc split train")
    if frame.duplicated(list(KEY)).any():
        raise ContractError("HEALTH_INDEX_KEY", "Health Index", "trùng khóa observation")
    values = frame["health_index"].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ContractError("HEALTH_INDEX_NUMERIC", "Health Index", "Health Index phải hữu hạn")
