"""Train-only Health Index từ StandardScaler và PCA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cmapss.data.contracts import KEY, SENSORS, ContractError
from cmapss.data.validation import validate_columns, validate_observations
from cmapss.timeseries.contracts import HEALTH_INDEX_COLUMNS, SELECTED_ENGINE_COLUMNS


@dataclass(frozen=True)
class HealthIndexResult:
    """Health Index và transformer đã fit chỉ trên train observations."""

    health_index: pd.DataFrame
    scaler: StandardScaler
    pca: PCA
    orientation_sign: int


def _orient_scores(scores: np.ndarray, cycles: np.ndarray) -> tuple[np.ndarray, int]:
    """Chọn hướng để Health Index có xu hướng giảm theo cycle."""
    correlation = float(np.corrcoef(scores, cycles)[0, 1])
    if not np.isfinite(correlation) or correlation <= 0.0:
        return scores, 1
    return -scores, -1


def fit_health_index(observations: pd.DataFrame) -> HealthIndexResult:
    """Fit scaler/PCA trên train sensors và trả Health Index chuẩn hóa.

    Test observations vẫn có thể tồn tại trong input Silver, nhưng luôn bị
    loại trước khi fit và không xuất hiện trong output.
    """
    validate_observations(observations, source="timeseries observations")
    train = (
        observations.loc[observations["split"].eq("train")]
        .sort_values(list(KEY), kind="mergesort")
        .reset_index(drop=True)
    )
    if train.empty:
        raise ContractError("TRAIN_ONLY", "health index", "cần train observations để fit PCA")

    values = train.loc[:, list(SENSORS)].to_numpy(dtype="float64")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    pca = PCA(n_components=1, svd_solver="full")
    raw_scores = pca.fit_transform(scaled).ravel()
    oriented_scores, orientation_sign = _orient_scores(
        raw_scores,
        train["cycle"].to_numpy(dtype="float64"),
    )
    scale = float(np.std(oriented_scores, ddof=0))
    if not np.isfinite(scale) or scale == 0.0:
        raise ContractError("HEALTH_INDEX_SCALE", "health index", "PC1 không có độ biến thiên hữu hạn")
    normalized = (oriented_scores - float(np.mean(oriented_scores))) / scale
    result = train.loc[:, ["split", "engine_id", "cycle"]].copy()
    result["health_index"] = normalized.astype("float64")
    validate_health_index(result)
    return HealthIndexResult(
        health_index=result.loc[:, list(HEALTH_INDEX_COLUMNS)],
        scaler=scaler,
        pca=pca,
        orientation_sign=orientation_sign,
    )


def validate_health_index(frame: pd.DataFrame) -> None:
    """Kiểm tra schema, train-only, khóa và giá trị Health Index."""
    validate_columns(frame, HEALTH_INDEX_COLUMNS, "health index")
    if not frame["split"].eq("train").all():
        raise ContractError("TRAIN_ONLY", "health index", "Health Index phải chỉ thuộc split train")
    if frame.duplicated(list(KEY)).any():
        raise ContractError("HEALTH_INDEX_KEY", "health index", "trùng khóa observation")
    values = frame["health_index"].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ContractError("HEALTH_INDEX_NUMERIC", "health index", "Health Index phải hữu hạn")


def select_validation_engines(
    observations: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    *,
    n_engines: int,
) -> pd.DataFrame:
    """Chọn engine đại diện theo fold và lifetime quantiles trước forecast.

    Mỗi fold nhận quota cân bằng; trong từng fold, các engine được chọn tại
    vị trí lifetime cách đều trong thứ tự tăng dần. Quy tắc chỉ dùng metadata
    train và không nhìn vào forecast hoặc metric forecast.
    """
    validate_observations(observations, source="timeseries engine selection")
    required_folds = {"split", "engine_id", "fold"}
    if not required_folds.issubset(fold_assignments.columns):
        raise ContractError("FOLD_ASSIGNMENT", "timeseries engine selection", "thiếu split, engine_id hoặc fold")
    train = observations.loc[observations["split"].eq("train")]
    lifetimes = (
        train.groupby(["split", "engine_id"], sort=True)["cycle"]
        .max()
        .rename("n_cycles")
        .reset_index()
    )
    folds = fold_assignments.loc[:, ["split", "engine_id", "fold"]].copy()
    folds["split"] = folds["split"].astype("string")
    merged = lifetimes.merge(folds, on=["split", "engine_id"], how="inner", validate="one_to_one")
    if len(merged) != len(lifetimes):
        raise ContractError("FOLD_ASSIGNMENT", "timeseries engine selection", "mọi train engine phải có đúng một fold")
    available = merged.sort_values(["fold", "n_cycles", "engine_id"], kind="mergesort")
    fold_values = sorted(int(value) for value in available["fold"].unique())
    if n_engines < 1 or n_engines > len(available):
        raise ValueError("n_engines phải nằm trong [1, số train engines]")
    base, remainder = divmod(n_engines, len(fold_values))
    records: list[dict[str, int | str]] = []
    for fold_position, fold in enumerate(fold_values):
        quota = base + (1 if fold_position < remainder else 0)
        candidates = available.loc[available["fold"].eq(fold)].sort_values(
            ["n_cycles", "engine_id"], kind="mergesort"
        )
        if quota > len(candidates):
            raise ValueError(f"fold {fold} không đủ engine cho quota {quota}")
        positions = np.linspace(0, len(candidates) - 1, quota).round().astype(int)
        selected = candidates.iloc[positions].drop_duplicates(["split", "engine_id"])
        for rank, row in enumerate(selected.itertuples(index=False), start=1):
            records.append(
                {
                    "split": str(row.split),
                    "engine_id": int(row.engine_id),
                    "fold": int(row.fold),
                    "n_cycles": int(row.n_cycles),
                    "selection_rank": rank,
                }
            )
    result = pd.DataFrame(records, columns=SELECTED_ENGINE_COLUMNS)
    result = result.sort_values(["fold", "selection_rank", "engine_id"], kind="mergesort").reset_index(drop=True)
    return result.astype(
        {"split": "string", "engine_id": "int64", "fold": "int64", "n_cycles": "int64", "selection_rank": "int64"}
    )
