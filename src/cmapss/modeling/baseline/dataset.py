"""Chuẩn bị train dataset, trọng số và fold theo engine; không đọc test labels."""

from __future__ import annotations

from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY, TRAIN_TARGET_COLUMNS, ContractError
from cmapss.data.validation import validate_columns, validate_integer
from cmapss.features.contracts import ROLLING_FEATURE_COLUMNS
from cmapss.features.rolling import validate_rolling_features
from cmapss.modeling.contracts import FOLD_ASSIGNMENT_COLUMNS, MODELING_FEATURE_COLUMNS


def _validate_train_targets_frame(targets: pd.DataFrame) -> None:
    validate_columns(targets, TRAIN_TARGET_COLUMNS, "modeling train targets")
    if not targets["split"].eq("train").fillna(False).all():
        raise ContractError("TRAIN_ONLY", "nhãn train cho mô hình", "targets phải thuộc split train")
    for column in ("engine_id", "cycle", "rul"):
        validate_integer(targets, column, 0 if column == "rul" else 1, "modeling train targets")
    if targets.duplicated(list(KEY)).any():
        raise ContractError("MODELING_TARGET_KEYS", "nhãn train cho mô hình", "trùng khóa target")


def prepare_modeling_dataset(features: pd.DataFrame, train_targets: pd.DataFrame) -> pd.DataFrame:
    """Join train rolling features với train RUL targets, không đọc test labels."""
    validate_rolling_features(features)
    _validate_train_targets_frame(train_targets)
    train_features = features.loc[features["split"].eq("train")].copy()
    joined = train_features.merge(
        train_targets.loc[:, list(TRAIN_TARGET_COLUMNS)],
        on=list(KEY),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = joined["_merge"].ne("both")
    if missing.any():
        key = joined.loc[missing, list(KEY)].iloc[0].to_dict()
        raise ContractError(
            "MODELING_TARGET_KEYS",
            "dữ liệu mô hình",
            f"thiếu RUL train cho rolling feature {key}",
        )
    joined = joined.drop(columns=["_merge"])
    return joined.loc[:, list(ROLLING_FEATURE_COLUMNS) + ["rul"]].reset_index(drop=True)


def engine_equal_sample_weights(dataset: pd.DataFrame) -> np.ndarray:
    """Gán tổng trọng số bằng 1 cho mỗi engine trong training fold."""
    counts = dataset.groupby("engine_id", sort=False)["cycle"].transform("size")
    return (1.0 / counts.to_numpy(dtype="float64")).astype("float64")


def build_group_folds(dataset: pd.DataFrame, *, n_splits: int) -> pd.DataFrame:
    """Tạo fold assignment theo engine, mỗi engine thuộc đúng một validation fold."""
    groups = dataset["engine_id"].to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    records: list[dict[str, int | str]] = []
    X = dataset.loc[:, list(MODELING_FEATURE_COLUMNS)]
    y = dataset["rul"].to_numpy()
    for fold, (_, validation_index) in enumerate(splitter.split(X, y, groups), start=1):
        engines = sorted(int(value) for value in dataset.iloc[validation_index]["engine_id"].unique())
        records.extend({"split": "train", "engine_id": engine, "fold": fold} for engine in engines)
    return pd.DataFrame(records, columns=FOLD_ASSIGNMENT_COLUMNS).astype(
        {"split": "string", "engine_id": "int64", "fold": "int64"}
    )
