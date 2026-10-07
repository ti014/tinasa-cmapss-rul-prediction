"""Tạo nhãn train RUL từ lịch sử run-to-failure, không đọc file nhãn test."""

import pandas as pd

from cmapss.data.contracts import KEY, TRAIN_TARGET_COLUMNS, ContractError
from cmapss.data.validation import (
    validate_columns,
    validate_integer,
    validate_observations,
)


def _validate_train(observations: pd.DataFrame) -> None:
    validate_observations(observations)
    if not observations["split"].eq("train").all():
        raise ContractError("TRAIN_ONLY", "train RUL", "không áp dụng max_cycle - cycle cho test")


def make_train_targets(observations: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng khóa và raw RUL từ toàn bộ lịch sử train, không cap RUL.

    Args:
        observations: Observations train đầy đủ đến khi hỏng, trước khi lọc window.

    Returns:
        Bảng split, engine_id, cycle, rul; giữ số dòng và thứ tự đầu vào.

    Raises:
        ContractError: Observations sai contract hoặc chứa split test.
    """
    _validate_train(observations)
    targets = observations.loc[:, list(KEY)].copy()
    last_cycle = observations.groupby("engine_id")["cycle"].transform("max")
    targets["rul"] = last_cycle - observations["cycle"]
    return targets


def validate_train_targets(observations: pd.DataFrame, targets: pd.DataFrame) -> None:
    """Kiểm tra khóa và công thức RUL theo engine, độc lập thứ tự dòng targets."""
    _validate_train(observations)
    validate_columns(targets, TRAIN_TARGET_COLUMNS, "train targets")
    if not targets["split"].eq("train").fillna(False).all():
        raise ContractError("TRAIN_ONLY", "train targets", "targets phải thuộc split train")
    validate_integer(targets, "engine_id", 1, "train targets")
    validate_integer(targets, "cycle", 1, "train targets")
    validate_integer(targets, "rul", 0, "train targets")
    if targets.duplicated(list(KEY)).any():
        raise ContractError("UNIQUE_KEY", "train targets", "trùng khóa observation")
    joined = observations.loc[:, list(KEY)].merge(
        targets,
        on=list(KEY),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise ContractError("TARGET_KEYS", "train targets", "khóa targets phải khớp toàn bộ observations")
    last_cycle = joined.groupby("engine_id")["cycle"].transform("max")
    expected_rul = last_cycle - joined["cycle"]
    incorrect = joined["rul"].ne(expected_rul)
    if incorrect.any():
        key = joined.loc[incorrect, list(KEY)].iloc[0].to_dict()
        raise ContractError("TRAIN_RUL", "train targets", f"RUL không bằng max_cycle - cycle tại {key}")
