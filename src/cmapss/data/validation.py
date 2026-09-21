"""Validator cho observations, nhãn test và profile FD001 đầy đủ."""

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from cmapss.data.contracts import (
    FD001_ENGINE_COUNT,
    FD001_ROWS,
    KEY,
    MEASUREMENTS,
    OBSERVATION_COLUMNS,
    SPLITS,
    TEST_LABEL_COLUMNS,
    ContractError,
    Split,
)


def validate_columns(frame: pd.DataFrame, columns: tuple[str, ...], source: str) -> None:
    """Yêu cầu đúng tên, thứ tự cột và ít nhất một dòng."""
    if tuple(frame.columns) != columns:
        raise ContractError("SCHEMA", source, f"cần các cột {columns}, nhận {tuple(frame.columns)}")
    if frame.empty:
        raise ContractError("NONEMPTY", source, "bảng không có dữ liệu")


def validate_integer(frame: pd.DataFrame, column: str, minimum: int, source: str) -> None:
    """Kiểm tra cột số nguyên không null và giới hạn dưới."""
    values = frame[column]
    if not is_integer_dtype(values.dtype):
        raise ContractError("NUMERIC", f"{source}.{column}", "cần dtype số nguyên")
    if values.isna().any() or (values < minimum).any():
        raise ContractError("NUMERIC", f"{source}.{column}", f"cần số nguyên không null >= {minimum}")


def validate_observations(frame: pd.DataFrame, *, source: str = "observations") -> None:
    """Kiểm tra bảng đầy đủ trước window; cho phép nhiều split và thứ tự dòng bất kỳ.

    Cycle phải liên tục sau khi sắp xếp trong từng (split, engine_id).
    Contract chung không áp đặt số dòng FD001 hoặc cycle bắt đầu bằng 1.
    """
    validate_columns(frame, OBSERVATION_COLUMNS, source)
    if not frame["split"].isin(SPLITS).all():
        raise ContractError("SPLIT", source, "split phải là train hoặc test, không null")
    for column in ("engine_id", "cycle"):
        validate_integer(frame, column, 1, source)
    for column in MEASUREMENTS:
        values = frame[column]
        if not (is_float_dtype(values.dtype) or is_integer_dtype(values.dtype)):
            raise ContractError("NUMERIC", f"{source}.{column}", "cần dtype số thực hoặc số nguyên")
        if values.isna().any() or not np.isfinite(values).all():
            raise ContractError("NUMERIC", f"{source}.{column}", "cần số hữu hạn, không null")

    duplicated = frame.duplicated(list(KEY), keep=False)
    if duplicated.any():
        key = frame.loc[duplicated, list(KEY)].iloc[0].to_dict()
        raise ContractError("UNIQUE_KEY", source, f"trùng khóa {key}")
    ordered = frame.sort_values(list(KEY))
    differences = ordered.groupby(["split", "engine_id"])["cycle"].diff()
    gaps = differences.notna() & differences.ne(1)
    if gaps.any():
        key = ordered.loc[gaps, list(KEY)].iloc[0].to_dict()
        raise ContractError("CYCLE_CONTINUITY", source, f"cycle không liên tục tại {key}")


def validate_fd001_profile(frame: pd.DataFrame, *, split: Split) -> None:
    """Đối chiếu file FD001 đầy đủ với số dòng, engine ID và cycle đầu."""
    validate_observations(frame, source=f"FD001/{split}")
    if split not in SPLITS or not frame["split"].eq(split).all():
        raise ContractError("SPLIT", "FD001", f"cần một split duy nhất: {split}")
    if len(frame) != FD001_ROWS[split]:
        raise ContractError("FD001_PROFILE", split, f"cần {FD001_ROWS[split]} dòng, nhận {len(frame)}")
    expected_ids = set(range(1, FD001_ENGINE_COUNT + 1))
    if set(frame["engine_id"]) != expected_ids:
        raise ContractError("FD001_PROFILE", split, "cần đúng engine ID 1–100")
    starts = frame.groupby("engine_id")["cycle"].min()
    if not starts.eq(1).all():
        raise ContractError("FD001_PROFILE", split, "mỗi engine trong file đầy đủ phải bắt đầu tại cycle 1")


def validate_test_labels(
    frame: pd.DataFrame,
    *,
    source: str = "test_labels",
    fd001: bool = False,
) -> None:
    """Kiểm tra nhãn engine-level; tùy chọn đối chiếu đúng 100 nhãn FD001."""
    validate_columns(frame, TEST_LABEL_COLUMNS, source)
    if not frame["split"].eq("test").fillna(False).all():
        raise ContractError("SPLIT", source, "engine-level test labels chỉ thuộc split test")
    validate_integer(frame, "engine_id", 1, source)
    validate_integer(frame, "rul", 0, source)
    if frame.duplicated(["split", "engine_id"]).any():
        raise ContractError("UNIQUE_KEY", source, "trùng khóa (split, engine_id)")
    if set(frame["engine_id"]) != set(range(1, len(frame) + 1)):
        raise ContractError("LABEL_MAPPING", source, "engine ID phải tương ứng các dòng 1–N của file nhãn")
    if fd001 and len(frame) != FD001_ENGINE_COUNT:
        raise ContractError("FD001_PROFILE", source, f"cần 100 nhãn, nhận {len(frame)}")
