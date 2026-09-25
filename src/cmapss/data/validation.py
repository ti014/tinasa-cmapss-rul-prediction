"""Validator cho observations, nhãn test và profile C-MAPSS."""

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from cmapss.data.contracts import (
    DATASET_IDS,
    DATASET_PROFILES,
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


def validate_dataset_profile(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    split: Split,
    profile_rule: str = "DATASET_PROFILE",
) -> None:
    """Đối chiếu một file C-MAPSS đầy đủ với profile subset đã biết."""
    dataset_id = dataset_id.upper()
    if dataset_id not in DATASET_IDS:
        raise ContractError("DATASET_ID", dataset_id, f"dataset_id phải thuộc {DATASET_IDS}")
    validate_observations(frame, source=f"{dataset_id}/{split}")
    if split not in SPLITS or not frame["split"].eq(split).all():
        raise ContractError("SPLIT", dataset_id, f"cần một split duy nhất: {split}")
    profile = DATASET_PROFILES[dataset_id]
    expected_rows = profile[f"{split}_rows"]
    expected_engine_count = profile[f"{split}_engines"]
    if len(frame) != expected_rows:
        raise ContractError(profile_rule, f"{dataset_id}/{split}", f"cần {expected_rows} dòng, nhận {len(frame)}")
    expected_ids = set(range(1, expected_engine_count + 1))
    if set(frame["engine_id"]) != expected_ids:
        raise ContractError(profile_rule, f"{dataset_id}/{split}", f"cần đúng engine ID 1–{expected_engine_count}")
    starts = frame.groupby("engine_id")["cycle"].min()
    if not starts.eq(1).all():
        raise ContractError(profile_rule, f"{dataset_id}/{split}", "mỗi engine trong file đầy đủ phải bắt đầu tại cycle 1")


def validate_fd001_profile(frame: pd.DataFrame, *, split: Split) -> None:
    """Tương thích ngược cho validator FD001 cũ."""
    validate_dataset_profile(frame, dataset_id="FD001", split=split, profile_rule="FD001_PROFILE")


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
    if fd001 and len(frame) != DATASET_PROFILES["FD001"]["test_engines"]:
        raise ContractError("FD001_PROFILE", source, f"cần 100 nhãn, nhận {len(frame)}")
