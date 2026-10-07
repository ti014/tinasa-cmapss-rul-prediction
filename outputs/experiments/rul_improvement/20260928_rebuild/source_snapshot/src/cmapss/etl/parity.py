"""Parity checks giữa các engine ETL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


class ParityError(ValueError):
    """Lỗi so sánh schema, khóa hoặc giá trị giữa hai artifact."""


@dataclass(frozen=True)
class FrameParity:
    """Kết quả parity của một bảng."""

    name: str
    rows: int
    key_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    max_abs_diff: float

    def as_dict(self) -> dict[str, object]:
        """Trả payload JSON-serializable."""
        return {
            "name": self.name,
            "status": "passed",
            "rows": self.rows,
            "key_columns": list(self.key_columns),
            "numeric_columns": list(self.numeric_columns),
            "max_abs_diff": self.max_abs_diff,
        }


def _sorted(frame: pd.DataFrame, key_columns: tuple[str, ...]) -> pd.DataFrame:
    """Sắp xếp và reset index để so sánh deterministic."""
    return frame.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)


def compare_frames(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    name: str,
    key_columns: Iterable[str],
    numeric_columns: Iterable[str],
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> FrameParity:
    """Đối chiếu schema, khóa, số dòng và giá trị số của hai bảng.

    Diagnostic nêu rõ boundary bị lệch để caller biết cần sửa engine nào.
    """
    keys = tuple(key_columns)
    numeric = tuple(numeric_columns)
    if tuple(reference.columns) != tuple(candidate.columns):
        raise ParityError(
            f"{name}: schema không khớp; reference={tuple(reference.columns)}, "
            f"candidate={tuple(candidate.columns)}"
        )
    if len(reference) != len(candidate):
        raise ParityError(
            f"{name}: row count không khớp; reference={len(reference)}, "
            f"candidate={len(candidate)}"
        )
    for frame_name, frame in (("reference", reference), ("candidate", candidate)):
        if frame.duplicated(list(keys)).any():
            raise ParityError(f"{name}: {frame_name} có khóa trùng trên {keys}")

    left = _sorted(reference, keys)
    right = _sorted(candidate, keys)
    for column in keys:
        if not left[column].astype("string").equals(right[column].astype("string")):
            raise ParityError(f"{name}: khóa {column} không khớp sau khi sắp xếp")

    max_abs_diff = 0.0
    for column in reference.columns:
        if column in numeric:
            left_values = left[column].to_numpy(dtype="float64")
            right_values = right[column].to_numpy(dtype="float64")
            differences = np.abs(left_values - right_values)
            finite = np.isfinite(differences)
            if not finite.all() or not np.allclose(
                left_values,
                right_values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                max_difference = float(np.nanmax(differences)) if finite.any() else float("inf")
                raise ParityError(
                    f"{name}: giá trị {column} lệch quá tolerance; "
                    f"max_abs_diff={max_difference:g}, rtol={rtol:g}, atol={atol:g}"
                )
            if finite.any():
                max_abs_diff = max(max_abs_diff, float(differences[finite].max()))
        elif not left[column].astype("string").equals(right[column].astype("string")):
            raise ParityError(f"{name}: giá trị non-numeric {column} không khớp")

    return FrameParity(
        name=name,
        rows=len(reference),
        key_columns=keys,
        numeric_columns=numeric,
        max_abs_diff=max_abs_diff,
    )
