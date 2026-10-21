"""Phối hợp trơn chỉ theo dự đoán hiện tại, không dùng nhãn/tương lai và không cắt RUL đầu ra."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY

from .contracts import GateSpec


def _vector(values: Any, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} phải là vector thực một chiều không rỗng")
    result = raw.astype("float64", copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} phải hữu hạn")
    return result


def capped_weights(uncapped_predictions: Any, spec: GateSpec) -> np.ndarray:
    """
    Trọng số smoothstep cap125: max_w × r² × (3−2r), r=(upper−clip(p))/(upper−lower);
    (60,150,0,5) cho trọng số [0,5;0,25;0] tại [60;105;150].
    """
    if not isinstance(spec, GateSpec):
        raise TypeError("spec phải là GateSpec")
    uncapped = _vector(uncapped_predictions, "uncapped_predictions")
    # Clip trước khi trừ để dự đoán hữu hạn cực lớn không gây tràn số.
    bounded = np.clip(uncapped, spec.lower_rul, spec.upper_rul)
    ramp = (spec.upper_rul - bounded) / (spec.upper_rul - spec.lower_rul)
    smooth = ramp * ramp * (3.0 - 2.0 * ramp)
    return spec.max_capped_weight * smooth


def blend_vectors(
    uncapped_predictions: Any, capped_predictions: Any, spec: GateSpec,
) -> np.ndarray:
    """Trả (1−w) × uncapped + w × capped; hai vector phải cùng thứ tự dòng."""
    uncapped = _vector(uncapped_predictions, "uncapped_predictions")
    capped = _vector(capped_predictions, "capped_predictions")
    if uncapped.shape != capped.shape:
        raise ValueError("Các vector dự đoán phải cùng shape")
    weights = capped_weights(uncapped, spec)
    result = (1.0 - weights) * uncapped + weights * capped
    if not np.isfinite(result).all():
        raise ValueError("Dự đoán phối hợp phải hữu hạn")
    return result


def _prediction_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = set(KEY) | {"y_pred"}
    if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
            or frame.empty or not required.issubset(frame.columns)):
        raise ValueError(f"{name} cần cột, khóa duy nhất không rỗng và y_pred")
    if frame.loc[:, list(KEY)].isna().any().any() or frame.duplicated(list(KEY)).any():
        raise ValueError(f"{name} khóa phải không thiếu và duy nhất")
    for key in ("engine_id", "cycle"):
        vector = _vector(frame[key].to_numpy(), key)
        if (vector < 1).any() or not np.equal(vector, np.floor(vector)).all():
            raise ValueError(f"{key} phải chứa số nguyên dương")
    _vector(frame.y_pred.to_numpy(), "y_pred")
    if "y_true" in frame:
        truth = _vector(frame.y_true.to_numpy(), "y_true")
        if (truth < 0).any():
            raise ValueError("y_true gốc phải không âm")
    return frame.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)


def blend_predictions(
    uncapped_predictions: pd.DataFrame, capped_predictions: pd.DataFrame,
    spec: GateSpec,
) -> pd.DataFrame:
    """
    Ghép hai bảng theo khóa rồi phối hợp; nhãn/fold phải khớp, nhãn chỉ dùng đánh giá,
    bỏ các cột sai số đầu vào cũ.
    """
    uncapped = _prediction_frame(uncapped_predictions, "uncapped_predictions")
    capped = _prediction_frame(capped_predictions, "capped_predictions")
    preserved = list(KEY)
    for column in ("y_true", "inner_fold", "outer_fold"):
        if (column in uncapped) != (column in capped):
            raise ValueError(f"Hai bảng dự đoán phải cùng có hoặc cùng bỏ {column}")
        if column in uncapped:
            if uncapped[column].isna().any() or capped[column].isna().any():
                raise ValueError(f"Giá trị {column} ghép cặp phải không thiếu")
            preserved.append(column)
    try:
        pd.testing.assert_frame_equal(
            uncapped.loc[:, preserved], capped.loc[:, preserved], check_exact=True,
        )
    except AssertionError as error:
        raise ValueError("Dự đoán ghép cặp cần khớp chính xác khóa, nhãn gốc và fold") from error
    old = uncapped.y_pred.to_numpy(dtype="float64")
    cap = capped.y_pred.to_numpy(dtype="float64")
    result = uncapped.loc[:, preserved].copy()
    result["uncapped_prediction"] = old
    result["cap125_prediction"] = cap
    result["capped_weight"] = capped_weights(old, spec)
    result["y_pred"] = blend_vectors(old, cap, spec)
    result["candidate_id"] = spec.candidate_id
    return result


__all__ = ["capped_weights", "blend_vectors", "blend_predictions"]
