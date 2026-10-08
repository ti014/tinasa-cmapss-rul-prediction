"""Blend uncapped and cap125 HGB predictions with a smooth gate.

The weight depends only on the uncapped prediction of the current row; labels
and future cycles never enter it. The blended RUL is not clipped.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY

from .contracts import GateSpec


def _vector(values: Any, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional real vector")
    result = raw.astype("float64", copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def capped_weights(uncapped_predictions: Any, spec: GateSpec) -> np.ndarray:
    """Smoothstep weight for cap125: max_capped_weight at lower_rul, 0 at upper_rul.

    With r = (upper - clip(p)) / (upper - lower), w = max_w * r^2 * (3 - 2r).
    Example: GateSpec(60, 150, 0.5) gives [0.5, 0.25, 0.0] for p = [60, 105, 150].
    """
    if not isinstance(spec, GateSpec):
        raise TypeError("spec must be a GateSpec")
    uncapped = _vector(uncapped_predictions, "uncapped_predictions")
    # Clip before subtraction to keep extreme finite predictions from overflow.
    bounded = np.clip(uncapped, spec.lower_rul, spec.upper_rul)
    ramp = (spec.upper_rul - bounded) / (spec.upper_rul - spec.lower_rul)
    smooth = ramp * ramp * (3.0 - 2.0 * ramp)
    return spec.max_capped_weight * smooth


def blend_vectors(
    uncapped_predictions: Any, capped_predictions: Any, spec: GateSpec,
) -> np.ndarray:
    """Return (1 - w) * uncapped + w * capped; both vectors must share row order."""
    uncapped = _vector(uncapped_predictions, "uncapped_predictions")
    capped = _vector(capped_predictions, "capped_predictions")
    if uncapped.shape != capped.shape:
        raise ValueError("prediction vectors must have identical shapes")
    weights = capped_weights(uncapped, spec)
    result = (1.0 - weights) * uncapped + weights * capped
    if not np.isfinite(result).all():
        raise ValueError("blended predictions must be finite")
    return result


def _prediction_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = set(KEY) | {"y_pred"}
    if (not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique
            or frame.empty or not required.issubset(frame.columns)):
        raise ValueError(f"{name} requires nonempty unique columns, keys and y_pred")
    if frame.loc[:, list(KEY)].isna().any().any() or frame.duplicated(list(KEY)).any():
        raise ValueError(f"{name} keys must be nonmissing and unique")
    for key in ("engine_id", "cycle"):
        vector = _vector(frame[key].to_numpy(), key)
        if (vector < 1).any() or not np.equal(vector, np.floor(vector)).all():
            raise ValueError(f"{key} must contain positive integers")
    _vector(frame.y_pred.to_numpy(), "y_pred")
    if "y_true" in frame:
        truth = _vector(frame.y_true.to_numpy(), "y_true")
        if (truth < 0).any():
            raise ValueError("raw y_true must be nonnegative")
    return frame.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)


def blend_predictions(
    uncapped_predictions: pd.DataFrame, capped_predictions: pd.DataFrame,
    spec: GateSpec,
) -> pd.DataFrame:
    """Pair two prediction frames by key and blend them.

    `y_true`, `inner_fold` and `outer_fold` must match exactly when present;
    `y_true` is only carried along for scoring. Output columns: keys, optional
    truth/folds, uncapped_prediction, cap125_prediction, capped_weight, y_pred,
    candidate_id. Old error columns from the inputs are dropped.
    """
    uncapped = _prediction_frame(uncapped_predictions, "uncapped_predictions")
    capped = _prediction_frame(capped_predictions, "capped_predictions")
    preserved = list(KEY)
    for column in ("y_true", "inner_fold", "outer_fold"):
        if (column in uncapped) != (column in capped):
            raise ValueError(f"paired predictions must both include or omit {column}")
        if column in uncapped:
            if uncapped[column].isna().any() or capped[column].isna().any():
                raise ValueError(f"paired {column} values must be nonmissing")
            preserved.append(column)
    try:
        pd.testing.assert_frame_equal(
            uncapped.loc[:, preserved], capped.loc[:, preserved], check_exact=True,
        )
    except AssertionError as error:
        raise ValueError("paired predictions require exact keys, raw truth and folds") from error
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
