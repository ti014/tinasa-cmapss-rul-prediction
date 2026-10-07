"""Blend two frozen HGB predictions using a smooth prediction-only gate.

Weights depend only on the uncapped prediction for the current observation.
Labels, future observations, engine lifetime and RUL regions do not enter the
gate. No clipping of the resulting RUL prediction is introduced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY

from .contracts import GateSpec


def _vector(values: Any, name: str) -> np.ndarray:
    """Require a finite nonempty one-dimensional real prediction vector."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional real vector")
    result = raw.astype("float64", copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def capped_weights(uncapped_predictions: Any, spec: GateSpec) -> np.ndarray:
    """Compute a bounded smooth weight without accessing any target.

    The weight decreases from its maximum at lower_rul to zero at upper_rul.
    Smoothstep has zero slope at both boundaries; this smoothness does not
    imply that the resulting prediction trajectory is monotonic over time.

    Args:
        uncapped_predictions: Current-observation uncapped HGB predictions.
        spec: A gate specification frozen before outer evaluation.

    Returns:
        A float64 vector with weights in [0, max_capped_weight].

    Raises:
        ValueError: Predictions are empty, malformed, nonnumeric or nonfinite.
        TypeError: The supplied specification is not a GateSpec.

    Examples:
        >>> capped_weights([60, 105, 150], GateSpec(60, 150, 0.5)).tolist()
        [0.5, 0.25, 0.0]
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
    """Combine matched predictions with a current-observation smooth gate.

    Args:
        uncapped_predictions: One-dimensional uncapped HGB vector.
        capped_predictions: Cap125 HGB vector in exactly the same row order.
        spec: A gate specification frozen before evaluation.

    Returns:
        Convex combinations of the two prediction vectors, without RUL clipping.

    Raises:
        ValueError: Vectors are malformed, nonfinite or have different shapes.
        TypeError: The supplied specification is not a GateSpec.

    Examples:
        >>> blend_vectors([60, 150], [50, 120], GateSpec(60, 150, 0.5)).tolist()
        [55.0, 150.0]
    """
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
    """Validate and sort one prediction frame before exact paired alignment."""
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
    """Align stored vectors by exact keys and retain gate inputs for audit.

    Raw truth and optional inner/outer fold columns must match exactly when
    present. Truth is copied for subsequent scoring and does not affect weights.
    Inference frames without any labels are accepted.

    Args:
        uncapped_predictions: Unique key/y_pred rows from uncapped HGB.
        capped_predictions: Matching key/y_pred rows from cap125 HGB.
        spec: A frozen gate specification.

    Returns:
        Sorted keys, optional truth/folds, component predictions, weights and
        the blended y_pred. Stale input error columns are omitted.

    Raises:
        ValueError: Frames have inconsistent keys, truth or fold membership.
        TypeError: The supplied specification is not a GateSpec.

    Examples:
        Inputs may be supplied in different row orders; unique keys determine
        the pairing. A missing engine-cycle row is rejected.
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
