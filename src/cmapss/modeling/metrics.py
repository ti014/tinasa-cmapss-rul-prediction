"""Metric RUL dùng trong cross-validation và final evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def nasa_score_components(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Tính NASA asymmetric score theo lỗi `predicted - true`.

    Nhánh dự đoán thấp hơn thực tế dùng mẫu số 13; nhánh dự đoán cao hơn hoặc
    bằng thực tế dùng mẫu số 10 theo công thức C-MAPSS thường dùng.
    """
    true = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    error = pred - true
    return np.where(error < 0.0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)


def add_prediction_errors(frame: pd.DataFrame) -> pd.DataFrame:
    """Thêm absolute/squared error và NASA score component vào bảng prediction."""
    result = frame.copy()
    result["absolute_error"] = (result["y_pred"] - result["y_true"]).abs()
    result["squared_error"] = (result["y_pred"] - result["y_true"]) ** 2
    result["nasa_score_component"] = nasa_score_components(
        result["y_true"].to_numpy(dtype="float64"),
        result["y_pred"].to_numpy(dtype="float64"),
    )
    return result


def summarize_prediction_errors(frame: pd.DataFrame) -> dict[str, float | int]:
    """Tóm tắt metric cho một bảng prediction đã có `y_true` và `y_pred`."""
    scored = add_prediction_errors(frame)
    engine_mae = scored.groupby("engine_id", sort=True)["absolute_error"].mean()
    return {
        "n_rows": int(len(scored)),
        "n_engines": int(scored["engine_id"].nunique()),
        "mean_window_mae": float(scored["absolute_error"].mean()),
        "mean_engine_mae": float(engine_mae.mean()),
        "rmse": float(np.sqrt(scored["squared_error"].mean())),
        "nasa_score": float(scored["nasa_score_component"].sum()),
    }


def bootstrap_metric_intervals(
    evaluation: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Bootstrap final metrics theo engine, dùng một dòng cho mỗi test engine."""
    rng = np.random.default_rng(seed)
    rows = evaluation.reset_index(drop=True)
    n_rows = len(rows)
    values = {
        "mean_engine_mae": np.empty(n_resamples, dtype="float64"),
        "rmse": np.empty(n_resamples, dtype="float64"),
        "nasa_score": np.empty(n_resamples, dtype="float64"),
    }
    for index in range(n_resamples):
        sample = rows.iloc[rng.integers(0, n_rows, size=n_rows)]
        metrics = summarize_prediction_errors(sample)
        for name in values:
            values[name][index] = float(metrics[name])

    return {
        name: {
            "ci_lower": float(np.quantile(metric_values, 0.025)),
            "ci_upper": float(np.quantile(metric_values, 0.975)),
        }
        for name, metric_values in values.items()
    }
