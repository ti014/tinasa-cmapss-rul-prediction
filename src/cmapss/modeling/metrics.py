"""Metric RUL dùng trong cross-validation và final evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def nasa_score_components(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """NASA asymmetric score từng dòng, với lỗi `d = pred - true`.

    d < 0: exp(-d/13) - 1; d >= 0: exp(d/10) - 1 (dự đoán trễ bị phạt nặng hơn).
    """
    true = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    error = pred - true
    return np.where(error < 0.0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)


def add_prediction_errors(frame: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột absolute_error, squared_error và nasa_score_component."""
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
    """Khoảng tin cậy 95% bằng bootstrap theo engine.

    Mỗi resample chọn engine có hoàn lại; engine được chọn nhiều lần thì các dòng của nó
    cũng được tính nhiều lần.
    """
    rng = np.random.default_rng(seed)
    rows = add_prediction_errors(evaluation.reset_index(drop=True))
    if rows.empty:
        raise ValueError("[BOOTSTRAP] evaluation phải có ít nhất một dòng")
    engine_groups = {
        engine_id: group.reset_index(drop=True)
        for engine_id, group in rows.groupby("engine_id", sort=True)
    }
    engine_ids = np.asarray(list(engine_groups), dtype="int64")
    engine_mae = rows.groupby("engine_id", sort=True)["absolute_error"].mean()
    n_engines = len(engine_ids)
    if len(rows) == n_engines:
        ordered = rows.sort_values("engine_id", kind="mergesort").reset_index(drop=True)
        indices = rng.integers(0, n_engines, size=(n_resamples, n_engines))
        absolute_errors = ordered["absolute_error"].to_numpy(dtype="float64")
        squared_errors = ordered["squared_error"].to_numpy(dtype="float64")
        score_components = ordered["nasa_score_component"].to_numpy(dtype="float64")
        fast_values = {
            "mean_engine_mae": absolute_errors[indices].mean(axis=1),
            "rmse": np.sqrt(squared_errors[indices].mean(axis=1)),
            "nasa_score": score_components[indices].sum(axis=1),
        }
        return {
            name: {
                "ci_lower": float(np.quantile(metric_values, 0.025)),
                "ci_upper": float(np.quantile(metric_values, 0.975)),
            }
            for name, metric_values in fast_values.items()
        }
    values = {
        "mean_engine_mae": np.empty(n_resamples, dtype="float64"),
        "rmse": np.empty(n_resamples, dtype="float64"),
        "nasa_score": np.empty(n_resamples, dtype="float64"),
    }
    for index in range(n_resamples):
        sampled_engine_ids = rng.choice(engine_ids, size=n_engines, replace=True)
        sampled = pd.concat(
            [engine_groups[int(engine_id)] for engine_id in sampled_engine_ids],
            ignore_index=True,
        )
        values["mean_engine_mae"][index] = float(
            engine_mae.loc[sampled_engine_ids].mean()
        )
        values["rmse"][index] = float(np.sqrt(sampled["squared_error"].mean()))
        values["nasa_score"][index] = float(sampled["nasa_score_component"].sum())

    return {
        name: {
            "ci_lower": float(np.quantile(metric_values, 0.025)),
            "ci_upper": float(np.quantile(metric_values, 0.975)),
        }
        for name, metric_values in values.items()
    }
