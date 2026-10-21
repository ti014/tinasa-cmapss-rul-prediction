"""Đánh giá baseline dự báo điểm trên cùng holdout Health Index với ARIMA."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_forecast_baselines(
    fit_values: pd.Series | np.ndarray,
    holdout_values: pd.Series | np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """
    Đánh giá last_value và drift không fit holdout;
    drift=last+horizon×(last−first)/(n_fit−1), cả hai dự báo toàn bộ holdout từ cuối
    prefix.
    """
    fit = np.asarray(fit_values, dtype="float64")
    actual = np.asarray(holdout_values, dtype="float64")
    if fit.ndim != 1 or len(fit) < 2 or not np.isfinite(fit).all():
        raise ValueError("[BASELINE_PREFIX] cần ít nhất hai fit values hữu hạn")
    if actual.ndim != 1 or len(actual) < 1 or not np.isfinite(actual).all():
        raise ValueError("[BASELINE_HOLDOUT] cần ít nhất một holdout value hữu hạn")
    horizons = np.arange(1, len(actual) + 1, dtype="float64")
    forecasts = {
        "last_value": np.full(len(actual), fit[-1], dtype="float64"),
        "drift": fit[-1] + horizons * (fit[-1] - fit[0]) / (len(fit) - 1),
    }
    return {
        name: {
            "n_train": len(fit),
            "n_holdout": len(actual),
            "mae": float(np.mean(np.abs(predicted - actual))),
            "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        }
        for name, predicted in forecasts.items()
    }
