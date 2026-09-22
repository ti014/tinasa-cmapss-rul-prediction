"""ARIMA forecast, diagnostics và metric cho Health Index train-only."""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from cmapss.timeseries.contracts import (
    DIAGNOSTIC_COLUMNS,
    FORECAST_COLUMNS,
    FORECAST_METRIC_COLUMNS,
    TIMESERIES_ARIMA_D_VALUES,
    TIMESERIES_ARIMA_P_MAX,
    TIMESERIES_ARIMA_Q_MAX,
    TIMESERIES_MIN_TRAIN_SIZE,
    TIMESERIES_PREDICTION_ALPHA,
)


def split_forecast_series(
    cycles: pd.Series,
    values: pd.Series,
    *,
    train_fraction: float,
    minimum_train_size: int = TIMESERIES_MIN_TRAIN_SIZE,
) -> tuple[pd.Series, pd.Series]:
    """Chia một chuỗi theo thứ tự cycle, không xáo trộn hoặc overlap."""
    if len(cycles) != len(values) or len(cycles) < 2:
        raise ValueError("[FORECAST_SPLIT] cycles và values phải cùng chiều dài >= 2")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("[FORECAST_SPLIT] train_fraction phải nằm giữa 0 và 1")
    n_train = int(np.floor(len(values) * train_fraction))
    if n_train < minimum_train_size or n_train >= len(values):
        raise ValueError(
            f"[FORECAST_SPLIT] cần n_train >= {minimum_train_size} và còn ít nhất một holdout"
        )
    ordered = pd.DataFrame({"cycle": cycles.to_numpy(), "value": values.to_numpy()}, index=cycles.index)
    if not ordered["cycle"].is_monotonic_increasing or ordered["cycle"].duplicated().any():
        raise ValueError("[FORECAST_SPLIT] cycle phải tăng dần và không trùng")
    return ordered["value"].iloc[:n_train], ordered["value"].iloc[n_train:]


def _order_grid() -> list[tuple[int, int, int]]:
    return [
        (p, d, q)
        for p, d, q in product(
            range(TIMESERIES_ARIMA_P_MAX + 1),
            TIMESERIES_ARIMA_D_VALUES,
            range(TIMESERIES_ARIMA_Q_MAX + 1),
        )
    ]


def select_arima_order(values: pd.Series | np.ndarray) -> tuple[int, int, int]:
    """Chọn order có AIC nhỏ nhất, chỉ trên fit segment."""
    series = np.asarray(values, dtype="float64")
    candidates: list[tuple[float, tuple[int, int, int]]] = []
    for order in _order_grid():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(
                    series,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
            aic = float(fitted.aic)
            if np.isfinite(aic):
                candidates.append((aic, order))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not candidates:
        raise ValueError("[ARIMA_ORDER] không fit được candidate order nào trên fit segment")
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def evaluate_arima_forecast(
    actual: np.ndarray,
    forecast: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Tính MAE, RMSE và coverage của prediction interval."""
    actual = np.asarray(actual, dtype="float64")
    forecast = np.asarray(forecast, dtype="float64")
    lower = np.asarray(lower, dtype="float64")
    upper = np.asarray(upper, dtype="float64")
    if not (len(actual) == len(forecast) == len(lower) == len(upper) and len(actual) > 0):
        raise ValueError("[FORECAST_METRIC] các vector forecast phải cùng chiều dài dương")
    error = forecast - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "coverage_95": float(np.mean((actual >= lower) & (actual <= upper))),
    }


def _diagnostics(values: np.ndarray) -> dict[str, float | int]:
    """Tính ADF trên fit segment và Ljung--Box trên residual."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_result = adfuller(values, autolag="AIC")
    return {
        "adf_statistic": float(adf_result[0]),
        "adf_pvalue": float(adf_result[1]),
        "adf_usedlag": int(adf_result[2]),
        "adf_nobs": int(adf_result[3]),
    }


def fit_engine_arima(
    series: pd.DataFrame,
    *,
    split: str,
    engine_id: int,
    fold: int,
    train_fraction: float,
    minimum_train_size: int = TIMESERIES_MIN_TRAIN_SIZE,
    alpha: float = TIMESERIES_PREDICTION_ALPHA,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Fit/forecast một engine và trả forecast rows, metrics, diagnostics."""
    ordered = series.sort_values("cycle", kind="mergesort").reset_index(drop=True)
    train_values, holdout_values = split_forecast_series(
        ordered["cycle"],
        ordered["health_index"],
        train_fraction=train_fraction,
        minimum_train_size=minimum_train_size,
    )
    n_train = len(train_values)
    n_holdout = len(holdout_values)
    order = select_arima_order(train_values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(
            train_values.to_numpy(dtype="float64"),
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
    forecast_result = fitted.get_forecast(steps=n_holdout)
    predicted = np.asarray(forecast_result.predicted_mean, dtype="float64")
    interval = forecast_result.conf_int(alpha=alpha)
    lower = np.asarray(interval[:, 0], dtype="float64")
    upper = np.asarray(interval[:, 1], dtype="float64")
    metrics = evaluate_arima_forecast(
        holdout_values.to_numpy(dtype="float64"), predicted, lower, upper
    )
    metrics_row: dict[str, Any] = {
        "split": split,
        "engine_id": engine_id,
        "fold": fold,
        "n_train": n_train,
        "n_holdout": n_holdout,
        "order": str(order),
        **metrics,
        "status": "ok",
        "error": "",
    }
    try:
        diagnostics = _diagnostics(np.asarray(train_values, dtype="float64"))
        residuals = np.asarray(fitted.resid, dtype="float64")
        lag = min(10, max(1, len(residuals) // 5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ljung = acorr_ljungbox(residuals, lags=[lag], return_df=True)
        diagnostics.update(
            {
                "ljung_box_lag": lag,
                "ljung_box_statistic": float(ljung["lb_stat"].iloc[-1]),
                "ljung_box_pvalue": float(ljung["lb_pvalue"].iloc[-1]),
                "status": "ok",
                "error": "",
            }
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        diagnostics = {
            "adf_statistic": np.nan,
            "adf_pvalue": np.nan,
            "adf_usedlag": -1,
            "adf_nobs": n_train,
            "ljung_box_lag": -1,
            "ljung_box_statistic": np.nan,
            "ljung_box_pvalue": np.nan,
            "status": "diagnostic_failed",
            "error": str(error),
        }
    diagnostics_row: dict[str, Any] = {
        "split": split,
        "engine_id": engine_id,
        "fold": fold,
        "n_train": n_train,
        "order": str(order),
        **diagnostics,
    }
    forecast_rows = pd.DataFrame(
        {
            "split": split,
            "engine_id": engine_id,
            "cycle": ordered.loc[ordered.index[n_train:], "cycle"].to_numpy(dtype="int64"),
            "health_index_true": holdout_values.to_numpy(dtype="float64"),
            "health_index_pred": predicted,
            "lower_95": lower,
            "upper_95": upper,
            "is_holdout": True,
        },
        columns=FORECAST_COLUMNS,
    )
    return forecast_rows, metrics_row, diagnostics_row
