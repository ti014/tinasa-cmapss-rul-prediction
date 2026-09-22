"""Pipeline PCA--ARIMA và ghi artifact train-only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

from cmapss.data.validation import validate_observations
from cmapss.timeseries.arima import fit_engine_arima
from cmapss.timeseries.contracts import (
    DIAGNOSTIC_COLUMNS,
    FORECAST_COLUMNS,
    FORECAST_METRIC_COLUMNS,
    TIMESERIES_MIN_TRAIN_SIZE,
    TIMESERIES_N_ENGINES,
    TIMESERIES_PREDICTION_ALPHA,
    TIMESERIES_SCHEMA_VERSION,
    TIMESERIES_SEED,
    TIMESERIES_TRAIN_FRACTION,
)
from cmapss.timeseries.health_index import (
    HealthIndexResult,
    fit_health_index,
    select_validation_engines,
    validate_health_index,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class TimeseriesConfig:
    """Cấu hình PCA--ARIMA đã chốt trong active plan."""

    n_engines: int = TIMESERIES_N_ENGINES
    train_fraction: float = TIMESERIES_TRAIN_FRACTION
    minimum_train_size: int = TIMESERIES_MIN_TRAIN_SIZE
    prediction_alpha: float = TIMESERIES_PREDICTION_ALPHA
    seed: int = TIMESERIES_SEED


@dataclass(frozen=True)
class TimeseriesResult:
    """Artifact trong bộ nhớ trước khi ghi outputs/timeseries."""

    config: TimeseriesConfig
    health_index_result: HealthIndexResult
    selected_engines: pd.DataFrame
    forecasts: pd.DataFrame
    forecast_metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    input_sha256: dict[str, str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _plot_health_index(health_index: pd.DataFrame, selected: pd.DataFrame, path: Path) -> None:
    """Vẽ Health Index của 10 engine đã chọn."""
    fig, ax = plt.subplots(figsize=(11, 6))
    selected_ids = selected["engine_id"].tolist()
    for engine_id in selected_ids:
        series = health_index.loc[health_index["engine_id"].eq(engine_id)]
        ax.plot(series["cycle"], series["health_index"], linewidth=1.0, label=f"engine {engine_id}")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Health Index (PC1 chuẩn hóa)")
    ax.set_title("Health Index trên các validation engines được chọn")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_forecasts(
    health_index: pd.DataFrame,
    selected: pd.DataFrame,
    forecasts: pd.DataFrame,
    path: Path,
) -> None:
    """Vẽ actual và forecast cho từng engine đã chọn."""
    n = len(selected)
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(15, 7), squeeze=False)
    for axis, row in zip(axes.ravel(), selected.itertuples(index=False)):
        series = health_index.loc[health_index["engine_id"].eq(row.engine_id)]
        forecast = forecasts.loc[forecasts["engine_id"].eq(row.engine_id)]
        axis.plot(series["cycle"], series["health_index"], color="0.55", linewidth=1.0, label="actual")
        if not forecast.empty:
            axis.plot(forecast["cycle"], forecast["health_index_pred"], color="#b2182b", label="forecast")
            axis.fill_between(
                forecast["cycle"].to_numpy(),
                forecast["lower_95"].to_numpy(),
                forecast["upper_95"].to_numpy(),
                color="#ef8a62",
                alpha=0.25,
                label="95% interval",
            )
            axis.axvline(int(forecast["cycle"].min()), color="#2166ac", linestyle="--", linewidth=0.8)
        axis.set_title(f"Engine {row.engine_id}")
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[n:]:
        axis.axis("off")
    axes[0, 0].set_ylabel("Health Index")
    axes[1, 0].set_ylabel("Health Index")
    for axis in axes[-1, :]:
        axis.set_xlabel("Cycle")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("ARIMA forecast 30% cuối chuỗi Health Index", y=1.07)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_timeseries_pipeline(
    observations_path: str | Path,
    fold_assignments_path: str | Path,
    *,
    config: TimeseriesConfig | None = None,
) -> TimeseriesResult:
    """Chạy PCA--ARIMA chỉ trên train observations."""
    run_config = config or TimeseriesConfig()
    observation_path = Path(observations_path)
    fold_path = Path(fold_assignments_path)
    observations = pd.read_parquet(observation_path)
    validate_observations(observations, source=str(observation_path))
    folds = pd.read_csv(fold_path)
    health_result = fit_health_index(observations)
    validate_health_index(health_result.health_index)
    selected = select_validation_engines(
        observations,
        folds,
        n_engines=run_config.n_engines,
    )
    forecast_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        series = health_result.health_index.loc[
            health_result.health_index["engine_id"].eq(row.engine_id),
            ["cycle", "health_index"],
        ]
        try:
            forecast, metrics, diagnostics = fit_engine_arima(
                series,
                split=str(row.split),
                engine_id=int(row.engine_id),
                fold=int(row.fold),
                train_fraction=run_config.train_fraction,
                minimum_train_size=run_config.minimum_train_size,
                alpha=run_config.prediction_alpha,
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            metrics = {
                "split": str(row.split),
                "engine_id": int(row.engine_id),
                "fold": int(row.fold),
                "n_train": int(np.floor(len(series) * run_config.train_fraction)),
                "n_holdout": len(series) - int(np.floor(len(series) * run_config.train_fraction)),
                "order": "",
                "mae": np.nan,
                "rmse": np.nan,
                "coverage_95": np.nan,
                "status": "fit_failed",
                "error": str(error),
            }
            diagnostics = {
                "split": str(row.split),
                "engine_id": int(row.engine_id),
                "fold": int(row.fold),
                "n_train": metrics["n_train"],
                "adf_statistic": np.nan,
                "adf_pvalue": np.nan,
                "adf_usedlag": -1,
                "adf_nobs": metrics["n_train"],
                "ljung_box_lag": -1,
                "ljung_box_statistic": np.nan,
                "ljung_box_pvalue": np.nan,
                "order": "",
                "status": "fit_failed",
                "error": str(error),
            }
        else:
            forecast_rows.append(forecast)
        metric_rows.append(metrics)
        diagnostic_rows.append(diagnostics)
    forecasts = (
        pd.concat(forecast_rows, ignore_index=True).loc[:, list(FORECAST_COLUMNS)]
        if forecast_rows
        else pd.DataFrame(columns=FORECAST_COLUMNS)
    )
    metrics = pd.DataFrame(metric_rows, columns=FORECAST_METRIC_COLUMNS)
    diagnostics = pd.DataFrame(diagnostic_rows, columns=DIAGNOSTIC_COLUMNS)
    return TimeseriesResult(
        config=run_config,
        health_index_result=health_result,
        selected_engines=selected,
        forecasts=forecasts,
        forecast_metrics=metrics,
        diagnostics=diagnostics,
        input_sha256={observation_path.name: _sha256(observation_path), fold_path.name: _sha256(fold_path)},
    )


def write_timeseries_artifacts(result: TimeseriesResult, output_dir: str | Path) -> dict[str, Any]:
    """Ghi bảng, forecast Parquet, hình và metadata PCA--ARIMA."""
    output_path = Path(output_dir).resolve()
    raw_path = Path("data/raw").resolve()
    if output_path == raw_path or raw_path in output_path.parents:
        raise ValueError("Timeseries output không được ghi vào data/raw để bảo vệ Bronze immutable")
    table_dir = output_path / "tables"
    prediction_dir = output_path / "predictions"
    figure_dir = output_path / "figures"
    for directory in (table_dir, prediction_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "selected_engines": table_dir / "selected_engines.csv",
        "health_index": table_dir / "health_index.csv",
        "forecast_metrics": table_dir / "forecast_metrics.csv",
        "diagnostics": table_dir / "diagnostics.csv",
        "arima_forecasts": prediction_dir / "arima_forecasts.parquet",
        "health_index_figure": figure_dir / "health_index_selected_engines.png",
        "forecast_figure": figure_dir / "arima_forecasts.png",
        "scaler": output_path / "models" / "health_index_scaler.joblib",
        "pca": output_path / "models" / "health_index_pca.joblib",
        "run_info": output_path / "run_info.json",
    }
    paths["models"] = output_path / "models"
    paths["models"].mkdir(parents=True, exist_ok=True)
    result.selected_engines.to_csv(paths["selected_engines"], index=False)
    result.health_index_result.health_index.to_csv(paths["health_index"], index=False)
    result.forecast_metrics.to_csv(paths["forecast_metrics"], index=False)
    result.diagnostics.to_csv(paths["diagnostics"], index=False)
    result.forecasts.to_parquet(paths["arima_forecasts"], index=False)
    joblib.dump(result.health_index_result.scaler, paths["scaler"])
    joblib.dump(result.health_index_result.pca, paths["pca"])
    _plot_health_index(result.health_index_result.health_index, result.selected_engines, paths["health_index_figure"])
    _plot_forecasts(
        result.health_index_result.health_index,
        result.selected_engines,
        result.forecasts,
        paths["forecast_figure"],
    )
    ok_metrics = result.forecast_metrics.loc[result.forecast_metrics["status"].eq("ok")]
    payload: dict[str, Any] = {
        "schema_version": TIMESERIES_SCHEMA_VERSION,
        "pipeline": "train_only_pca_arima_health_index",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": result.input_sha256,
        "config": {
            "n_engines": result.config.n_engines,
            "train_fraction": result.config.train_fraction,
            "minimum_train_size": result.config.minimum_train_size,
            "prediction_interval": 1.0 - result.config.prediction_alpha,
            "seed": result.config.seed,
            "order_selection": "AIC grid p,q=0..2 and d in {0,1} on fit segment only",
            "health_index": "standardized sensor columns -> PCA PC1; orientation chosen by train cycle correlation",
        },
        "selection": {
            "rule": "quota cân bằng theo GroupKFold fold; lifetime quantile positions trong từng fold",
            "forecast_independent": True,
            "n_selected": int(len(result.selected_engines)),
        },
        "row_counts": {
            "health_index": int(len(result.health_index_result.health_index)),
            "selected_engines": int(len(result.selected_engines)),
            "forecasts": int(len(result.forecasts)),
            "forecast_metrics": int(len(result.forecast_metrics)),
            "diagnostics": int(len(result.diagnostics)),
            "successful_engines": int(len(ok_metrics)),
        },
        "outputs": {
            name: str(path.relative_to(output_path))
            for name, path in paths.items()
            if name not in {"run_info", "models"}
        },
        "isolation": {
            "uses_test_observations": False,
            "uses_test_labels": False,
            "forecast_is_rul": False,
            "health_index_fit_split": "train",
        },
    }
    paths["run_info"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    return payload
