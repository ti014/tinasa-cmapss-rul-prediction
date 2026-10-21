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

from cmapss.presentation import COLORS, apply_plot_style, style_figure
from cmapss.data.validation import validate_observations
from cmapss.timeseries.arima import fit_engine_arima
from cmapss.timeseries.baselines import evaluate_forecast_baselines
from cmapss.timeseries.contracts import (
    BASELINE_METRIC_COLUMNS,
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
    """Cấu hình PCA--ARIMA."""

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
    baseline_metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    input_sha256: dict[str, str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _plot_health_index(health_index: pd.DataFrame, selected: pd.DataFrame, path: Path) -> None:
    """Vẽ Health Index của các engine đã chọn."""
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    selected_ids = selected["engine_id"].tolist()
    for engine_id in selected_ids:
        series = health_index.loc[health_index["engine_id"].eq(engine_id)]
        ax.plot(series["cycle"], series["health_index"], linewidth=1.0, label=f"Động cơ {engine_id}")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Health Index (PC1 chuẩn hóa)")
    ax.set_title("Health Index trên các động cơ đánh giá đã chọn")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    style_figure(fig)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_forecasts(
    health_index: pd.DataFrame,
    selected: pd.DataFrame,
    forecasts: pd.DataFrame,
    path: Path,
) -> None:
    """Vẽ Health Index thực tế và dự báo cho từng động cơ đã chọn."""
    apply_plot_style()
    n = len(selected)
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(15, 7), squeeze=False)
    for axis, row in zip(axes.ravel(), selected.itertuples(index=False)):
        series = health_index.loc[health_index["engine_id"].eq(row.engine_id)]
        forecast = forecasts.loc[forecasts["engine_id"].eq(row.engine_id)]
        axis.plot(series["cycle"], series["health_index"], color="0.55", linewidth=1.0, label="Thực tế")
        if not forecast.empty:
            axis.plot(forecast["cycle"], forecast["health_index_pred"], color=COLORS["red"], label="ARIMA dự báo")
            axis.fill_between(
                forecast["cycle"].to_numpy(),
                forecast["lower_95"].to_numpy(),
                forecast["upper_95"].to_numpy(),
                color=COLORS["orange"],
                alpha=0.25,
                label="Khoảng dự đoán 95%",
            )
            axis.axvline(int(forecast["cycle"].min()), color=COLORS["blue"], linestyle="--", linewidth=0.8)
        axis.set_title(f"Động cơ {row.engine_id}")
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[n:]:
        axis.axis("off")
    axes[0, 0].set_ylabel("Health Index")
    axes[1, 0].set_ylabel("Health Index")
    for axis in axes[-1, :]:
        axis.set_xlabel("Cycle")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("ARIMA dự báo 30% cuối chuỗi Health Index", y=1.07)
    fig.tight_layout()
    style_figure(fig)
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
    selected = select_validation_engines(
        observations,
        folds,
        n_engines=run_config.n_engines,
    )
    selected_engine_ids = set(int(value) for value in selected["engine_id"])
    train_engine_ids = set(
        int(value)
        for value in observations.loc[observations["split"].eq("train"), "engine_id"].unique()
    )
    fit_engine_ids = train_engine_ids.difference(selected_engine_ids)
    health_result = fit_health_index(observations, fit_engine_ids=fit_engine_ids)
    validate_health_index(health_result.health_index)
    forecast_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        series = health_result.health_index.loc[
            health_result.health_index["engine_id"].eq(row.engine_id),
            ["cycle", "health_index"],
        ]
        n_train = int(np.floor(len(series) * run_config.train_fraction))
        for name, baseline in evaluate_forecast_baselines(
            series["health_index"].iloc[:n_train],
            series["health_index"].iloc[n_train:],
        ).items():
            baseline_rows.append(
                {
                    "split": str(row.split),
                    "engine_id": int(row.engine_id),
                    "fold": int(row.fold),
                    "baseline_name": name,
                    **baseline,
                }
            )
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
            warning_count = int(getattr(error, "warning_count", 0))
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
                "fit_converged": False,
                "fit_warning_count": warning_count,
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
                "model_df": -1,
                "fit_converged": False,
                "fit_warning_count": warning_count,
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
    baseline_metrics = pd.DataFrame(baseline_rows, columns=BASELINE_METRIC_COLUMNS)
    diagnostics = pd.DataFrame(diagnostic_rows, columns=DIAGNOSTIC_COLUMNS)
    return TimeseriesResult(
        config=run_config,
        health_index_result=health_result,
        selected_engines=selected,
        forecasts=forecasts,
        forecast_metrics=metrics,
        baseline_metrics=baseline_metrics,
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
        "baseline_metrics": table_dir / "baseline_metrics.csv",
        "diagnostics": table_dir / "diagnostics.csv",
        "arima_forecasts": prediction_dir / "arima_forecasts.parquet",
        "health_index_figure": figure_dir / "health_index_selected_engines.png",
        "forecast_figure": figure_dir / "arima_forecasts.png",
        "scaler": output_path / "models" / "health_index_scaler.joblib",
        "pca": output_path / "models" / "health_index_pca.joblib",
        "health_index_parameters": output_path / "models" / "health_index_parameters.json",
        "run_info": output_path / "run_info.json",
    }
    paths["models"] = output_path / "models"
    paths["models"].mkdir(parents=True, exist_ok=True)
    result.selected_engines.to_csv(paths["selected_engines"], index=False)
    result.health_index_result.health_index.to_csv(paths["health_index"], index=False)
    result.forecast_metrics.to_csv(paths["forecast_metrics"], index=False)
    result.baseline_metrics.to_csv(paths["baseline_metrics"], index=False)
    result.diagnostics.to_csv(paths["diagnostics"], index=False)
    result.forecasts.to_parquet(paths["arima_forecasts"], index=False)
    joblib.dump(result.health_index_result.scaler, paths["scaler"])
    joblib.dump(result.health_index_result.pca, paths["pca"])
    paths["health_index_parameters"].write_text(
        json.dumps(
            {
                "orientation_sign": result.health_index_result.orientation_sign,
                "normalization_mean": result.health_index_result.normalization_mean,
                "normalization_scale": result.health_index_result.normalization_scale,
                "fit_engine_ids": list(result.health_index_result.fit_engine_ids),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _plot_health_index(result.health_index_result.health_index, result.selected_engines, paths["health_index_figure"])
    _plot_forecasts(
        result.health_index_result.health_index,
        result.selected_engines,
        result.forecasts,
        paths["forecast_figure"],
    )
    ok_metrics = result.forecast_metrics.loc[result.forecast_metrics["status"].eq("ok")]
    baseline_summary = (
        result.baseline_metrics.groupby("baseline_name", sort=True)
        .agg(mean_engine_mae=("mae", "mean"), mean_rmse=("rmse", "mean"))
        .reset_index()
    )
    arima_by_engine = result.forecast_metrics.loc[
        result.forecast_metrics["status"].eq("ok"), ["engine_id", "mae"]
    ].rename(columns={"mae": "arima_mae"})
    comparison_rows = result.baseline_metrics.merge(arima_by_engine, on="engine_id", how="inner")
    comparison_rows["arima_lower_mae"] = comparison_rows["arima_mae"] < comparison_rows["mae"]
    baseline_comparison = (
        comparison_rows.groupby("baseline_name", sort=True)["arima_lower_mae"].sum().astype(int).to_dict()
    )
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
            "point_forecast_baselines": "last_value and endpoint drift; same fit prefix and holdout as ARIMA; no prediction intervals",
            "health_index": "standardized sensor columns -> PCA PC1; orientation and normalization fit on non-selected train engines",
        },
        "selection": {
            "rule": "quota cân bằng theo GroupKFold fold; lifetime quantile positions trong từng fold",
            "forecast_independent": True,
            "n_selected": int(len(result.selected_engines)),
            "selected_engine_ids": [int(value) for value in result.selected_engines["engine_id"]],
            "fit_engine_ids": [int(value) for value in result.health_index_result.fit_engine_ids],
        },
        "row_counts": {
            "health_index": int(len(result.health_index_result.health_index)),
            "selected_engines": int(len(result.selected_engines)),
            "forecasts": int(len(result.forecasts)),
            "forecast_metrics": int(len(result.forecast_metrics)),
            "baseline_metrics": int(len(result.baseline_metrics)),
            "diagnostics": int(len(result.diagnostics)),
            "successful_engines": int(len(ok_metrics)),
            "baseline_summary": baseline_summary.to_dict(orient="records"),
            "arima_lower_mae_engines_by_baseline": baseline_comparison,
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
            "health_index_fit_excludes_selected_engines": True,
            "health_index_normalization_mean": result.health_index_result.normalization_mean,
            "health_index_normalization_scale": result.health_index_result.normalization_scale,
        },
    }
    paths["run_info"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    return payload
