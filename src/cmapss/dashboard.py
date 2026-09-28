"""Read-only artifact layer for the Streamlit dashboard.

Every loader reads persisted outputs under ``outputs/``. Nothing here fits a
model, reads raw test labels, or writes files, so opening a dashboard page can
never change an experiment result.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


IMPROVEMENT_RUN = Path("outputs/experiments/rul_improvement/20260927_full")
TRAJECTORY_SENSOR_STATS = ("last", "mean")


class DashboardDataError(RuntimeError):
    """Raised when a dashboard input artifact is missing or malformed."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DashboardDataError(f"Missing dashboard artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"Could not read dashboard artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise DashboardDataError(f"Dashboard JSON must contain an object: {path}")
    return payload


def _read_table(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a non-empty CSV or Parquet artifact, chosen by file suffix."""

    if not path.is_file():
        raise DashboardDataError(f"Missing dashboard artifact: {path}")
    try:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path, columns=columns)
        else:
            frame = pd.read_csv(path, usecols=columns)
    except (OSError, ValueError, ImportError) as exc:
        raise DashboardDataError(f"Could not read dashboard artifact: {path}") from exc
    if frame.empty:
        raise DashboardDataError(f"Dashboard artifact is empty: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise DashboardDataError(f"{source} is missing columns: {', '.join(missing)}")


def _read_checked(path: Path, columns: set[str]) -> pd.DataFrame:
    frame = _read_table(path)
    _require_columns(frame, columns, path.name)
    return frame


@dataclass(frozen=True)
class DashboardData:
    """Final FD001 modeling, early-warning and time-series artifacts."""

    model_metrics: dict[str, Any]
    model_comparison: pd.DataFrame
    warning_events: pd.DataFrame
    warning_summary: dict[str, Any]
    model_predictions: pd.DataFrame
    model_evaluation: pd.DataFrame
    oof_predictions: pd.DataFrame
    health_index: pd.DataFrame
    forecast_metrics: pd.DataFrame
    baseline_metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    forecasts: pd.DataFrame
    model_run_info: dict[str, Any]
    timeseries_run_info: dict[str, Any]

    @property
    def selected_model(self) -> str:
        return str(self.model_run_info.get("selected_model", "n/a"))


def load_dashboard_data(repo_root: str | Path = ".") -> DashboardData:
    """Load dashboard inputs from persisted modeling and time-series outputs."""

    root = Path(repo_root).resolve()
    modeling = root / "outputs" / "modeling"
    timeseries = root / "outputs" / "timeseries"
    key = {"split", "engine_id", "cycle"}
    return DashboardData(
        model_predictions=_read_checked(
            modeling / "predictions" / "final_predictions.parquet", key | {"model_name", "rul_pred"}
        ),
        model_evaluation=_read_checked(
            modeling / "predictions" / "final_evaluation.parquet",
            key | {"y_true", "y_pred", "absolute_error"},
        ),
        oof_predictions=_read_checked(
            modeling / "predictions" / "oof_predictions.parquet",
            key | {"fold", "model_name", "y_true", "y_pred"},
        ),
        model_comparison=_read_checked(
            modeling / "tables" / "model_metrics.csv",
            {"model_name", "mean_engine_mae", "std_engine_mae", "selected"},
        ),
        warning_events=_read_checked(
            modeling / "tables" / "warning_events.csv",
            {
                "engine_id",
                "first_warning_cycle",
                "actual_rul_at_first_warning",
                "lead_time_to_failure_cycles",
                "missed_warning",
                "threshold_delay_cycles",
                "warning_timing",
            },
        ),
        health_index=_read_checked(timeseries / "tables" / "health_index.csv", key | {"health_index"}),
        forecast_metrics=_read_checked(
            timeseries / "tables" / "forecast_metrics.csv", {"engine_id", "mae", "rmse", "coverage_95", "status"}
        ),
        baseline_metrics=_read_checked(
            timeseries / "tables" / "baseline_metrics.csv", {"engine_id", "baseline_name", "mae", "rmse"}
        ),
        diagnostics=_read_checked(
            timeseries / "tables" / "diagnostics.csv", {"engine_id", "adf_pvalue", "ljung_box_pvalue", "status"}
        ),
        forecasts=_read_checked(
            timeseries / "predictions" / "arima_forecasts.parquet",
            {"engine_id", "cycle", "health_index_true", "health_index_pred", "lower_95", "upper_95"},
        ),
        model_metrics=_read_json(modeling / "tables" / "final_metrics.json"),
        warning_summary=_read_json(modeling / "tables" / "warning_summary.json"),
        model_run_info=_read_json(modeling / "run_info.json"),
        timeseries_run_info=_read_json(timeseries / "run_info.json"),
    )


@dataclass(frozen=True)
class ImprovementData:
    """Completed nested RUL improvement experiment shared with the report."""

    root: Path
    run_info: dict[str, Any]
    decision: dict[str, Any]
    ablation: pd.DataFrame
    paired: pd.DataFrame
    test: pd.DataFrame
    selection: pd.DataFrame
    predictions: pd.DataFrame


def load_improvement_data(repo_root: str | Path = ".") -> ImprovementData:
    """Read the completed, explicitly named nested experiment shared with report."""

    root = Path(repo_root).resolve() / IMPROVEMENT_RUN
    info = _read_json(root / "run_info.json")
    if not info.get("test_label_joined"):
        raise DashboardDataError("Thực nghiệm chưa hoàn tất đánh giá cuối.")
    return ImprovementData(
        root=root,
        run_info=info,
        decision=_read_json(root / "decision.json"),
        ablation=_read_checked(
            root / "ablation_metrics.csv",
            {"stage", "mean_engine_mae", "low_rul_engine_mae", "late_or_missed_count", "early_alarm_burden"},
        ),
        paired=_read_checked(root / "paired_comparison.csv", {"stage", "metric", "delta", "ci_lower", "ci_upper"}),
        test=_read_table(root / "test_comparison.csv"),
        selection=_read_checked(root / "final_selection_metrics.csv", {"stage", "candidate_id", "selected"}),
        predictions=_read_checked(
            root / "outer_predictions.parquet", {"stage", "engine_id", "cycle", "y_true", "y_pred"}
        ),
    )


@dataclass(frozen=True)
class ExplorationData:
    """Train-only descriptive artifacts used by the data exploration page."""

    engine_lifetimes: pd.DataFrame
    sensor_rul_summary: pd.DataFrame
    sensor_trajectories: pd.DataFrame
    window_cycles: int


def load_exploration_data(repo_root: str | Path = ".") -> ExplorationData:
    """Load EDA tables and per-cycle train sensor values from Gold features.

    Only ``sensor_*__last`` and ``sensor_*__mean`` columns of the train split are
    kept. ``__last`` equals the raw sensor reading at that cycle; the first
    ``window_cycles - 1`` cycles of every engine are absent because rolling
    features require a full trailing window.
    """

    root = Path(repo_root).resolve()
    eda = root / "outputs" / "eda" / "tables"
    features_path = root / "outputs" / "features" / "gold" / "rolling_features.parquet"
    feature_info = _read_json(root / "outputs" / "features" / "run_info.json")
    try:
        import pyarrow.parquet as pq

        schema_names = pq.read_schema(features_path).names
    except (OSError, ImportError) as exc:
        raise DashboardDataError(f"Could not read dashboard artifact: {features_path}") from exc
    columns = ["split", "engine_id", "cycle"] + [
        name for name in schema_names if name.rsplit("__", 1)[-1] in TRAJECTORY_SENSOR_STATS
    ]
    features = _read_table(features_path, columns=columns)
    trajectories = features.loc[features["split"] == "train"].drop(columns="split").reset_index(drop=True)
    return ExplorationData(
        engine_lifetimes=_read_checked(eda / "engine_lifetimes.csv", {"engine_id", "n_observations"}),
        sensor_rul_summary=_read_checked(
            eda / "sensor_rul_summary.csv", {"sensor", "defined_engines", "median", "q25", "q75"}
        ),
        sensor_trajectories=trajectories,
        window_cycles=int(feature_info["window_cycles"]),
    )


def load_cross_dataset(repo_root: str | Path = ".") -> pd.DataFrame:
    """Flatten ``outputs/cross_dataset/matrix.json`` into one row per evaluation."""

    matrix = _read_json(Path(repo_root).resolve() / "outputs" / "cross_dataset" / "matrix.json")
    evaluations = matrix.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise DashboardDataError("matrix.json has no evaluations")
    rows = []
    for item in evaluations:
        metrics = item["metrics"]
        interval = item.get("bootstrap", {}).get("intervals", {}).get("mean_engine_mae", {})
        rows.append(
            {
                "source": item["source_dataset"],
                "target": item["target_dataset"],
                "evaluation_type": item["evaluation_type"],
                "n_engines": int(metrics["n_engines"]),
                "mean_engine_mae": float(metrics["mean_engine_mae"]),
                "mae_ci_lower": interval.get("ci_lower"),
                "mae_ci_upper": interval.get("ci_upper"),
                "rmse": float(metrics["rmse"]),
                "nasa_score": float(metrics["nasa_score"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["window_cycles"] = matrix.get("window_cycles")
    return frame


def format_metric(value: float | int, decimals: int = 4) -> str:
    """Format a numeric value consistently for dashboard metric cards."""

    return f"{float(value):,.{decimals}f}"


def selected_engine_ids(data: DashboardData) -> list[int]:
    """Return deterministic engine choices available in persisted artifacts."""

    ids = pd.concat(
        [data.model_evaluation["engine_id"], data.health_index["engine_id"]],
        ignore_index=True,
    ).dropna()
    return sorted({int(engine_id) for engine_id in ids})


def engine_rul_view(data: DashboardData, engine_id: int) -> pd.DataFrame:
    """Return the final RUL evaluation row for one test engine."""

    view = data.model_evaluation.loc[data.model_evaluation["engine_id"] == engine_id].copy()
    return view.sort_values(["split", "cycle"]).reset_index(drop=True)


def engine_oof_trajectory(data: DashboardData, engine_id: int, model_name: str | None = None) -> pd.DataFrame:
    """Return per-cycle out-of-fold RUL predictions for one train engine."""

    model = model_name or data.selected_model
    oof = data.oof_predictions
    view = oof.loc[(oof["engine_id"] == engine_id) & (oof["model_name"] == model)]
    return view.sort_values("cycle").reset_index(drop=True)


def engine_timeseries_view(data: DashboardData, engine_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Health Index history and ARIMA holdout forecast for one engine."""

    history = data.health_index.loc[data.health_index["engine_id"] == engine_id].copy()
    forecast = data.forecasts.loc[data.forecasts["engine_id"] == engine_id].copy()
    return (
        history.sort_values("cycle").reset_index(drop=True),
        forecast.sort_values("cycle").reset_index(drop=True),
    )


def forecast_comparison(data: DashboardData, engine_id: int) -> pd.DataFrame:
    """Return ARIMA and naive baseline holdout errors for one engine."""

    arima = data.forecast_metrics.loc[data.forecast_metrics["engine_id"] == engine_id, ["mae", "rmse"]]
    arima = arima.assign(method="arima")
    baselines = data.baseline_metrics.loc[data.baseline_metrics["engine_id"] == engine_id]
    baselines = baselines.rename(columns={"baseline_name": "method"})[["method", "mae", "rmse"]]
    return pd.concat([arima[["method", "mae", "rmse"]], baselines], ignore_index=True)


def sensor_trajectory(exploration: ExplorationData, engine_id: int, sensors: list[str], stat: str) -> pd.DataFrame:
    """Return a long frame ``cycle, sensor, value`` for selected train sensors."""

    if stat not in TRAJECTORY_SENSOR_STATS:
        raise ValueError(f"stat must be one of {TRAJECTORY_SENSOR_STATS}")
    frame = exploration.sensor_trajectories
    engine = frame.loc[frame["engine_id"] == engine_id]
    columns = {f"{sensor}__{stat}": sensor for sensor in sensors if f"{sensor}__{stat}" in engine.columns}
    long = engine[["cycle", *columns]].rename(columns=columns).melt("cycle", var_name="sensor", value_name="value")
    return long.sort_values(["sensor", "cycle"]).reset_index(drop=True)
