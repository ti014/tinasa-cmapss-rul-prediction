"""Đọc artifact đã lưu trong outputs/ cho dashboard Streamlit, không ghi dữ liệu."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

TRAJECTORY_SENSOR_STATS = ("last", "mean")


class DashboardDataError(RuntimeError):
    """Báo lỗi khi artifact đầu vào của dashboard bị thiếu hoặc sai cấu trúc."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DashboardDataError(f"Thiếu artifact dashboard: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"Không đọc được artifact dashboard: {path}") from exc
    if not isinstance(payload, dict):
        raise DashboardDataError(f"JSON dashboard phải chứa một object: {path}")
    return payload


def _read_table(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Đọc bảng CSV hoặc Parquet không rỗng theo phần mở rộng của file."""
    if not path.is_file():
        raise DashboardDataError(f"Thiếu artifact dashboard: {path}")
    try:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path, columns=columns)
        else:
            frame = pd.read_csv(path, usecols=columns)
    except (OSError, ValueError, ImportError) as exc:
        raise DashboardDataError(f"Không đọc được artifact dashboard: {path}") from exc
    if frame.empty:
        raise DashboardDataError(f"Artifact dashboard rỗng: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise DashboardDataError(f"{source} thiếu các cột: {', '.join(missing)}")


def _read_checked(path: Path, columns: set[str]) -> pd.DataFrame:
    frame = _read_table(path)
    _require_columns(frame, columns, path.name)
    return frame


@dataclass(frozen=True)
class DashboardData:
    """Artifact cuối của mô hình FD001 và cảnh báo sớm."""

    model_metrics: dict[str, Any]
    model_comparison: pd.DataFrame
    warning_events: pd.DataFrame
    warning_summary: dict[str, Any]
    model_predictions: pd.DataFrame
    model_evaluation: pd.DataFrame
    oof_predictions: pd.DataFrame
    model_run_info: dict[str, Any]

    @property
    def selected_model(self) -> str:
        return str(self.model_run_info.get("selected_model", "n/a"))


def load_dashboard_data(repo_root: str | Path = ".") -> DashboardData:
    root = Path(repo_root).resolve()
    modeling = root / "outputs" / "modeling"
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
        model_metrics=_read_json(modeling / "tables" / "final_metrics.json"),
        warning_summary=_read_json(modeling / "tables" / "warning_summary.json"),
        model_run_info=_read_json(modeling / "run_info.json"),
    )


@dataclass(frozen=True)
class ExplorationData:
    """Artifact thống kê mô tả chỉ từ train cho trang khám phá dữ liệu."""

    engine_lifetimes: pd.DataFrame
    sensor_rul_summary: pd.DataFrame
    sensor_trajectories: pd.DataFrame
    window_cycles: int


def load_exploration_data(repo_root: str | Path = ".") -> ExplorationData:
    """
    Đọc bảng EDA và sensor train từ Gold features; __last là phép đo gốc, __mean là
    trung bình window; thiếu window_cycles−1 cycle đầu mỗi động cơ.
    """
    root = Path(repo_root).resolve()
    eda = root / "outputs" / "eda" / "tables"
    features_path = root / "outputs" / "features" / "gold" / "rolling_features.parquet"
    feature_info = _read_json(root / "outputs" / "features" / "run_info.json")
    try:
        import pyarrow.parquet as pq

        schema_names = pq.read_schema(features_path).names
    except (OSError, ImportError) as exc:
        raise DashboardDataError(f"Không đọc được artifact dashboard: {features_path}") from exc
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


def format_metric(value: float | int, decimals: int = 4) -> str:
    """Định dạng số kiểu Việt: phẩy thập phân, chấm hàng nghìn."""
    from cmapss.presentation import format_number_vi

    return format_number_vi(value, decimals)


def engine_rul_view(data: DashboardData, engine_id: int) -> pd.DataFrame:
    view = data.model_evaluation.loc[data.model_evaluation["engine_id"] == engine_id].copy()
    return view.sort_values(["split", "cycle"]).reset_index(drop=True)


def engine_oof_trajectory(data: DashboardData, engine_id: int, model_name: str | None = None) -> pd.DataFrame:
    """Dự đoán RUL OOF theo cycle của một động cơ train."""
    model = model_name or data.selected_model
    oof = data.oof_predictions
    view = oof.loc[(oof["engine_id"] == engine_id) & (oof["model_name"] == model)]
    return view.sort_values("cycle").reset_index(drop=True)


def sensor_trajectory(exploration: ExplorationData, engine_id: int, sensors: list[str], stat: str) -> pd.DataFrame:
    """Bảng dài cycle, sensor, value cho các sensor train được chọn."""
    if stat not in TRAJECTORY_SENSOR_STATS:
        raise ValueError(f"stat phải thuộc {TRAJECTORY_SENSOR_STATS}")
    frame = exploration.sensor_trajectories
    engine = frame.loc[frame["engine_id"] == engine_id]
    columns = {f"{sensor}__{stat}": sensor for sensor in sensors if f"{sensor}__{stat}" in engine.columns}
    long = engine[["cycle", *columns]].rename(columns=columns).melt("cycle", var_name="sensor", value_name="value")
    return long.sort_values(["sensor", "cycle"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Khử nhiễu, phát hiện bất thường và lịch bảo trì
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DegradationData:
    """Đầu ra của python -m cmapss.degradation trên train FD001."""

    summary: dict[str, Any]
    noise: pd.DataFrame
    events: pd.DataFrame
    signals: pd.DataFrame
    sensitivity: pd.DataFrame

    @property
    def sensors(self) -> list[str]:
        return [column.removesuffix("__raw") for column in self.signals.columns if column.endswith("__raw")]


def load_degradation_data(repo_root: str | Path = ".") -> DegradationData:
    """Đọc tín hiệu gốc/EWMA, Health Index và cycle phát hiện bất thường."""
    base = Path(repo_root).resolve() / "outputs" / "degradation"
    return DegradationData(
        summary=_read_json(base / "summary.json"),
        noise=_read_checked(base / "tables" / "noise_summary.csv", {"sensor", "diff_std_raw", "diff_std_smoothed", "reduction_pct"}),
        events=_read_checked(
            base / "tables" / "anomaly_events.csv",
            {"engine_id", "failure_cycle", "baseline_mean", "baseline_std", "detection_cycle", "lead_cycles", "detected"},
        ),
        signals=_read_checked(
            base / "tables" / "signals.parquet", {"engine_id", "cycle", "health_index_raw", "health_index_ewma"}
        ),
        sensitivity=_read_checked(
            base / "tables" / "anomaly_sensitivity.csv",
            {"signal", "z_threshold", "persistence", "n_engines", "n_detected", "lead_min", "lead_median", "life_fraction_median", "n_early"},
        ),
    )


def engine_signal_view(data: DegradationData, engine_id: int, sensor: str) -> pd.DataFrame:
    """Long frame (cycle, Tín hiệu, value) of one sensor before and after EWMA."""
    if sensor not in data.sensors:
        raise ValueError(f"sensor không có trong artifact khử nhiễu: {sensor}")
    engine = data.signals.loc[data.signals["engine_id"] == engine_id, ["cycle", f"{sensor}__raw", f"{sensor}__ewma"]]
    renamed = engine.rename(columns={f"{sensor}__raw": "Tín hiệu gốc", f"{sensor}__ewma": "Sau EWMA"})
    return renamed.melt("cycle", var_name="Tín hiệu", value_name="value").sort_values(["Tín hiệu", "cycle"])


@dataclass(frozen=True)
class MaintenanceData:
    """Đầu ra của python -m cmapss.maintenance cho 100 động cơ test."""

    summary: dict[str, Any]
    schedule: pd.DataFrame
    planning: pd.DataFrame
    check: pd.DataFrame
    test_health_index: pd.DataFrame
    test_anomaly: pd.DataFrame


def load_maintenance_data(repo_root: str | Path = ".") -> MaintenanceData:
    """
    Đọc lịch, kế hoạch, bảng đối chiếu đã tính sẵn và Health Index test; không mở nhãn
    raw test.
    """
    base = Path(repo_root).resolve() / "outputs" / "maintenance"
    tables = base / "tables"
    return MaintenanceData(
        summary=_read_json(base / "summary.json"),
        schedule=_read_checked(
            tables / "maintenance_schedule.csv",
            {"priority", "engine_id", "current_cycle", "rul_pred", "cycles_to_maintenance", "maintenance_cycle",
             "predicted_failure_cycle", "anomaly_detected", "anomaly_cycle", "status"},
        ),
        planning=_read_checked(tables / "maintenance_planning.csv", {"window"}),
        check=_read_checked(tables / "status_vs_actual.csv", {"status"}),
        test_health_index=_read_checked(tables / "test_health_index.parquet", {"engine_id", "cycle", "health_index"}),
        test_anomaly=_read_checked(tables / "test_anomaly_status.csv", {"engine_id", "anomaly_evaluable", "threshold"}),
    )
