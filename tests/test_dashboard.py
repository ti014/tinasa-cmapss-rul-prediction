from pathlib import Path

import pandas as pd
import pytest

from cmapss.dashboard import (
    DashboardDataError,
    _require_columns,
    engine_oof_trajectory,
    format_metric,
    load_dashboard_data,
    load_exploration_data,
    load_degradation_data,
    load_maintenance_data,
    engine_signal_view,
    sensor_trajectory,
)


ROOT = Path(__file__).resolve().parents[1]


def test_format_metric_is_stable() -> None:
    assert format_metric(1234.56789) == "1.234,5679"
    assert format_metric(42, decimals=0) == "42"


def test_dashboard_loader_reports_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(DashboardDataError, match="Thiếu artifact dashboard"):
        load_dashboard_data(tmp_path)


def test_dashboard_loader_reads_persisted_artifacts() -> None:
    data = load_dashboard_data(ROOT)
    assert len(data.model_evaluation) == 100
    assert data.model_comparison.loc[data.model_comparison["selected"] == True, "model_name"].tolist() == ["hist_gradient_boosting"]
    assert data.selected_model == "hist_gradient_boosting"
    assert len(data.warning_events) == 100
    assert data.warning_summary["split"] == "train_oof"


def test_dashboard_loader_rejects_missing_columns() -> None:
    with pytest.raises(DashboardDataError, match="final_predictions.parquet thiếu các cột"):
        _require_columns(pd.DataFrame({"engine_id": [1]}), {"engine_id", "cycle"}, "final_predictions.parquet")


def test_oof_trajectory_uses_selected_model_and_is_sorted() -> None:
    data = load_dashboard_data(ROOT)
    trajectory = engine_oof_trajectory(data, 1)
    assert set(trajectory["model_name"]) == {"hist_gradient_boosting"}
    assert trajectory["cycle"].is_monotonic_increasing
    assert trajectory["fold"].nunique() == 1


def test_exploration_data_keeps_train_only_sensor_values() -> None:
    exploration = load_exploration_data(ROOT)
    assert exploration.window_cycles == 30
    assert "split" not in exploration.sensor_trajectories.columns
    assert exploration.sensor_trajectories["engine_id"].nunique() == 100
    long = sensor_trajectory(exploration, 1, ["sensor_11", "sensor_4"], "mean")
    assert set(long["sensor"]) == {"sensor_11", "sensor_4"}
    with pytest.raises(ValueError):
        sensor_trajectory(exploration, 1, ["sensor_11"], "std")


def test_degradation_data_has_raw_and_smoothed_signals() -> None:
    data = load_degradation_data(ROOT)
    assert len(data.sensors) == 14 and "sensor_11" in data.sensors
    assert data.signals["engine_id"].nunique() == 100
    assert len(data.sensitivity) == 18 and set(data.sensitivity["signal"]) == {"ewma", "raw"}
    chosen = data.sensitivity.query("signal == 'ewma' and z_threshold == 3.0 and persistence == 5").iloc[0]
    assert chosen["lead_median"] == data.summary["anomaly_smoothed"]["lead_median"]
    view = engine_signal_view(data, 1, "sensor_11")
    assert set(view["Tín hiệu"]) == {"Tín hiệu gốc", "Sau EWMA"}
    with pytest.raises(ValueError):
        engine_signal_view(data, 1, "sensor_1")


def test_maintenance_data_covers_every_test_engine() -> None:
    data = load_maintenance_data(ROOT)
    assert len(data.schedule) == 100 and data.schedule["engine_id"].is_unique
    assert data.schedule["priority"].tolist() == list(range(1, 101))
    assert int(data.check.drop(columns="status").to_numpy().sum()) == 100
    assert data.summary["labels_read_after_schedule_written"] is True
    assert data.test_health_index["engine_id"].nunique() == 100


@pytest.mark.parametrize(
    "page",
    ["Tổng quan", "Dữ liệu & EDA", "Khử nhiễu", "Xu hướng & bất thường", "Dự đoán RUL", "Cảnh báo sớm", "Lịch bảo trì", "Provenance"],
)
def test_every_dashboard_page_renders_without_exception(page: str) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    app.run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [item.value for item in app.exception]
    assert not app.error, [item.value for item in app.error]


def test_app_entry_point_imports_without_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    """cmapss.reproducibility imports load_dashboard_data from app.py without the dashboard extra."""
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "streamlit", None)
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.delitem(sys.modules, "app", raising=False)
    app = importlib.import_module("app")
    assert app.load_dashboard_data is load_dashboard_data
