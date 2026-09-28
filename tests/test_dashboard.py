from pathlib import Path

import pandas as pd
import pytest

from cmapss.dashboard import (
    DashboardDataError,
    _require_columns,
    engine_oof_trajectory,
    forecast_comparison,
    format_metric,
    load_cross_dataset,
    load_dashboard_data,
    load_exploration_data,
    load_improvement_data,
    sensor_trajectory,
)


ROOT = Path(__file__).resolve().parents[1]


def test_format_metric_is_stable() -> None:
    assert format_metric(1234.56789) == "1,234.5679"
    assert format_metric(42, decimals=0) == "42"


def test_dashboard_loader_reports_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(DashboardDataError, match="Missing dashboard artifact"):
        load_dashboard_data(tmp_path)


def test_dashboard_loader_reads_persisted_artifacts() -> None:
    data = load_dashboard_data(ROOT)
    assert len(data.model_evaluation) == 100
    assert data.model_comparison.loc[data.model_comparison["selected"] == True, "model_name"].tolist() == ["hist_gradient_boosting"]
    assert data.selected_model == "hist_gradient_boosting"
    assert len(data.health_index) == 20_631
    assert len(data.forecasts) == 704
    assert len(data.warning_events) == 100
    assert data.warning_summary["split"] == "train_oof"
    assert data.timeseries_run_info["isolation"]["uses_test_labels"] is False


def test_dashboard_loader_rejects_missing_columns() -> None:
    with pytest.raises(DashboardDataError, match="final_predictions.parquet is missing columns"):
        _require_columns(pd.DataFrame({"engine_id": [1]}), {"engine_id", "cycle"}, "final_predictions.parquet")


def test_improvement_dashboard_refuses_unfinished_run(tmp_path: Path) -> None:
    target = tmp_path / "outputs/experiments/rul_improvement/20260927_full"
    target.mkdir(parents=True)
    (target / "run_info.json").write_text('{"test_label_joined": false}', encoding="utf-8")
    with pytest.raises(DashboardDataError, match="chưa hoàn tất"):
        load_improvement_data(tmp_path)


def test_oof_trajectory_uses_selected_model_and_is_sorted() -> None:
    data = load_dashboard_data(ROOT)
    trajectory = engine_oof_trajectory(data, 1)
    assert set(trajectory["model_name"]) == {"hist_gradient_boosting"}
    assert trajectory["cycle"].is_monotonic_increasing
    assert trajectory["fold"].nunique() == 1


def test_forecast_comparison_lists_arima_and_baselines() -> None:
    data = load_dashboard_data(ROOT)
    engine_id = int(data.forecast_metrics["engine_id"].iloc[0])
    assert forecast_comparison(data, engine_id)["method"].tolist() == ["arima", "last_value", "drift"]


def test_exploration_data_keeps_train_only_sensor_values() -> None:
    exploration = load_exploration_data(ROOT)
    assert exploration.window_cycles == 30
    assert "split" not in exploration.sensor_trajectories.columns
    assert exploration.sensor_trajectories["engine_id"].nunique() == 100
    long = sensor_trajectory(exploration, 1, ["sensor_11", "sensor_4"], "mean")
    assert set(long["sensor"]) == {"sensor_11", "sensor_4"}
    with pytest.raises(ValueError):
        sensor_trajectory(exploration, 1, ["sensor_11"], "std")


def test_cross_dataset_matrix_has_within_and_transfer_rows() -> None:
    matrix = load_cross_dataset(ROOT)
    assert len(matrix) == 7
    assert matrix.attrs["window_cycles"] == 10
    assert (matrix["evaluation_type"] == "transfer").sum() == 3
    assert (matrix["mae_ci_lower"] <= matrix["mean_engine_mae"]).all()


@pytest.mark.parametrize(
    "page",
    ["Tổng quan", "Dữ liệu & EDA", "Đánh giá RUL", "Cảnh báo sớm", "Cross-dataset", "Thử nghiệm cải thiện", "Health Index / ARIMA", "Provenance"],
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
