from pathlib import Path

import pandas as pd
import pytest

from app import DashboardDataError, _require_columns, format_metric, load_dashboard_data


def test_format_metric_is_stable() -> None:
    assert format_metric(1234.56789) == "1,234.5679"
    assert format_metric(42, decimals=0) == "42"


def test_dashboard_loader_reports_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(DashboardDataError, match="Missing dashboard artifact"):
        load_dashboard_data(tmp_path)


def test_dashboard_loader_reads_persisted_artifacts() -> None:
    data = load_dashboard_data(Path(__file__).resolve().parents[1])
    assert len(data.model_evaluation) == 100
    assert len(data.health_index) == 20_631
    assert len(data.forecasts) == 704
    assert data.timeseries_run_info["isolation"]["uses_test_labels"] is False


def test_dashboard_loader_rejects_missing_columns(tmp_path: Path) -> None:
    del tmp_path
    with pytest.raises(DashboardDataError, match="final_predictions.parquet is missing columns"):
        _require_columns(pd.DataFrame({"engine_id": [1]}), {"engine_id", "cycle"}, "final_predictions.parquet")
