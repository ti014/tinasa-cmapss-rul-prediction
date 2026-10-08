"""Chạy lại cmapss.degradation và cmapss.maintenance, so với artifact đã lưu trong outputs/."""

import json
from pathlib import Path

import pandas as pd
import pytest

from cmapss.degradation.cli import main as degradation_main
from cmapss.maintenance.cli import main as maintenance_main

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

pytestmark = pytest.mark.skipif(not (RAW / "train_FD001.txt").is_file(), reason="thiếu dữ liệu raw FD001")


def _same_table(actual: Path, stored: Path) -> None:
    read = pd.read_parquet if actual.suffix == ".parquet" else pd.read_csv
    pd.testing.assert_frame_equal(read(actual), read(stored), check_exact=False, rtol=1e-9)


def test_degradation_artifacts_match_a_fresh_run(tmp_path: Path) -> None:
    degradation_main(["--train", str(RAW / "train_FD001.txt"), "--output-dir", str(tmp_path)])
    stored = ROOT / "outputs" / "degradation"
    for name in ("noise_summary.csv", "anomaly_events.csv", "anomaly_sensitivity.csv", "signals.parquet"):
        _same_table(tmp_path / "tables" / name, stored / "tables" / name)
    fresh = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    saved = json.loads((stored / "summary.json").read_text(encoding="utf-8"))
    for key in ("noise_reduction_pct_median", "anomaly_config", "anomaly_smoothed", "anomaly_raw_for_comparison"):
        assert fresh[key] == pytest.approx(saved[key]), key


def test_maintenance_artifacts_match_a_fresh_run(tmp_path: Path) -> None:
    maintenance_main(
        [
            "--train", str(RAW / "train_FD001.txt"),
            "--test", str(RAW / "test_FD001.txt"),
            "--labels", str(RAW / "RUL_FD001.txt"),
            "--predictions", str(ROOT / "outputs/modeling/predictions/final_predictions.parquet"),
            "--output-dir", str(tmp_path),
        ]
    )
    stored = ROOT / "outputs" / "maintenance" / "tables"
    for name in (
        "maintenance_schedule.csv",
        "maintenance_planning.csv",
        "status_vs_actual.csv",
        "test_anomaly_status.csv",
        "test_health_index.parquet",
    ):
        _same_table(tmp_path / "tables" / name, stored / name)
