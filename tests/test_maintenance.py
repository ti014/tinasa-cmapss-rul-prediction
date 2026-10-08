"""Kiểm thử lịch bảo trì bằng fixture tổng hợp."""

import pandas as pd
import pytest

from cmapss.data.contracts import ContractError
from cmapss.degradation import AnomalyConfig
from cmapss.maintenance import (
    STATUS_ORDER,
    MaintenanceConfig,
    anomaly_status,
    build_schedule,
    evaluate_schedule,
)

NEEDS, WATCH, NORMAL = STATUS_ORDER


def _anomalies(flags: dict[int, bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {"engine_id": list(flags), "anomaly_evaluable": True, "anomaly_detected": list(flags.values()),
         "anomaly_cycle": [50 if f else None for f in flags.values()]}
    ).astype({"anomaly_cycle": "Int64"})


def _predictions(rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["engine_id", "cycle", "rul_pred"])


def test_schedule_arithmetic_and_status_priority() -> None:
    predictions = _predictions([(1, 100, 25.0), (2, 120, 75.9), (3, 80, 140.2)])
    schedule = build_schedule(predictions, _anomalies({1: True, 2: True, 3: False}), MaintenanceConfig())
    by_engine = schedule.set_index("engine_id")
    # RUL 25 ≤ 30: cần bảo trì ngay, dù cũng có bất thường.
    assert by_engine.loc[1, "status"] == NEEDS and by_engine.loc[1, "cycles_to_maintenance"] == 0
    # RUL 75,9: còn floor(45,9) = 45 cycle → cycle 165; có bất thường nên theo dõi.
    assert by_engine.loc[2, "status"] == WATCH
    assert by_engine.loc[2, "cycles_to_maintenance"] == 45 and by_engine.loc[2, "maintenance_cycle"] == 165
    assert by_engine.loc[2, "predicted_failure_cycle"] == 195
    assert by_engine.loc[3, "status"] == NORMAL
    # Sắp theo mức ưu tiên: cần bảo trì → theo dõi → bình thường.
    assert schedule["engine_id"].tolist() == [1, 2, 3]
    assert schedule["priority"].tolist() == [1, 2, 3]


def test_schedule_rejects_duplicate_or_missing_anomaly() -> None:
    with pytest.raises(ContractError):
        build_schedule(_predictions([(1, 10, 5.0), (1, 11, 4.0)]), _anomalies({1: False}), MaintenanceConfig())
    with pytest.raises(ContractError):
        build_schedule(_predictions([(1, 10, 5.0), (2, 10, 5.0)]), _anomalies({1: False}), MaintenanceConfig())


def test_short_engine_is_not_evaluable() -> None:
    config = AnomalyConfig(baseline_cycles=30, persistence=5)
    short = pd.DataFrame({"engine_id": 9, "cycle": range(1, 35), "health_index": [1.0 + 0.01 * (i % 2) for i in range(34)]})
    status = anomaly_status(short, config)
    assert not bool(status.loc[0, "anomaly_evaluable"]) and not bool(status.loc[0, "anomaly_detected"])


def test_evaluation_counts() -> None:
    predictions = _predictions([(1, 100, 20.0), (2, 100, 50.0), (3, 100, 120.0), (4, 100, 25.0)])
    schedule = build_schedule(predictions, _anomalies({1: True, 2: True, 3: False, 4: False}), MaintenanceConfig())
    labels = pd.DataFrame({"engine_id": [1, 2, 3, 4], "rul": [10, 28, 130, 45]})
    crosstab, summary = evaluate_schedule(schedule, labels, MaintenanceConfig())
    assert summary["n_actual_needs_maintenance"] == 2  # động cơ 1 và 2
    assert summary["n_correct_needs_maintenance"] == 1  # chỉ động cơ 1 được báo cần bảo trì
    assert summary["n_actual_needs_with_any_warning"] == 2  # động cơ 2 vẫn ở mức theo dõi
    assert summary["n_flagged_but_actual_above_threshold"] == 1  # động cơ 4 có RUL thật 45
    assert int(crosstab.to_numpy().sum()) == 4
