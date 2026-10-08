"""Kiểm thử khử nhiễu EWMA và phát hiện bất thường bằng fixture tổng hợp."""

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import ContractError
from cmapss.degradation import AnomalyConfig, detect_anomalies, ewma_smooth, noise_summary, sensitivity_grid


def _frame(values_by_engine: dict[int, list[float]]) -> pd.DataFrame:
    rows = [
        {"engine_id": engine, "cycle": cycle, "x": value}
        for engine, values in values_by_engine.items()
        for cycle, value in enumerate(values, start=1)
    ]
    return pd.DataFrame(rows)


def test_ewma_matches_hand_computation() -> None:
    # span=3 → alpha=0,5: 0; 0,5*4+0,5*0=2; 0,5*8+0,5*2=5.
    smoothed = ewma_smooth(_frame({1: [0.0, 4.0, 8.0]}), ["x"], span=3)
    assert smoothed["x"].tolist() == pytest.approx([0.0, 2.0, 5.0])


def test_ewma_is_causal_and_per_engine() -> None:
    base = _frame({1: [1.0, 2.0, 3.0, 4.0], 2: [100.0, 100.0]})
    changed = base.copy()
    changed.loc[(changed["engine_id"] == 1) & (changed["cycle"] == 4), "x"] = 999.0
    a = ewma_smooth(base, ["x"], span=3)
    b = ewma_smooth(changed, ["x"], span=3)
    # Đổi cycle 4 không làm đổi cycle 1-3 của cùng động cơ.
    assert a.loc[a["cycle"] < 4, "x"].tolist() == b.loc[b["cycle"] < 4, "x"].tolist()
    # Động cơ 2 không bị kéo bởi giá trị của động cơ 1.
    assert a.loc[a["engine_id"] == 2, "x"].tolist() == pytest.approx([100.0, 100.0])


def test_ewma_rejects_invalid_span_and_missing_column() -> None:
    with pytest.raises(ValueError):
        ewma_smooth(_frame({1: [1.0]}), ["x"], span=0)
    with pytest.raises(ContractError):
        ewma_smooth(_frame({1: [1.0]}), ["y"], span=3)


def test_noise_summary_reports_reduction() -> None:
    rng = np.random.default_rng(0)
    raw = _frame({1: list(rng.normal(size=200)), 2: list(rng.normal(size=200))})
    summary = noise_summary(raw, ewma_smooth(raw, ["x"], span=10), ["x"])
    assert summary.loc[0, "diff_std_smoothed"] < summary.loc[0, "diff_std_raw"]
    assert 0.0 < summary.loc[0, "reduction_pct"] < 100.0


def _health_index(n_flat: int, n_drop: int, *, spike_at: int | None = None) -> pd.DataFrame:
    values = [1.0 + 0.01 * ((-1) ** i) for i in range(n_flat)]
    values += [1.0 - 0.5 * (i + 1) for i in range(n_drop)]
    if spike_at is not None:
        values[spike_at] = -10.0
    frame = _frame({7: values})
    return frame.rename(columns={"x": "hi"})


def test_detects_first_persistent_drop() -> None:
    config = AnomalyConfig(baseline_cycles=30, z_threshold=3.0, persistence=5)
    events = detect_anomalies(_health_index(40, 20), config, column="hi")
    row = events.iloc[0]
    # Giảm bắt đầu ở cycle 41; đủ 5 cycle liên tiếp tại cycle 45.
    assert bool(row["detected"]) and row["detection_cycle"] == 45
    assert row["failure_cycle"] == 60 and row["lead_cycles"] == 15


def test_single_spike_does_not_trigger() -> None:
    config = AnomalyConfig(baseline_cycles=30, z_threshold=3.0, persistence=5)
    events = detect_anomalies(_health_index(50, 0, spike_at=40), config, column="hi")
    assert not bool(events.iloc[0]["detected"])
    assert pd.isna(events.iloc[0]["lead_cycles"])


def test_rejects_short_or_flat_baseline() -> None:
    config = AnomalyConfig(baseline_cycles=30)
    with pytest.raises(ContractError):
        detect_anomalies(_health_index(20, 0), config, column="hi")
    flat = _frame({1: [1.0] * 40}).rename(columns={"x": "hi"})
    with pytest.raises(ContractError):
        detect_anomalies(flat, config, column="hi")


def test_sensitivity_grid_tracks_persistence_and_early_alarms() -> None:
    # Giảm bắt đầu ngay cycle 31; đủ 3/5/10 cycle liên tiếp tại cycle 33/35/40.
    grid = sensitivity_grid(
        _health_index(30, 20), column="hi", z_thresholds=(2.0, 3.0), persistences=(3, 5, 10), early_window=5
    )
    assert len(grid) == 6 and grid["n_detected"].eq(1).all()
    by_p = grid.loc[grid["z_threshold"].eq(3.0)].set_index("persistence")
    assert by_p["lead_median"].to_dict() == {3: 17.0, 5: 15.0, 10: 10.0}
    # Cửa sổ sớm là cycle 31-35: cấu hình 3 và 5 cycle bị tính là báo sớm, 10 cycle thì không.
    assert by_p["n_early"].to_dict() == {3: 1, 5: 1, 10: 0}
    assert by_p.loc[10, "life_fraction_median"] == pytest.approx(40 / 50)
