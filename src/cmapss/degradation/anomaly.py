"""Phát hiện dấu hiệu bất thường khi Health Index rời khỏi mức khỏe ban đầu."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cmapss.data.contracts import ContractError


@dataclass(frozen=True)
class AnomalyConfig:
    """Tham số của quy tắc phát hiện bất thường.

    `persistence` là số cycle liên tiếp phải vượt ngưỡng trước khi báo, để một
    dao động đơn lẻ không gây báo động.
    """

    baseline_cycles: int = 30
    z_threshold: float = 3.0
    persistence: int = 5


def detect_anomalies(health_index: pd.DataFrame, config: AnomalyConfig, *, column: str) -> pd.DataFrame:
    """Tìm cycle đầu tiên Health Index giảm bất thường so với đầu đời.

    Với mỗi động cơ, mean và std của ``column`` trên ``baseline_cycles`` cycle
    đầu là mức khỏe tham chiếu. ``z_t = (h_t - mean) / std``. Health Index giảm
    khi động cơ suy giảm, nên cycle t vượt ngưỡng khi ``z_t <= -z_threshold``.
    Báo bất thường khi điều kiện giữ đủ ``persistence`` cycle liên tiếp.

    Trả về mỗi động cơ một dòng, gồm ``detection_cycle`` (NA nếu không phát hiện)
    và ``lead_cycles = failure_cycle - detection_cycle``.
    """
    required = {"engine_id", "cycle", column}
    if not required.issubset(health_index.columns):
        raise ContractError("COLUMNS", "detect_anomalies", f"cần các cột {sorted(required)}")
    records = []
    ordered = health_index.sort_values(["engine_id", "cycle"], kind="mergesort")
    for engine_id, engine in ordered.groupby("engine_id", sort=True):
        values = engine[column].to_numpy(dtype="float64")
        cycles = engine["cycle"].to_numpy(dtype="int64")
        if len(values) <= config.baseline_cycles:
            raise ContractError(
                "ANOMALY_BASELINE", f"engine {engine_id}", "chuỗi phải dài hơn đoạn tham chiếu"
            )
        baseline = values[: config.baseline_cycles]
        mean = float(baseline.mean())
        std = float(baseline.std(ddof=1))
        if not np.isfinite(std) or std == 0.0:
            raise ContractError("ANOMALY_BASELINE", f"engine {engine_id}", "std tham chiếu phải dương")
        below = (values - mean) / std <= -config.z_threshold
        below[: config.baseline_cycles] = False
        run = 0
        detection = None
        for position, flag in enumerate(below):
            run = run + 1 if flag else 0
            if run >= config.persistence:
                detection = int(cycles[position])
                break
        failure = int(cycles[-1])
        records.append(
            {
                "engine_id": int(engine_id),
                "failure_cycle": failure,
                "baseline_mean": mean,
                "baseline_std": std,
                "detection_cycle": detection,
                "lead_cycles": None if detection is None else failure - detection,
                "detected": detection is not None,
            }
        )
    result = pd.DataFrame.from_records(records)
    return result.astype({"detection_cycle": "Int64", "lead_cycles": "Int64"})


def sensitivity_grid(
    health_index: pd.DataFrame,
    *,
    column: str,
    z_thresholds: tuple[float, ...] = (2.0, 3.0, 4.0),
    persistences: tuple[int, ...] = (3, 5, 10),
    baseline_cycles: int = 30,
    early_window: int = 10,
) -> pd.DataFrame:
    """Chạy lại ``detect_anomalies`` trên lưới ngưỡng z và số cycle liên tiếp.

    Không có nhãn bất thường nên không đo trực tiếp được tỷ lệ dương tính giả.
    ``n_early`` đếm động cơ bị báo trong ``early_window`` cycle ngay sau đoạn tham
    chiếu; báo sớm như vậy thường do dao động, nên dùng làm chỉ báo gián tiếp cho
    độ nhạy quá mức. ``life_fraction_median`` là trung vị cycle phát hiện / cycle hỏng.
    """
    rows = []
    for z_threshold in z_thresholds:
        for persistence in persistences:
            config = AnomalyConfig(baseline_cycles=baseline_cycles, z_threshold=z_threshold, persistence=persistence)
            events = detect_anomalies(health_index, config, column=column)
            detected = events.loc[events["detected"]]
            detection = detected["detection_cycle"].astype("float64")
            leads = detected["lead_cycles"].astype("float64")
            rows.append(
                {
                    "z_threshold": float(z_threshold),
                    "persistence": int(persistence),
                    "n_engines": int(len(events)),
                    "n_detected": int(len(detected)),
                    "lead_min": float(leads.min()) if len(detected) else float("nan"),
                    "lead_median": float(leads.median()) if len(detected) else float("nan"),
                    "life_fraction_median": float((detection / detected["failure_cycle"]).median())
                    if len(detected)
                    else float("nan"),
                    "n_early": int((detection <= baseline_cycles + early_window).sum()),
                }
            )
    return pd.DataFrame.from_records(rows)
