"""Tạo lịch bảo trì và trạng thái cảnh báo cho từng động cơ."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cmapss.data.contracts import SENSORS, ContractError
from cmapss.degradation.anomaly import AnomalyConfig, detect_anomalies
from cmapss.timeseries.health_index import HealthIndexResult

NEEDS_MAINTENANCE = "Cần bảo trì"
WATCH = "Theo dõi tăng cường"
NORMAL = "Bình thường"
STATUS_ORDER = (NEEDS_MAINTENANCE, WATCH, NORMAL)


@dataclass(frozen=True)
class MaintenanceConfig:
    """Ngưỡng RUL (cycle) để đánh dấu cần bảo trì."""

    rul_threshold: int = 30


def transform_health_index(fitted: HealthIndexResult, observations: pd.DataFrame) -> pd.DataFrame:
    """Áp dụng scaler, PCA, hướng và chuẩn hóa đã fit trên train cho dữ liệu mới.

    Không fit lại bất kỳ tham số nào trên ``observations``, nên dùng được cho động cơ
    test mà không rò rỉ thông tin.
    """
    missing = [c for c in ("engine_id", "cycle", *SENSORS) if c not in observations.columns]
    if missing:
        raise ContractError("COLUMNS", "transform_health_index", f"thiếu cột {missing}")
    ordered = observations.sort_values(["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    values = ordered.loc[:, list(SENSORS)].to_numpy(dtype="float64")
    scores = fitted.pca.transform(fitted.scaler.transform(values)).ravel() * fitted.orientation_sign
    result = ordered.loc[:, ["engine_id", "cycle"]].copy()
    result["health_index"] = (scores - fitted.normalization_mean) / fitted.normalization_scale
    return result


def anomaly_status(health_index: pd.DataFrame, config: AnomalyConfig) -> pd.DataFrame:
    """Xác định động cơ đã có dấu hiệu bất thường tính đến cycle cuối quan sát.

    Động cơ ngắn hơn ``baseline_cycles + persistence`` không thể đạt điều kiện báo, nên
    được ghi ``anomaly_evaluable = False`` thay vì coi là bình thường. Cột ``threshold``
    là mức Health Index ứng với ``z = -z_threshold`` của động cơ đó, dùng để vẽ ngưỡng
    trên biểu đồ.
    """
    minimum = config.baseline_cycles + config.persistence
    records = []
    for engine_id, engine in health_index.groupby("engine_id", sort=True):
        if len(engine) < minimum:
            records.append({"engine_id": int(engine_id), "anomaly_evaluable": False,
                            "anomaly_detected": False, "anomaly_cycle": None,
                            "threshold": None})
            continue
        event = detect_anomalies(engine, config, column="health_index").iloc[0]
        records.append({
            "engine_id": int(engine_id),
            "anomaly_evaluable": True,
            "anomaly_detected": bool(event["detected"]),
            "anomaly_cycle": None if pd.isna(event["detection_cycle"]) else int(event["detection_cycle"]),
            "threshold": float(event["baseline_mean"] - config.z_threshold * event["baseline_std"]),
        })
    return pd.DataFrame.from_records(records).astype({"anomaly_cycle": "Int64"})


def build_schedule(predictions: pd.DataFrame, anomalies: pd.DataFrame, config: MaintenanceConfig) -> pd.DataFrame:
    """Ghép dự đoán RUL và trạng thái bất thường thành lịch bảo trì.

    ``cycles_to_maintenance`` là số cycle còn lại đến khi RUL dự đoán chạm ngưỡng, làm
    tròn xuống để lịch nghiêng về phía an toàn; bằng 0 khi RUL dự đoán đã không vượt
    ngưỡng. Trạng thái xét theo thứ tự ưu tiên: cần bảo trì (RUL dự đoán <= ngưỡng),
    theo dõi tăng cường (đã có bất thường), bình thường. ``predictions`` có một dòng mỗi
    động cơ, ``cycle`` là cycle cuối quan sát. Kết quả sắp theo mức ưu tiên, động cơ cần
    bảo trì sớm nhất lên đầu.
    """
    required = {"engine_id", "cycle", "rul_pred"}
    if not required.issubset(predictions.columns):
        raise ContractError("COLUMNS", "build_schedule", f"cần các cột {sorted(required)}")
    if predictions["engine_id"].duplicated().any():
        raise ContractError("KEY", "build_schedule", "mỗi động cơ chỉ được có một dự đoán")
    merged = predictions.loc[:, ["engine_id", "cycle", "rul_pred"]].merge(
        anomalies, on="engine_id", how="left", validate="one_to_one"
    )
    if merged["anomaly_evaluable"].isna().any():
        raise ContractError("KEY", "build_schedule", "thiếu trạng thái bất thường cho một số động cơ")
    threshold = config.rul_threshold
    rul = merged["rul_pred"].to_numpy(dtype="float64")
    if not np.isfinite(rul).all():
        raise ContractError("NUMERIC", "build_schedule", "RUL dự đoán phải hữu hạn")
    schedule = pd.DataFrame({
        "engine_id": merged["engine_id"].astype("int64"),
        "current_cycle": merged["cycle"].astype("int64"),
        "rul_pred": rul,
    })
    schedule["cycles_to_maintenance"] = [max(math.floor(value - threshold), 0) for value in rul]
    schedule["maintenance_cycle"] = schedule["current_cycle"] + schedule["cycles_to_maintenance"]
    schedule["predicted_failure_cycle"] = schedule["current_cycle"] + np.floor(rul).astype("int64")
    schedule["anomaly_evaluable"] = merged["anomaly_evaluable"].astype(bool).to_numpy()
    schedule["anomaly_detected"] = merged["anomaly_detected"].astype(bool).to_numpy()
    schedule["anomaly_cycle"] = merged["anomaly_cycle"].astype("Int64").array
    schedule["status"] = np.select(
        [rul <= threshold, schedule["anomaly_detected"].to_numpy()],
        [NEEDS_MAINTENANCE, WATCH],
        default=NORMAL,
    )
    rank = {status: index for index, status in enumerate(STATUS_ORDER)}
    schedule = schedule.assign(_rank=schedule["status"].map(rank)).sort_values(
        ["_rank", "cycles_to_maintenance", "rul_pred", "engine_id"], kind="mergesort"
    ).drop(columns="_rank").reset_index(drop=True)
    schedule.insert(0, "priority", np.arange(1, len(schedule) + 1))
    return schedule


def evaluate_schedule(schedule: pd.DataFrame, labels: pd.DataFrame, config: MaintenanceConfig) -> tuple[pd.DataFrame, dict]:
    """Đối chiếu lịch bảo trì với RUL thật.

    Trả về bảng chéo trạng thái theo nhóm RUL thật và dict chỉ số tóm tắt (số báo đúng,
    báo nhầm, MAE của số cycle đến lúc cần bảo trì).
    """
    threshold = config.rul_threshold
    joined = schedule.merge(labels.loc[:, ["engine_id", "rul"]], on="engine_id", how="left", validate="one_to_one")
    if joined["rul"].isna().any():
        raise ContractError("KEY", "evaluate_schedule", "thiếu nhãn RUL cho một số động cơ")
    actual = joined["rul"].astype("float64")
    bins = pd.cut(actual, [-np.inf, threshold, 2 * threshold, np.inf],
                  labels=[f"≤ {threshold}", f"{threshold + 1}–{2 * threshold}", f"> {2 * threshold}"])
    crosstab = pd.crosstab(pd.Categorical(joined["status"], categories=STATUS_ORDER), bins, dropna=False)
    crosstab.index.name, crosstab.columns.name = "status", "actual_rul"
    needs = actual <= threshold
    flagged = joined["status"].eq(NEEDS_MAINTENANCE)
    any_warning = joined["status"].isin([NEEDS_MAINTENANCE, WATCH])
    actual_cycles = np.maximum(np.floor(actual - threshold), 0)
    summary = {
        "n_engines": int(len(joined)),
        "n_actual_needs_maintenance": int(needs.sum()),
        "n_flagged_needs_maintenance": int(flagged.sum()),
        "n_correct_needs_maintenance": int((needs & flagged).sum()),
        "n_actual_needs_with_any_warning": int((needs & any_warning).sum()),
        "n_flagged_but_actual_above_threshold": int((flagged & ~needs).sum()),
        "max_actual_rul_among_flagged": float(actual[flagged].max()) if flagged.any() else None,
        "cycles_to_maintenance_mae": float(np.mean(np.abs(joined["cycles_to_maintenance"] - actual_cycles))),
        "median_actual_rul_by_status": {
            status: float(actual[joined["status"].eq(status)].median()) if joined["status"].eq(status).any() else None
            for status in STATUS_ORDER
        },
    }
    return crosstab, summary
