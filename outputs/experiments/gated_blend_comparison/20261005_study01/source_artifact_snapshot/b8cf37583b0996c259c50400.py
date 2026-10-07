"""Train-only early-warning summaries from out-of-fold RUL predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


WARNING_EVENT_COLUMNS = (
    "split",
    "engine_id",
    "model_name",
    "rul_threshold",
    "failure_cycle",
    "actual_threshold_cycle",
    "first_warning_cycle",
    "first_warning_pred",
    "actual_rul_at_first_warning",
    "lead_time_to_failure_cycles",
    "threshold_delay_cycles",
    "warning_timing",
    "warning_before_actual_threshold",
    "missed_warning",
    "false_alarm",
    "n_warning_rows",
    "n_rows",
)


def build_warning_events(
    oof_predictions: pd.DataFrame,
    *,
    model_name: str,
    rul_threshold: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate a predeclared low-RUL warning rule by train engine.

    The first predicted value at or below ``rul_threshold`` is the warning
    event. Lead time is the observed RUL at that event, which is a proxy for
    cycles until the run-to-failure endpoint in FD001. ``false_alarm`` is only
    true when no observed low-RUL event exists; this denominator is zero for
    complete run-to-failure train trajectories and is reported explicitly.
    """
    required = {"split", "engine_id", "cycle", "model_name", "y_true", "y_pred"}
    if not required.issubset(oof_predictions.columns):
        missing = sorted(required.difference(oof_predictions.columns))
        raise ValueError(f"[WARNING_INPUT] thiếu cột: {', '.join(missing)}")
    if isinstance(rul_threshold, bool) or not isinstance(rul_threshold, (int, np.integer)) or int(rul_threshold) < 0:
        raise ValueError("[WARNING_CONFIG] rul_threshold phải là số nguyên không âm")
    rul_threshold = int(rul_threshold)
    frame = oof_predictions.loc[oof_predictions["model_name"].eq(model_name)].copy()
    if frame.empty or not frame["split"].eq("train").all():
        raise ValueError("[WARNING_INPUT] cần OOF predictions của một model trên split train")
    if frame["engine_id"].isna().any():
        raise ValueError("[WARNING_INPUT] thiếu engine_id; không được bỏ qua engine khi tổng hợp")
    frame = frame.sort_values(["engine_id", "cycle"], kind="mergesort")
    records: list[dict[str, Any]] = []
    for (split, engine_id), group in frame.groupby(["split", "engine_id"], sort=True):
        if group[["cycle", "y_true", "y_pred"]].isna().any().any():
            raise ValueError(f"[WARNING_TRAJECTORY] engine {engine_id} có giá trị thiếu")
        actual = group["y_true"].to_numpy(dtype="float64")
        predicted = group["y_pred"].to_numpy(dtype="float64")
        cycle_values = group["cycle"].to_numpy(dtype="float64")
        if not np.isfinite(cycle_values).all() or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError(f"[WARNING_TRAJECTORY] engine {engine_id} có giá trị không hữu hạn")
        if (cycle_values < 1).any() or not np.equal(cycle_values, np.floor(cycle_values)).all():
            raise ValueError(f"[WARNING_TRAJECTORY] engine {engine_id} cần cycle nguyên dương")
        cycles = cycle_values.astype("int64")
        if len(np.unique(cycles)) != len(cycles) or np.any(np.diff(cycles) != 1):
            raise ValueError(f"[WARNING_TRAJECTORY] engine {engine_id} có cycle trùng hoặc không liên tục")
        if actual[-1] != 0.0 or (actual < 0).any() or not np.array_equal(
            actual, cycles[-1] - cycles
        ):
            raise ValueError(f"[WARNING_TRAJECTORY] engine {engine_id} không khớp run-to-failure RUL")
        low_rul = actual <= rul_threshold
        warning = predicted <= rul_threshold
        actual_threshold_index = int(np.flatnonzero(low_rul)[0]) if low_rul.any() else None
        warning_index = int(np.flatnonzero(warning)[0]) if warning.any() else None
        failure_cycle = int(cycles[-1])
        first_warning_cycle = int(cycles[warning_index]) if warning_index is not None else -1
        actual_rul = float(actual[warning_index]) if warning_index is not None else np.nan
        actual_threshold_cycle = int(cycles[actual_threshold_index]) if actual_threshold_index is not None else -1
        if warning_index is None:
            warning_timing = "no_warning"
            threshold_delay = np.nan
        elif actual_threshold_index is None:
            warning_timing = "false_alarm"
            threshold_delay = np.nan
        elif first_warning_cycle < actual_threshold_cycle:
            warning_timing = "before_threshold"
            threshold_delay = first_warning_cycle - actual_threshold_cycle
        elif first_warning_cycle == failure_cycle:
            warning_timing = "at_failure"
            threshold_delay = first_warning_cycle - actual_threshold_cycle
        elif first_warning_cycle == actual_threshold_cycle:
            warning_timing = "at_threshold"
            threshold_delay = 0
        else:
            warning_timing = "after_threshold"
            threshold_delay = first_warning_cycle - actual_threshold_cycle
        records.append(
            {
                "split": str(split),
                "engine_id": int(engine_id),
                "model_name": model_name,
                "rul_threshold": int(rul_threshold),
                "failure_cycle": failure_cycle,
                "actual_threshold_cycle": actual_threshold_cycle,
                "first_warning_cycle": first_warning_cycle,
                "first_warning_pred": float(predicted[warning_index]) if warning_index is not None else np.nan,
                "actual_rul_at_first_warning": actual_rul,
                "lead_time_to_failure_cycles": actual_rul,
                "threshold_delay_cycles": threshold_delay,
                "warning_timing": warning_timing,
                "warning_before_actual_threshold": bool(
                    warning_index is not None
                    and actual_threshold_index is not None
                    and warning_index < actual_threshold_index
                ),
                "missed_warning": bool(warning_index is None),
                "false_alarm": bool(warning_index is not None and actual_threshold_index is None),
                "n_warning_rows": int(warning.sum()),
                "n_rows": int(len(group)),
            }
        )
    events = pd.DataFrame(records, columns=WARNING_EVENT_COLUMNS)
    warned = events.loc[~events["missed_warning"]]
    lead_times = warned["lead_time_to_failure_cycles"].dropna().to_numpy(dtype="float64")
    if len(lead_times):
        quantiles = np.quantile(lead_times, [0.25, 0.5, 0.75])
        lead_time_distribution = {
            "n": int(len(lead_times)), "min": float(lead_times.min()),
            "q25": float(quantiles[0]), "median": float(quantiles[1]),
            "q75": float(quantiles[2]), "max": float(lead_times.max()),
        }
    else:
        lead_time_distribution = {"n": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    timing_counts = {
        key: int(events["warning_timing"].eq(key).sum())
        for key in ("before_threshold", "at_threshold", "after_threshold", "at_failure", "no_warning")
    }
    n_negative = int((events["actual_threshold_cycle"] < 0).sum())
    summary: dict[str, Any] = {
        "schema_version": 2,
        "model_name": model_name,
        "rul_threshold": int(rul_threshold),
        "split": "train_oof",
        "n_engines": int(len(events)),
        "n_warning": int((~events["missed_warning"]).sum()),
        "n_missed_warning": int(events["missed_warning"].sum()),
        "n_warning_before_actual_threshold": int(events["warning_before_actual_threshold"].sum()),
        "n_false_alarm": int(events["false_alarm"].sum()),
        "n_negative_engines": n_negative,
        "false_alarm_rate": float(events["false_alarm"].sum() / n_negative) if n_negative else None,
        "timing_counts": timing_counts,
        "lead_time_distribution": lead_time_distribution,
        "warning_rate": float((~events["missed_warning"]).mean()),
        "mean_lead_time_cycles": float(warned["lead_time_to_failure_cycles"].mean()) if not warned.empty else None,
        "false_alarm_definition": "predicted low-RUL warning without an observed low-RUL event; complete train runs make denominator zero",
        "timing_definition": "relative to first actual RUL <= threshold; not a minimum maintenance lead-time requirement",
        "missed_warning_definition": "no predicted RUL <= threshold anywhere in the observed trajectory, including the failure cycle",
        "evaluation_limit": "OOF predictions exclude each engine from fitting, but the same OOF was used for model selection; descriptive internal evaluation only",
    }
    return events, summary
