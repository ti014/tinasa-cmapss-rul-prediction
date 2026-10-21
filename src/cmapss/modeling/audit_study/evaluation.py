"""Đánh giá predictions và warning policy theo protocol."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from cmapss.modeling.audit_study.config import AuditStudyConfig, WarningPolicy
from cmapss.modeling.metrics import add_prediction_errors, summarize_prediction_errors


def build_endpoint_panel(predictions: pd.DataFrame,
                         fractions: tuple[float, ...]) -> pd.DataFrame:
    """
    Lấy endpoint tại floor(fraction × tuổi thọ), đẩy lên cycle khả dụng đầu tiên nếu
    cần; tuổi thọ bằng cycle + y_true.
    """
    if predictions.empty:
        raise ValueError("Đầu vào dự đoán endpoint rỗng")
    records = []
    for (_, engine), group in predictions.groupby(["split", "engine_id"], sort=True):
        lifetimes = group.cycle.to_numpy(dtype="float64") + group.y_true.to_numpy(dtype="float64")
        if not np.isfinite(lifetimes).all() or not np.all(lifetimes == lifetimes[0]):
            raise ValueError(f"Tuổi thọ không nhất quán với động cơ{engine}")
        lifetime = int(lifetimes[0])
        for fraction in fractions:
            requested = int(np.floor(fraction * lifetime))
            origin = max(int(group.cycle.min()), requested)
            row = group.loc[group.cycle.eq(origin)]
            if len(row) != 1:
                raise ValueError(f"Thiếu hoặc trùng động cơ endpoint{engine}/cycle{origin}")
            record = row.iloc[0].to_dict()
            record.update(endpoint_fraction=fraction, requested_cycle=requested,
                          origin_clamped_to_available_prefix=origin != requested)
            records.append(record)
    return pd.DataFrame(records)


def _prediction_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    metrics = summarize_prediction_errors(frame)
    scored = add_prediction_errors(frame)
    mean_nasa = scored.groupby("engine_id").nasa_score_component.mean().mean()
    return {**{f"{prefix}_{key}": value for key, value in metrics.items()},
            f"{prefix}_mean_engine_nasa": float(mean_nasa)}


def _model_readout(predictions: pd.DataFrame, config: AuditStudyConfig) -> dict[str, Any]:
    endpoint = build_endpoint_panel(predictions, config.endpoint_fractions)
    return {**_prediction_metrics(endpoint, "endpoint"),
            **_prediction_metrics(predictions, "trajectory")}


def select_candidate(records: pd.DataFrame) -> str:
    """Chọn endpoint MAE thấp nhất; hòa thì xét NASA score rồi ID ứng viên."""
    columns = ["endpoint_mean_engine_mae", "endpoint_mean_engine_nasa"]
    if records.empty or not np.isfinite(records[columns].to_numpy()).all():
        raise ValueError("Metric chọn mô hình inner không hợp lệ")
    return str(records.sort_values(columns + ["candidate_id"], kind="mergesort").iloc[0].candidate_id)


def _alarm_flags(group: pd.DataFrame, policy: WarningPolicy) -> np.ndarray:
    """Trả True khi persistence dự đoán cuối đều không vượt ngưỡng."""
    below = group.y_pred.le(policy.threshold)
    return below.rolling(policy.persistence, min_periods=policy.persistence).sum().ge(policy.persistence).to_numpy()


def warning_events(predictions: pd.DataFrame, policy: WarningPolicy,
                   *, lead_requirement: int) -> pd.DataFrame:
    """
    Đo thời điểm cảnh báo, tỷ lệ cycle báo sớm và chuyển trạng thái theo động cơ;
    early_alarm_burden không phải tỷ lệ báo nhầm.
    """
    if predictions.empty or not predictions.split.eq("train").all():
        raise ValueError("Đánh giá cảnh báo cần quỹ đạo train đầy đủ")
    rows = []
    for engine, unsorted in predictions.groupby("engine_id", sort=True):
        group = unsorted.sort_values("cycle", kind="mergesort")
        cycles = group.cycle.to_numpy(dtype="int64")
        actual = group.y_true.to_numpy(dtype="float64")
        if (len(np.unique(cycles)) != len(cycles) or np.any(np.diff(cycles) != 1)
                or not np.array_equal(actual, cycles[-1] - cycles)
                or not np.isfinite(group.y_pred.to_numpy()).all()):
            raise ValueError(f"Quỹ đạo cảnh báo của động cơ{engine} không đầy đủ hoặc không liên tục")
        alarm = _alarm_flags(group, policy)
        first = np.flatnonzero(alarm)
        lead = float(actual[first[0]]) if len(first) else np.nan
        eligible = actual > lead_requirement
        transitions = np.diff(np.r_[False, alarm].astype("int64"))
        rows.append({
            "engine_id": int(engine), "threshold": policy.threshold,
            "persistence": policy.persistence,
            "first_warning_cycle": int(cycles[first[0]]) if len(first) else -1,
            "lead_time_cycles": lead, "missed_warning": not bool(len(first)),
            "on_time": bool(len(first) and lead >= lead_requirement),
            "early_alarm_burden": float(alarm[eligible].mean()) if eligible.any() else np.nan,
            "n_pre_danger_rows": int(eligible.sum()), "n_alarm_rows": int(alarm.sum()),
            "n_transitions": int(np.count_nonzero(transitions)),
            "n_retractions": int(np.sum(transitions == -1)),
            "n_alarm_episodes": int(np.sum(transitions == 1)), "n_rows": int(len(group)),
            "lead_requirement": lead_requirement,
        })
    return pd.DataFrame(rows)


def _warning_readout(events: pd.DataFrame) -> dict[str, Any]:
    lead = events.lead_time_cycles.dropna()
    return {
        "on_time_rate": float(events.on_time.mean()),
        "on_time_count": int(events.on_time.sum()),
        "late_or_missed_rate": float((~events.on_time).mean()),
        "missed_warning_count": int(events.missed_warning.sum()),
        "early_alarm_burden": float(events.early_alarm_burden.mean()),
        "mean_transitions": float(events.n_transitions.mean()),
        "mean_retractions": float(events.n_retractions.mean()),
        "engines_with_retractions": int(events.n_retractions.gt(0).sum()),
        "median_lead_time": float(lead.median()) if len(lead) else None,
        "min_lead_time": float(lead.min()) if len(lead) else None,
        "max_lead_time": float(lead.max()) if len(lead) else None,
        "n_engines": int(len(events)),
    }


def choose_policy(records: pd.DataFrame, target_rate: float) -> WarningPolicy:
    """
    Chọn tỷ lệ báo sớm thấp nhất trong các chính sách đạt target_rate; nếu không có,
    chọn tỷ lệ kịp thời cao nhất rồi báo sớm thấp nhất.
    """
    valid = records.dropna(subset=["on_time_rate", "early_alarm_burden"])
    if valid.empty:
        raise ValueError("Không có chính sách cảnh báo đánh giá được")
    feasible = valid.loc[valid.on_time_rate.ge(target_rate)]
    if len(feasible):
        ordered = feasible.sort_values(["early_alarm_burden", "threshold", "persistence"])
    else:
        ordered = valid.sort_values(["on_time_rate", "early_alarm_burden", "threshold", "persistence"],
                                   ascending=[False, True, True, True])
    winner = ordered.iloc[0]
    return WarningPolicy(int(winner.threshold), int(winner.persistence))


def _policy_grid(predictions: pd.DataFrame, config: AuditStudyConfig) -> tuple[WarningPolicy, pd.DataFrame]:
    """Đánh giá mọi chính sách trên inner OOF và chọn một chính sách."""
    rows = []
    for threshold in config.thresholds:
        for persistence in config.persistences:
            policy = WarningPolicy(threshold, persistence)
            events = warning_events(predictions, policy, lead_requirement=config.lead_requirement)
            rows.append({**asdict(policy), **_warning_readout(events)})
    table = pd.DataFrame(rows)
    policy = choose_policy(table, config.target_on_time_rate)
    table["selected"] = table.threshold.eq(policy.threshold) & table.persistence.eq(policy.persistence)
    table["inner_target_met"] = table.on_time_rate.ge(config.target_on_time_rate)
    return policy, table


def _engine_readout(predictions: pd.DataFrame, events: pd.DataFrame,
                     config: AuditStudyConfig) -> pd.DataFrame:
    """Metric endpoint, quỹ đạo và cảnh báo theo động cơ."""
    endpoint = add_prediction_errors(build_endpoint_panel(predictions, config.endpoint_fractions))
    trajectory = add_prediction_errors(predictions)
    result = endpoint.groupby("engine_id").agg(
        endpoint_mean_engine_mae=("absolute_error", "mean"),
        endpoint_mean_engine_nasa=("nasa_score_component", "mean"),
    ).join(trajectory.groupby("engine_id").absolute_error.mean().rename("trajectory_mean_engine_mae"))
    result = result.join(trajectory[trajectory.y_true.le(30)].groupby("engine_id")
                         .absolute_error.mean().rename("low_rul_engine_mae"))
    warning = events.set_index("engine_id")[["on_time", "early_alarm_burden", "n_transitions", "n_retractions"]]
    warning = warning.rename(columns={"on_time": "on_time_rate", "n_transitions": "mean_transitions",
                                      "n_retractions": "mean_retractions"})
    return result.join(warning).astype("float64")


def _paired_comparison(baseline: pd.DataFrame, candidate: pd.DataFrame,
                       config: AuditStudyConfig) -> pd.DataFrame:
    """Bootstrap CI của chênh lệch ứng viên − baseline theo động cơ."""
    if not baseline.index.equals(candidate.index):
        raise ValueError("Metric ghép cặp cần cùng ID động cơ theo đúng thứ tự")
    rng = np.random.default_rng(config.seed)
    indices = rng.integers(0, len(baseline), size=(config.bootstrap_resamples, len(baseline)))
    rows = []
    for metric in baseline.columns:
        differences = (candidate[metric] - baseline[metric]).to_numpy(dtype="float64")
        valid = np.isfinite(differences)
        if not valid.any():
            rows.append({"metric": metric, "baseline": None, "candidate": None, "delta": None,
                         "ci_lower": None, "ci_upper": None, "n_engines": 0})
            continue
        with np.errstate(invalid="ignore"):
            sampled = differences[indices]
            counts = np.isfinite(sampled).sum(axis=1)
            draws = np.divide(np.nansum(sampled, axis=1), counts,
                              out=np.full(len(counts), np.nan), where=counts > 0)
        finite = draws[np.isfinite(draws)]
        rows.append({"metric": metric, "baseline": float(baseline[metric].mean()),
                     "candidate": float(candidate[metric].mean()),
                     "delta": float(differences[valid].mean()),
                     "ci_lower": float(np.quantile(finite, .025)),
                     "ci_upper": float(np.quantile(finite, .975)),
                     "n_engines": int(valid.sum()), "resamples": config.bootstrap_resamples,
                     "seed": config.seed,
                     "interval_scope": "paired engines conditional on stored outer predictions; excludes retraining variability"})
    return pd.DataFrame(rows)
