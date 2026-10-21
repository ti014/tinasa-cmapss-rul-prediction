"""Chia fold, tính metrics và chọn candidate theo tiêu chí cố định."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import time

from sklearn.model_selection import GroupKFold
import numpy as np
import pandas as pd

from cmapss.modeling.early_warning import build_warning_events
from cmapss.modeling.improvement.config import ImprovementConfig
from cmapss.modeling.improvement.training import _fit_predict
from cmapss.modeling.metrics import add_prediction_errors


def _metrics(predictions: pd.DataFrame, threshold: int) -> dict[str, Any]:
    """Metric sai số và cảnh báo dùng để chọn ứng viên."""
    scored = add_prediction_errors(predictions)
    engine_mae = scored.groupby("engine_id", sort=True)["absolute_error"].mean()
    low = scored.loc[scored["y_true"].le(threshold)]
    low_engine_mae = low.groupby("engine_id", sort=True)["absolute_error"].mean()
    event_input = scored.loc[:, ["split", "engine_id", "cycle", "model_name", "y_true", "y_pred"]]
    events, summary = build_warning_events(
        event_input,
        model_name="candidate",
        rul_threshold=threshold,
    )
    late = events["warning_timing"].isin(["after_threshold", "at_failure", "no_warning"])
    burden_values: list[float] = []
    for _, group in scored.groupby("engine_id", sort=True):
        eligible = group.loc[group["y_true"].gt(threshold)]
        if not eligible.empty:
            burden_values.append(float(eligible["y_pred"].le(threshold).mean()))
    return {
        "n_rows": int(len(scored)),
        "n_engines": int(scored["engine_id"].nunique()),
        "mean_engine_mae": float(engine_mae.mean()),
        "rmse": float(np.sqrt(scored["squared_error"].mean())),
        "nasa_score": float(scored["nasa_score_component"].sum()),
        "low_rul_engine_mae": float(low_engine_mae.mean()) if not low_engine_mae.empty else None,
        "n_low_rul_engines": int(low_engine_mae.size),
        "late_or_missed_count": int(late.sum()),
        "late_or_missed_rate": float(late.mean()),
        "early_alarm_burden": float(np.mean(burden_values)) if burden_values else None,
        "n_burden_engines": int(len(burden_values)),
        "warning_summary": summary,
    }


def _aggregate_prediction_metrics(predictions: list[pd.DataFrame], threshold: int) -> dict[str, Any]:
    if not predictions:
        raise ValueError("Không có dự đoán để tổng hợp")
    return _metrics(pd.concat(predictions, ignore_index=True), threshold)


def _inner_splits(frame: pd.DataFrame, config: ImprovementConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame["engine_id"].to_numpy()
    splitter = GroupKFold(n_splits=config.inner_splits)
    return list(splitter.split(frame, frame["rul"], groups))


def _evaluate_candidate(
    frame: pd.DataFrame,
    params: dict[str, Any],
    weight_scheme: str,
    config: ImprovementConfig,
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    started = time.perf_counter()
    fold_metrics: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(_inner_splits(frame, config), start=1):
        prediction = _fit_predict(
            frame.iloc[train_index],
            frame.iloc[validation_index],
            params,
            weight_scheme,
            config.seed,
            cache_dir,
            config.fit_threads,
        )
        prediction["fold"] = fold
        predictions.append(prediction)
        metrics = _metrics(prediction, config.low_rul_threshold)
        fold_metrics.append({"fold": fold, **{key: value for key, value in metrics.items() if key != "warning_summary"}})
    aggregate = _aggregate_prediction_metrics(predictions, config.low_rul_threshold)
    aggregate["fit_seconds"] = float(time.perf_counter() - started)
    return aggregate, pd.concat(predictions, ignore_index=True)


def _select_candidate(records: list[dict[str, Any]], baseline: dict[str, Any]) -> int:
    """Chọn MAE vùng RUL thấp nhỏ nhất trong các ứng viên không kém baseline ở chỉ số kiểm soát."""
    eligible = []
    for index, record in enumerate(records):
        controls = (
            record["mean_engine_mae"] <= baseline["mean_engine_mae"]
            and record["late_or_missed_rate"] <= baseline["late_or_missed_rate"]
            and (record["early_alarm_burden"] or 0.0) <= (baseline["early_alarm_burden"] or 0.0)
        )
        if controls:
            eligible.append(index)
    pool = eligible or list(range(len(records)))
    return min(
        pool,
        key=lambda index: (
            records[index]["low_rul_engine_mae"] if records[index]["low_rul_engine_mae"] is not None else float("inf"),
            records[index]["late_or_missed_rate"],
            records[index]["mean_engine_mae"],
            records[index]["candidate_id"],
        ),
    )


def _evaluate_stage(
    frame: pd.DataFrame,
    candidates: list[tuple[str, dict[str, Any], str]],
    config: ImprovementConfig,
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for candidate_id, params, weight_scheme in candidates:
        print(f"  candidate {candidate_id}", flush=True)
        metrics, predictions = _evaluate_candidate(frame, params, weight_scheme, config, cache_dir)
        record = {"candidate_id": candidate_id, "weight_scheme": weight_scheme, **params, **metrics}
        records.append({key: value for key, value in record.items() if key != "warning_summary"})
        predictions["candidate_id"] = candidate_id
        prediction_frames.append(predictions)
    baseline_index = next((index for index, record in enumerate(records) if record["candidate_id"] == "baseline"), 0)
    selected_index = _select_candidate(records, records[baseline_index])
    selected = records[selected_index].copy()
    selected["selected"] = True
    result_table = pd.DataFrame(records)
    result_table["selected"] = False
    result_table.loc[selected_index, "selected"] = True
    result_table["selection_pool"] = "eligible_controls"
    prediction_table = pd.concat(prediction_frames, ignore_index=True)
    return selected, result_table, prediction_table


def _outer_assignments(frame: pd.DataFrame, config: ImprovementConfig) -> pd.DataFrame:
    splitter = GroupKFold(n_splits=config.outer_splits)
    rows: list[dict[str, int]] = []
    for fold, (_, validation_index) in enumerate(
        splitter.split(frame, frame["rul"], frame["engine_id"]), start=1
    ):
        for engine_id in sorted(frame.iloc[validation_index]["engine_id"].unique()):
            rows.append({"engine_id": int(engine_id), "outer_fold": fold})
    return pd.DataFrame(rows).sort_values("engine_id", kind="mergesort").reset_index(drop=True)


def _candidate_groups(
    window: int,
    params: dict[str, Any],
    weight_scheme: str,
) -> tuple[str, dict[str, Any], str]:
    return (f"w{window}_hgb_{weight_scheme}", params, weight_scheme)
