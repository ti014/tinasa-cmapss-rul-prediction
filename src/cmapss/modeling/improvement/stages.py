"""Các giai đoạn nested CV và chọn cấu hình cuối trên train."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import pandas as pd

from cmapss.modeling.improvement.config import (
    BASELINE_PARAMS,
    ImprovementConfig,
    WINDOWS,
    make_hgb_candidates,
)
from cmapss.modeling.improvement.selection import (
    _evaluate_candidate,
    _evaluate_stage,
    _metrics,
    _select_candidate,
)
from cmapss.modeling.improvement.training import _fit_predict


def _run_outer_stages(
    datasets: dict[int, pd.DataFrame],
    config: ImprovementConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_frame = datasets[30]
    outer_assignments = pd.read_csv(output_dir / "outer_assignments.csv")
    cache_dir = output_dir / "fit_cache"
    outer_records: list[dict[str, Any]] = []
    outer_predictions: list[pd.DataFrame] = []
    inner_tables: list[pd.DataFrame] = []
    hgb_candidates = make_hgb_candidates(config)
    for outer_fold in range(1, config.outer_splits + 1):
        print(f"outer fold {outer_fold}/{config.outer_splits}", flush=True)
        validation_engines = set(outer_assignments.loc[outer_assignments["outer_fold"].eq(outer_fold), "engine_id"])
        selected_window_records: list[dict[str, Any]] = []
        selected_window_predictions: list[pd.DataFrame] = []
        for window in WINDOWS:
            frame = datasets[window]
            train = frame.loc[~frame["engine_id"].isin(validation_engines)].reset_index(drop=True)
            metrics, predictions = _evaluate_candidate(frame=train, params=BASELINE_PARAMS, weight_scheme="uniform", config=config, cache_dir=cache_dir)
            selected_window_records.append({"candidate_id": f"window_{window}", "window": window, **{k: v for k, v in metrics.items() if k != "warning_summary"}})
            predictions["outer_fold"] = outer_fold
            predictions["window"] = window
            selected_window_predictions.append(predictions)
        # Window 30 is the reference for the control checks.
        window_baseline = next(record for record in selected_window_records if record["window"] == 30)
        selected_window_index = _select_candidate(selected_window_records, window_baseline)
        selected_window = int(selected_window_records[selected_window_index]["window"])
        selected_window_table = pd.DataFrame(selected_window_records)
        selected_window_table["selected"] = selected_window_table["window"].eq(selected_window)
        selected_window_table["stage"] = "windows"
        selected_window_table["outer_fold"] = outer_fold
        inner_tables.append(selected_window_table)

        selected_frame = datasets[selected_window].loc[
            ~datasets[selected_window]["engine_id"].isin(validation_engines)
        ].reset_index(drop=True)
        config_candidates = [
            ("baseline" if index == 0 else f"hgb_{index:02d}", params, "uniform")
            for index, params in enumerate(hgb_candidates)
        ]
        selected_params, parameter_table, parameter_predictions = _evaluate_stage(
            selected_frame,
            config_candidates,
            config,
            cache_dir,
        )
        parameter_table["outer_fold"] = outer_fold
        parameter_table["stage"] = "parameters"
        inner_tables.append(parameter_table)
        best_params = {key: selected_params[key] for key in BASELINE_PARAMS}
        weight_candidates = [
            ("baseline", best_params, "uniform"),
            ("moderate_low_rul", best_params, "moderate_low_rul"),
            ("strong_low_rul", best_params, "strong_low_rul"),
        ]
        selected_weight, weight_table, weight_predictions = _evaluate_stage(
            selected_frame,
            weight_candidates,
            config,
            cache_dir,
        )
        weight_table["outer_fold"] = outer_fold
        weight_table["stage"] = "weights"
        inner_tables.append(weight_table)
        final_params = {key: selected_weight[key] for key in BASELINE_PARAMS}
        final_weight = str(selected_weight["weight_scheme"])
        stage_configs = [
            ("baseline", 30, BASELINE_PARAMS, "uniform"),
            ("windows", selected_window, BASELINE_PARAMS, "uniform"),
            ("parameters", selected_window, best_params, "uniform"),
            ("weights", selected_window, final_params, final_weight),
        ]
        for stage, window, params, weight in stage_configs:
            frame = datasets[window]
            outer_train = frame.loc[~frame.engine_id.isin(validation_engines)].reset_index(drop=True)
            outer_validation = frame.loc[frame.engine_id.isin(validation_engines)].reset_index(drop=True)
            prediction = _fit_predict(outer_train, outer_validation, params, weight, config.seed, cache_dir, config.fit_threads)
            prediction["outer_fold"] = outer_fold
            prediction["stage"] = stage
            prediction["selected_window"] = window
            prediction["weight_scheme"] = weight
            outer_predictions.append(prediction)
            metrics = _metrics(prediction, config.low_rul_threshold)
            outer_records.append({"outer_fold": outer_fold, "stage": stage, "selected_window": window,
                                  "weight_scheme": weight, **params,
                                  **{k: v for k, v in metrics.items() if k != "warning_summary"}})
        pd.DataFrame(outer_records).to_csv(output_dir / "outer_metrics.csv", index=False)
        pd.concat(outer_predictions, ignore_index=True).to_parquet(output_dir / "outer_predictions.parquet", index=False)
        pd.concat(inner_tables, ignore_index=True).to_csv(output_dir / "inner_candidate_metrics.csv", index=False)
        checkpoint = {
            "outer_fold": outer_fold,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_window": selected_window,
            "weight_scheme": final_weight,
        }
        (output_dir / f"checkpoint_outer_{outer_fold}.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    return (
        pd.DataFrame(outer_records),
        pd.concat(outer_predictions, ignore_index=True),
        {"inner_tables": pd.concat(inner_tables, ignore_index=True), "outer_assignments": outer_assignments},
    )


def _select_final_configuration(
    datasets: dict[int, pd.DataFrame],
    config: ImprovementConfig,
    cache_dir: Path | None = None,
) -> tuple[int, dict[str, Any], str, pd.DataFrame]:
    """Select window, HGB parameters and weights with CV on all train engines."""
    window_records: list[dict[str, Any]] = []
    for window in WINDOWS:
        metrics, _ = _evaluate_candidate(datasets[window], BASELINE_PARAMS, "uniform", config, cache_dir)
        window_records.append({"candidate_id": f"window_{window}", "window": window, **{k: v for k, v in metrics.items() if k != "warning_summary"}})
    window_baseline = next(record for record in window_records if record["window"] == 30)
    selected_window = int(window_records[_select_candidate(window_records, window_baseline)]["window"])
    selected_frame = datasets[selected_window]
    hgb_candidates = [
        ("baseline" if index == 0 else f"hgb_{index:02d}", params, "uniform")
        for index, params in enumerate(make_hgb_candidates(config))
    ]
    selected_params, parameter_table, _ = _evaluate_stage(selected_frame, hgb_candidates, config, cache_dir)
    best_params = {key: selected_params[key] for key in BASELINE_PARAMS}
    weight_candidates = [
        ("baseline", best_params, "uniform"),
        ("moderate_low_rul", best_params, "moderate_low_rul"),
        ("strong_low_rul", best_params, "strong_low_rul"),
    ]
    selected_weight, weight_table, _ = _evaluate_stage(selected_frame, weight_candidates, config, cache_dir)
    table = pd.concat(
        [
            pd.DataFrame(window_records).assign(stage="windows", selected=lambda frame: frame["window"].eq(selected_window)),
            parameter_table.assign(stage="parameters"),
            weight_table.assign(stage="weights"),
        ],
        ignore_index=True,
        sort=False,
    )
    return selected_window, {key: selected_weight[key] for key in BASELINE_PARAMS}, str(selected_weight["weight_scheme"]), table
