"""Giao thức đánh giá và join test labels sau khi final predictions đã được ghi."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmapss.data.contracts import TEST_LABEL_COLUMNS, ContractError
from cmapss.data.readers import read_test_labels
from cmapss.data.validation import validate_columns, validate_test_labels
from cmapss.modeling.contracts import (
    EVALUATION_PROTOCOL_COLUMNS,
    FINAL_EVALUATION_COLUMNS,
    FINAL_PREDICTION_COLUMNS,
)
from cmapss.modeling.metrics import add_prediction_errors


def build_evaluation_protocol() -> pd.DataFrame:
    """Bảng mô tả các giao thức đánh giá dùng trong repo."""
    rows = [
        {
            "protocol_id": "rul_cv",
            "target": "RUL (cycles)",
            "unit_of_analysis": "engine; windows aggregated within engine",
            "group_key": "engine_id",
            "split_and_horizon": "5-fold GroupKFold on train windows; no temporal horizon",
            "metrics": "mean_engine_mae (primary); RMSE; NASA score",
            "selection_or_reporting_role": "OOF model selection and train-only comparison",
            "uncertainty_or_repetition": "fold mean and standard deviation",
            "artifact": "tables/fold_metrics.csv; tables/model_metrics.csv",
        },
        {
            "protocol_id": "rul_final",
            "target": "RUL (cycles)",
            "unit_of_analysis": "test engine; one prediction at final observed cycle",
            "group_key": "engine_id",
            "split_and_horizon": "refit on all train windows; 100 FD001 test engines",
            "metrics": "mean_engine_mae; RMSE; NASA score",
            "selection_or_reporting_role": "final held-out evaluation after model selection",
            "uncertainty_or_repetition": "10,000 engine-cluster bootstrap resamples",
            "artifact": "predictions/final_evaluation.parquet; tables/final_metrics.json",
        },
        {
            "protocol_id": "health_index_forecast",
            "target": "Health Index (standardized PC1)",
            "unit_of_analysis": "engine trajectory; cycles in holdout suffix",
            "group_key": "engine_id",
            "split_and_horizon": "10 selected train engines; 70% prefix fit and 30% suffix holdout",
            "metrics": "MAE; RMSE; ARIMA coverage 95%; baseline point metrics",
            "selection_or_reporting_role": "secondary temporal analysis, not RUL evaluation",
            "uncertainty_or_repetition": "ARIMA prediction interval; baselines have no interval",
            "artifact": "tables/forecast_metrics.csv; tables/baseline_metrics.csv",
        },
        {
            "protocol_id": "rul_warning_oof",
            "target": "low RUL event (RUL <= 30 cycles)",
            "unit_of_analysis": "engine trajectory; first warning event",
            "group_key": "engine_id",
            "split_and_horizon": "selected-model OOF predictions on train windows",
            "metrics": "ever-warning rate; no-warning count; threshold timing; lead-time distribution",
            "selection_or_reporting_role": "descriptive early-warning evidence only",
            "uncertainty_or_repetition": "fixed threshold declared before scoring; no test-label tuning",
            "artifact": "tables/warning_events.csv; tables/warning_summary.json",
        },
        {
            "protocol_id": "etl_benchmark",
            "target": "build_engine_summary transform",
            "unit_of_analysis": "benchmark run",
            "group_key": "none",
            "split_and_horizon": "synthetic in-memory 1x, 10x and 100x replicas",
            "metrics": "duration_seconds; input/output row counts",
            "selection_or_reporting_role": "descriptive Pandas benchmark only",
            "uncertainty_or_repetition": "two cold/warm repetitions per scale",
            "artifact": "outputs/etl/benchmark.json",
        },
    ]
    return pd.DataFrame(rows, columns=EVALUATION_PROTOCOL_COLUMNS)


def evaluate_final_predictions(
    final_predictions: pd.DataFrame,
    test_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join 100 final predictions với NASA test labels và tính lỗi từng engine."""
    validate_columns(final_predictions, FINAL_PREDICTION_COLUMNS, "final predictions")
    validate_columns(test_labels, TEST_LABEL_COLUMNS, "test labels")
    validate_test_labels(test_labels, source="modeling final evaluation labels")
    joined = final_predictions.merge(
        test_labels,
        on=["split", "engine_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        key = joined.loc[joined["_merge"].ne("both"), ["split", "engine_id"]].iloc[0].to_dict()
        raise ContractError("FINAL_LABEL_JOIN", "final evaluation", f"thiếu test label cho {key}")
    expected_engines = int(test_labels["engine_id"].nunique())
    if len(joined) != expected_engines:
        raise ContractError(
            "FINAL_PREDICTION_COUNT",
            "final predictions",
            f"cần đúng {expected_engines} test engines theo nhãn đầu vào",
        )
    evaluation = joined.rename(columns={"rul": "y_true", "rul_pred": "y_pred"}).drop(columns=["_merge"])
    evaluation = add_prediction_errors(evaluation)
    return evaluation.loc[:, list(FINAL_EVALUATION_COLUMNS)].reset_index(drop=True)


def evaluate_predictions_from_path(
    final_predictions_path: str | Path,
    test_labels_path: str | Path,
) -> pd.DataFrame:
    """Đọc final predictions đã ghi rồi mới đọc NASA test labels."""
    final_predictions = pd.read_parquet(final_predictions_path)
    labels = read_test_labels(test_labels_path)
    return evaluate_final_predictions(final_predictions, labels)
