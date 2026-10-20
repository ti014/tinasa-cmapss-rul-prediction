"""Đánh giá predictions và warning policy theo protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cmapss.modeling import audit_study as audit
from cmapss.modeling.comparison_runner.artifact_io import _read_json, _sha, _write_json
from cmapss.modeling.comparison_runner.contracts import ComparisonBackend, ComparisonConfig
from cmapss.modeling.metrics import (
    add_prediction_errors,
    bootstrap_metric_intervals,
    summarize_prediction_errors,
)


def evaluate_final_test(
    predictions: pd.DataFrame, labels_path: Path, config: ComparisonConfig, output: Path, backend: ComparisonBackend,
) -> dict[str, Any]:
    """Score final predictions on test labels; requires all models and receipts on disk."""
    required = [output / "selection.json", output / "final_predictions.parquet",
                output / "final_predictions_receipt.json"]
    required.extend(output / "models" / arm / name
                    for arm in backend.arms for name in ("model.json", "model.joblib"))
    if not all(path.is_file() for path in required):
        raise ValueError("all choices/models/predictions must precede test label access")
    receipt = _read_json(output / "final_predictions_receipt.json")
    selection = _read_json(output / "selection.json")
    if (selection.get("test_labels_read_for_selection") is not False
            or selection.get("source_identity") != receipt.get("source_identity")):
        raise ValueError("final selection must be frozen before test label access")
    if receipt.get("test_labels_read") is not False or receipt.get("saved_model_replay_exact") is not True:
        raise ValueError("final prediction receipt does not prove pre-label replay")
    if receipt.get("predictions_sha256") != _sha(output / "final_predictions.parquet"):
        raise ValueError("final prediction receipt checksum mismatch")
    try:
        pd.testing.assert_frame_equal(predictions, pd.read_parquet(output / "final_predictions.parquet"),
                                      check_exact=True)
    except AssertionError as error:
        raise ValueError("test reporting must use the exact persisted predictions") from error
    for arm in backend.arms:
        model_path = output / "models" / arm / "model.joblib"
        if receipt.get("models_sha256", {}).get(arm) != _sha(model_path):
            raise ValueError(f"final model checksum mismatch before test read: {arm}")
    labels = backend.read_test_labels(labels_path).rename(columns={"rul": "y_true"})
    evaluations, summaries = [], []
    for procedure, frame in predictions.groupby("procedure", sort=True):
        joined = frame.merge(labels, on=["split", "engine_id"], how="outer",
                             validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError("test prediction and label engines differ")
        evaluation = add_prediction_errors(joined.drop(columns="_merge"))
        evaluations.append(evaluation)
        metrics = {"procedure": procedure, **summarize_prediction_errors(evaluation),
                   "bootstrap_intervals": bootstrap_metric_intervals(
                       evaluation, n_resamples=config.bootstrap_resamples, seed=config.seed)}
        summaries.append(metrics)
        folder = output / "test" / procedure
        folder.mkdir(parents=True, exist_ok=True)
        audit._write_parquet(frame.rename(columns={"y_pred": "rul_pred"}), folder / "final_predictions.parquet")
        audit._write_parquet(evaluation, folder / "final_evaluation.parquet")
        _write_json(folder / "final_metrics.json", metrics)
    audit._write_parquet(pd.concat(evaluations, ignore_index=True), output / "final_evaluation.parquet")
    pd.DataFrame([{key: value for key, value in row.items() if key != "bootstrap_intervals"}
                  for row in summaries]).to_csv(output / "test_metrics.csv", index=False)
    result = {"test_labels_sha256": _sha(labels_path), "procedures": summaries,
              "selection_used_test_labels": False, "outer_gate_changed_by_test": False,
              "reporting_only": True, "historically_inspected_test": True,
              "interval_scope": "engine bootstrap conditional on fitted model; excludes retraining variability"}
    _write_json(output / "test_evaluation.json", result)
    return result
