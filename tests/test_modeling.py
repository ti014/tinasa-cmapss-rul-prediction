"""Positive/negative proof cho modeling pipeline RUL."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import ContractError
from cmapss.features.contracts import ROLLING_FEATURE_COLUMNS
from cmapss.modeling.contracts import (
    FINAL_PREDICTION_COLUMNS,
    FOLD_ASSIGNMENT_COLUMNS,
    MODEL_METRIC_COLUMNS,
    OOF_PREDICTION_COLUMNS,
)
from cmapss.modeling.metrics import bootstrap_metric_intervals, nasa_score_components
from cmapss.modeling.early_warning import build_warning_events
from cmapss.modeling.pipeline import (
    ModelingConfig,
    build_group_folds,
    engine_equal_sample_weights,
    evaluate_final_predictions,
    prepare_modeling_dataset,
    run_modeling_pipeline,
    write_final_evaluation_artifacts,
    write_modeling_artifacts,
    build_evaluation_protocol,
)


def _feature_rows(split: str, engine_lengths: dict[int, int]) -> pd.DataFrame:
    rows = []
    for engine_id, length in engine_lengths.items():
        for cycle in range(30, 30 + length):
            row = {
                "split": split,
                "engine_id": engine_id,
                "cycle": cycle,
                **{column: float(engine_id * 10 + cycle) for column in ROLLING_FEATURE_COLUMNS[3:]},
            }
            row["sensor_2__last"] = float(cycle)
            rows.append(row)
    return pd.DataFrame(rows, columns=ROLLING_FEATURE_COLUMNS).astype(
        {
            "split": "string",
            "engine_id": "int64",
            "cycle": "int64",
            **dict.fromkeys(ROLLING_FEATURE_COLUMNS[3:], "float64"),
        }
    )


def _targets_from_features(features: pd.DataFrame) -> pd.DataFrame:
    train = features.loc[features["split"].eq("train"), ["split", "engine_id", "cycle"]].copy()
    train["rul"] = (train.groupby("engine_id")["cycle"].transform("max") - train["cycle"]).astype("int64")
    return train.astype({"split": "string", "engine_id": "int64", "cycle": "int64", "rul": "int64"})


def test_nasa_score_uses_fd001_piecewise_penalty() -> None:
    y_true = np.array([100.0, 100.0, 100.0])
    y_pred = np.array([90.0, 100.0, 110.0])

    score = nasa_score_components(y_true, y_pred)

    assert score[0] == pytest.approx(np.exp(10.0 / 13.0) - 1.0)
    assert score[1] == 0.0
    assert score[2] == pytest.approx(np.exp(10.0 / 10.0) - 1.0)


def test_bootstrap_metric_intervals_preserve_duplicate_engine_draws() -> None:
    evaluation = pd.DataFrame(
        {
            "engine_id": [1, 1, 2, 3],
            "y_true": [0.0, 0.0, 0.0, 0.0],
            "y_pred": [0.0, 0.0, 10.0, 20.0],
        }
    )

    intervals = bootstrap_metric_intervals(evaluation, n_resamples=1, seed=60)

    # The draw is [engine 1, engine 1, engine 2]. Collapsing duplicate engine 1
    # would yield 5, while the correct engine-level bootstrap MAE is 10 / 3.
    assert intervals["mean_engine_mae"] == {
        "ci_lower": pytest.approx(10.0 / 3.0),
        "ci_upper": pytest.approx(10.0 / 3.0),
    }
    assert intervals["rmse"]["ci_lower"] == pytest.approx(np.sqrt(20.0))


def test_prepare_modeling_dataset_joins_only_train_features() -> None:
    features = pd.concat(
        [
            _feature_rows("train", {1: 3, 2: 2}),
            _feature_rows("test", {1: 2}),
        ],
        ignore_index=True,
    ).sort_values(["split", "engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    targets = _targets_from_features(features)

    dataset = prepare_modeling_dataset(features, targets)

    assert len(dataset) == 5
    assert dataset["split"].eq("train").all()
    assert "rul" in dataset.columns


def test_prepare_modeling_dataset_rejects_missing_target() -> None:
    features = _feature_rows("train", {1: 3})
    targets = _targets_from_features(features).iloc[:-1]

    with pytest.raises(ContractError, match="MODELING_TARGET_KEYS"):
        prepare_modeling_dataset(features, targets)


def test_engine_equal_sample_weights_sum_to_one_per_engine() -> None:
    features = _feature_rows("train", {1: 3, 2: 2})
    dataset = prepare_modeling_dataset(features, _targets_from_features(features))

    weights = engine_equal_sample_weights(dataset)
    weight_sums = pd.Series(weights).groupby(dataset["engine_id"].to_numpy()).sum()

    assert weight_sums.to_dict() == {1: pytest.approx(1.0), 2: pytest.approx(1.0)}


def test_evaluation_protocol_has_distinct_estimands() -> None:
    protocol = build_evaluation_protocol()

    assert set(protocol["protocol_id"]) == {
        "rul_cv",
        "rul_final",
        "health_index_forecast",
        "rul_warning_oof",
        "etl_benchmark",
    }
    assert protocol.loc[protocol["protocol_id"].eq("rul_cv"), "unit_of_analysis"].item().startswith("engine")
    assert "100 FD001 test engines" in protocol.loc[protocol["protocol_id"].eq("rul_final"), "split_and_horizon"].item()


def test_warning_events_are_aggregated_by_engine_and_keep_lead_time() -> None:
    predictions = pd.DataFrame(
        {
            "split": ["train"] * 5,
            "engine_id": [1] * 5,
            "cycle": [1, 2, 3, 4, 5],
            "model_name": ["selected"] * 5,
            "y_true": [4.0, 3.0, 2.0, 1.0, 0.0],
            "y_pred": [4.5, 3.2, 2.5, 1.5, 0.5],
        }
    )

    events, summary = build_warning_events(predictions, model_name="selected", rul_threshold=3)

    row = events.iloc[0]
    assert row["first_warning_cycle"] == 3
    assert row["actual_rul_at_first_warning"] == pytest.approx(2.0)
    assert row["lead_time_to_failure_cycles"] == pytest.approx(2.0)
    assert bool(row["warning_before_actual_threshold"]) is False
    assert summary["n_engines"] == 1
    assert summary["n_missed_warning"] == 0


def test_warning_events_reject_non_train_or_unknown_model() -> None:
    predictions = pd.DataFrame(
        {
            "split": ["test"],
            "engine_id": [1],
            "cycle": [1],
            "model_name": ["other"],
            "y_true": [0.0],
            "y_pred": [0.0],
        }
    )

    with pytest.raises(ValueError, match="WARNING_INPUT"):
        build_warning_events(predictions, model_name="selected", rul_threshold=30)


def test_group_folds_keep_engine_boundaries_disjoint() -> None:
    features = _feature_rows("train", {engine: 2 for engine in range(1, 7)})
    dataset = prepare_modeling_dataset(features, _targets_from_features(features))

    folds = build_group_folds(dataset, n_splits=3)

    assert tuple(folds.columns) == FOLD_ASSIGNMENT_COLUMNS
    for fold in sorted(folds["fold"].unique()):
        validation_engines = set(folds.loc[folds["fold"].eq(fold), "engine_id"])
        training_engines = set(dataset["engine_id"]) - validation_engines
        assert validation_engines.isdisjoint(training_engines)


def test_modeling_pipeline_writes_artifacts_and_final_evaluation(tmp_path: Path) -> None:
    features = pd.concat(
        [
            _feature_rows("train", {engine: 3 for engine in range(1, 7)}),
            _feature_rows("test", {engine: 2 for engine in range(1, 4)}),
        ],
        ignore_index=True,
    ).sort_values(["split", "engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    targets = _targets_from_features(features)
    labels = pd.DataFrame(
        {"split": pd.Series(["test"] * 3, dtype="string"), "engine_id": [1, 2, 3], "rul": [10, 20, 30]}
    ).astype({"engine_id": "int64", "rul": "int64"})
    feature_path = tmp_path / "features.parquet"
    target_path = tmp_path / "targets.parquet"
    features.to_parquet(feature_path, index=False)
    targets.to_parquet(target_path, index=False)

    result = run_modeling_pipeline(
        feature_path,
        target_path,
        config=ModelingConfig(n_splits=3, candidate_models=("median_baseline", "ridge_alpha_1")),
    )
    payload = write_modeling_artifacts(result, tmp_path / "modeling")
    final_predictions = pd.read_parquet(tmp_path / "modeling" / "predictions" / "final_predictions.parquet")
    evaluation = evaluate_final_predictions(final_predictions, labels)
    final_payload = write_final_evaluation_artifacts(evaluation, tmp_path / "modeling")

    assert tuple(result.fold_assignments.columns) == FOLD_ASSIGNMENT_COLUMNS
    assert tuple(result.oof_predictions.columns) == OOF_PREDICTION_COLUMNS
    assert tuple(result.model_metrics.columns) == MODEL_METRIC_COLUMNS
    assert tuple(final_predictions.columns) == FINAL_PREDICTION_COLUMNS
    assert payload["selected_model"] in {"median_baseline", "ridge_alpha_1"}
    assert payload["row_counts"]["final_predictions"] == 3
    assert final_payload["metrics"]["n_engines"] == 3
    assert (tmp_path / "modeling" / "models" / "selected_model.joblib").exists()
