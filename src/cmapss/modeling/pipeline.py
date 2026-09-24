"""Leakage-safe modeling pipeline cho RUL FD001."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cmapss.data.contracts import KEY, TEST_LABEL_COLUMNS, TRAIN_TARGET_COLUMNS, ContractError
from cmapss.data.readers import read_test_labels
from cmapss.data.validation import validate_columns, validate_integer, validate_test_labels
from cmapss.features.contracts import ROLLING_FEATURE_COLUMNS
from cmapss.features.rolling import validate_rolling_features
from cmapss.modeling.contracts import (
    FINAL_EVALUATION_COLUMNS,
    FINAL_PREDICTION_COLUMNS,
    FOLD_ASSIGNMENT_COLUMNS,
    FOLD_METRIC_COLUMNS,
    MODELING_BOOTSTRAP_RESAMPLES,
    MODELING_FEATURE_COLUMNS,
    MODELING_N_SPLITS,
    MODELING_PRIMARY_METRIC,
    MODELING_SCHEMA_VERSION,
    MODELING_SEED,
    MODELING_WARNING_RUL_THRESHOLD,
    MODEL_METRIC_COLUMNS,
    OOF_PREDICTION_COLUMNS,
    EVALUATION_PROTOCOL_COLUMNS,
)
from cmapss.modeling.metrics import (
    add_prediction_errors,
    bootstrap_metric_intervals,
    summarize_prediction_errors,
)
from cmapss.modeling.early_warning import build_warning_events

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class ModelingConfig:
    """Cấu hình modeling đã cố định cho mốc RUL đầu tiên."""

    n_splits: int = MODELING_N_SPLITS
    seed: int = MODELING_SEED
    candidate_models: tuple[str, ...] = (
        "median_baseline",
        "ridge_alpha_1",
        "hist_gradient_boosting",
    )
    ridge_alpha: float = 1.0
    hgb_learning_rate: float = 0.05
    hgb_max_iter: int = 200
    hgb_max_leaf_nodes: int = 31
    hgb_min_samples_leaf: int = 20
    bootstrap_resamples: int = MODELING_BOOTSTRAP_RESAMPLES


@dataclass(frozen=True)
class ModelingResult:
    """Artifact trong bộ nhớ trước khi ghi ra `outputs/modeling`."""

    config: ModelingConfig
    selected_model: str
    selected_estimator: Any
    input_sha256: dict[str, str]
    fold_assignments: pd.DataFrame
    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    model_metrics: pd.DataFrame
    final_predictions: pd.DataFrame


class WeightedMedianRegressor(BaseEstimator, RegressorMixin):
    """Baseline hằng bằng weighted median RUL của training fold."""

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "WeightedMedianRegressor":
        del X
        self.constant_ = _weighted_median(np.asarray(y, dtype="float64"), sample_weight)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.full(len(X), self.constant_, dtype="float64")


def build_evaluation_protocol() -> pd.DataFrame:
    """Return the repository-wide evaluation matrix used by the report."""
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _weighted_median(values: np.ndarray, sample_weight: np.ndarray | None) -> float:
    if sample_weight is None:
        return float(np.median(values))
    weights = np.asarray(sample_weight, dtype="float64")
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * ordered_weights.sum()
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left")])


def _validate_train_targets_frame(targets: pd.DataFrame) -> None:
    validate_columns(targets, TRAIN_TARGET_COLUMNS, "modeling train targets")
    if not targets["split"].eq("train").fillna(False).all():
        raise ContractError("TRAIN_ONLY", "modeling train targets", "targets phải thuộc split train")
    for column in ("engine_id", "cycle", "rul"):
        validate_integer(targets, column, 0 if column == "rul" else 1, "modeling train targets")
    if targets.duplicated(list(KEY)).any():
        raise ContractError("MODELING_TARGET_KEYS", "modeling train targets", "trùng khóa target")


def prepare_modeling_dataset(features: pd.DataFrame, train_targets: pd.DataFrame) -> pd.DataFrame:
    """Join train rolling features với train RUL targets, không đọc test labels."""
    validate_rolling_features(features)
    _validate_train_targets_frame(train_targets)
    train_features = features.loc[features["split"].eq("train")].copy()
    joined = train_features.merge(
        train_targets.loc[:, list(TRAIN_TARGET_COLUMNS)],
        on=list(KEY),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = joined["_merge"].ne("both")
    if missing.any():
        key = joined.loc[missing, list(KEY)].iloc[0].to_dict()
        raise ContractError(
            "MODELING_TARGET_KEYS",
            "modeling dataset",
            f"thiếu train RUL cho rolling feature {key}",
        )
    joined = joined.drop(columns=["_merge"])
    return joined.loc[:, list(ROLLING_FEATURE_COLUMNS) + ["rul"]].reset_index(drop=True)


def engine_equal_sample_weights(dataset: pd.DataFrame) -> np.ndarray:
    """Gán tổng trọng số bằng 1 cho mỗi engine trong training fold."""
    counts = dataset.groupby("engine_id", sort=False)["cycle"].transform("size")
    return (1.0 / counts.to_numpy(dtype="float64")).astype("float64")


def build_group_folds(dataset: pd.DataFrame, *, n_splits: int) -> pd.DataFrame:
    """Tạo fold assignment theo engine, mỗi engine thuộc đúng một validation fold."""
    groups = dataset["engine_id"].to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    records: list[dict[str, int | str]] = []
    X = dataset.loc[:, list(MODELING_FEATURE_COLUMNS)]
    y = dataset["rul"].to_numpy()
    for fold, (_, validation_index) in enumerate(splitter.split(X, y, groups), start=1):
        engines = sorted(int(value) for value in dataset.iloc[validation_index]["engine_id"].unique())
        records.extend({"split": "train", "engine_id": engine, "fold": fold} for engine in engines)
    return pd.DataFrame(records, columns=FOLD_ASSIGNMENT_COLUMNS).astype(
        {"split": "string", "engine_id": "int64", "fold": "int64"}
    )


def _make_estimator(name: str, config: ModelingConfig) -> Any:
    if name == "median_baseline":
        return WeightedMedianRegressor()
    if name == "ridge_alpha_1":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=config.ridge_alpha)),
            ]
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=config.hgb_learning_rate,
            max_iter=config.hgb_max_iter,
            max_leaf_nodes=config.hgb_max_leaf_nodes,
            min_samples_leaf=config.hgb_min_samples_leaf,
            early_stopping=False,
            random_state=config.seed,
        )
    raise ValueError(f"Unknown candidate model: {name}")


def _fit_estimator(estimator: Any, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray) -> Any:
    if isinstance(estimator, Pipeline):
        return estimator.fit(X, y, model__sample_weight=sample_weight)
    return estimator.fit(X, y, sample_weight=sample_weight)


def _prediction_frame(
    rows: pd.DataFrame,
    *,
    fold: int,
    model_name: str,
    predictions: np.ndarray,
) -> pd.DataFrame:
    frame = rows.loc[:, list(KEY)].copy()
    frame["fold"] = fold
    frame["model_name"] = model_name
    frame["y_true"] = rows["rul"].to_numpy(dtype="float64")
    frame["y_pred"] = predictions.astype("float64")
    frame = add_prediction_errors(frame)
    return frame.loc[:, list(OOF_PREDICTION_COLUMNS)]


def _write_warning_figure(
    oof_predictions: pd.DataFrame,
    event: pd.Series,
    path: Path,
) -> None:
    """Plot one train OOF trajectory with the fixed warning threshold."""
    frame = oof_predictions.loc[
        oof_predictions["engine_id"].eq(int(event["engine_id"]))
        & oof_predictions["model_name"].eq(str(event["model_name"]))
    ].sort_values("cycle", kind="mergesort")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame["cycle"], frame["y_true"], label="Actual RUL", color="#2166ac")
    ax.plot(frame["cycle"], frame["y_pred"], label="OOF prediction", color="#b2182b")
    ax.axhline(int(event["rul_threshold"]), color="#f28e2b", linestyle="--", label="Warning threshold")
    ax.axvline(int(event["failure_cycle"]), color="black", linestyle="--", linewidth=0.8, label="Failure proxy")
    if int(event["first_warning_cycle"]) >= 0:
        ax.axvline(int(event["first_warning_cycle"]), color="#762a83", linestyle=":", label="First warning")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Train OOF early-warning trajectory — engine {int(event['engine_id'])}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _cross_validate_candidates(
    dataset: pd.DataFrame,
    *,
    config: ModelingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    X = dataset.loc[:, list(MODELING_FEATURE_COLUMNS)]
    y = dataset["rul"]
    groups = dataset["engine_id"].to_numpy()
    splitter = GroupKFold(n_splits=config.n_splits)
    oof_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []

    for model_name in config.candidate_models:
        for fold, (train_index, validation_index) in enumerate(splitter.split(X, y, groups), start=1):
            train_rows = dataset.iloc[train_index]
            validation_rows = dataset.iloc[validation_index]
            estimator = _make_estimator(model_name, config)
            weights = engine_equal_sample_weights(train_rows)
            fitted = _fit_estimator(
                estimator,
                train_rows.loc[:, list(MODELING_FEATURE_COLUMNS)],
                train_rows["rul"],
                weights,
            )
            predictions = fitted.predict(validation_rows.loc[:, list(MODELING_FEATURE_COLUMNS)])
            prediction_frame = _prediction_frame(
                validation_rows,
                fold=fold,
                model_name=model_name,
                predictions=np.asarray(predictions, dtype="float64"),
            )
            oof_frames.append(prediction_frame)
            metric_records.append(
                {
                    "model_name": model_name,
                    "fold": fold,
                    **summarize_prediction_errors(prediction_frame),
                }
            )

    oof = pd.concat(oof_frames, ignore_index=True).loc[:, list(OOF_PREDICTION_COLUMNS)]
    fold_metrics = pd.DataFrame(metric_records, columns=FOLD_METRIC_COLUMNS)
    model_metrics = (
        fold_metrics.groupby("model_name", sort=True)
        .agg(
            n_folds=("fold", "nunique"),
            mean_engine_mae=("mean_engine_mae", "mean"),
            std_engine_mae=("mean_engine_mae", "std"),
            mean_rmse=("rmse", "mean"),
            mean_nasa_score=("nasa_score", "mean"),
        )
        .reset_index()
    )
    model_metrics["std_engine_mae"] = model_metrics["std_engine_mae"].fillna(0.0)
    selected_model = str(
        model_metrics.sort_values(
            ["mean_engine_mae", "mean_rmse", "mean_nasa_score", "model_name"],
            kind="mergesort",
        )
        .iloc[0]["model_name"]
    )
    model_metrics["selected"] = model_metrics["model_name"].eq(selected_model)
    model_metrics = model_metrics.loc[:, list(MODEL_METRIC_COLUMNS)]
    return oof, fold_metrics, model_metrics, selected_model


def _select_final_test_windows(features: pd.DataFrame, model_name: str, estimator: Any) -> pd.DataFrame:
    test_features = (
        features.loc[features["split"].eq("test")]
        .sort_values(list(KEY), kind="mergesort")
        .groupby("engine_id", as_index=False, sort=True)
        .tail(1)
        .reset_index(drop=True)
    )
    predictions = estimator.predict(test_features.loc[:, list(MODELING_FEATURE_COLUMNS)])
    final = test_features.loc[:, list(KEY)].copy()
    final["model_name"] = model_name
    final["rul_pred"] = np.asarray(predictions, dtype="float64")
    if not np.isfinite(final["rul_pred"].to_numpy(dtype="float64")).all():
        raise ContractError("MODELING_PREDICTION", "final predictions", "mọi prediction phải hữu hạn")
    return final.loc[:, list(FINAL_PREDICTION_COLUMNS)].reset_index(drop=True)


def run_modeling_pipeline(
    rolling_features_path: str | Path,
    train_targets_path: str | Path,
    *,
    config: ModelingConfig | None = None,
) -> ModelingResult:
    """Chạy CV, chọn model và tạo 100 final predictions chưa join test labels."""
    run_config = config or ModelingConfig()
    feature_path = Path(rolling_features_path)
    target_path = Path(train_targets_path)
    features = pd.read_parquet(feature_path)
    train_targets = pd.read_parquet(target_path)
    dataset = prepare_modeling_dataset(features, train_targets)
    fold_assignments = build_group_folds(dataset, n_splits=run_config.n_splits)
    oof_predictions, fold_metrics, model_metrics, selected_model = _cross_validate_candidates(
        dataset,
        config=run_config,
    )
    estimator = _make_estimator(selected_model, run_config)
    selected_estimator = _fit_estimator(
        estimator,
        dataset.loc[:, list(MODELING_FEATURE_COLUMNS)],
        dataset["rul"],
        engine_equal_sample_weights(dataset),
    )
    final_predictions = _select_final_test_windows(features, selected_model, selected_estimator)
    return ModelingResult(
        config=run_config,
        selected_model=selected_model,
        selected_estimator=selected_estimator,
        input_sha256={feature_path.name: _sha256(feature_path), target_path.name: _sha256(target_path)},
        fold_assignments=fold_assignments,
        oof_predictions=oof_predictions,
        fold_metrics=fold_metrics,
        model_metrics=model_metrics,
        final_predictions=final_predictions,
    )


def write_modeling_artifacts(result: ModelingResult, output_dir: str | Path) -> dict[str, Any]:
    """Ghi CV metrics, final predictions chưa join labels và selected model."""
    output_path = Path(output_dir).resolve()
    raw_path = Path("data/raw").resolve()
    if output_path == raw_path or raw_path in output_path.parents:
        raise ValueError("Modeling output không được ghi vào data/raw để bảo vệ Bronze immutable")

    table_dir = output_path / "tables"
    prediction_dir = output_path / "predictions"
    model_dir = output_path / "models"
    figure_dir = output_path / "figures"
    for directory in (table_dir, prediction_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "fold_assignments": table_dir / "fold_assignments.csv",
        "fold_metrics": table_dir / "fold_metrics.csv",
        "model_metrics": table_dir / "model_metrics.csv",
        "oof_predictions": prediction_dir / "oof_predictions.parquet",
        "final_predictions": prediction_dir / "final_predictions.parquet",
        "selected_model": model_dir / "selected_model.joblib",
        "run_info": output_path / "run_info.json",
        "evaluation_protocol": table_dir / "evaluation_protocol.csv",
        "warning_events": table_dir / "warning_events.csv",
        "warning_summary": table_dir / "warning_summary.json",
        "warning_figure": figure_dir / "warning_trajectory.png",
    }
    result.fold_assignments.to_csv(paths["fold_assignments"], index=False)
    result.fold_metrics.to_csv(paths["fold_metrics"], index=False)
    result.model_metrics.to_csv(paths["model_metrics"], index=False)
    build_evaluation_protocol().to_csv(paths["evaluation_protocol"], index=False)
    warning_events, warning_summary = build_warning_events(
        result.oof_predictions,
        model_name=result.selected_model,
        rul_threshold=MODELING_WARNING_RUL_THRESHOLD,
    )
    warning_events.to_csv(paths["warning_events"], index=False)
    paths["warning_summary"].write_text(
        json.dumps(warning_summary, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    _write_warning_figure(result.oof_predictions, warning_events.iloc[0], paths["warning_figure"])
    result.oof_predictions.to_parquet(paths["oof_predictions"], index=False)
    result.final_predictions.to_parquet(paths["final_predictions"], index=False)
    joblib.dump(result.selected_estimator, paths["selected_model"])

    payload: dict[str, Any] = {
        "schema_version": MODELING_SCHEMA_VERSION,
        "pipeline": "sklearn_grouped_rul_modeling",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": result.input_sha256,
        "config": {
            "n_splits": result.config.n_splits,
            "seed": result.config.seed,
            "candidate_models": list(result.config.candidate_models),
            "primary_metric": MODELING_PRIMARY_METRIC,
            "engine_equal_sample_weighting": True,
            "prediction_clipping": "none",
            "early_warning": {
                "target": "low RUL",
                "threshold_cycles": MODELING_WARNING_RUL_THRESHOLD,
                "source": "selected-model OOF predictions on train only",
            },
        },
        "selected_model": result.selected_model,
        "row_counts": {
            "fold_assignments": int(len(result.fold_assignments)),
            "oof_predictions": int(len(result.oof_predictions)),
            "fold_metrics": int(len(result.fold_metrics)),
            "model_metrics": int(len(result.model_metrics)),
            "final_predictions": int(len(result.final_predictions)),
            "warning_events": int(len(warning_events)),
        },
        "outputs": {
            name: str(path.relative_to(output_path))
            for name, path in paths.items()
            if name != "run_info"
        },
        "test_label_join": {
            "status": "not_yet_joined",
            "reason": "final_predictions được ghi trước; test labels chỉ đọc trong bước final evaluation.",
        },
    }
    paths["run_info"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    return payload


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
    if len(joined) != 100 and test_labels["engine_id"].nunique() == 100:
        raise ContractError("FINAL_PREDICTION_COUNT", "final predictions", "cần đúng 100 test engines")
    evaluation = joined.rename(columns={"rul": "y_true", "rul_pred": "y_pred"}).drop(columns=["_merge"])
    evaluation = add_prediction_errors(evaluation)
    return evaluation.loc[:, list(FINAL_EVALUATION_COLUMNS)].reset_index(drop=True)


def _write_final_figures(evaluation: pd.DataFrame, figure_dir: Path) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = figure_dir / "final_true_vs_predicted.png"
    residual_path = figure_dir / "final_residuals_by_rul.png"

    plt.figure(figsize=(7, 5))
    plt.scatter(evaluation["y_true"], evaluation["y_pred"], s=28, alpha=0.75)
    max_value = float(max(evaluation["y_true"].max(), evaluation["y_pred"].max()))
    plt.plot([0, max_value], [0, max_value], color="black", linewidth=1)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=160)
    plt.close()

    residual = evaluation["y_pred"] - evaluation["y_true"]
    plt.figure(figsize=(7, 5))
    plt.scatter(evaluation["y_true"], residual, s=28, alpha=0.75)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("True RUL")
    plt.ylabel("Prediction error")
    plt.tight_layout()
    plt.savefig(residual_path, dpi=160)
    plt.close()

    return {
        "final_true_vs_predicted": str(scatter_path),
        "final_residuals_by_rul": str(residual_path),
    }


def write_final_evaluation_artifacts(
    evaluation: pd.DataFrame,
    output_dir: str | Path,
    *,
    test_label_path: str | Path | None = None,
    bootstrap_resamples: int = MODELING_BOOTSTRAP_RESAMPLES,
    seed: int = MODELING_SEED,
) -> dict[str, Any]:
    """Ghi final evaluation, bootstrap CI, figures và cập nhật run_info."""
    output_path = Path(output_dir).resolve()
    prediction_dir = output_path / "predictions"
    table_dir = output_path / "tables"
    figure_dir = output_path / "figures"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "final_evaluation": prediction_dir / "final_evaluation.parquet",
        "final_metrics": table_dir / "final_metrics.json",
        "run_info": output_path / "run_info.json",
    }
    evaluation.to_parquet(paths["final_evaluation"], index=False)
    metrics = summarize_prediction_errors(evaluation)
    intervals = bootstrap_metric_intervals(
        evaluation,
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    figure_paths = _write_final_figures(evaluation, figure_dir)
    payload: dict[str, Any] = {
        "schema_version": MODELING_SCHEMA_VERSION,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_label_sha256": _sha256(Path(test_label_path)) if test_label_path is not None else None,
        "metrics": metrics,
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": seed,
            "method": "engine_cluster_resampling_with_replacement; mean_engine_mae preserves duplicate engine draws",
            "intervals": intervals,
        },
        "outputs": {
            "final_evaluation": str(paths["final_evaluation"].relative_to(output_path)),
            "final_metrics": str(paths["final_metrics"].relative_to(output_path)),
            **{
                name: str(Path(path).relative_to(output_path))
                for name, path in figure_paths.items()
            },
        },
    }
    paths["final_metrics"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    if paths["run_info"].exists():
        run_info = json.loads(paths["run_info"].read_text(encoding="utf-8"))
        run_info["test_label_join"] = {
            "status": "completed_after_final_predictions",
            "test_label_sha256": payload["test_label_sha256"],
            "final_metrics": payload["metrics"],
            "bootstrap": payload["bootstrap"],
        }
        run_info["outputs"] = {**run_info.get("outputs", {}), **payload["outputs"]}
        paths["run_info"].write_text(
            json.dumps(run_info, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
            encoding="utf-8",
        )
    return payload


def evaluate_predictions_from_path(
    final_predictions_path: str | Path,
    test_labels_path: str | Path,
) -> pd.DataFrame:
    """Đọc final predictions đã ghi rồi mới đọc NASA test labels."""
    final_predictions = pd.read_parquet(final_predictions_path)
    labels = read_test_labels(test_labels_path)
    return evaluate_final_predictions(final_predictions, labels)
