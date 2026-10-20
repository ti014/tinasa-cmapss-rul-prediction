"""Ghi bảng, predictions, model, metadata và figures của modeling run."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json

from matplotlib import pyplot as plt
import joblib
import matplotlib
import pandas as pd

from cmapss.modeling.baseline.config import ModelingResult
from cmapss.modeling.baseline.evaluation import build_evaluation_protocol
from cmapss.modeling.contracts import (
    MODELING_BOOTSTRAP_RESAMPLES,
    MODELING_PRIMARY_METRIC,
    MODELING_SCHEMA_VERSION,
    MODELING_SEED,
    MODELING_WARNING_RUL_THRESHOLD,
)
from cmapss.modeling.early_warning import build_warning_events
from cmapss.modeling.metrics import bootstrap_metric_intervals, summarize_prediction_errors


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_warning_figure(
    oof_predictions: pd.DataFrame,
    event: pd.Series,
    path: Path,
) -> None:
    frame = oof_predictions.loc[
        oof_predictions["engine_id"].eq(int(event["engine_id"]))
        & oof_predictions["model_name"].eq(str(event["model_name"]))
    ].sort_values("cycle", kind="mergesort")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame["cycle"], frame["y_true"], label="RUL thực tế", color="#2166ac")
    ax.plot(frame["cycle"], frame["y_pred"], label="RUL dự đoán (OOF)", color="#b2182b")
    ax.axhline(int(event["rul_threshold"]), color="#f28e2b", linestyle="--", label="Ngưỡng cảnh báo")
    ax.axvline(int(event["failure_cycle"]), color="black", linestyle="--", linewidth=0.8, label="Thời điểm hỏng hóc")
    if int(event["first_warning_cycle"]) >= 0:
        ax.axvline(int(event["first_warning_cycle"]), color="#762a83", linestyle=":", label="Cảnh báo đầu tiên")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycle)")
    ax.set_title(f"Quỹ đạo cảnh báo sớm trên dự đoán OOF — động cơ train {int(event['engine_id'])}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_modeling_artifacts(result: ModelingResult, output_dir: str | Path) -> dict[str, Any]:
    """Ghi CV metrics, final predictions (chưa join labels), selected model và run_info."""
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


def _write_final_figures(evaluation: pd.DataFrame, figure_dir: Path) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = figure_dir / "final_true_vs_predicted.png"
    residual_path = figure_dir / "final_residuals_by_rul.png"

    plt.figure(figsize=(7, 5))
    plt.scatter(evaluation["y_true"], evaluation["y_pred"], s=28, alpha=0.75)
    max_value = float(max(evaluation["y_true"].max(), evaluation["y_pred"].max()))
    plt.plot([0, max_value], [0, max_value], color="black", linewidth=1)
    plt.xlabel("RUL thực tế (cycle)")
    plt.ylabel("RUL dự đoán (cycle)")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=160)
    plt.close()

    residual = evaluation["y_pred"] - evaluation["y_true"]
    plt.figure(figsize=(7, 5))
    plt.scatter(evaluation["y_true"], residual, s=28, alpha=0.75)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("RUL thực tế (cycle)")
    plt.ylabel("Sai số dự đoán: dự đoán − thực tế (cycle)")
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
