"""Read completed follow-up artifacts consistently across notebooks and UI."""

import json
from pathlib import Path

import pandas as pd

from cmapss.project import FOLLOWUP_RUN


def load_followup_results(repo_root: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require all three studies to finish before exposing a combined comparison."""
    root = Path(repo_root) / FOLLOWUP_RUN
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete" or set(completion.get("datasets", {})) != {"FD001", "FD002", "FD004"}:
        raise ValueError("Follow-up chưa hoàn tất đủ ba dataset")
    rows, comparisons = [], []
    root_repo = Path(repo_root)
    original_fd001 = json.loads((root_repo / "outputs/modeling/tables/final_metrics.json").read_text(encoding="utf-8"))["metrics"]
    matrix = json.loads((root_repo / "outputs/cross_dataset/matrix.json").read_text(encoding="utf-8"))["evaluations"]
    for dataset in ("FD001", "FD002", "FD004"):
        run = root / dataset
        info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
        if not info.get("test_label_joined") or not info.get("saved_model_predictions_match"):
            raise ValueError(f"Follow-up {dataset} chưa hoàn tất đánh giá")
        decision = json.loads((run / "decision.json").read_text(encoding="utf-8"))
        historical = original_fd001 if dataset == "FD001" else next(
            item["metrics"] for item in matrix if item["source_dataset"] == item["target_dataset"] == dataset)
        outer = pd.read_csv(run / "outer_metrics.csv").set_index("stage")
        rows.append({"dataset": dataset, "selected_final": info["selected_final"]["name"],
                     "n_features": info["n_features"], "baseline_outer_mae": outer.loc["baseline", "mean_engine_mae"],
                     "selected_outer_mae": outer.loc["selected", "mean_engine_mae"],
                     "baseline_outer_rmse": outer.loc["baseline", "rmse"],
                     "selected_outer_rmse": outer.loc["selected", "rmse"],
                     "baseline_outer_nasa": outer.loc["baseline", "nasa_score"],
                     "selected_outer_nasa": outer.loc["selected", "nasa_score"],
                     "baseline_low_mae": outer.loc["baseline", "low_rul_engine_mae"],
                     "selected_low_mae": outer.loc["selected", "low_rul_engine_mae"],
                     "baseline_late": int(outer.loc["baseline", "late_or_missed_count"]),
                     "selected_late": int(outer.loc["selected", "late_or_missed_count"]),
                     "test_mae": info["test_metrics"]["mean_engine_mae"],
                     "test_rmse": info["test_metrics"]["rmse"],
                     "test_nasa_score": info["test_metrics"]["nasa_score"],
                     "historical_test_mae": historical["mean_engine_mae"],
                     "historical_test_nasa_score": historical["nasa_score"],
                     "outer_gate_passed": decision["meets_predeclared_outer_gate"]})
        comparisons.append(pd.read_csv(run / "paired_comparison.csv").assign(dataset=dataset))
    return pd.DataFrame(rows), pd.concat(comparisons, ignore_index=True)
