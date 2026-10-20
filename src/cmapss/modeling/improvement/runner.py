"""Điều phối study và giữ thứ tự ghi predictions trước khi đọc test labels."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import platform

import joblib
import pandas as pd

from cmapss.modeling.baseline.artifact_io import write_final_evaluation_artifacts
from cmapss.modeling.baseline.evaluation import evaluate_predictions_from_path
from cmapss.modeling.improvement.artifact_io import _json_safe, _prepare_run, _sha256, _write_json
from cmapss.modeling.improvement.config import ImprovementConfig
from cmapss.modeling.improvement.dataset import load_window_datasets
from cmapss.modeling.improvement.selection import _inner_splits, _outer_assignments
from cmapss.modeling.improvement.stages import _run_outer_stages, _select_final_configuration
from cmapss.modeling.improvement.training import _final_test_predictions


def _save_assignments(datasets: dict[int, pd.DataFrame], output: Path, config: ImprovementConfig) -> pd.DataFrame:
    frame = datasets[30]
    if config.outer_splits == 5:
        assignments = pd.read_csv(output / "baseline_snapshot/tables/fold_assignments.csv")
        assignments = assignments.loc[:, ["engine_id", "fold"]].rename(columns={"fold": "outer_fold"})
    else:
        assignments = _outer_assignments(frame, config)
    if assignments.engine_id.duplicated().any() or set(assignments.engine_id) != set(frame.engine_id):
        raise ValueError("outer assignments must cover each train engine exactly once")
    destination = output / "outer_assignments.csv"
    if destination.exists():
        pd.testing.assert_frame_equal(pd.read_csv(destination), assignments)
    assignments.to_csv(destination, index=False)
    records = []
    for outer in range(config.outer_splits + 1):
        excluded = set(assignments.loc[assignments.outer_fold.eq(outer), "engine_id"]) if outer else set()
        train = frame.loc[~frame.engine_id.isin(excluded)].reset_index(drop=True)
        for inner, (fit_idx, val_idx) in enumerate(_inner_splits(train, config), start=1):
            fit_ids = set(train.iloc[fit_idx].engine_id)
            val_ids = set(train.iloc[val_idx].engine_id)
            if fit_ids & val_ids or (fit_ids | val_ids) & excluded:
                raise ValueError("nested engine separation failed")
            records.extend({"outer_fold": outer, "inner_fold": inner, "engine_id": int(engine)} for engine in sorted(val_ids))
    inner_path = output / "inner_assignments.csv"
    inner_frame = pd.DataFrame(records)
    if inner_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(inner_path), inner_frame)
    inner_frame.to_csv(inner_path, index=False)
    return assignments


def run_improvement_experiment(
    observations_path: str | Path = "outputs/etl/silver/observations.parquet",
    targets_path: str | Path = "outputs/etl/silver/train_targets.parquet",
    *,
    test_labels_path: str | Path = "data/raw/RUL_FD001.txt",
    output_dir: str | Path = "outputs/experiments/rul_improvement/20260927",
    config: ImprovementConfig | None = None,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    """Run the full experiment and write all artifacts to ``output_dir``."""
    run_config = config or ImprovementConfig()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    observation_path = Path(observations_path).resolve()
    target_path = Path(targets_path).resolve()
    test_label_path = Path(test_labels_path).resolve()
    _prepare_run(output_path, observation_path, target_path, run_config)
    datasets = load_window_datasets(observation_path, target_path, min_cycle=run_config.min_cycle)
    _save_assignments(datasets, output_path, run_config)
    outer_metrics, outer_predictions, extra = _run_outer_stages(datasets, run_config, output_path)
    from cmapss.modeling.improvement.analysis import write_outer_comparison
    write_outer_comparison(outer_predictions, output_path, run_config.bootstrap_resamples, run_config.seed)
    print("selecting final configuration on all train engines", flush=True)
    selected_window, selected_params, selected_weight, final_selection = _select_final_configuration(datasets, run_config, output_path / "fit_cache")
    final_train = datasets[selected_window]
    final_predictions, final_model = _final_test_predictions(
        observation_path,
        selected_window,
        selected_params,
        selected_weight,
        final_train,
        run_config,
    )
    outer_predictions.to_parquet(output_path / "outer_predictions.parquet", index=False)
    final_predictions.to_parquet(output_path / "final_predictions.parquet", index=False)
    joblib.dump(final_model, output_path / "selected_model.joblib")
    outer_metrics.to_csv(output_path / "outer_metrics.csv", index=False)
    extra["inner_tables"].to_csv(output_path / "inner_candidate_metrics.csv", index=False)
    final_selection.to_csv(output_path / "final_selection_metrics.csv", index=False)
    manifest = {
        "schema_version": 1,
        "run_id": output_path.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "config": asdict(run_config),
        "inputs": {
            "observations": str(observation_path),
            "targets": str(target_path),
            "test_labels": str(test_label_path),
            "observations_sha256": _sha256(observation_path),
            "targets_sha256": _sha256(target_path),
        },
        "windows": {str(window): {"rows": int(len(frame)), "engines": int(frame["engine_id"].nunique())} for window, frame in datasets.items()},
        "selected_final": {"window": selected_window, "weight_scheme": selected_weight, **selected_params},
        "outputs": {
            "outer_metrics": "outer_metrics.csv",
            "outer_predictions": "outer_predictions.parquet",
            "inner_candidate_metrics": "inner_candidate_metrics.csv",
            "final_selection_metrics": "final_selection_metrics.csv",
            "final_predictions": "final_predictions.parquet",
            "selected_model": "selected_model.joblib",
            "outer_assignments": "outer_assignments.csv",
        },
        "test_label_joined": False,
    }
    (output_path / "run_info.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_safe) + "\n", encoding="utf-8")
    if not evaluate_test:
        return manifest
    # Test labels are read only after final_predictions is saved.
    evaluation = evaluate_predictions_from_path(output_path / "final_predictions.parquet", test_label_path)
    final_metrics = write_final_evaluation_artifacts(
        evaluation,
        output_path,
        test_label_path=test_label_path,
        bootstrap_resamples=run_config.bootstrap_resamples,
        seed=run_config.seed,
    )
    from cmapss.modeling.improvement.analysis import write_test_comparison
    write_test_comparison(evaluation, output_path, run_config.bootstrap_resamples, run_config.seed)
    manifest = json.loads((output_path / "run_info.json").read_text(encoding="utf-8"))
    manifest["test_label_joined"] = True
    manifest["final_metrics"] = final_metrics["metrics"]
    manifest["outputs"].update(final_metrics["outputs"])
    _write_json(output_path / "run_info.json", manifest)
    return manifest
