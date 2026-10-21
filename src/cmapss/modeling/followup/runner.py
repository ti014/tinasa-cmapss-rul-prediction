"""Điều phối study và giữ thứ tự ghi predictions trước khi đọc test labels."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import platform
import time

from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import joblib
import pandas as pd

from cmapss.modeling.baseline.artifact_io import write_final_evaluation_artifacts
from cmapss.modeling.baseline.evaluation import evaluate_predictions_from_path
from cmapss.modeling.followup.artifact_io import freeze_manifest
from cmapss.modeling.followup.config import StudyConfig, study_candidates
from cmapss.modeling.followup.estimators import fit_recipe
from cmapss.modeling.followup.features import FeatureStore
from cmapss.modeling.followup.training import fit_pair, inner_selection
from cmapss.modeling.improvement.analysis import engine_metrics, paired_comparison
from cmapss.modeling.improvement.artifact_io import _write_json
from cmapss.modeling.improvement.selection import _metrics


def run_study(dataset_id: str, repo: Path, output: Path, config: StudyConfig, *, evaluate_test: bool = True) -> dict[str, Any]:
    """Chạy nested cross-validation một dataset và ghi mọi artifact vào output."""
    started = time.perf_counter()
    source = repo / ("outputs/etl" if dataset_id == "FD001" else f"outputs/cross_dataset/{dataset_id}/etl")
    observation_path, target_path = source / "silver/observations.parquet", source / "silver/train_targets.parquet"
    candidates = study_candidates(dataset_id)
    identity = freeze_manifest(repo, output, dataset_id, observation_path, target_path, candidates, config)
    observations, targets = pd.read_parquet(observation_path), pd.read_parquet(target_path)
    store = FeatureStore(observations, targets)
    frame = store.features(candidates[0])
    if dataset_id == "FD001" and config.outer_splits == 5:
        assignments = pd.read_csv(repo / "outputs/modeling/tables/fold_assignments.csv")[["engine_id", "fold"]]
    else:
        assignments = pd.DataFrame([
            {"engine_id": int(engine), "fold": fold}
            for fold, (_, validation) in enumerate(GroupKFold(config.outer_splits).split(frame, frame.rul, frame.engine_id), 1)
            for engine in frame.iloc[validation].engine_id.unique()
        ])
    assignments.sort_values("engine_id").to_csv(output / "outer_assignments.csv", index=False)
    outer_frames: list[pd.DataFrame] = []
    selection_tables: list[pd.DataFrame] = []
    chosen: list[dict[str, Any]] = []
    for fold in sorted(assignments.fold.unique()):
        validation_ids = set(map(int, assignments.loc[assignments.fold.eq(fold), "engine_id"]))
        train_ids = set(map(int, frame.engine_id.unique())) - validation_ids
        training = frame.loc[frame.engine_id.isin(train_ids)].reset_index(drop=True)
        selected, table = inner_selection(store, training, candidates, dataset_id, config, output, identity, int(fold))
        selection_tables.append(table)
        chosen.append({"outer_fold": int(fold), **asdict(selected)})
        for stage, recipe in (("baseline", candidates[0]), ("selected", selected)):
            prediction = fit_pair(store, recipe, train_ids, validation_ids, config, output / "fit_cache", identity)
            outer_frames.append(prediction.assign(stage=stage, outer_fold=int(fold), candidate_id=recipe.name))
        pd.concat(outer_frames, ignore_index=True).to_parquet(output / "outer_predictions.parquet", index=False)
        pd.concat(selection_tables, ignore_index=True).to_csv(output / "inner_candidate_metrics.csv", index=False)
        _write_json(output / f"checkpoint_outer_{fold}.json", {"completed": True, "selected": asdict(selected)})
    outer = pd.concat(outer_frames, ignore_index=True)
    summaries, per_engine = [], {}
    for stage in ("baseline", "selected"):
        prediction = outer.loc[outer.stage.eq(stage)].sort_values(["engine_id", "cycle"])
        metrics = _metrics(prediction, 30)
        summaries.append({"stage": stage, **{k: v for k, v in metrics.items() if k != "warning_summary"}})
        _write_json(output / f"warning_summary_{stage}.json", metrics["warning_summary"])
        per_engine[stage] = engine_metrics(prediction)
        per_engine[stage].to_csv(output / f"engine_metrics_{stage}.csv")
    pd.DataFrame(summaries).to_csv(output / "outer_metrics.csv", index=False)
    paired = paired_comparison(per_engine["baseline"], per_engine["selected"], config.bootstrap_resamples, config.seed)
    paired.to_csv(output / "paired_comparison.csv", index=False)
    comparison = paired.set_index("metric")
    accepted = bool(comparison.loc["mean_engine_mae", "ci_upper"] < 0)
    if dataset_id == "FD001":
        accepted = bool(all(comparison.loc[key, "ci_upper"] < 0 for key in ("low_rul_engine_mae", "late_or_missed_rate"))
                        and all(comparison.loc[key, "delta"] <= 0 for key in ("mean_engine_mae", "early_alarm_burden")))
    _write_json(output / "decision.json", {"meets_predeclared_outer_gate": accepted,
        "recommendation": "candidate_supported_for_further_validation" if accepted else "retain_baseline",
        "interval_scope": "conditional on fitted outer predictions", "test_used_for_selection": False})
    selected, final_selection = inner_selection(store, frame, candidates, dataset_id, config, output, identity, 0)
    final_selection.to_csv(output / "final_selection_metrics.csv", index=False)
    all_ids = set(map(int, frame.engine_id.unique()))
    with threadpool_limits(limits=config.fit_threads):
        train, _, normalizer = store.fold_features(selected, all_ids, set(), config.seed)
        model = fit_recipe(train, selected, normalizer, config)
        test = observations.loc[observations.split.eq("test")]
        predictions = model.predict(test).groupby("engine_id", sort=True).tail(1).reset_index(drop=True)
    joblib.dump(model, output / "selected_model.joblib")
    prediction_path = output / "final_predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    with threadpool_limits(limits=config.fit_threads):
        reloaded = joblib.load(output / "selected_model.joblib").predict(test).groupby("engine_id", sort=True).tail(1).reset_index(drop=True)
    pd.testing.assert_frame_equal(predictions, reloaded)
    info: dict[str, Any] = {
        "dataset_id": dataset_id, "selected_final": asdict(selected), "outer_choices": chosen,
        "n_features": len(model.feature_columns), "config": asdict(config), "identity": identity,
        "test_label_joined": False, "saved_model_predictions_match": True,
        "python": platform.python_version(), "platform": platform.platform(),
        "selection_protocol": "nested GroupKFold by engine; preprocessing fitted inside every training fold",
    }
    _write_json(output / "run_info.json", info)
    if evaluate_test:
        labels = repo / "data/raw" / f"RUL_{dataset_id}.txt"
        evaluation = evaluate_predictions_from_path(prediction_path, labels)
        final = write_final_evaluation_artifacts(evaluation, output, test_label_path=labels,
                                                bootstrap_resamples=config.bootstrap_resamples, seed=config.seed)
        info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
        info["test_label_joined"] = True
        info["test_metrics"] = final["metrics"]
    info["seconds"] = time.perf_counter() - started
    _write_json(output / "run_info.json", info)
    print(json.dumps({"dataset": dataset_id, "selected": selected.name, "seconds": info["seconds"], "test": info.get("test_metrics")}), flush=True)
    return info
