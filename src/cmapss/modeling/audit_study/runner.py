"""Điều phối study và giữ thứ tự ghi predictions trước khi đọc test labels."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import time

from threadpoolctl import threadpool_limits
import joblib
import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.data.readers import read_test_labels
from cmapss.data.targets import validate_train_targets
from cmapss.modeling.audit_study.artifact_io import (
    _freeze_selection,
    _relative,
    _sha256,
    _write_comparisons,
    _write_json,
    _write_parquet,
)
from cmapss.modeling.audit_study.config import (
    AuditStudyConfig,
    BASE_FEATURES,
    HISTORY_FEATURES,
    WarningPolicy,
    make_candidate_specs,
)
from cmapss.modeling.audit_study.evaluation import (
    _paired_comparison,
    _policy_grid,
    build_endpoint_panel,
    warning_events,
)
from cmapss.modeling.audit_study.features import build_study_features
from cmapss.modeling.audit_study.provenance import BASELINE_FILES, _initialize_run
from cmapss.modeling.audit_study.training import (
    _final_predictor,
    _fit_predict,
    _inner_evaluation,
    _outer_assignments,
)
from cmapss.modeling.metrics import (
    add_prediction_errors,
    bootstrap_metric_intervals,
    summarize_prediction_errors,
)


def _baseline_replay(predictions: pd.DataFrame, config: AuditStudyConfig,
                     repo_root: Path, output: Path) -> dict[str, Any]:
    """Kiểm tra baseline_fixed tái tạo đúng dự đoán HGB OOF của pipeline chính."""
    if config.outer_splits != 5 or config.hgb_max_iter != 200 or config.seed != 42:
        result = {"applicable": False, "reason": "reduced/custom dry-run configuration"}
    else:
        historical = pd.read_parquet(repo_root / BASELINE_FILES[1])
        historical = historical[historical.model_name.eq("hist_gradient_boosting")]
        current = predictions[predictions.procedure.eq("baseline_fixed")]
        columns = list(KEY)
        joined = historical[columns + ["y_true", "y_pred"]].merge(
            current[columns + ["y_true", "y_pred"]], on=columns,
            how="outer", validate="one_to_one", indicator=True,
            suffixes=("_old", "_new"),
        )
        keys_match = bool(joined._merge.eq("both").all())
        truth_match = bool(keys_match and np.array_equal(joined.y_true_old, joined.y_true_new))
        delta = float((joined.y_pred_old - joined.y_pred_new).abs().max()) if keys_match else None
        passed = bool(truth_match and delta is not None and delta <= 1e-8)
        result = {"applicable": True, "keys_match": keys_match, "raw_truth_match": truth_match,
                  "prediction_max_abs_delta": delta, "tolerance": 1e-8, "passed": passed,
                  "historical_model": "hist_gradient_boosting", "n_rows": len(joined)}
    _write_json(output / "baseline_replay.json", result)
    if result.get("applicable") and not result.get("passed"):
        raise ValueError("Tái tạo OOF baseline công bố thất bại; giữ output và kiểm tra trước khi đánh giá test")
    return result


def _evaluate_test(predictions: pd.DataFrame, labels_path: Path,
                   config: AuditStudyConfig, output: Path,
                   repo_root: Path) -> dict[str, Any]:
    """Đánh giá test sau khi selection.json và dự đoán cuối đã được ghi ra đĩa."""
    if not (output / "selection.json").is_file() or not (output / "final_predictions.parquet").is_file():
        raise ValueError("Chỉ đọc nhãn test sau khi lưu lựa chọn và dự đoán cuối")
    labels = read_test_labels(labels_path)
    joined = predictions.merge(labels.rename(columns={"rul": "y_true"}),
                               on=["split", "engine_id"], how="outer",
                               validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Động cơ trong dự đoán test cuối và nhãn không khớp")
    evaluation = add_prediction_errors(joined.drop(columns="_merge")
                                       .rename(columns={"rul_pred": "y_pred"}))
    _write_parquet(evaluation, output / "final_evaluation.parquet")
    metrics = {**summarize_prediction_errors(evaluation),
               "bootstrap_intervals": bootstrap_metric_intervals(
                   evaluation, n_resamples=config.bootstrap_resamples, seed=config.seed),
               "test_labels_sha256": _sha256(labels_path),
               "test_history": "historically inspected FD001 test; reporting only, never used as selection or gate",
               "bootstrap_scope": "engine resampling conditional on fitted model; excludes retraining variability"}
    _write_json(output / "final_metrics.json", metrics)
    historical = pd.read_parquet(repo_root / BASELINE_FILES[2])
    columns = list(KEY)
    pair = historical[columns + ["y_true", "absolute_error", "nasa_score_component"]].merge(
        evaluation[columns + ["y_true", "absolute_error", "nasa_score_component"]],
        on=columns, how="outer", validate="one_to_one", indicator=True,
        suffixes=("_old", "_new"),
    )
    if not pair._merge.eq("both").all() or not np.array_equal(pair.y_true_old, pair.y_true_new):
        raise ValueError("Khóa hoặc nhãn gốc baseline đối chiếu test không khớp")
    pair = pair.sort_values("engine_id").set_index("engine_id")
    baseline = pd.DataFrame({"test_endpoint_mae": pair.absolute_error_old,
                             "test_endpoint_nasa": pair.nasa_score_component_old})
    candidate = pd.DataFrame({"test_endpoint_mae": pair.absolute_error_new,
                              "test_endpoint_nasa": pair.nasa_score_component_new})
    _paired_comparison(baseline, candidate, config).assign(
        reporting_only=True, promotion_gate=False,
    ).to_csv(output / "paired_test_comparison.csv", index=False)
    return metrics


def run_audit_study(
    *,
    repo_root: str | Path | None = None,
    observations_path: str | Path = "outputs/etl/silver/observations.parquet",
    targets_path: str | Path = "outputs/etl/silver/train_targets.parquet",
    test_labels_path: str | Path = "data/raw/RUL_FD001.txt",
    output_dir: str | Path = "outputs/experiments/audit_remediation/20261005",
    config: AuditStudyConfig | None = None,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    """
    Chạy nghiên cứu dưới outputs/experiments/audit_remediation; đường dẫn theo
    repo_root, nhãn test đọc sau khi lưu lựa chọn và dự đoán cuối.
    """
    started = time.perf_counter()
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[4]
    config = config or AuditStudyConfig()
    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return (root / path).resolve() if not path.is_absolute() else path.resolve()
    observations_path = resolve(observations_path)
    targets_path = resolve(targets_path)
    labels_path = resolve(test_labels_path)
    output = resolve(output_dir)
    _relative(labels_path, root)  # Nhãn phải nằm trong repository.
    manifest, identity = _initialize_run(output, root, (observations_path, targets_path), config)
    observations = pd.read_parquet(observations_path)
    targets = pd.read_parquet(targets_path)
    train_observations = observations[observations.split.eq("train")].copy()
    test_observations = observations[observations.split.eq("test")].copy()
    validate_train_targets(train_observations, targets)
    if set(observations.split) != {"train", "test"}:
        raise ValueError("Observations nghiên cứu phải có đúng split train và test")
    features = build_study_features(observations)
    train = features[features.split.eq("train")].merge(
        targets, on=list(KEY), how="left", validate="one_to_one",
    ).sort_values(["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    if train.rul.isna().any():
        raise ValueError("Features window thiếu nhãn train gốc")
    test_features = features[features.split.eq("test")]
    final_test = test_features.sort_values(["engine_id", "cycle"]).groupby(
        "engine_id", sort=True, as_index=False).tail(1).reset_index(drop=True)
    if final_test.engine_id.nunique() != test_observations.engine_id.nunique():
        raise ValueError("Động cơ test thiếu prefix 30 cycle cần thiết")
    expected_train = len(train_observations) - 29 * train_observations.engine_id.nunique()
    if len(train) != expected_train:
        raise ValueError("Số window train khác mốc window 30 đầy đủ")
    boundaries = {
        "raw_observations": len(observations), "raw_train_rows": len(train_observations),
        "raw_test_rows": len(test_observations), "invalid_raw_rows_removed": 0,
        "invalid_raw_policy": "fail fast, never silently drop observations",
        "train_engines": int(train.engine_id.nunique()), "test_engines": int(final_test.engine_id.nunique()),
        "window_cycles": 30, "feature_rows": len(features), "train_window_rows": len(train),
        "test_window_rows": len(test_features), "train_prefix_rows_excluded": len(train_observations) - len(train),
        "test_prefix_rows_excluded": len(test_observations) - len(test_features),
        "base_feature_count": len(BASE_FEATURES), "history_feature_count": len(HISTORY_FEATURES),
        "fold_local_pruning": "learned on fitting engines only; explicit selected columns in each fit cache",
        "endpoint_fractions": config.endpoint_fractions,
        "outer_endpoint_rows": int(train.engine_id.nunique() * len(config.endpoint_fractions)),
        "test_endpoint_rows": len(final_test), "target_scale": "raw RUL for every evaluation",
    }
    _write_json(output / "data_boundaries.json", boundaries)
    assignments = _outer_assignments(train, config, root)
    assignments.to_csv(output / "outer_assignments.csv", index=False)
    specs = {spec.candidate_id: spec for spec in make_candidate_specs()}
    all_predictions, all_events, inner_tables, inner_assignments, policy_tables, selections = [], [], [], [], [], []
    for fold in range(1, config.outer_splits + 1):
        val_ids = set(assignments.loc[assignments.outer_fold.eq(fold), "engine_id"])
        outer_train = train[~train.engine_id.isin(val_ids)].reset_index(drop=True)
        outer_validation = train[train.engine_id.isin(val_ids)].reset_index(drop=True)
        selected, inner_predictions, table, inner_assignment = _inner_evaluation(
            outer_train, config, output, identity, fold)
        inner_tables.append(table)
        inner_assignments.append(inner_assignment)
        baseline_policy, baseline_grid = _policy_grid(inner_predictions["hgb_baseline"], config)
        selected_policy, selected_grid = _policy_grid(inner_predictions[selected], config)
        policy_tables.extend([baseline_grid.assign(outer_fold=fold, role="baseline", candidate_id="hgb_baseline"),
                              selected_grid.assign(outer_fold=fold, role="selected", candidate_id=selected)])
        selections.append({"outer_fold": fold, "selected_candidate_id": selected,
                           "baseline_calibrated_policy": asdict(baseline_policy),
                           "selected_calibrated_policy": asdict(selected_policy),
                           "outer_validation_engine_ids": sorted(map(int, val_ids)),
                           "outer_train_engine_ids": sorted(map(int, outer_train.engine_id.unique()))})
        baseline_predictions = _fit_predict(outer_train, outer_validation, specs["hgb_baseline"],
                                            config, output / "fit_cache", identity)
        selected_predictions = _fit_predict(outer_train, outer_validation, specs[selected],
                                            config, output / "fit_cache", identity)
        procedures = (("baseline_fixed", baseline_predictions, WarningPolicy(30, 1)),
                      ("baseline_calibrated", baseline_predictions, baseline_policy),
                      ("selected_calibrated", selected_predictions, selected_policy))
        for procedure, predictions, policy in procedures:
            all_predictions.append(predictions.assign(procedure=procedure, outer_fold=fold,
                                                       threshold=policy.threshold, persistence=policy.persistence))
            events = warning_events(predictions, policy, lead_requirement=config.lead_requirement)
            all_events.append(events.assign(procedure=procedure, outer_fold=fold,
                                             candidate_id=predictions.candidate_id.iloc[0]))
        # Lưu checkpoint sau mỗi outer fold; resume dựng lại từ fit-cache.
        _write_parquet(pd.concat(all_predictions, ignore_index=True), output / "outer_predictions.parquet")
        _write_parquet(pd.concat(all_events, ignore_index=True), output / "warning_events.parquet")
        pd.concat(inner_tables, ignore_index=True).to_csv(output / "inner_candidate_metrics.csv", index=False)
        pd.concat(inner_assignments, ignore_index=True).to_csv(output / "inner_assignments.csv", index=False)
        pd.concat(policy_tables, ignore_index=True).to_csv(output / "inner_policy_metrics.csv", index=False)
        _write_json(output / "outer_selections.json", selections)
        _write_json(output / "progress.json", {"completed_outer_folds": fold, "total_outer_folds": config.outer_splits})
    outer_predictions = pd.concat(all_predictions, ignore_index=True).sort_values(
        ["procedure", "engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    events = pd.concat(all_events, ignore_index=True).sort_values(
        ["procedure", "engine_id"], kind="mergesort").reset_index(drop=True)
    _write_parquet(outer_predictions, output / "outer_predictions.parquet")
    _write_parquet(events, output / "warning_events.parquet")
    endpoints = pd.concat([build_endpoint_panel(frame, config.endpoint_fractions)
                           for _, frame in outer_predictions.groupby("procedure")], ignore_index=True)
    _write_parquet(endpoints, output / "outer_endpoints.parquet")
    decision = _write_comparisons(outer_predictions, events, config, output)
    replay = _baseline_replay(outer_predictions, config, root, output)
    selected, final_inner, final_table, final_assignment = _inner_evaluation(train, config, output, identity, 0)
    final_baseline_policy, final_baseline_grid = _policy_grid(final_inner["hgb_baseline"], config)
    final_policy, final_grid = _policy_grid(final_inner[selected], config)
    final_table.to_csv(output / "final_inner_candidate_metrics.csv", index=False)
    final_assignment.to_csv(output / "final_inner_assignments.csv", index=False)
    pd.concat([final_baseline_grid.assign(role="baseline", candidate_id="hgb_baseline"),
               final_grid.assign(role="selected", candidate_id=selected)], ignore_index=True).to_csv(
        output / "final_inner_policy_metrics.csv", index=False)
    chosen_grid = final_grid.loc[final_grid.selected].iloc[0]
    selection = {
        "candidate": asdict(specs[selected]), "config": asdict(config),
        "selected_calibrated_policy": asdict(final_policy),
        "baseline_calibrated_policy": asdict(final_baseline_policy),
        "inner_policy_target_met": bool(chosen_grid.inner_target_met),
        "inner_policy_on_time_rate": float(chosen_grid.on_time_rate),
        "promotion_gate_met": decision["promotion_gate_met"],
        "selection_source": "train only: all-train grouped inner CV; outer gate evaluated separately",
        "test_labels_read_for_selection": False, "source_identity": identity,
    }
    _freeze_selection(output / "selection.json", selection)
    fitted = _final_predictor(train, specs[selected], config, output, identity)
    with threadpool_limits(limits=config.fit_threads):
        prediction = fitted.predict(final_test)
        restored_prediction = joblib.load(output / "selected_model.joblib").predict(final_test)
    np.testing.assert_array_equal(prediction, restored_prediction)
    final_predictions = final_test.loc[:, list(KEY)].copy()
    final_predictions["model_name"] = selected
    final_predictions["rul_pred"] = prediction
    _write_parquet(final_predictions, output / "final_predictions.parquet")
    metrics = _evaluate_test(final_predictions, labels_path, config, output, root) if evaluate_test else None
    current_baseline = {name: _sha256(root / name) for name in BASELINE_FILES}
    if current_baseline != manifest["identity"]["baseline_sha256"]:
        raise ValueError("Baseline lịch sử đã đổi trong lúc nghiên cứu")
    current_sources = {name: _sha256(root / name) for name in manifest["identity"]["source_sha256"]}
    fit_metadata = [json.loads(path.read_text(encoding="utf-8")) for path in (output / "fit_cache").glob("*.json")]
    summary = {
        "status": "completed", "output_dir": _relative(output, root),
        "manifest": _relative(output / "manifest.json", root),
        "selection": _relative(output / "selection.json", root),
        "selected_model": _relative(output / "selected_model.joblib", root),
        "final_predictions": _relative(output / "final_predictions.parquet", root),
        "source_identity": identity, "decision": decision, "selected_candidate_id": selected,
        "warning_policy": asdict(final_policy), "baseline_replay": replay,
        "test_evaluation_completed": evaluate_test, "final_metrics": metrics,
        "baseline_bytes_unchanged": True,
        "current_used_source_equivalent_to_snapshot": current_sources == manifest["identity"]["source_sha256"],
        "data_boundaries": boundaries, "elapsed_seconds": time.perf_counter() - started,
        "unique_cached_cv_fits": len(fit_metadata),
        "sum_cached_cv_fit_seconds": sum(item["seconds"] for item in fit_metadata),
        "fit_upper_bound": len(config.candidate_ids) * config.inner_splits * (config.outer_splits + 1)
                           + 2 * config.outer_splits + 1,
        "bootstrap_scope": "paired engines conditional on predictions, not full procedure retraining uncertainty",
    }
    _write_json(output / "run_info.json", summary)
    _write_json(output / "progress.json", {"status": "completed", "completed_outer_folds": config.outer_splits})
    return summary
