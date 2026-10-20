"""Điều phối study và giữ thứ tự ghi predictions trước khi đọc test labels."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any
import multiprocessing
import shutil
import time

from threadpoolctl import threadpool_limits
import joblib
import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.modeling import audit_study as audit
from cmapss.modeling.comparison_runner.artifact_io import (
    _freeze_json,
    _progress,
    _read_json,
    _sha,
    _write_json,
)
from cmapss.modeling.comparison_runner.contracts import AUTHOR, ComparisonBackend, ComparisonConfig
from cmapss.modeling.comparison_runner.evaluation import evaluate_final_test
from cmapss.modeling.comparison_runner.provenance import _assert_immutable, freeze_inputs
from cmapss.modeling.comparison_runner.training import fit_batch, inner_stage


def run_comparison(
    root: Path, output: Path, observations: Path, targets: Path, labels: Path,
    config: ComparisonConfig, evaluate_test: bool, backend: ComparisonBackend,
) -> dict[str, Any]:
    """Run outer folds, the final inner selection, final models and optional test scoring."""
    manifest, identity, train, test, assignments = freeze_inputs(
        root, output, observations, targets, config, backend)
    started = time.perf_counter()
    candidate_specs = {item.candidate_id: item for item in backend.active_specs(config)}
    predictions, events, selections, stage_tables = [], [], [], []
    with ProcessPoolExecutor(max_workers=config.max_workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for fold in range(1, config.outer_splits + 1):
            validation_ids = tuple(sorted(map(int, assignments.loc[
                assignments.outer_fold.eq(fold), "engine_id"])))
            fitting_ids = tuple(sorted(set(map(int, train.engine_id.unique())) - set(validation_ids)))
            fitting = train[train.engine_id.isin(fitting_ids)].reset_index(drop=True)
            choices, _, selection = inner_stage(
                fitting, fold, pool, root, output, manifest, identity, config, backend)
            selections.append({**selection, "outer_fit_engine_ids": fitting_ids,
                               "outer_validation_engine_ids": validation_ids})
            stage_tables.append(output / "stages" / f"outer_{fold}")
            tasks = [(candidate_specs[choices[arm]["candidate_id"]], fitting_ids,
                      validation_ids, fold) for arm in backend.arms]
            paths = fit_batch(pool, tasks, root, output, manifest, identity, config,
                               f"outer_{fold}_evaluation", backend)
            for arm in backend.arms:
                candidate = choices[arm]["candidate_id"]
                policy = audit.WarningPolicy(**choices[arm]["policy"])
                prediction = pd.read_parquet(paths[candidate, fold]).assign(
                    procedure=arm, outer_fold=fold, threshold=policy.threshold,
                    persistence=policy.persistence)
                warning = audit.warning_events(prediction, policy,
                                                lead_requirement=config.lead_requirement).assign(
                    procedure=arm, outer_fold=fold, candidate_id=candidate)
                predictions.append(prediction)
                events.append(warning)
                if arm == selection["procedure"]["selected_procedure_id"]:
                    predictions.append(prediction.assign(procedure="selected_pipeline", selected_arm=arm))
                    events.append(warning.assign(procedure="selected_pipeline", selected_arm=arm))
            audit._write_parquet(pd.concat(predictions, ignore_index=True), output / "outer_predictions.parquet")
            audit._write_parquet(pd.concat(events, ignore_index=True), output / "warning_events.parquet")
            _write_json(output / "outer_selections.json", selections)
            for name in ("inner_candidate_metrics.csv", "inner_assignments.csv", "inner_policy_metrics.csv"):
                pd.concat([pd.read_csv(folder / name) for folder in stage_tables], ignore_index=True).to_csv(
                    output / name, index=False)
            _progress(output, "outer_completed", completed_outer_folds=fold,
                      total_outer_folds=config.outer_splits)
        outer = pd.concat(predictions, ignore_index=True).sort_values(
            ["procedure", "engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
        warnings = pd.concat(events, ignore_index=True).sort_values(
            ["procedure", "engine_id"], kind="mergesort").reset_index(drop=True)
        audit._write_parquet(outer, output / "outer_predictions.parquet")
        audit._write_parquet(warnings, output / "warning_events.parquet")
        endpoints = pd.concat([audit.build_endpoint_panel(group, config.endpoint_fractions)
                               for _, group in outer.groupby("procedure", sort=True)], ignore_index=True)
        audit._write_parquet(endpoints, output / "outer_endpoints.parquet")
        _assert_immutable(root, output, manifest)
        decision = backend.write_readouts(outer, warnings, config, output)
        _freeze_json(output / "outer_decision_before_test.json", decision)
        choices, _, final_selection = inner_stage(train, 0, pool, root, output,
                                                    manifest, identity, config, backend)
        for name in ("inner_candidate_metrics.csv", "inner_assignments.csv", "inner_policy_metrics.csv"):
            shutil.copy2(output / "stages/final_inner" / name, output / f"final_{name}")
        final_selection.update(config=asdict(config), outer_decision=decision,
                               author=AUTHOR, test_labels_read_for_selection=False)
        _freeze_json(output / "selection.json", final_selection)
        _assert_immutable(root, output, manifest)
        futures = {pool.submit(backend.final_model_task, output / "frozen_train_features.parquet",
                               candidate_specs[choices[arm]["candidate_id"]], config,
                               output / "models" / arm, identity): arm for arm in backend.arms}
        model_paths = {}
        for completed, future in enumerate(as_completed(futures), start=1):
            arm = futures[future]
            model_paths[arm] = Path(future.result())
            _assert_immutable(root, output, manifest)
            _progress(output, "final_models", completed_fits=completed, total_fits=len(backend.arms), procedure=arm)
    final_test = test.groupby("engine_id", sort=True, as_index=False).tail(1).reset_index(drop=True)
    if final_test.engine_id.nunique() != test.engine_id.nunique():
        raise ValueError("final test endpoints do not cover all engines")
    final_predictions = []
    for arm in backend.arms:
        fitted = joblib.load(model_paths[arm])
        with threadpool_limits(limits=config.fit_threads):
            predicted = fitted.predict(final_test)
            replayed = joblib.load(model_paths[arm]).predict(final_test)
        np.testing.assert_array_equal(predicted, replayed)
        frame = final_test[list(KEY)].copy()
        frame["candidate_id"] = choices[arm]["candidate_id"]
        frame["procedure"] = arm
        frame["y_pred"] = predicted
        final_predictions.append(frame)
        if arm == final_selection["procedure"]["selected_procedure_id"]:
            final_predictions.append(frame.assign(procedure="selected_pipeline", selected_arm=arm))
    final = pd.concat(final_predictions, ignore_index=True)
    audit._write_parquet(final, output / "final_predictions.parquet")
    _freeze_json(output / "final_predictions_receipt.json", {
        "source_identity": identity, "predictions_sha256": _sha(output / "final_predictions.parquet"),
        "test_labels_read": False, "saved_model_replay_exact": True,
        "models_sha256": {arm: _sha(path) for arm, path in model_paths.items()},
    })
    _assert_immutable(root, output, manifest)
    if evaluate_test:
        evaluate_final_test(final, labels, config, output, backend)
    if _read_json(output / "outer_decision_before_test.json") != audit._json_value(decision):
        raise ValueError("outer decision changed after test reporting")
    result = {"status": "completed", "author": AUTHOR, "source_identity": identity,
              "output_dir": output.relative_to(root).as_posix(), "decision": decision,
              "test_evaluated": evaluate_test, "test_reporting_only": True,
              "selected_procedure": final_selection["procedure"]["selected_procedure_id"],
              "seconds": time.perf_counter() - started,
              "historical_baseline_overwritten": False}
    _write_json(output / "run_info.json", result)
    _progress(output, "completed", **result)
    return result
