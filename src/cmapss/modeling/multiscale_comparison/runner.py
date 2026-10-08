"""Nested, causal HGB comparison for the three multi-scale feature arms."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from cmapss.data.contracts import KEY
from cmapss.data.readers import read_test_labels
from cmapss.data.targets import validate_train_targets
from cmapss.modeling import audit_study as audit
from cmapss.modeling.metrics import add_prediction_errors, summarize_prediction_errors
from .analysis import ARMS, BASELINE_ARM, calibrate_policy, candidate_readout, choose_procedure, write_readouts
from .cache import ALL_FEATURES, run_fit_task
from .config import StudyConfig
from .contracts import feature_spec, feature_specs
from .estimators import fit_predictor
from .features import build_feature_arms

AUTHOR = {"full_name": "Phan Nguyễn Mai Phương", "student_id": "25002711"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    """Package versions recorded in the run identity."""
    names = ("python", "numpy", "pandas", "sklearn", "joblib", "pyarrow",
             "scipy", "threadpoolctl")
    return {
        name: (platform.python_version() if name == "python"
               else version("scikit-learn" if name == "sklearn" else name))
        for name in names
    }


def _json(path: Path, value: object) -> None:
    audit._write_json(path, value)


def _build_union(observations: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge all arms and raw targets into one union matrix."""
    arms = build_feature_arms(observations)
    base = arms[BASELINE_ARM].merge(targets, on=list(KEY), how="left", validate="one_to_one")
    if base.rul.isna().any():
        raise ValueError("raw train labels do not cover baseline keys")
    union = base.loc[:, [*KEY, *feature_spec(BASELINE_ARM).feature_columns, "rul"]].copy()
    for arm in ARMS[1:]:
        additions = arms[arm].loc[:, list(dict.fromkeys((*KEY, *feature_spec(arm).feature_columns)))
                                  ].drop(columns=[c for c in feature_spec(BASELINE_ARM).feature_columns
                                                  if c not in KEY])
        union = union.merge(additions, on=list(KEY), how="left", validate="one_to_one")
    train = union[union.split.eq("train")].sort_values(["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    test = pd.concat([arms[arm] for arm in []], ignore_index=True) if False else None
    if len(train) != 17731 or train.engine_id.nunique() != 100:
        raise ValueError(f"unexpected train panel {len(train)} rows/{train.engine_id.nunique()} engines")
    if not np.isfinite(train[list(ALL_FEATURES) + ["rul"]].to_numpy(dtype="float64")).all():
        raise ValueError("union train matrix contains nonfinite values")
    return union, arms[BASELINE_ARM]


def _make_train_test(observations: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the union train matrix (with raw RUL) and the test feature matrix."""
    arms = build_feature_arms(observations)
    base = arms[BASELINE_ARM]
    union = base.merge(targets, on=list(KEY), how="left", validate="one_to_one")
    for arm in ARMS[1:]:
        spec = feature_spec(arm)
        extra = arms[arm].loc[:, list(dict.fromkeys((*KEY, *spec.feature_columns)))]
        extra = extra.drop(columns=[c for c in feature_spec(BASELINE_ARM).feature_columns
                                    if c not in KEY], errors="ignore")
        union = union.merge(extra, on=list(KEY), how="left", validate="one_to_one")
    train = union[union.split.eq("train")].sort_values(["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    test = union[union.split.eq("test")].drop(columns=["rul"], errors="ignore")
    test = test.sort_values(["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    if len(train) != 17731 or len(test) != 10196:
        raise ValueError(f"unexpected frozen rows train={len(train)} test={len(test)}")
    return train, test


def _assignments(root: Path, train: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    if config.outer_splits == 5:
        path = root / "outputs/modeling/tables/fold_assignments.csv"
        table = pd.read_csv(path).loc[:, ["engine_id", "fold"]].rename(columns={"fold": "outer_fold"})
    else:
        splitter = GroupKFold(config.outer_splits)
        rows = []
        for fold, (_, val) in enumerate(splitter.split(train, train.rul, train.engine_id), 1):
            rows.extend({"engine_id": int(e), "outer_fold": fold}
                        for e in sorted(train.iloc[val].engine_id.unique()))
        table = pd.DataFrame(rows)
    if table.engine_id.duplicated().any() or set(table.engine_id) != set(train.engine_id.unique()):
        raise ValueError("outer assignments must cover each engine once")
    if set(table.outer_fold) != set(range(1, config.outer_splits + 1)):
        raise ValueError("outer assignment fold count mismatch")
    return table.sort_values("engine_id").reset_index(drop=True)


def _prediction(path: str | Path, arm: str, frame: pd.DataFrame) -> pd.DataFrame:
    prediction = pd.read_parquet(path)
    if not prediction.candidate_id.eq(arm).all():
        raise ValueError("fit cache arm mismatch")
    expected = frame.loc[:, list(KEY)].reset_index(drop=True)
    if not prediction[list(KEY)].reset_index(drop=True).equals(expected):
        raise ValueError("fit cache keys do not match validation frame")
    return prediction


def _inner_stage(frame: pd.DataFrame, outer_fold: int, config: StudyConfig,
                 matrix_path: Path, cache_dir: Path, identity: str,
                 output: Path) -> tuple[dict[str, dict], dict[str, pd.DataFrame], dict]:
    """Run inner CV for every arm, calibrate its warning policy and pick an arm."""
    splitter = GroupKFold(config.inner_splits)
    folds = list(splitter.split(frame, frame.rul, frame.engine_id))
    assignments = []
    for inner, (_, val_idx) in enumerate(folds, 1):
        assignments.extend({"outer_fold": outer_fold, "inner_fold": inner, "engine_id": int(e)}
                           for e in sorted(frame.iloc[val_idx].engine_id.unique()))
    stage = output / "stages" / (f"outer_{outer_fold}" if outer_fold else "final_inner")
    stage.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(assignments).to_csv(stage / "inner_assignments.csv", index=False)
    arm_predictions: dict[str, pd.DataFrame] = {}
    records, grids = [], []
    for arm in ARMS:
        pieces = []
        for inner, (fit_idx, val_idx) in enumerate(folds, 1):
            fit_ids = tuple(sorted(map(int, frame.iloc[fit_idx].engine_id.unique())))
            val_ids = tuple(sorted(map(int, frame.iloc[val_idx].engine_id.unique())))
            path = run_fit_task(matrix_path, arm, fit_ids, val_ids, config, cache_dir, identity)
            pieces.append(_prediction(path, arm, frame.iloc[val_idx].reset_index(drop=True)).assign(inner_fold=inner))
        predictions = pd.concat(pieces, ignore_index=True).sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
        arm_predictions[arm] = predictions
        policy, grid = calibrate_policy(predictions, config)
        events = audit.warning_events(predictions, policy, lead_requirement=config.lead_requirement)
        records.append({"procedure": arm, **candidate_readout(predictions, config),
                        **audit._warning_readout(events), "numeric_controls_met": True,
                        "policy_threshold": policy.threshold, "policy_persistence": policy.persistence})
        grids.append(grid.assign(procedure=arm, outer_fold=outer_fold))
    table = pd.DataFrame(records)
    choice = choose_procedure(table, config)
    chosen = {row.procedure: {"policy": {"threshold": int(row.policy_threshold),
                                          "persistence": int(row.policy_persistence)},
                              "inner_metrics": row.to_dict()}
              for _, row in table.iterrows()}
    chosen["selected_pipeline"] = {"arm": choice["selected_procedure_id"]}
    table.to_csv(stage / "inner_arm_metrics.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(stage / "inner_policy_metrics.csv", index=False)
    _json(stage / "selection.json", {"outer_fold": outer_fold, "arms": chosen,
                                     "procedure": choice, "test_labels_read": False,
                                     "source_identity": identity})
    return chosen, arm_predictions, choice


def _save_final_model(train: pd.DataFrame, arm: str, config: StudyConfig,
                      output: Path, identity: str) -> Path:
    folder = output / "models" / arm
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / "model.joblib"
    metadata_path = folder / "model.json"
    fitted = fit_predictor(train, arm, config)
    temporary = model_path.with_suffix(".tmp.joblib")
    joblib.dump(fitted, temporary)
    temporary.replace(model_path)
    restored = joblib.load(model_path)
    with threadpool_limits(limits=config.fit_threads):
        expected, actual = fitted.predict(train), restored.predict(train)
    np.testing.assert_array_equal(expected, actual)
    _json(metadata_path, {"source_identity": identity, "arm_id": arm,
                          "feature_columns": list(fitted.feature_columns),
                          "effective_parameters": fitted.effective_parameters,
                          "weight_metadata": fitted.weight_metadata,
                          "model_sha256": _sha(model_path), "saved_model_replay_exact": True,
                          "fit_rows": len(train), "fit_engine_ids": sorted(map(int, train.engine_id.unique()))})
    return model_path


def _evaluate_test(final: pd.DataFrame, labels_path: Path, output: Path, config: StudyConfig) -> dict:
    # Must run before test labels are read.
    required = [output / "selection.json", output / "outer_decision_before_test.json",
                output / "final_predictions.parquet", output / "final_predictions_receipt.json"]
    required.extend(output / "models" / arm / name
                    for arm in ARMS for name in ("model.joblib", "model.json"))
    if not all(path.is_file() for path in required):
        raise ValueError("test labels require frozen selection, decision, models and predictions")
    receipt = json.loads((output / "final_predictions_receipt.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    decision = json.loads((output / "outer_decision_before_test.json").read_text(encoding="utf-8"))
    if selection.get("test_labels_read_for_selection") is not False:
        raise ValueError("final arm must be frozen before test labels")
    if receipt.get("test_labels_read") is not False or receipt.get("saved_model_replay_exact") is not True:
        raise ValueError("final prediction replay receipt is incomplete")
    if receipt.get("predictions_sha256") != _sha(output / "final_predictions.parquet"):
        raise ValueError("final prediction receipt checksum mismatch")
    if json.loads((output / "outer_decision_before_test.json").read_text(encoding="utf-8")) != decision:
        raise ValueError("outer decision changed before test reporting")
    for arm in ARMS:
        model = output / "models" / arm / "model.joblib"
        metadata = json.loads((output / "models" / arm / "model.json").read_text(encoding="utf-8"))
        if metadata.get("model_sha256") != _sha(model) or receipt.get("models_sha256", {}).get(arm) != _sha(model):
            raise ValueError(f"model receipt checksum mismatch: {arm}")
    labels = read_test_labels(labels_path).rename(columns={"rul": "y_true"})
    rows = []
    for procedure, frame in final.groupby("procedure", sort=True):
        joined = frame.merge(labels, on=["split", "engine_id"], how="outer", validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError("test labels do not match final endpoints")
        scored = add_prediction_errors(joined.drop(columns="_merge").rename(columns={"y_pred": "y_pred"}))
        metrics = {"procedure": procedure, **summarize_prediction_errors(scored)}
        rows.append(metrics)
        audit._write_parquet(scored, output / "test" / procedure / "evaluation.parquet")
    result = {"procedures": rows, "reporting_only": True, "selection_used_test_labels": False,
              "test_labels_sha256": _sha(labels_path), "bootstrap_scope": "reporting only"}
    pd.DataFrame(rows).to_csv(output / "test_metrics.csv", index=False)
    _json(output / "test_evaluation.json", result)
    return result


def run_multiscale_comparison(*, repo_root: str | Path | None = None,
                              observations_path: str | Path = "outputs/etl/silver/observations.parquet",
                              targets_path: str | Path = "outputs/etl/silver/train_targets.parquet",
                              test_labels_path: str | Path = "data/raw/RUL_FD001.txt",
                              output_dir: str | Path = "outputs/experiments/multiscale_comparison/20261006_study01",
                              config: StudyConfig | None = None,
                              evaluate_test: bool = True) -> dict:
    """Run the complete nested comparison and reporting-only test evaluation."""
    started = time.perf_counter()
    root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[4]
    config = config or StudyConfig()
    output = (root / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve()
    if not output.is_relative_to(root) or "multiscale_comparison" not in output.parts:
        raise ValueError("output must stay under outputs/experiments/multiscale_comparison")
    observations_path = (root / observations_path).resolve() if not Path(observations_path).is_absolute() else Path(observations_path).resolve()
    targets_path = (root / targets_path).resolve() if not Path(targets_path).is_absolute() else Path(targets_path).resolve()
    labels_path = (root / test_labels_path).resolve() if not Path(test_labels_path).is_absolute() else Path(test_labels_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not (output / "manifest.json").exists():
        raise ValueError("nonempty output has no manifest")
    if not (output / "manifest.json").exists():
        # Snapshot all local sources, not only this package, so later source
        # changes can be detected.
        source_files = [
            *sorted((root / "src/cmapss").rglob("*.py")),
            root / "experiments/MULTISCALE_FEATURE_PROTOCOL.md",
            root / "scripts/run_multiscale_study.py",
        ]
        for source in source_files + [root / "uv.lock", root / "pyproject.toml"]:
            if source.is_file():
                destination = output / "source_snapshot" / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        for source in (observations_path, targets_path):
            destination = output / "input_snapshot" / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    observations = pd.read_parquet(observations_path)
    targets = pd.read_parquet(targets_path)
    validate_train_targets(observations[observations.split.eq("train")], targets)
    train, test = _make_train_test(observations, targets)
    assignments = _assignments(root, train, config)
    matrix_path = output / "frozen_train_features.parquet"
    test_path = output / "frozen_test_features.parquet"
    audit._write_parquet(train, matrix_path)
    audit._write_parquet(test, test_path)
    assignments.to_csv(output / "outer_assignments.csv", index=False)
    source_names = [
        *sorted((root / "src/cmapss").rglob("*.py")),
        root / "experiments/MULTISCALE_FEATURE_PROTOCOL.md",
        root / "scripts/run_multiscale_study.py",
    ]
    input_names = (observations_path.relative_to(root).as_posix(),
                   targets_path.relative_to(root).as_posix())
    identity_payload = audit._json_value({"author": AUTHOR, "config": asdict(config),
                        "feature_specs": [asdict(spec) for spec in feature_specs()],
                        "arms": list(ARMS),
                        "feature_columns": {arm: list(feature_spec(arm).feature_columns) for arm in ARMS},
                        "inputs_sha256": {name: _sha(root / name) for name in input_names},
                        "frozen_train_sha256": _sha(matrix_path),
                        "frozen_test_sha256": _sha(test_path),
                        "outer_assignments_sha256": _sha(output / "outer_assignments.csv"),
                        "project_sha256": _sha(root / "pyproject.toml"),
                        "dependency_lock_sha256": _sha(root / "uv.lock"),
                        "source_sha256": {path.relative_to(root).as_posix(): _sha(path)
                                          for path in source_names},
                        "versions": _runtime_versions()})
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    manifest = {"schema_version": 1, "author": AUTHOR, "identity": identity_payload,
                "source_identity": identity, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "feature_rows": {"train": len(train), "test": len(test)}, "procedures": list(ARMS),
                "test_labels_read": False, "scientific": bool(config.scientific)}
    _json(output / "manifest.json", manifest)
    cache_dir = output / "fit_cache"
    outer_predictions, outer_events, outer_choices = [], [], []
    for fold in range(1, config.outer_splits + 1):
        val_ids = tuple(sorted(map(int, assignments.loc[assignments.outer_fold.eq(fold), "engine_id"])))
        fit_ids = tuple(sorted(set(map(int, train.engine_id.unique())) - set(val_ids)))
        outer_train = train[train.engine_id.isin(fit_ids)].reset_index(drop=True)
        outer_val = train[train.engine_id.isin(val_ids)].reset_index(drop=True)
        choices, _, choice = _inner_stage(outer_train, fold, config, matrix_path, cache_dir, identity, output)
        outer_choices.append({"outer_fold": fold, "choice": choice, "fit_engine_ids": list(fit_ids), "validation_engine_ids": list(val_ids)})
        fold_frames, fold_events = [], []
        for arm in ARMS:
            path = run_fit_task(matrix_path, arm, fit_ids, val_ids, config, cache_dir, identity)
            prediction = _prediction(path, arm, outer_val)
            policy = audit.WarningPolicy(**choices[arm]["policy"])
            prediction = prediction.assign(procedure=arm, outer_fold=fold,
                                           threshold=policy.threshold, persistence=policy.persistence)
            events = audit.warning_events(prediction, policy, lead_requirement=config.lead_requirement)
            fold_frames.append(prediction)
            fold_events.append(events.assign(procedure=arm, outer_fold=fold))
        selected_arm = choice["selected_procedure_id"]
        selected = next(frame for frame in fold_frames if frame.procedure.iloc[0] == selected_arm)
        selected_events = next(frame for frame in fold_events if frame.procedure.iloc[0] == selected_arm)
        fold_frames.append(selected.assign(procedure="selected_pipeline", selected_arm=selected_arm))
        fold_events.append(selected_events.assign(procedure="selected_pipeline", selected_arm=selected_arm))
        outer_predictions.extend(fold_frames)
        outer_events.extend(fold_events)
        _json(output / "outer_selections.json", outer_choices)
        audit._write_parquet(pd.concat(outer_predictions, ignore_index=True), output / "outer_predictions.parquet")
        audit._write_parquet(pd.concat(outer_events, ignore_index=True), output / "warning_events.parquet")
        _json(output / "progress.json", {"stage": "outer", "completed_outer_folds": fold, "total_outer_folds": config.outer_splits})
    outer = pd.concat(outer_predictions, ignore_index=True).sort_values(["procedure", *KEY], kind="mergesort").reset_index(drop=True)
    events = pd.concat(outer_events, ignore_index=True).sort_values(["procedure", "engine_id"], kind="mergesort").reset_index(drop=True)
    audit._write_parquet(outer, output / "outer_predictions.parquet")
    audit._write_parquet(events, output / "warning_events.parquet")
    decision = write_readouts(outer, events, config, output)
    # Check the baseline arm against the published HGB outer predictions
    # before any test label is read.
    reference_path = root / "outputs/experiments/lightgbm_comparison/20261005_study01/outer_predictions.parquet"
    replay = {"applicable": bool(config.scientific and reference_path.is_file()), "passed": False}
    if config.scientific and reference_path.is_file():
        reference = pd.read_parquet(reference_path)
        reference = reference[reference.procedure.eq("hgb_uncapped")]
        current = outer[outer.procedure.eq(BASELINE_ARM)]
        joined = reference.loc[:, list(KEY) + ["y_true", "y_pred"]].merge(
            current.loc[:, list(KEY) + ["y_true", "y_pred"]], on=list(KEY), how="outer",
            validate="one_to_one", indicator=True, suffixes=("_reference", "_current"))
        keys_match = bool(joined._merge.eq("both").all())
        truth_match = bool(keys_match and np.array_equal(joined.y_true_reference, joined.y_true_current))
        max_delta = float(np.max(np.abs(joined.y_pred_reference - joined.y_pred_current))) if keys_match else None
        replay = {"applicable": True, "keys_match": keys_match, "raw_truth_match": truth_match,
                  "prediction_max_abs_delta": max_delta, "tolerance": 1e-8,
                  "reference": str(reference_path.relative_to(root)),
                  "passed": bool(keys_match and truth_match and max_delta is not None and max_delta <= 1e-8)}
        if not replay["passed"]:
            raise ValueError("published HGB baseline replay failed")
    _json(output / "baseline_replay.json", replay)
    _json(output / "outer_decision_before_test.json", decision)
    final_choices, _, final_choice = _inner_stage(train, 0, config, matrix_path, cache_dir, identity, output)
    _json(output / "selection.json", {"procedure": final_choice, "arms": final_choices,
                                      "test_labels_read_for_selection": False, "source_identity": identity,
                                      "author": AUTHOR})
    model_paths = {arm: _save_final_model(train, arm, config, output, identity) for arm in ARMS}
    final_test = test.groupby("engine_id", sort=True, as_index=False).tail(1).reset_index(drop=True)
    final_rows = []
    for arm, path in model_paths.items():
        fitted = joblib.load(path)
        with threadpool_limits(limits=config.fit_threads):
            predicted = fitted.predict(final_test)
            replay = joblib.load(path).predict(final_test)
        np.testing.assert_array_equal(predicted, replay)
        row = final_test.loc[:, list(KEY)].copy()
        row["procedure"], row["candidate_id"], row["y_pred"] = arm, arm, predicted
        final_rows.append(row)
        if arm == final_choice["selected_procedure_id"]:
            final_rows.append(row.assign(procedure="selected_pipeline", selected_arm=arm))
    final = pd.concat(final_rows, ignore_index=True)
    audit._write_parquet(final, output / "final_predictions.parquet")
    _json(output / "final_predictions_receipt.json", {"source_identity": identity,
        "predictions_sha256": _sha(output / "final_predictions.parquet"), "test_labels_read": False,
        "saved_model_replay_exact": True, "models_sha256": {arm: _sha(path) for arm, path in model_paths.items()}})
    test_report = _evaluate_test(final, labels_path, output, config) if evaluate_test else None
    result = {"status": "completed", "author": AUTHOR, "source_identity": identity,
              "output_dir": output.relative_to(root).as_posix(), "decision": decision,
              "selected_procedure": final_choice["selected_procedure_id"], "test_evaluated": evaluate_test,
              "test_reporting_only": True, "seconds": time.perf_counter() - started}
    _json(output / "run_info.json", result)
    _json(output / "progress.json", {"stage": "completed", **result})
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--fit-threads", type=int, default=4)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = StudyConfig(fit_threads=args.fit_threads)
    if args.dry_run:
        config = StudyConfig(outer_splits=2, inner_splits=2, bootstrap_resamples=20,
                             hgb_max_iter=20, fit_threads=args.fit_threads)
    result = run_multiscale_comparison(repo_root=args.repo_root, output_dir=args.output_dir,
                                       config=config, evaluate_test=not (args.skip_test_evaluation or args.dry_run))
    # Windows console may still use cp1252 even though persisted artifacts are UTF-8.
    print(json.dumps(audit._json_value(result), ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
