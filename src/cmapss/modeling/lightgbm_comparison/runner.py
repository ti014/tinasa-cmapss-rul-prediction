"""Run the nested HGB/LightGBM comparison on the shared 169-feature matrix.

Inner folds tune LightGBM and the warning policy; outer folds evaluate the
held-out engines. Test labels are read only after final predictions are saved.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from cmapss.data.readers import read_test_labels
from cmapss.modeling import audit_study as audit
from cmapss.modeling import comparison_runner as common
from cmapss.modeling.comparison_runner import (
    AUTHOR, _assert_immutable, _freeze_json, _progress, _read_json, _resolve,
    _sha, _write_json,
)
from cmapss.modeling.lightgbm_comparison.analysis import (
    calibrate_policy, candidate_readout, choose_procedure, select_lgbm_candidate,
    write_readouts,
)
from cmapss.modeling.lightgbm_comparison.cache import load_frozen_features, run_fit_task
from cmapss.modeling.lightgbm_comparison.contracts import FEATURES, StudyConfig, TreeSpec, specs
from cmapss.modeling.lightgbm_comparison.estimators import fit_predictor

ARMS = ("hgb_uncapped", "hgb_cap125", "lgbm_uncapped", "lgbm_cap125")


def _backend() -> common.ComparisonBackend:
    """Bundle this study's grid, selectors and workers for the shared runner."""
    return common.ComparisonBackend(
        study_name="lightgbm_comparison", arms=ARMS, features=FEATURES,
        specs=specs, active_specs=_active_specs, source_files=_source_files,
        versions=_versions, fit_task=run_fit_task, final_model_task=_final_model_task,
        load_frozen_features=load_frozen_features, fit_predictor=fit_predictor,
        candidate_readout=candidate_readout, select_candidate=select_lgbm_candidate,
        calibrate_policy=calibrate_policy, choose_procedure=choose_procedure,
        write_readouts=write_readouts, read_test_labels=read_test_labels,
    )


def _source_files(root: Path) -> tuple[str, ...]:
    """Source files hashed into the run manifest."""
    return common.source_files(root, "lightgbm_comparison", "scripts/run_lightgbm_study.py")


def _versions() -> dict[str, str]:
    """Package versions to record, including lightgbm."""
    return common.installed_versions("lightgbm")


def _freeze_inputs(
    root: Path, output: Path, observations_path: Path, targets_path: Path,
    config: StudyConfig,
) -> tuple[dict[str, Any], str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return common.freeze_inputs(root, output, observations_path, targets_path,
                                config, _backend())


def _active_specs(config: StudyConfig) -> tuple[TreeSpec, ...]:
    """Full grid, or the explicitly listed smoke candidates."""
    declared = specs(config)
    if config.scientific:
        return declared
    keep = {
        "hgb_uncapped", "hgb_cap125",
        "lgbm_uncapped_leaves31_lr0.05_l20",
        "lgbm_uncapped_leaves31_lr0.05_l210",
        "lgbm_cap125_leaves31_lr0.05_l20",
        "lgbm_cap125_leaves31_lr0.05_l210",
    }
    selected = tuple(item for item in declared if item.candidate_id in keep)
    if len(selected) != len(keep):
        raise ValueError("smoke grid does not match declared candidate IDs")
    return selected


def _fit_batch(
    pool: ProcessPoolExecutor,
    tasks: list[tuple[TreeSpec, tuple[int, ...], tuple[int, ...], int]],
    root: Path, output: Path, manifest: dict[str, Any], identity: str,
    config: StudyConfig, stage: str,
) -> dict[tuple[str, int], str]:
    return common.fit_batch(pool, tasks, root, output, manifest, identity,
                            config, stage, _backend())


def _inner_stage(
    frame: pd.DataFrame, fold: int, pool: ProcessPoolExecutor, root: Path,
    output: Path, manifest: dict[str, Any], identity: str, config: StudyConfig,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame], dict[str, Any]]:
    return common.inner_stage(frame, fold, pool, root, output, manifest,
                              identity, config, _backend())


def _final_model_task(
    feature_path: Path, spec: TreeSpec, config: StudyConfig, folder: Path, identity: str,
) -> str:
    """Defined at module level so spawned worker processes can pickle it."""
    return common.final_model_task(feature_path, spec, config, folder,
                                   identity, _backend())


def _evaluate_final_test(
    predictions: pd.DataFrame, labels_path: Path, config: StudyConfig, output: Path,
) -> dict[str, Any]:
    return common.evaluate_final_test(predictions, labels_path, config,
                                      output, _backend())


def run_lightgbm_comparison(
    *, repo_root: str | Path | None = None,
    observations_path: str | Path = "outputs/etl/silver/observations.parquet",
    targets_path: str | Path = "outputs/etl/silver/train_targets.parquet",
    test_labels_path: str | Path = "data/raw/RUL_FD001.txt",
    output_dir: str | Path = "outputs/experiments/lightgbm_comparison/20261005",
    config: StudyConfig | None = None, evaluate_test: bool = True,
) -> dict[str, Any]:
    """Run the full study and return the run summary.

    Relative paths are resolved against ``repo_root``. An existing compatible
    ``output_dir`` is resumed. ``evaluate_test=False`` skips test scoring.
    """
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[4]
    config = config or StudyConfig()
    output = _resolve(root, output_dir)
    observations, targets, labels = (_resolve(root, path) for path in
                                     (observations_path, targets_path, test_labels_path))
    return common.run_comparison(root, output, observations, targets, labels,
                                 config, evaluate_test, _backend())


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point; ``--dry-run`` uses a small smoke config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--observations", default="outputs/etl/silver/observations.parquet")
    parser.add_argument("--targets", default="outputs/etl/silver/train_targets.parquet")
    parser.add_argument("--test-labels", default="data/raw/RUL_FD001.txt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-threads", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = StudyConfig(fit_threads=args.fit_threads, max_workers=args.max_workers)
    if args.dry_run:
        config = replace(config, outer_splits=2, inner_splits=2, bootstrap_resamples=20,
                         hgb_max_iter=20, lgbm_n_estimators=20)
    run_lightgbm_comparison(repo_root=args.repo_root, observations_path=args.observations,
                        targets_path=args.targets, test_labels_path=args.test_labels,
                        output_dir=args.output_dir, config=config,
                        evaluate_test=not args.skip_test_evaluation)
