"""Predeclared nested studies for capped targets, history and operating regimes.

Run with ``python -m cmapss.modeling.followup --datasets FD001 FD002 FD004``.
Every run freezes inputs, source and candidates before fitting. Test labels are
read only after the final model and endpoint predictions have been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from importlib.metadata import version
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from cmapss.data.contracts import KEY
from cmapss.features.enhanced import FeatureRecipe, OperatingConditionNormalizer, build_recipe_features
from cmapss.modeling.improvement import BASELINE_PARAMS, _metrics, _select_candidate, _write_json
from cmapss.modeling.improvement_analysis import engine_metrics, paired_comparison
from cmapss.modeling.metrics import add_prediction_errors
from cmapss.modeling.pipeline import engine_equal_sample_weights, evaluate_predictions_from_path, write_final_evaluation_artifacts
from cmapss.project import FOLLOWUP_RUN


@dataclass(frozen=True)
class StudyConfig:
    """Nested evaluation settings frozen in each study manifest."""

    outer_splits: int = 5
    inner_splits: int = 3
    seed: int = 42
    bootstrap_resamples: int = 10_000
    fit_threads: int = 4
    max_iter: int = 200

    def __post_init__(self) -> None:
        if min(self.outer_splits, self.inner_splits) < 2:
            raise ValueError("At least two inner and outer folds are required")
        if min(self.bootstrap_resamples, self.fit_threads, self.max_iter) < 1:
            raise ValueError("Resamples, threads and iterations must be positive")


def study_candidates(dataset_id: str) -> tuple[FeatureRecipe, ...]:
    """Return the complete fixed search space declared before observing results."""
    if dataset_id == "FD001":
        return tuple(
            FeatureRecipe(
                name=("baseline" if cap is None and not history else f"cap_{cap or 'none'}_history_{int(history)}"),
                target_cap=cap,
                history=history,
            )
            for history in (False, True)
            for cap in (None, 110, 125, 140)
        )
    if dataset_id not in {"FD002", "FD004"}:
        raise ValueError("The predeclared study covers FD001, FD002 and FD004")
    return (
        FeatureRecipe("baseline", window=10),
        FeatureRecipe("raw_settings", window=10, settings=True),
        FeatureRecipe("condition_normalized", window=10, settings=True, normalize_conditions=True),
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _join_targets(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    joined = features.merge(targets.loc[:, [*KEY, "rul"]], on=list(KEY), how="left", validate="one_to_one")
    if joined.rul.isna().any():
        raise ValueError("Missing RUL target for a training feature row")
    return joined


class FeatureStore:
    """Cache causal raw representations; learned normalization remains fold-local."""

    def __init__(self, observations: pd.DataFrame, targets: pd.DataFrame) -> None:
        self.observations = observations.loc[observations.split.eq("train")].copy()
        self.targets = targets
        self.raw: dict[tuple[int, bool, bool], pd.DataFrame] = {}

    def features(self, recipe: FeatureRecipe) -> pd.DataFrame:
        """Build a representation with no learned preprocessing once per study."""
        if recipe.normalize_conditions:
            raise ValueError("Normalized features must be constructed inside a fold")
        key = (recipe.window, recipe.history, recipe.settings)
        if key not in self.raw:
            self.raw[key] = _join_targets(build_recipe_features(self.observations, recipe), self.targets)
        return self.raw[key]

    def fold_features(
        self, recipe: FeatureRecipe, train_ids: set[int], validation_ids: set[int], seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, OperatingConditionNormalizer | None]:
        """Fit preprocessing on training IDs and transform the requested engine groups."""
        if train_ids & validation_ids:
            raise ValueError("Training and validation engines overlap")
        normalizer = None
        if recipe.normalize_conditions:
            training = self.observations.loc[self.observations.engine_id.isin(train_ids)]
            normalizer = OperatingConditionNormalizer(seed=seed).fit(training)
            pair = self.observations.loc[self.observations.engine_id.isin(train_ids | validation_ids)]
            frame = _join_targets(build_recipe_features(pair, recipe, normalizer=normalizer), self.targets)
        else:
            frame = self.features(recipe)
        train = frame.loc[frame.engine_id.isin(train_ids)].reset_index(drop=True)
        validation = frame.loc[frame.engine_id.isin(validation_ids)].reset_index(drop=True)
        return train, validation, normalizer


@dataclass
class RecipeModel:
    """Persist the feature recipe, learned preprocessing and fitted regressor together."""

    recipe: FeatureRecipe
    estimator: HistGradientBoostingRegressor
    normalizer: OperatingConditionNormalizer | None
    feature_columns: tuple[str, ...]

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Return one prediction per available trailing window with its original keys."""
        features = build_recipe_features(observations, self.recipe, normalizer=self.normalizer)
        result = features.loc[:, list(KEY)].copy()
        result["model_name"] = self.recipe.name
        result["rul_pred"] = self.estimator.predict(features.loc[:, list(self.feature_columns)])
        if not np.isfinite(result.rul_pred).all():
            raise ValueError("Non-finite prediction")
        return result


def fit_recipe(
    train: pd.DataFrame, recipe: FeatureRecipe, normalizer: OperatingConditionNormalizer | None, config: StudyConfig,
) -> RecipeModel:
    """Fit with capped training labels while keeping the source targets unchanged."""
    columns = tuple(column for column in train if "__" in column)
    target = train.rul.to_numpy(dtype="float64", copy=True)
    if recipe.target_cap is not None:
        target = np.minimum(target, recipe.target_cap)
    params = {**BASELINE_PARAMS, "max_iter": config.max_iter}
    estimator = HistGradientBoostingRegressor(**params, early_stopping=False, random_state=config.seed)
    with threadpool_limits(limits=config.fit_threads):
        estimator.fit(train.loc[:, list(columns)], target, sample_weight=engine_equal_sample_weights(train))
    return RecipeModel(recipe, estimator, normalizer, columns)


def fit_pair(
    store: FeatureStore, recipe: FeatureRecipe, train_ids: set[int], validation_ids: set[int],
    config: StudyConfig, cache: Path, identity: str,
) -> pd.DataFrame:
    """Cache fitted-fold predictions using source, data, config and engine identities."""
    if train_ids & validation_ids:
        raise ValueError("Training and validation engines overlap")
    record = {"identity": identity, "recipe": asdict(recipe), "train_engines": sorted(train_ids), "validation_engines": sorted(validation_ids)}
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    path, metadata = cache / f"{key}.parquet", cache / f"{key}.json"
    if path.is_file() and metadata.is_file():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved["identity_record"] != record or saved["prediction_sha256"] != _digest(path):
            raise ValueError("Fit cache metadata or predictions were modified")
        return pd.read_parquet(path)
    started = time.perf_counter()
    with threadpool_limits(limits=config.fit_threads):
        train, validation, normalizer = store.fold_features(recipe, train_ids, validation_ids, config.seed)
        model = fit_recipe(train, recipe, normalizer, config)
        prediction = model.estimator.predict(validation.loc[:, list(model.feature_columns)])
    result = validation.loc[:, list(KEY)].copy()
    result["y_true"] = validation.rul.to_numpy(dtype="float64")
    result["y_pred"] = prediction
    result["model_name"] = "candidate"
    result = add_prediction_errors(result)
    cache.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    result.to_parquet(temporary, index=False)
    temporary.replace(path)
    _write_json(metadata, {
        "status": "completed", "identity_record": record, "prediction_sha256": _digest(path),
        "seconds": time.perf_counter() - started, "n_features": len(model.feature_columns),
        "normalizer_fit_engines": list(normalizer.fit_engine_ids_) if normalizer else [],
    })
    return result


def select_record(records: list[dict[str, Any]], dataset_id: str) -> int:
    """Use the selection rule recorded in the journal, without consulting outer data."""
    if dataset_id == "FD001":
        baseline = next(record for record in records if record["candidate_id"] == "baseline")
        return _select_candidate(records, baseline)
    return min(range(len(records)), key=lambda i: (records[i]["mean_engine_mae"], records[i]["low_rul_engine_mae"], records[i]["candidate_id"]))


def inner_selection(
    store: FeatureStore, frame: pd.DataFrame, candidates: tuple[FeatureRecipe, ...], dataset_id: str,
    config: StudyConfig, output: Path, identity: str, outer_fold: int,
) -> tuple[FeatureRecipe, pd.DataFrame]:
    """Select from grouped inner predictions and save all candidate scores."""
    splits = list(GroupKFold(config.inner_splits).split(frame, frame.rul, frame.engine_id))
    records: list[dict[str, Any]] = []
    for recipe in candidates:
        print(f"{dataset_id} outer={outer_fold} candidate={recipe.name}", flush=True)
        predictions = [
            fit_pair(store, recipe, set(map(int, frame.iloc[train].engine_id.unique())),
                     set(map(int, frame.iloc[validation].engine_id.unique())), config, output / "fit_cache", identity)
            for train, validation in splits
        ]
        metrics = _metrics(pd.concat(predictions, ignore_index=True), 30)
        records.append({"candidate_id": recipe.name, "outer_fold": outer_fold, **{k: v for k, v in metrics.items() if k != "warning_summary"}})
    selected = select_record(records, dataset_id)
    table = pd.DataFrame(records)
    table["selected"] = False
    table.loc[selected, "selected"] = True
    return candidates[selected], table


def freeze_manifest(
    repo: Path, output: Path, dataset_id: str, observations: Path, targets: Path,
    candidates: tuple[FeatureRecipe, ...], config: StudyConfig,
) -> str:
    """Freeze relevant source and input bytes, and refuse incompatible resumes."""
    source_paths = sorted(path for folder in ("data", "features", "modeling") for path in (repo / "src/cmapss" / folder).glob("*.py"))
    input_paths = [observations, targets]
    if dataset_id == "FD001" and config.outer_splits == 5:
        input_paths.append(repo / "outputs/modeling/tables/fold_assignments.csv")
    identity = {
        "dataset_id": dataset_id, "config": asdict(config), "candidates": [asdict(recipe) for recipe in candidates],
        "inputs": {str(path.relative_to(repo).as_posix()): _digest(path) for path in input_paths},
        "runtime_versions": {package: version(package) for package in ("numpy", "pandas", "scikit-learn", "joblib")},
        "source_sha256": {path.relative_to(repo).as_posix(): _digest(path) for path in source_paths},
        "dependencies_sha256": _digest(repo / "uv.lock"),
    }
    path = output / "search_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("Resume mismatch: source, inputs, config or dependencies changed; use a new run directory")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _write_json(path, {"created_at_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
                           "test_history": "Historical test results were already inspected; this is an exploratory follow-up.",
                           "interval_scope": "Paired engine bootstrap conditional on stored outer predictions; excludes refit variability."})
        for source in [*source_paths, repo / "uv.lock"]:
            target = output / "source_snapshot" / source.relative_to(repo)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def run_study(dataset_id: str, repo: Path, output: Path, config: StudyConfig, *, evaluate_test: bool = True) -> dict[str, Any]:
    """Run one nested study and write a complete, independently inspectable artifact set."""
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


def main() -> None:
    """Execute each requested dataset sequentially with the same frozen protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("FD001", "FD002", "FD004"), default=["FD001", "FD002", "FD004"])
    parser.add_argument("--output-root", type=Path, default=FOLLOWUP_RUN)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-test-evaluation", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = (repo / args.output_root).resolve()
    results = {}
    for dataset in args.datasets:
        results[dataset] = run_study(dataset, repo, output / dataset, StudyConfig(), evaluate_test=not args.skip_test_evaluation)
    _write_json(output / "completion.json", {"datasets": results, "status": "complete"})


if __name__ == "__main__":
    main()
