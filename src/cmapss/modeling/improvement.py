"""Nested CV experiment to improve FD001 RUL predictions.

Tunes window, HGB parameters and low-RUL weights in stages. Results go to
``outputs/experiments``; the main model artifacts are not touched.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from cmapss.data.contracts import KEY
from cmapss.features.rolling import build_rolling_features, read_silver_observations
from cmapss.modeling.early_warning import build_warning_events
from cmapss.modeling.metrics import add_prediction_errors
from cmapss.modeling.pipeline import (
    evaluate_predictions_from_path,
    write_final_evaluation_artifacts,
)


WINDOWS = (10, 20, 30)
LOW_RUL_THRESHOLD = 30
BASELINE_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}
WEIGHT_SCHEMES = {
    "uniform": (1.0, 1.0, 1.0),
    "moderate_low_rul": (1.0, 1.5, 2.0),
    "strong_low_rul": (1.0, 2.0, 3.0),
}


@dataclass(frozen=True)
class ImprovementConfig:
    """Settings for one improvement run."""

    seed: int = 42
    outer_splits: int = 5
    inner_splits: int = 3
    low_rul_threshold: int = LOW_RUL_THRESHOLD
    min_cycle: int = 30
    n_hgb_candidates: int = 24
    bootstrap_resamples: int = 10_000
    fit_threads: int = 4

    def __post_init__(self) -> None:
        if not 1 <= self.n_hgb_candidates <= 243:
            raise ValueError("n_hgb_candidates must be between 1 and 243")
        if self.min_cycle != 30 or self.low_rul_threshold != 30:
            raise ValueError("the accepted protocol fixes min_cycle and threshold at 30")
        if min(self.inner_splits, self.outer_splits) < 2:
            raise ValueError("at least two folds are required")
        if self.bootstrap_resamples < 1 or self.fit_threads < 1:
            raise ValueError("resamples and fit_threads must be positive")


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_hgb_candidates(config: ImprovementConfig) -> list[dict[str, Any]]:
    """Baseline params plus random unique HGB configs (seeded)."""
    candidates: list[dict[str, Any]] = [{**BASELINE_PARAMS}]
    rng = np.random.default_rng(config.seed)
    choices = {
        "learning_rate": (0.03, 0.05, 0.1),
        "max_iter": (100, 200, 400),
        "max_leaf_nodes": (15, 31, 63),
        "min_samples_leaf": (10, 20, 40),
        "l2_regularization": (0.0, 0.01, 0.1),
    }
    seen = {tuple(BASELINE_PARAMS.items())}
    while len(candidates) < config.n_hgb_candidates:
        candidate = {
            name: values[int(rng.integers(0, len(values)))]
            for name, values in choices.items()
        }
        key = tuple(candidate.items())
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def low_rul_weights(dataset: pd.DataFrame, scheme: str) -> np.ndarray:
    """Row weights that sum to one per engine.

    The scheme gives factors for RUL > 30, 16..30 and <= 15, in that order.
    """
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"unknown weight scheme: {scheme}")
    high, medium, low = WEIGHT_SCHEMES[scheme]
    rul = dataset["rul"].to_numpy(dtype="float64")
    factors = np.where(rul <= 15, low, np.where(rul <= 30, medium, high))
    totals = pd.Series(factors).groupby(dataset["engine_id"].to_numpy(), sort=False).transform("sum")
    weights = factors / totals.to_numpy(dtype="float64")
    return weights.astype("float64")


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if "__" in column]


def load_window_datasets(
    observations_path: str | Path,
    targets_path: str | Path,
    windows: Iterable[int] = WINDOWS,
    *,
    min_cycle: int = 30,
) -> dict[int, pd.DataFrame]:
    """Train feature tables per window, all starting at ``min_cycle`` so rows match."""
    observations = read_silver_observations(observations_path)
    targets = pd.read_parquet(targets_path)
    datasets: dict[int, pd.DataFrame] = {}
    for window in windows:
        features = build_rolling_features(observations, window_cycles=int(window))
        train = features.loc[
            features["split"].eq("train") & features["cycle"].ge(min_cycle)
        ].copy()
        train = train.merge(
            targets.loc[:, ["split", "engine_id", "cycle", "rul"]],
            on=list(KEY),
            how="left",
            validate="one_to_one",
        )
        if train["rul"].isna().any():
            raise ValueError(f"window {window} has missing train targets")
        datasets[int(window)] = train.reset_index(drop=True)
    keys = [set(frame.loc[:, list(KEY)].itertuples(index=False, name=None)) for frame in datasets.values()]
    if any(key != keys[0] for key in keys[1:]):
        raise ValueError("window datasets do not share the same evaluation keys")
    return datasets


def _fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    params: dict[str, Any],
    weight_scheme: str,
    seed: int,
    cache_dir: Path | None = None,
    fit_threads: int = 4,
) -> pd.DataFrame:
    if set(train.engine_id) & set(validation.engine_id):
        raise ValueError("training and validation engines overlap")
    identity = hashlib.sha256()
    for frame in (train, validation):
        identity.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
        identity.update(str(list(frame.columns)).encode())
    identity.update(json.dumps([params, weight_scheme, seed, fit_threads], sort_keys=True, default=_json_safe).encode())
    cache_key = identity.hexdigest()
    cache_path = cache_dir / (cache_key + ".parquet") if cache_dir else None
    if cache_path is not None and cache_path.exists():
        return pd.read_parquet(cache_path)
    columns = _feature_columns(train)
    model = HistGradientBoostingRegressor(
        **params,
        early_stopping=False,
        random_state=seed,
    )
    # Total weight per engine is one, so long trajectories do not dominate.
    weights = low_rul_weights(train, weight_scheme)
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=fit_threads):
            model.fit(train.loc[:, columns], train["rul"], sample_weight=weights)
            values = model.predict(validation.loc[:, columns]).astype("float64")
    except Exception as error:
        if cache_dir:
            _write_json(cache_dir / (cache_key + ".failed.json"), {"status": "failed", "error": str(error), "params": params})
        raise
    output = validation.loc[:, list(KEY)].copy()
    output["y_true"] = validation["rul"].to_numpy(dtype="float64")
    output["y_pred"] = values
    output["model_name"] = "candidate"
    output = add_prediction_errors(output)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.parquet")
        output.to_parquet(temporary, index=False)
        temporary.replace(cache_path)
        _write_json(cache_path.with_suffix(".json"), {
            "status": "completed", "seconds": time.perf_counter() - started,
            "params": params, "weight_scheme": weight_scheme,
            "train_engines": sorted(map(int, train.engine_id.unique())),
            "validation_engines": sorted(map(int, validation.engine_id.unique())),
        })
    return output


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe) + "\n", encoding="utf-8")
    temporary.replace(path)


def _metrics(predictions: pd.DataFrame, threshold: int) -> dict[str, Any]:
    """Error and warning metrics used for candidate selection."""
    scored = add_prediction_errors(predictions)
    engine_mae = scored.groupby("engine_id", sort=True)["absolute_error"].mean()
    low = scored.loc[scored["y_true"].le(threshold)]
    low_engine_mae = low.groupby("engine_id", sort=True)["absolute_error"].mean()
    event_input = scored.loc[:, ["split", "engine_id", "cycle", "model_name", "y_true", "y_pred"]]
    events, summary = build_warning_events(
        event_input,
        model_name="candidate",
        rul_threshold=threshold,
    )
    late = events["warning_timing"].isin(["after_threshold", "at_failure", "no_warning"])
    burden_values: list[float] = []
    for _, group in scored.groupby("engine_id", sort=True):
        eligible = group.loc[group["y_true"].gt(threshold)]
        if not eligible.empty:
            burden_values.append(float(eligible["y_pred"].le(threshold).mean()))
    return {
        "n_rows": int(len(scored)),
        "n_engines": int(scored["engine_id"].nunique()),
        "mean_engine_mae": float(engine_mae.mean()),
        "rmse": float(np.sqrt(scored["squared_error"].mean())),
        "nasa_score": float(scored["nasa_score_component"].sum()),
        "low_rul_engine_mae": float(low_engine_mae.mean()) if not low_engine_mae.empty else None,
        "n_low_rul_engines": int(low_engine_mae.size),
        "late_or_missed_count": int(late.sum()),
        "late_or_missed_rate": float(late.mean()),
        "early_alarm_burden": float(np.mean(burden_values)) if burden_values else None,
        "n_burden_engines": int(len(burden_values)),
        "warning_summary": summary,
    }


def _aggregate_prediction_metrics(predictions: list[pd.DataFrame], threshold: int) -> dict[str, Any]:
    if not predictions:
        raise ValueError("no predictions to aggregate")
    return _metrics(pd.concat(predictions, ignore_index=True), threshold)


def _inner_splits(frame: pd.DataFrame, config: ImprovementConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = frame["engine_id"].to_numpy()
    splitter = GroupKFold(n_splits=config.inner_splits)
    return list(splitter.split(frame, frame["rul"], groups))


def _evaluate_candidate(
    frame: pd.DataFrame,
    params: dict[str, Any],
    weight_scheme: str,
    config: ImprovementConfig,
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    started = time.perf_counter()
    fold_metrics: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(_inner_splits(frame, config), start=1):
        prediction = _fit_predict(
            frame.iloc[train_index],
            frame.iloc[validation_index],
            params,
            weight_scheme,
            config.seed,
            cache_dir,
            config.fit_threads,
        )
        prediction["fold"] = fold
        predictions.append(prediction)
        metrics = _metrics(prediction, config.low_rul_threshold)
        fold_metrics.append({"fold": fold, **{key: value for key, value in metrics.items() if key != "warning_summary"}})
    aggregate = _aggregate_prediction_metrics(predictions, config.low_rul_threshold)
    aggregate["fit_seconds"] = float(time.perf_counter() - started)
    return aggregate, pd.concat(predictions, ignore_index=True)


def _select_candidate(records: list[dict[str, Any]], baseline: dict[str, Any]) -> int:
    """Pick the lowest low-RUL MAE among candidates no worse than baseline on the controls."""
    eligible = []
    for index, record in enumerate(records):
        controls = (
            record["mean_engine_mae"] <= baseline["mean_engine_mae"]
            and record["late_or_missed_rate"] <= baseline["late_or_missed_rate"]
            and (record["early_alarm_burden"] or 0.0) <= (baseline["early_alarm_burden"] or 0.0)
        )
        if controls:
            eligible.append(index)
    pool = eligible or list(range(len(records)))
    return min(
        pool,
        key=lambda index: (
            records[index]["low_rul_engine_mae"] if records[index]["low_rul_engine_mae"] is not None else float("inf"),
            records[index]["late_or_missed_rate"],
            records[index]["mean_engine_mae"],
            records[index]["candidate_id"],
        ),
    )


def _evaluate_stage(
    frame: pd.DataFrame,
    candidates: list[tuple[str, dict[str, Any], str]],
    config: ImprovementConfig,
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for candidate_id, params, weight_scheme in candidates:
        print(f"  candidate {candidate_id}", flush=True)
        metrics, predictions = _evaluate_candidate(frame, params, weight_scheme, config, cache_dir)
        record = {"candidate_id": candidate_id, "weight_scheme": weight_scheme, **params, **metrics}
        records.append({key: value for key, value in record.items() if key != "warning_summary"})
        predictions["candidate_id"] = candidate_id
        prediction_frames.append(predictions)
    baseline_index = next((index for index, record in enumerate(records) if record["candidate_id"] == "baseline"), 0)
    selected_index = _select_candidate(records, records[baseline_index])
    selected = records[selected_index].copy()
    selected["selected"] = True
    result_table = pd.DataFrame(records)
    result_table["selected"] = False
    result_table.loc[selected_index, "selected"] = True
    result_table["selection_pool"] = "eligible_controls"
    prediction_table = pd.concat(prediction_frames, ignore_index=True)
    return selected, result_table, prediction_table


def _outer_assignments(frame: pd.DataFrame, config: ImprovementConfig) -> pd.DataFrame:
    splitter = GroupKFold(n_splits=config.outer_splits)
    rows: list[dict[str, int]] = []
    for fold, (_, validation_index) in enumerate(
        splitter.split(frame, frame["rul"], frame["engine_id"]), start=1
    ):
        for engine_id in sorted(frame.iloc[validation_index]["engine_id"].unique()):
            rows.append({"engine_id": int(engine_id), "outer_fold": fold})
    return pd.DataFrame(rows).sort_values("engine_id", kind="mergesort").reset_index(drop=True)


def _candidate_groups(
    window: int,
    params: dict[str, Any],
    weight_scheme: str,
) -> tuple[str, dict[str, Any], str]:
    return (f"w{window}_hgb_{weight_scheme}", params, weight_scheme)


def _run_outer_stages(
    datasets: dict[int, pd.DataFrame],
    config: ImprovementConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_frame = datasets[30]
    outer_assignments = pd.read_csv(output_dir / "outer_assignments.csv")
    cache_dir = output_dir / "fit_cache"
    outer_records: list[dict[str, Any]] = []
    outer_predictions: list[pd.DataFrame] = []
    inner_tables: list[pd.DataFrame] = []
    hgb_candidates = make_hgb_candidates(config)
    for outer_fold in range(1, config.outer_splits + 1):
        print(f"outer fold {outer_fold}/{config.outer_splits}", flush=True)
        validation_engines = set(outer_assignments.loc[outer_assignments["outer_fold"].eq(outer_fold), "engine_id"])
        selected_window_records: list[dict[str, Any]] = []
        selected_window_predictions: list[pd.DataFrame] = []
        for window in WINDOWS:
            frame = datasets[window]
            train = frame.loc[~frame["engine_id"].isin(validation_engines)].reset_index(drop=True)
            metrics, predictions = _evaluate_candidate(frame=train, params=BASELINE_PARAMS, weight_scheme="uniform", config=config, cache_dir=cache_dir)
            selected_window_records.append({"candidate_id": f"window_{window}", "window": window, **{k: v for k, v in metrics.items() if k != "warning_summary"}})
            predictions["outer_fold"] = outer_fold
            predictions["window"] = window
            selected_window_predictions.append(predictions)
        # Window 30 is the reference for the control checks.
        window_baseline = next(record for record in selected_window_records if record["window"] == 30)
        selected_window_index = _select_candidate(selected_window_records, window_baseline)
        selected_window = int(selected_window_records[selected_window_index]["window"])
        selected_window_table = pd.DataFrame(selected_window_records)
        selected_window_table["selected"] = selected_window_table["window"].eq(selected_window)
        selected_window_table["stage"] = "windows"
        selected_window_table["outer_fold"] = outer_fold
        inner_tables.append(selected_window_table)

        selected_frame = datasets[selected_window].loc[
            ~datasets[selected_window]["engine_id"].isin(validation_engines)
        ].reset_index(drop=True)
        config_candidates = [
            ("baseline" if index == 0 else f"hgb_{index:02d}", params, "uniform")
            for index, params in enumerate(hgb_candidates)
        ]
        selected_params, parameter_table, parameter_predictions = _evaluate_stage(
            selected_frame,
            config_candidates,
            config,
            cache_dir,
        )
        parameter_table["outer_fold"] = outer_fold
        parameter_table["stage"] = "parameters"
        inner_tables.append(parameter_table)
        best_params = {key: selected_params[key] for key in BASELINE_PARAMS}
        weight_candidates = [
            ("baseline", best_params, "uniform"),
            ("moderate_low_rul", best_params, "moderate_low_rul"),
            ("strong_low_rul", best_params, "strong_low_rul"),
        ]
        selected_weight, weight_table, weight_predictions = _evaluate_stage(
            selected_frame,
            weight_candidates,
            config,
            cache_dir,
        )
        weight_table["outer_fold"] = outer_fold
        weight_table["stage"] = "weights"
        inner_tables.append(weight_table)
        final_params = {key: selected_weight[key] for key in BASELINE_PARAMS}
        final_weight = str(selected_weight["weight_scheme"])
        stage_configs = [
            ("baseline", 30, BASELINE_PARAMS, "uniform"),
            ("windows", selected_window, BASELINE_PARAMS, "uniform"),
            ("parameters", selected_window, best_params, "uniform"),
            ("weights", selected_window, final_params, final_weight),
        ]
        for stage, window, params, weight in stage_configs:
            frame = datasets[window]
            outer_train = frame.loc[~frame.engine_id.isin(validation_engines)].reset_index(drop=True)
            outer_validation = frame.loc[frame.engine_id.isin(validation_engines)].reset_index(drop=True)
            prediction = _fit_predict(outer_train, outer_validation, params, weight, config.seed, cache_dir, config.fit_threads)
            prediction["outer_fold"] = outer_fold
            prediction["stage"] = stage
            prediction["selected_window"] = window
            prediction["weight_scheme"] = weight
            outer_predictions.append(prediction)
            metrics = _metrics(prediction, config.low_rul_threshold)
            outer_records.append({"outer_fold": outer_fold, "stage": stage, "selected_window": window,
                                  "weight_scheme": weight, **params,
                                  **{k: v for k, v in metrics.items() if k != "warning_summary"}})
        pd.DataFrame(outer_records).to_csv(output_dir / "outer_metrics.csv", index=False)
        pd.concat(outer_predictions, ignore_index=True).to_parquet(output_dir / "outer_predictions.parquet", index=False)
        pd.concat(inner_tables, ignore_index=True).to_csv(output_dir / "inner_candidate_metrics.csv", index=False)
        checkpoint = {
            "outer_fold": outer_fold,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_window": selected_window,
            "weight_scheme": final_weight,
        }
        (output_dir / f"checkpoint_outer_{outer_fold}.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    return (
        pd.DataFrame(outer_records),
        pd.concat(outer_predictions, ignore_index=True),
        {"inner_tables": pd.concat(inner_tables, ignore_index=True), "outer_assignments": outer_assignments},
    )


def _select_final_configuration(
    datasets: dict[int, pd.DataFrame],
    config: ImprovementConfig,
    cache_dir: Path | None = None,
) -> tuple[int, dict[str, Any], str, pd.DataFrame]:
    """Select window, HGB parameters and weights with CV on all train engines."""
    window_records: list[dict[str, Any]] = []
    for window in WINDOWS:
        metrics, _ = _evaluate_candidate(datasets[window], BASELINE_PARAMS, "uniform", config, cache_dir)
        window_records.append({"candidate_id": f"window_{window}", "window": window, **{k: v for k, v in metrics.items() if k != "warning_summary"}})
    window_baseline = next(record for record in window_records if record["window"] == 30)
    selected_window = int(window_records[_select_candidate(window_records, window_baseline)]["window"])
    selected_frame = datasets[selected_window]
    hgb_candidates = [
        ("baseline" if index == 0 else f"hgb_{index:02d}", params, "uniform")
        for index, params in enumerate(make_hgb_candidates(config))
    ]
    selected_params, parameter_table, _ = _evaluate_stage(selected_frame, hgb_candidates, config, cache_dir)
    best_params = {key: selected_params[key] for key in BASELINE_PARAMS}
    weight_candidates = [
        ("baseline", best_params, "uniform"),
        ("moderate_low_rul", best_params, "moderate_low_rul"),
        ("strong_low_rul", best_params, "strong_low_rul"),
    ]
    selected_weight, weight_table, _ = _evaluate_stage(selected_frame, weight_candidates, config, cache_dir)
    table = pd.concat(
        [
            pd.DataFrame(window_records).assign(stage="windows", selected=lambda frame: frame["window"].eq(selected_window)),
            parameter_table.assign(stage="parameters"),
            weight_table.assign(stage="weights"),
        ],
        ignore_index=True,
        sort=False,
    )
    return selected_window, {key: selected_weight[key] for key in BASELINE_PARAMS}, str(selected_weight["weight_scheme"]), table


def _final_test_predictions(
    observations_path: str | Path,
    selected_window: int,
    params: dict[str, Any],
    weight_scheme: str,
    train_dataset: pd.DataFrame,
    config: ImprovementConfig,
) -> tuple[pd.DataFrame, HistGradientBoostingRegressor]:
    observations = read_silver_observations(observations_path)
    features = build_rolling_features(observations, window_cycles=selected_window)
    test = features.loc[
        features["split"].eq("test") & features["cycle"].ge(config.min_cycle)
    ].copy()
    final_rows = (
        test.sort_values(list(KEY), kind="mergesort")
        .groupby("engine_id", sort=True, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    columns = _feature_columns(train_dataset)
    model = HistGradientBoostingRegressor(**params, early_stopping=False, random_state=config.seed)
    with threadpool_limits(limits=config.fit_threads):
        model.fit(train_dataset.loc[:, columns], train_dataset["rul"], sample_weight=low_rul_weights(train_dataset, weight_scheme))
        values = model.predict(final_rows.loc[:, columns]).astype("float64")
    result = final_rows.loc[:, list(KEY)].copy()
    result["model_name"] = "improved_candidate"
    result["rul_pred"] = values
    return result, model


def _prepare_run(output: Path, observations: Path, targets: Path, config: ImprovementConfig) -> None:
    """Write the run manifest and baseline snapshot; refuse to resume if inputs changed."""
    import sklearn
    identity = {
        "config": asdict(config),
        "inputs": {str(path): _sha256(path) for path in (observations, targets)},
        "source_sha256": {str(path): _sha256(path) for path in sorted(Path("src/cmapss").rglob("*.py"))},
        "dependencies_sha256": _sha256(Path("uv.lock")),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    manifest_path = output / "search_manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("resume mismatch: config, inputs, source or dependencies changed; use a new output directory")
        for name, digest in saved["baseline_sha256"].items():
            if _sha256(output / "baseline_snapshot" / name) != digest:
                raise ValueError("baseline snapshot checksum mismatch")
        return
    if any(output.iterdir()):
        raise ValueError("output directory is not empty and has no search manifest")
    shutil.copytree("outputs/modeling", output / "baseline_snapshot")
    baseline_hashes = {str(path.relative_to(output / "baseline_snapshot")): _sha256(path)
                       for path in (output / "baseline_snapshot").rglob("*") if path.is_file()}
    git_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (output / "working_tree.patch").write_text(subprocess.check_output(["git", "diff", "HEAD"], text=True, encoding="utf-8"), encoding="utf-8")
    (output / "git_status.txt").write_text(subprocess.check_output(["git", "status", "--short"], text=True, encoding="utf-8"), encoding="utf-8")
    source_dir = output / "source_snapshot"
    source_dir.mkdir()
    for path in Path(__file__).parent.glob("improvement*.py"):
        shutil.copy2(path, source_dir / path.name)
    shutil.copy2("uv.lock", source_dir / "uv.lock")
    _write_json(manifest_path, {
        "identity": identity, "git_revision": git_revision, "baseline_sha256": baseline_hashes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": make_hgb_candidates(config), "weights": WEIGHT_SCHEMES, "windows": WINDOWS,
        "selection_rule": "low-RUL engine MAE, then late/missed rate, overall engine MAE, candidate ID; all three controls no worse than stage baseline",
        "test_history": "FD001 test labels and baseline scores were examined before this study; this is not a pristine test set",
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "fit_threads": config.fit_threads},
    })


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
    from cmapss.modeling.improvement_analysis import write_outer_comparison
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
    from cmapss.modeling.improvement_analysis import write_test_comparison
    write_test_comparison(evaluation, output_path, run_config.bootstrap_resamples, run_config.seed)
    manifest = json.loads((output_path / "run_info.json").read_text(encoding="utf-8"))
    manifest["test_label_joined"] = True
    manifest["final_metrics"] = final_metrics["metrics"]
    manifest["outputs"].update(final_metrics["outputs"])
    _write_json(output_path / "run_info.json", manifest)
    return manifest
