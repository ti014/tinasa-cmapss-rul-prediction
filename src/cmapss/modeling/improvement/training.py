"""Huấn luyện theo fold, kiểm tra cache và lưu model."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import time

from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.features.rolling import build_rolling_features, read_silver_observations
from cmapss.modeling.improvement.artifact_io import _json_safe, _write_json
from cmapss.modeling.improvement.config import ImprovementConfig, WEIGHT_SCHEMES
from cmapss.modeling.improvement.dataset import _feature_columns
from cmapss.modeling.metrics import add_prediction_errors


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
