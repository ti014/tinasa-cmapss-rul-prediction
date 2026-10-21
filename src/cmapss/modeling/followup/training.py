"""Huấn luyện theo fold, kiểm tra cache và lưu model."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import hashlib
import json
import time

from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.features.enhanced import FeatureRecipe
from cmapss.modeling.followup.artifact_io import _digest
from cmapss.modeling.followup.config import StudyConfig
from cmapss.modeling.followup.estimators import fit_recipe
from cmapss.modeling.followup.features import FeatureStore
from cmapss.modeling.improvement.artifact_io import _write_json
from cmapss.modeling.improvement.selection import _metrics, _select_candidate
from cmapss.modeling.metrics import add_prediction_errors


def fit_pair(
    store: FeatureStore, recipe: FeatureRecipe, train_ids: set[int], validation_ids: set[int],
    config: StudyConfig, cache: Path, identity: str,
) -> pd.DataFrame:
    """Fit một fold và trả dự đoán validation; cache theo hash thiết lập fold."""
    if train_ids & validation_ids:
        raise ValueError("Động cơ train và validation giao nhau")
    record = {"identity": identity, "recipe": asdict(recipe), "train_engines": sorted(train_ids), "validation_engines": sorted(validation_ids)}
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    path, metadata = cache / f"{key}.parquet", cache / f"{key}.json"
    if path.is_file() and metadata.is_file():
        saved = json.loads(metadata.read_text(encoding="utf-8"))
        if saved["identity_record"] != record or saved["prediction_sha256"] != _digest(path):
            raise ValueError("Metadata hoặc dự đoán fit-cache đã bị sửa")
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
    """Chỉ số ứng viên được chọn chỉ bằng metric inner cross-validation."""
    if dataset_id == "FD001":
        baseline = next(record for record in records if record["candidate_id"] == "baseline")
        return _select_candidate(records, baseline)
    return min(range(len(records)), key=lambda i: (records[i]["mean_engine_mae"], records[i]["low_rul_engine_mae"], records[i]["candidate_id"]))


def inner_selection(
    store: FeatureStore, frame: pd.DataFrame, candidates: tuple[FeatureRecipe, ...], dataset_id: str,
    config: StudyConfig, output: Path, identity: str, outer_fold: int,
) -> tuple[FeatureRecipe, pd.DataFrame]:
    """Đánh giá ứng viên bằng inner GroupKFold rồi chọn một ứng viên."""
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
