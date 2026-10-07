"""Checksum-bound process workers; validation engines never enter fitting."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.modeling.metrics import add_prediction_errors
from cmapss.modeling.gated_blend.runner import _write_json
from cmapss.modeling.audit_study import _write_parquet
from cmapss.modeling.tree_comparison.cache import _frame_digest, _file_digest
from .config import StudyConfig
from .contracts import FEATURE_COLUMNS_BY_ARM, feature_spec
from .estimators import fit_predictor

ALL_FEATURES = tuple(dict.fromkeys(column for columns in FEATURE_COLUMNS_BY_ARM.values() for column in columns))


@lru_cache(maxsize=2)
def _load(path: str, identity: str, file_hash: str) -> tuple[pd.DataFrame, str]:
    """Load immutable union matrix once in each process."""
    if not identity:
        raise ValueError("source identity must be nonempty")
    frame = pd.read_parquet(path)
    columns = list(dict.fromkeys((*KEY, *ALL_FEATURES, "rul")))
    frame = frame.loc[:, columns]
    if frame.empty or frame.duplicated(list(KEY)).any() or not frame.split.eq("train").all():
        raise ValueError("frozen train keys must be unique and train-only")
    if not np.isfinite(frame[list(ALL_FEATURES) + ["rul"]].to_numpy(dtype="float64")).all() or frame.rul.lt(0).any():
        raise ValueError("frozen values must be finite and raw targets nonnegative")
    lifetime_counts = (frame.cycle + frame.rul).groupby(frame.engine_id).nunique()
    if not lifetime_counts.eq(1).all():
        raise ValueError("frozen targets must retain raw engine lifetimes")
    if _file_digest(Path(path)) != file_hash:
        raise ValueError("frozen matrix changed during load")
    return frame, _frame_digest(frame)


def load_frozen_features(path: Path, identity: str) -> pd.DataFrame:
    """Return a private copy, invalidating cached values when bytes change."""
    path = Path(path).resolve()
    return _load(str(path), identity, _file_digest(path))[0].copy(deep=True)


def _validate(predictions: pd.DataFrame, validation: pd.DataFrame, arm_id: str) -> None:
    """Reject cached misalignment, target changes and nonfinite outputs."""
    if not predictions[list(KEY)].reset_index(drop=True).equals(validation[list(KEY)].reset_index(drop=True)):
        raise ValueError("cache validation keys differ")
    if not np.array_equal(predictions.y_true, validation.rul):
        raise ValueError("cache raw truth differs")
    if not predictions.candidate_id.eq(arm_id).all() or not np.isfinite(predictions.y_pred).all():
        raise ValueError("cache candidate/prediction invalid")


def run_fit_task(feature_path: Path, arm_id: str, fit_ids: tuple[int, ...],
                 validation_ids: tuple[int, ...], config: StudyConfig,
                 cache_dir: Path, identity: str) -> str:
    """Fit one fixed arm and cache predictions with exact source/data binding."""
    spec = feature_spec(arm_id)
    if len(set(fit_ids)) != len(fit_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("engine IDs must not repeat")
    fit_ids, validation_ids = tuple(sorted(fit_ids)), tuple(sorted(validation_ids))
    if not fit_ids or not validation_ids or set(fit_ids) & set(validation_ids):
        raise ValueError("fit and validation engines must be disjoint and nonempty")
    path = Path(feature_path).resolve()
    matrix, source_hash = _load(str(path), identity, _file_digest(path))
    fitting = matrix[matrix.engine_id.isin(fit_ids)].reset_index(drop=True)
    validation = matrix[matrix.engine_id.isin(validation_ids)].reset_index(drop=True)
    if set(fitting.engine_id) != set(fit_ids) or set(validation.engine_id) != set(validation_ids):
        raise ValueError("requested engines missing from frozen matrix")
    payload = {"source_identity": identity, "source_hash": source_hash,
               "arm_id": arm_id, "feature_columns": list(spec.feature_columns),
               "config": asdict(config), "fit_engine_ids": list(fit_ids),
               "validation_engine_ids": list(validation_ids)}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet, meta = cache_dir / f"{key}.parquet", cache_dir / f"{key}.json"
    normalized = json.loads(json.dumps(payload))
    if parquet.exists() and meta.exists():
        metadata = json.loads(meta.read_text(encoding="utf-8"))
        if metadata.get("identity") != normalized or metadata.get("prediction_sha256") != _file_digest(parquet):
            raise ValueError("cache identity or prediction checksum changed")
        _validate(pd.read_parquet(parquet), validation, arm_id)
        return str(parquet)
    if parquet.exists() or meta.exists():
        quarantine = cache_dir / "interrupted_publications"
        quarantine.mkdir(exist_ok=True)
        stamp = time.time_ns()
        for artifact in (parquet, meta):
            if artifact.exists():
                artifact.replace(quarantine / f"{stamp}_{artifact.name}")
    started = time.perf_counter()
    fitted = fit_predictor(fitting, arm_id, config)
    prediction = validation[list(KEY)].copy()
    prediction["candidate_id"] = arm_id
    prediction["y_true"] = validation.rul.to_numpy(dtype="float64")
    prediction["y_pred"] = fitted.predict(validation)
    prediction = add_prediction_errors(prediction)
    _validate(prediction, validation, arm_id)
    _write_parquet(prediction, parquet)
    _write_json(meta, {"identity": normalized, "cache_key": key,
                       "prediction_sha256": _file_digest(parquet),
                       "effective_parameters": fitted.effective_parameters,
                       "weight_metadata": fitted.weight_metadata,
                       "seconds": time.perf_counter() - started})
    return str(parquet)


__all__ = ["ALL_FEATURES", "load_frozen_features", "run_fit_task"]
