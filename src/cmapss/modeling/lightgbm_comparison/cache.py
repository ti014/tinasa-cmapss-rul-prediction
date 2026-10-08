"""Fold fitting with an on-disk prediction cache shared across worker processes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.modeling.metrics import add_prediction_errors

from .contracts import FEATURES, StudyConfig, TreeSpec
from .estimators import fit_predictor


def _required_columns() -> tuple[str, ...]:
    """Key, feature and target columns, deduplicated (``cycle`` is in both KEY and FEATURES)."""
    return tuple(dict.fromkeys((*KEY, *FEATURES, "rul")))


def _frame_digest(frame: pd.DataFrame) -> str:
    """Hash column order, dtypes and values."""
    digest = hashlib.sha256()
    digest.update(repr(tuple(frame.columns)).encode("utf-8"))
    digest.update(repr(tuple(str(dtype) for dtype in frame.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _load_frozen_cached(
    path_string: str, identity: str, file_hash: str,
) -> tuple[pd.DataFrame, str]:
    """Read and validate the matrix once per worker; also return its content hash."""
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"frozen feature matrix not found: {path}")
    frame = pd.read_parquet(path)
    required = set(_required_columns())
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"frozen feature matrix misses columns: {missing[:5]}")
    ordered = frame.loc[:, list(_required_columns())].copy()
    if ordered.duplicated(list(KEY)).any():
        raise ValueError("frozen feature matrix contains duplicate observation keys")
    if "split" in ordered and not ordered["split"].eq("train").all():
        raise ValueError("tree comparison worker may read train rows only")
    numeric = ordered.loc[:, list(FEATURES) + ["rul"]].to_numpy(dtype="float64")
    if not np.isfinite(numeric).all() or (ordered["rul"] < 0).any():
        raise ValueError("frozen feature matrix contains nonfinite values or negative RUL")
    if not identity:
        raise ValueError("frozen source identity must be nonempty")
    # file_hash is part of the lru_cache key, so a parquet replaced at the
    # same path is reloaded instead of reusing the old matrix.
    if _file_digest(path) != file_hash:
        raise ValueError("frozen feature matrix changed while loading")
    return ordered, _frame_digest(ordered)


def load_frozen_features(path: Path, identity: str) -> pd.DataFrame:
    """Return a copy of the train matrix from the per-worker cache."""
    resolved = Path(path).resolve()
    frame, _ = _load_frozen_cached(str(resolved), identity, _file_digest(resolved))
    return frame.copy(deep=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    # Short temp name: deep worktrees plus a 64-char key can hit Windows MAX_PATH.
    temporary = path.with_name(f".{uuid4().hex}.tmp.json")
    temporary.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{uuid4().hex}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _quarantine_pair(parquet: Path, metadata: Path) -> None:
    """Move half-written cache files aside instead of deleting them."""
    if not parquet.exists() and not metadata.exists():
        return
    orphan = parquet.parent / "interrupted_publications"
    orphan.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    for path in (parquet, metadata):
        if path.exists():
            shutil.move(str(path), str(orphan / f"{stamp}_{path.name}"))


def _cache_identity(
    source_identity: str,
    source_hash: str,
    spec: TreeSpec,
    config: StudyConfig,
    fit_ids: tuple[int, ...],
    validation_ids: tuple[int, ...],
) -> str:
    payload = {
        "source_identity": source_identity,
        "source_hash": source_hash,
        "spec": asdict(spec),
        "config": asdict(config),
        "fit_engine_ids": sorted(map(int, fit_ids)),
        "validation_engine_ids": sorted(map(int, validation_ids)),
    }
    return hashlib.sha256(json.dumps(_json_value(payload), sort_keys=True).encode("utf-8")).hexdigest()


def _validate_cached(
    prediction: pd.DataFrame,
    validation: pd.DataFrame,
    spec: TreeSpec,
) -> None:
    expected = validation.loc[:, list(KEY)].reset_index(drop=True)
    actual = prediction.loc[:, list(KEY)].reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("fit-cache prediction keys do not match validation rows")
    if not np.array_equal(prediction["y_true"].to_numpy(dtype="float64"), validation["rul"].to_numpy(dtype="float64")):
        raise ValueError("fit-cache raw truth does not match validation RUL")
    if not prediction["candidate_id"].eq(spec.candidate_id).all():
        raise ValueError("fit-cache candidate mismatch")
    if not np.isfinite(prediction[["y_true", "y_pred"]].to_numpy(dtype="float64")).all():
        raise ValueError("fit-cache contains nonfinite prediction")


def run_fit_task(
    feature_path: Path,
    spec: TreeSpec,
    fit_ids: tuple[int, ...],
    validation_ids: tuple[int, ...],
    config: StudyConfig,
    cache_dir: Path,
    identity: str,
) -> str:
    """Fit one candidate on one fold and return the path of the cached predictions.

    ``fit_ids`` and ``validation_ids`` are engine IDs and must not overlap.
    """
    fit_ids = tuple(sorted(set(map(int, fit_ids))))
    validation_ids = tuple(sorted(set(map(int, validation_ids))))
    if not fit_ids or not validation_ids or set(fit_ids) & set(validation_ids):
        raise ValueError("fit and validation engine IDs must be nonempty and disjoint")
    resolved_feature_path = Path(feature_path).resolve()
    matrix, source_hash = _load_frozen_cached(
        str(resolved_feature_path), identity, _file_digest(resolved_feature_path)
    )
    train = matrix[matrix["engine_id"].isin(fit_ids)].reset_index(drop=True)
    validation = matrix[matrix["engine_id"].isin(validation_ids)].reset_index(drop=True)
    if set(train["engine_id"].unique()) != set(fit_ids):
        raise ValueError("fit engine IDs are not fully present in frozen matrix")
    if set(validation["engine_id"].unique()) != set(validation_ids):
        raise ValueError("validation engine IDs are not fully present in frozen matrix")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_identity(identity, source_hash, spec, config, fit_ids, validation_ids)
    parquet = cache_dir / f"{key}.parquet"
    metadata_path = cache_dir / f"{key}.json"
    expected_payload = {
        "cache_key": key,
        "source_identity": identity,
        "source_hash": source_hash,
        "candidate": asdict(spec),
        "config": asdict(config),
        "fit_engine_ids": list(fit_ids),
        "validation_engine_ids": list(validation_ids),
    }
    if parquet.exists() or metadata_path.exists():
        if not parquet.exists() or not metadata_path.exists():
            _quarantine_pair(parquet, metadata_path)
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if any(metadata.get(field) != _json_value(value) for field, value in expected_payload.items()):
                raise ValueError("fit-cache metadata identity mismatch")
            if _sha256(parquet) != metadata.get("prediction_sha256"):
                raise ValueError("fit-cache prediction checksum mismatch")
            cached = pd.read_parquet(parquet)
            _validate_cached(cached, validation, spec)
            return str(parquet)
    started = time.perf_counter()
    fitted = fit_predictor(train, spec, config)
    prediction_values = fitted.predict(validation)
    prediction = validation.loc[:, list(KEY)].copy()
    prediction["candidate_id"] = spec.candidate_id
    prediction["y_true"] = validation["rul"].to_numpy(dtype="float64")
    prediction["y_pred"] = prediction_values
    prediction = add_prediction_errors(prediction)
    _validate_cached(prediction, validation, spec)
    _atomic_parquet(parquet, prediction)
    metadata = {
        **expected_payload,
        "status": "completed",
        "prediction_sha256": _sha256(parquet),
        "seconds": time.perf_counter() - started,
        "n_rows": len(prediction),
        "n_features": len(FEATURES),
        "feature_columns": list(FEATURES),
        "effective_parameters": fitted.effective_parameters,
        "weight_metadata": fitted.weight_metadata,
    }
    _atomic_json(metadata_path, metadata)
    return str(parquet)


__all__ = ["load_frozen_features", "run_fit_task"]
