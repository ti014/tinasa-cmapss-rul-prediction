"""Huấn luyện theo fold, kiểm tra cache và lưu model."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time

from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import joblib
import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.modeling.audit_study.artifact_io import (
    _json_value,
    _sha256,
    _write_json,
    _write_parquet,
)
from cmapss.modeling.audit_study.config import (
    AuditStudyConfig,
    BASE_FEATURES,
    CandidateSpec,
    HISTORY_FEATURES,
    make_candidate_specs,
)
from cmapss.modeling.audit_study.estimators import FittedCandidate, fit_candidate
from cmapss.modeling.audit_study.evaluation import _model_readout, select_candidate
from cmapss.modeling.metrics import add_prediction_errors


def _fit_predict(train: pd.DataFrame, validation: pd.DataFrame, spec: CandidateSpec,
                 config: AuditStudyConfig, cache_dir: Path,
                 source_identity: str) -> pd.DataFrame:
    """Fit một fold và dự đoán validation; cache theo hash dữ liệu, cấu hình và mã nguồn."""
    if set(train.engine_id) & set(validation.engine_id):
        raise ValueError("Động cơ fit và validation giao nhau")
    if not validation.split.eq("train").all():
        raise ValueError("Validation cross-validation chỉ được chứa train")
    identity = hashlib.sha256(source_identity.encode())
    identity.update(json.dumps(_json_value(asdict(spec)), sort_keys=True).encode())
    for frame in (train, validation):
        identity.update(str(tuple(frame.columns)).encode())
        identity.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    cache_key = identity.hexdigest()
    parquet = cache_dir / f"{cache_key}.parquet"
    metadata_path = cache_dir / f"{cache_key}.json"
    if parquet.exists() or metadata_path.exists():
        if not parquet.exists() or not metadata_path.exists():
            # Chuyển cặp file ghi dở sang bên cạnh rồi fit lại khi lần chạy bị ngắt.
            orphan_dir = cache_dir / "interrupted_publications"
            orphan_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.time_ns()
            for orphan in (parquet, metadata_path):
                if orphan.exists():
                    orphan.replace(orphan_dir / f"{stamp}_{orphan.name}")
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (metadata["cache_key"] != cache_key
                    or metadata["source_identity"] != source_identity
                    or metadata["candidate"] != _json_value(asdict(spec))):
                raise ValueError("Định danh metadata fit-cache không khớp")
            if _sha256(parquet) != metadata["prediction_sha256"]:
                raise ValueError("Checksum dự đoán fit-cache không khớp")
            cached = pd.read_parquet(parquet)
            pd.testing.assert_frame_equal(cached[list(KEY)].reset_index(drop=True),
                                          validation[list(KEY)].reset_index(drop=True))
            np.testing.assert_array_equal(cached.y_true, validation.rul)
            if not cached.candidate_id.eq(spec.candidate_id).all():
                raise ValueError("Ứng viên fit-cache không khớp")
            return cached
    started = time.perf_counter()
    fitted = fit_candidate(train, spec, config)
    with threadpool_limits(limits=config.fit_threads):
        prediction = fitted.predict(validation)
    result = validation.loc[:, list(KEY)].copy().reset_index(drop=True)
    result["candidate_id"] = spec.candidate_id
    result["y_true"] = validation.rul.to_numpy(dtype="float64")
    result["y_pred"] = prediction
    result = add_prediction_errors(result)
    _write_parquet(result, parquet)
    _write_json(metadata_path, {
        "status": "completed", "cache_key": cache_key,
        "source_identity": source_identity, "candidate": asdict(spec),
        "prediction_sha256": _sha256(parquet), "seconds": time.perf_counter() - started,
        "fit_engine_ids": sorted(map(int, train.engine_id.unique())),
        "validation_engine_ids": sorted(map(int, validation.engine_id.unique())),
        "selected_feature_columns": list(fitted.feature_columns),
        "n_input_features": len(BASE_FEATURES) + (len(HISTORY_FEATURES) if spec.history else 0),
        "n_retained_features": len(fitted.feature_columns),
    })
    return result


def _inner_evaluation(frame: pd.DataFrame, config: AuditStudyConfig,
                      output: Path, identity: str,
                      outer_fold: int) -> tuple[str, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Đánh giá mọi ứng viên trên cùng các inner fold theo động cơ và chọn một ứng viên."""
    splitter = GroupKFold(n_splits=config.inner_splits)
    folds = list(splitter.split(frame, frame.rul, frame.engine_id))
    assignment_rows = []
    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        train_ids = set(frame.iloc[fit_idx].engine_id)
        val_ids = set(frame.iloc[val_idx].engine_id)
        if train_ids & val_ids:
            raise ValueError("Rò rỉ động cơ ở vòng inner")
        assignment_rows.extend({"outer_fold": outer_fold, "inner_fold": fold,
                                "engine_id": int(engine)} for engine in sorted(val_ids))
    rows, prediction_map = [], {}
    for spec in make_candidate_specs():
        if spec.candidate_id not in config.candidate_ids:
            continue
        print(f"outer{outer_fold}: candidate {spec.candidate_id}", flush=True)
        pieces = []
        for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
            prediction = _fit_predict(frame.iloc[fit_idx].reset_index(drop=True),
                                      frame.iloc[val_idx].reset_index(drop=True), spec,
                                      config, output / "fit_cache", identity)
            prediction = prediction.assign(inner_fold=fold)
            pieces.append(prediction)
        predictions = pd.concat(pieces, ignore_index=True).sort_values(["engine_id", "cycle"])
        prediction_map[spec.candidate_id] = predictions
        rows.append({"outer_fold": outer_fold, **asdict(spec), **_model_readout(predictions, config)})
    table = pd.DataFrame(rows)
    selected = select_candidate(table)
    table["selected"] = table.candidate_id.eq(selected)
    return selected, prediction_map, table, pd.DataFrame(assignment_rows)


def _outer_assignments(frame: pd.DataFrame, config: AuditStudyConfig,
                       repo_root: Path) -> pd.DataFrame:
    """Dùng lại năm fold của pipeline chính; nếu thiếu thì dùng GroupKFold."""
    if config.outer_splits == 5:
        assignments = pd.read_csv(repo_root / "outputs/modeling/tables/fold_assignments.csv")
        assignments = assignments[["engine_id", "fold"]].rename(columns={"fold": "outer_fold"})
    else:
        splitter = GroupKFold(n_splits=config.outer_splits)
        rows = []
        for fold, (_, validation) in enumerate(splitter.split(frame, frame.rul, frame.engine_id), start=1):
            rows.extend({"engine_id": int(engine), "outer_fold": fold}
                        for engine in sorted(frame.iloc[validation].engine_id.unique()))
        assignments = pd.DataFrame(rows)
    if (assignments.engine_id.duplicated().any()
            or set(assignments.engine_id) != set(frame.engine_id)
            or set(assignments.outer_fold) != set(range(1, config.outer_splits + 1))):
        raise ValueError("Outer assignment phải phủ đúng một lần mỗi động cơ")
    return assignments.sort_values("engine_id").reset_index(drop=True)


def _final_predictor(train: pd.DataFrame, spec: CandidateSpec,
                     config: AuditStudyConfig, output: Path,
                     identity: str) -> FittedCandidate:
    """Fit mô hình cuối trên mọi động cơ train hoặc đọc mô hình đã lưu."""
    model_path = output / "selected_model.joblib"
    metadata_path = output / "selected_model.json"
    if model_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata["source_identity"] != identity
                or metadata["candidate"] != _json_value(asdict(spec))
                or metadata["model_sha256"] != _sha256(model_path)):
            raise ValueError("Định danh hoặc checksum mô hình cuối không khớp")
        fitted = joblib.load(model_path)
        if not isinstance(fitted, FittedCandidate) or fitted.spec != spec:
            raise ValueError("Object mô hình cuối không khớp")
        return fitted
    if model_path.exists() or metadata_path.exists():
        interrupted = output / "interrupted_publications"
        interrupted.mkdir(exist_ok=True)
        for orphan in (model_path, metadata_path):
            if orphan.exists():
                orphan.replace(interrupted / f"{time.time_ns()}_{orphan.name}")
    started = time.perf_counter()
    fitted = fit_candidate(train, spec, config)
    temporary = model_path.with_suffix(".tmp.joblib")
    joblib.dump(fitted, temporary)
    temporary.replace(model_path)
    reloaded = joblib.load(model_path)
    with threadpool_limits(limits=config.fit_threads):
        replay_delta = float(np.max(np.abs(fitted.predict(train) - reloaded.predict(train))))
    if replay_delta != 0.0:
        raise ValueError("Mô hình cuối đã lưu không tái tạo dự đoán khớp tuyệt đối")
    _write_json(metadata_path, {
        "source_identity": identity, "candidate": asdict(spec),
        "model_sha256": _sha256(model_path), "seconds": time.perf_counter() - started,
        "selected_feature_columns": list(fitted.feature_columns),
        "fit_engine_ids": sorted(map(int, train.engine_id.unique())),
        "fit_row_count": len(train), "saved_model_replay_max_abs_delta": replay_delta,
        "input_contract": "build_study_features canonical observations; never pass lifetime or RUL as model input",
    })
    return fitted
