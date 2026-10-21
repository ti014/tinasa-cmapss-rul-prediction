"""Pipeline Pandas tham chiếu cho ETL Silver/Gold C-MAPSS.

Chuẩn hóa observations, tạo train targets và tổng hợp bảng engine summary. Rolling
features được tạo riêng trong `cmapss.features`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cmapss.artifacts import default_raw_dir, staged_artifact_bundle
from cmapss.data.contracts import (
    DATASET_IDS,
    KEY,
    OBSERVATION_COLUMNS,
    TRAIN_TARGET_COLUMNS,
    ContractError,
)
from cmapss.data.readers import read_observations
from cmapss.data.targets import make_train_targets, validate_train_targets
from cmapss.data.validation import validate_dataset_profile, validate_observations
from cmapss.etl.contracts import (
    ETL_SCHEMA_VERSION,
    GOLD_ENGINE_COLUMNS,
    GOLD_ENGINE_KEY,
    SILVER_OBSERVATION_COLUMNS,
    SILVER_TARGET_COLUMNS,
)

PANDAS_ARTIFACT_PATHS = {
    "silver_observations": Path("silver/observations.parquet"),
    "silver_train_targets": Path("silver/train_targets.parquet"),
    "gold_engine_summary": Path("gold/engine_summary.parquet"),
    "gold_sqlite": Path("gold/etl.sqlite"),
    "run_info": Path("run_info.json"),
}


@dataclass(frozen=True)
class PandasETLResult:
    """Các bảng canonical và metadata do pipeline Pandas tạo ra."""

    observations: pd.DataFrame
    train_targets: pd.DataFrame
    engine_summary: pd.DataFrame
    input_sha256: dict[str, str]
    dataset_id: str = "FD001"
    raw_dir: str | None = None

    @property
    def row_counts(self) -> dict[str, int]:
        """Trả số dòng của từng boundary ETL."""
        return {
            "observations": int(len(self.observations)),
            "train_targets": int(len(self.train_targets)),
            "engine_summary": int(len(self.engine_summary)),
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_observations(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Ghép observations, kiểm tra contract và sắp thứ tự canonical."""
    observations = pd.concat(frames, ignore_index=True)
    validate_observations(observations, source="ETL observations")
    return (
        observations.sort_values(list(KEY), kind="mergesort")
        .reset_index(drop=True)
        .loc[:, list(SILVER_OBSERVATION_COLUMNS)]
    )


def load_dataset(raw_dir: str | Path, dataset_id: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Đọc và kiểm tra `train_<id>.txt`, `test_<id>.txt` của một subset C-MAPSS.

    Trả về (observations canonical, train targets, SHA-256 của từng file input).
    """
    dataset_id = dataset_id.upper()
    if dataset_id not in DATASET_IDS:
        raise ContractError("DATASET_ID", dataset_id, f"dataset_id phải thuộc {DATASET_IDS}")
    raw_path = Path(raw_dir)
    frames: list[pd.DataFrame] = []
    input_sha256: dict[str, str] = {}
    for split in ("train", "test"):
        path = raw_path / f"{split}_{dataset_id}.txt"
        frame = read_observations(path, split=split)
        validate_dataset_profile(frame, dataset_id=dataset_id, split=split)
        frames.append(frame)
        input_sha256[path.name] = _sha256(path)

    observations = _canonical_observations(frames)
    train_observations = observations.loc[observations["split"].eq("train")].copy()
    train_targets = make_train_targets(train_observations)
    validate_train_targets(train_observations, train_targets)
    train_targets = train_targets.loc[:, list(SILVER_TARGET_COLUMNS)].reset_index(drop=True)
    return observations, train_targets, input_sha256


def load_fd001(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Tương thích ngược cho loader FD001 cũ."""
    return load_dataset(raw_dir, "FD001")


def build_engine_summary(observations: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng Gold một dòng cho mỗi `(split, engine_id)`, sắp theo split rồi ID."""
    validate_observations(observations, source="ETL engine summary")
    summary = (
        observations.groupby(list(GOLD_ENGINE_KEY), sort=True, as_index=False)
        .agg(
            n_cycles=("cycle", "size"),
            first_cycle=("cycle", "min"),
            last_cycle=("cycle", "max"),
        )
        .loc[:, list(GOLD_ENGINE_COLUMNS)]
    )
    summary["split"] = summary["split"].astype("string")
    summary = summary.astype(
        {
            "engine_id": "int64",
            "n_cycles": "int64",
            "first_cycle": "int64",
            "last_cycle": "int64",
        }
    )
    return summary.reset_index(drop=True)


def run_pandas_etl(raw_dir: str | Path, dataset_id: str = "FD001") -> PandasETLResult:
    """Chạy toàn bộ ETL Pandas trong bộ nhớ."""
    raw_path = Path(raw_dir).resolve()
    observations, train_targets, input_sha256 = load_dataset(raw_path, dataset_id)
    engine_summary = build_engine_summary(observations)
    return PandasETLResult(
        observations=observations,
        train_targets=train_targets,
        engine_summary=engine_summary,
        input_sha256=input_sha256,
        dataset_id=dataset_id.upper(),
        raw_dir=str(raw_path),
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Ghi Parquet và báo lỗi cài đặt có thể xử lý."""
    try:
        frame.to_parquet(path, index=False)
    except ImportError as error:
        raise RuntimeError(
            "Pandas ETL cần pyarrow để ghi Parquet; cài dependency ETL "
            "trong requirements.txt rồi chạy lại."
        ) from error


def _write_sqlite(result: PandasETLResult, path: Path) -> None:
    """Ghi Gold engine summary và train targets vào SQLite portable."""
    with closing(sqlite3.connect(path)) as connection, connection:
        result.engine_summary.to_sql("engine_summary", connection, if_exists="replace", index=False)
        result.train_targets.to_sql("train_targets", connection, if_exists="replace", index=False)
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_engine_summary_key
            ON engine_summary (split, engine_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_train_targets_key
            ON train_targets (split, engine_id, cycle)
            """
        )


def _json_safe(value: Any) -> Any:
    """Chuyển scalar NumPy/Pandas thành kiểu JSON chuẩn."""
    if hasattr(value, "item"):
        return value.item()
    return value


def write_pandas_artifacts(
    result: PandasETLResult,
    output_dir: str | Path,
    *,
    raw_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Ghi Silver Parquet, Gold Parquet/SQLite và `run_info.json`; trả về metadata.

    `output_dir` không được nằm trong thư mục raw. `raw_dir` dùng cho result tạo thủ
    công; nó được bảo vệ thêm chứ không thay `result.raw_dir`.
    """
    output_path = Path(output_dir).resolve()
    raw_path = Path(result.raw_dir or raw_dir or default_raw_dir()).resolve()
    protected_dirs = [raw_path]
    if raw_dir is not None:
        protected_dirs.append(Path(raw_dir).resolve())
    outputs = PANDAS_ARTIFACT_PATHS

    payload: dict[str, Any] = {
        "schema_version": ETL_SCHEMA_VERSION,
        "pipeline": "pandas",
        "dataset_id": result.dataset_id,
        "raw_dir": str(raw_path),
        "auxiliary_checks": {
            "benchmark": {"status": "not_run"},
            "parity": {"status": "not_run"},
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": result.input_sha256,
        "row_counts": result.row_counts,
        "engine_counts": {
            split: int(result.engine_summary.loc[
                result.engine_summary["split"].eq(split), "engine_id"
            ].nunique())
            for split in ("train", "test")
        },
        "schemas": {
            "silver_observations": list(OBSERVATION_COLUMNS),
            "silver_train_targets": list(TRAIN_TARGET_COLUMNS),
            "gold_engine_summary": list(GOLD_ENGINE_COLUMNS),
        },
        "outputs": {
            name: str(path)
            for name, path in outputs.items()
            if name != "run_info"
        },
        "rolling_features": {
            "status": "separate_artifact",
            "reason": "M3.1 ETL chỉ tạo engine summary; M3.2 ghi rolling features riêng tại outputs/features.",
        },
    }
    with staged_artifact_bundle(
        output_path, tuple(outputs.values()), protected_dirs=protected_dirs,
    ) as stage:
        _write_parquet(result.observations, stage / outputs["silver_observations"])
        _write_parquet(result.train_targets, stage / outputs["silver_train_targets"])
        _write_parquet(result.engine_summary, stage / outputs["gold_engine_summary"])
        _write_sqlite(result, stage / outputs["gold_sqlite"])
        (stage / outputs["run_info"]).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
            encoding="utf-8",
        )
    return payload
