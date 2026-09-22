"""Pipeline Pandas tham chiếu cho ETL Silver/Gold FD001.

Mốc ETL chỉ chuẩn hóa observations, tạo train targets và tổng hợp catalog
engine. Rolling features, predictions và metrics sẽ dùng catalog này ở các
mốc sau; chúng không được suy diễn là đã có từ artifact hiện tại.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cmapss.data.contracts import (
    KEY,
    OBSERVATION_COLUMNS,
    TRAIN_TARGET_COLUMNS,
    ContractError,
)
from cmapss.data.readers import read_observations
from cmapss.data.targets import make_train_targets, validate_train_targets
from cmapss.data.validation import validate_fd001_profile, validate_observations
from cmapss.etl.contracts import (
    ETL_SCHEMA_VERSION,
    GOLD_ENGINE_COLUMNS,
    GOLD_ENGINE_KEY,
    SILVER_OBSERVATION_COLUMNS,
    SILVER_TARGET_COLUMNS,
)


@dataclass(frozen=True)
class PandasETLResult:
    """Các bảng canonical và metadata do pipeline Pandas tạo ra."""

    observations: pd.DataFrame
    train_targets: pd.DataFrame
    engine_summary: pd.DataFrame
    input_sha256: dict[str, str]

    @property
    def row_counts(self) -> dict[str, int]:
        """Trả số dòng của từng boundary ETL."""
        return {
            "observations": int(len(self.observations)),
            "train_targets": int(len(self.train_targets)),
            "engine_summary": int(len(self.engine_summary)),
        }


def _sha256(path: Path) -> str:
    """Tính SHA-256 của một file đầu vào."""
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


def load_fd001(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Đọc và kiểm tra train/test FD001 trước khi tạo artifact Silver.

    Args:
        raw_dir: Thư mục chứa `train_FD001.txt` và `test_FD001.txt`.

    Returns:
        Tuple gồm observations canonical, train targets và SHA-256 input.

    Raises:
        ContractError: Nếu observations hoặc train targets vi phạm contract.
        OSError: Nếu không đọc được raw file.

    Example:
        >>> observations, targets, hashes = load_fd001("data/raw")
        >>> len(targets) == len(observations.query("split == 'train'"))
        True
    """
    raw_path = Path(raw_dir)
    frames: list[pd.DataFrame] = []
    input_sha256: dict[str, str] = {}
    for split in ("train", "test"):
        path = raw_path / f"{split}_FD001.txt"
        frame = read_observations(path, split=split)
        validate_fd001_profile(frame, split=split)
        frames.append(frame)
        input_sha256[path.name] = _sha256(path)

    observations = _canonical_observations(frames)
    train_observations = observations.loc[observations["split"].eq("train")].copy()
    train_targets = make_train_targets(train_observations)
    validate_train_targets(train_observations, train_targets)
    train_targets = train_targets.loc[:, list(SILVER_TARGET_COLUMNS)].reset_index(drop=True)
    return observations, train_targets, input_sha256


def build_engine_summary(observations: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng Gold engine-level từ observations đã kiểm tra.

    Args:
        observations: Bảng observations có schema data contract.

    Returns:
        Một dòng cho mỗi `(split, engine_id)`, canonical theo split rồi ID.

    Raises:
        ContractError: Nếu observations không đạt schema, khóa hoặc cycle.

    Example:
        >>> summary = build_engine_summary(observations)
        >>> {"split", "engine_id", "n_cycles"} <= set(summary.columns)
        True
    """
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


def run_pandas_etl(raw_dir: str | Path) -> PandasETLResult:
    """Chạy toàn bộ ETL Pandas trong bộ nhớ."""
    observations, train_targets, input_sha256 = load_fd001(raw_dir)
    engine_summary = build_engine_summary(observations)
    return PandasETLResult(
        observations=observations,
        train_targets=train_targets,
        engine_summary=engine_summary,
        input_sha256=input_sha256,
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
    with sqlite3.connect(path) as connection:
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


def write_pandas_artifacts(result: PandasETLResult, output_dir: str | Path) -> dict[str, Any]:
    """Ghi Silver Parquet, Gold Parquet/SQLite và metadata lần chạy.

    Args:
        result: Kết quả từ :func:`run_pandas_etl`.
        output_dir: Thư mục artifact ETL, không được nằm trong `data/raw`.

    Returns:
        Metadata JSON-serializable gồm row counts, schema và output paths.

    Raises:
        ValueError: Nếu output nằm trong thư mục raw.
        RuntimeError: Nếu thiếu engine Parquet.
        OSError: Nếu không tạo/ghi được artifact.
    """
    output_path = Path(output_dir).resolve()
    raw_path = Path("data/raw").resolve()
    if output_path == raw_path or raw_path in output_path.parents:
        raise ValueError("ETL output không được ghi vào data/raw để bảo vệ Bronze immutable")

    silver_dir = output_path / "silver"
    gold_dir = output_path / "gold"
    silver_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "silver_observations": silver_dir / "observations.parquet",
        "silver_train_targets": silver_dir / "train_targets.parquet",
        "gold_engine_summary": gold_dir / "engine_summary.parquet",
        "gold_sqlite": gold_dir / "etl.sqlite",
        "run_info": output_path / "run_info.json",
    }
    _write_parquet(result.observations, outputs["silver_observations"])
    _write_parquet(result.train_targets, outputs["silver_train_targets"])
    _write_parquet(result.engine_summary, outputs["gold_engine_summary"])
    _write_sqlite(result, outputs["gold_sqlite"])

    payload: dict[str, Any] = {
        "schema_version": ETL_SCHEMA_VERSION,
        "pipeline": "pandas",
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
            name: str(path.relative_to(output_path))
            for name, path in outputs.items()
            if name != "run_info"
        },
        "rolling_features": {
            "status": "not_implemented",
            "reason": "M3 ETL chỉ tạo engine summary; window 30 sẽ thuộc mốc kế tiếp.",
        },
    }
    outputs["run_info"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    return payload
