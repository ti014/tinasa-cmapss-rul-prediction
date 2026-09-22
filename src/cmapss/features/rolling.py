"""Tạo rolling features từ Silver observations.

M3.2 chỉ tạo đặc trưng từ observations đã kiểm tra. Module này không đọc
`RUL_FD001.txt`, không fit scaler, không chọn sensor theo toàn bộ train và
không huấn luyện mô hình.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cmapss.data.contracts import SENSORS, SPLITS, ContractError
from cmapss.data.validation import validate_columns, validate_integer, validate_observations
from cmapss.features.contracts import (
    FEATURE_SCHEMA_VERSION,
    ROLLING_FEATURE_COLUMNS,
    ROLLING_FEATURE_KEY,
    ROLLING_FEATURE_STATS,
    ROLLING_FEATURE_STD_DDOF,
    ROLLING_WINDOW_CYCLES,
)

_FEATURE_SOURCE = "rolling features"


@dataclass(frozen=True)
class RollingFeatureResult:
    """Bảng rolling features và metadata cấu hình."""

    features: pd.DataFrame
    input_sha256: dict[str, str]
    source_path: str | None = None

    @property
    def row_counts(self) -> dict[str, int]:
        """Trả số dòng feature theo toàn bộ bảng và từng split."""
        return {
            "rolling_features": int(len(self.features)),
            **{
                f"rolling_features_{split}": int(self.features["split"].eq(split).sum())
                for split in SPLITS
            },
        }


def _sha256(path: Path) -> str:
    """Tính SHA-256 của artifact đầu vào."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slope(values: np.ndarray) -> float:
    """Tính linear slope theo vị trí 1..window cho một rolling window."""
    positions = np.arange(1, len(values) + 1, dtype="float64")
    centered_positions = positions - positions.mean()
    denominator = float(np.dot(centered_positions, centered_positions))
    if denominator == 0.0:
        return 0.0
    centered_values = values.astype("float64", copy=False) - float(values.mean())
    return float(np.dot(centered_positions, centered_values) / denominator)


def _require_windowable_observations(
    observations: pd.DataFrame,
    *,
    window_cycles: int,
) -> pd.DataFrame:
    """Validate observations và yêu cầu mỗi engine đủ window."""
    if window_cycles < 2:
        raise ContractError(
            "ROLLING_WINDOW",
            _FEATURE_SOURCE,
            "window_cycles phải >= 2 để tính sample standard deviation và slope",
        )
    validate_observations(observations, source="rolling feature observations")
    counts = observations.groupby(["split", "engine_id"], sort=True)["cycle"].size()
    short = counts.loc[counts < window_cycles]
    if not short.empty:
        split, engine_id = short.index[0]
        raise ContractError(
            "ROLLING_WINDOW",
            f"split={split}, engine_id={int(engine_id)}",
            f"cần ít nhất {window_cycles} cycles để tạo một trailing window đầy đủ",
        )
    return observations.sort_values(list(ROLLING_FEATURE_KEY), kind="mergesort").reset_index(drop=True)


def _build_group_features(group: pd.DataFrame, *, window_cycles: int) -> pd.DataFrame:
    """Tạo features cho một `(split, engine_id)` đã sắp theo cycle."""
    feature_data: dict[str, Any] = {
        "split": group["split"].to_numpy(),
        "engine_id": group["engine_id"].to_numpy(),
        "cycle": group["cycle"].to_numpy(),
    }
    for sensor in SENSORS:
        values = group[sensor].astype("float64")
        rolling = values.rolling(window_cycles, min_periods=window_cycles)
        feature_data[f"{sensor}__last"] = values.to_numpy(dtype="float64")
        feature_data[f"{sensor}__mean"] = rolling.mean().to_numpy(dtype="float64")
        feature_data[f"{sensor}__std"] = rolling.std(ddof=ROLLING_FEATURE_STD_DDOF).to_numpy(
            dtype="float64"
        )
        feature_data[f"{sensor}__min"] = rolling.min().to_numpy(dtype="float64")
        feature_data[f"{sensor}__max"] = rolling.max().to_numpy(dtype="float64")
        feature_data[f"{sensor}__slope"] = rolling.apply(_slope, raw=True).to_numpy(dtype="float64")

    features = pd.DataFrame(feature_data)
    return features.loc[features["cycle"].rank(method="first").gt(window_cycles - 1)]


def build_rolling_features(
    observations: pd.DataFrame,
    *,
    window_cycles: int = ROLLING_WINDOW_CYCLES,
) -> pd.DataFrame:
    """Tạo trailing-window features cho mọi split và engine.

    Mỗi output row dùng đúng `window_cycles` observations kết thúc tại khóa
    `(split, engine_id, cycle)` của chính dòng đó. Window không đi qua engine,
    split hoặc cycle tương lai.

    Args:
        observations: Silver observations đã đạt data contract.
        window_cycles: Độ dài trailing window. M3.2 dùng mặc định 30.

    Returns:
        DataFrame gồm khóa observation và 126 feature columns theo mẫu
        `<sensor>__<stat>` cho 21 sensors và 6 statistics.

    Raises:
        ContractError: Nếu schema observations sai hoặc một engine không đủ
            số cycle để tạo window đầy đủ.

    Example:
        Với sensor bằng cycle và window 30, row tại cycle 30 có `last=30`,
        `mean=15.5`, `min=1`, `max=30` và `slope=1`.
    """
    ordered = _require_windowable_observations(observations, window_cycles=window_cycles)
    frames = [
        _build_group_features(group, window_cycles=window_cycles)
        for _, group in ordered.groupby(["split", "engine_id"], sort=True)
    ]
    features = pd.concat(frames, ignore_index=True)
    features = features.loc[:, list(ROLLING_FEATURE_COLUMNS)].reset_index(drop=True)
    features = features.astype(
        {
            "split": "string",
            "engine_id": "int64",
            "cycle": "int64",
            **{column: "float64" for column in ROLLING_FEATURE_COLUMNS[3:]},
        }
    )
    validate_rolling_features(features, observations=ordered, window_cycles=window_cycles)
    return features


def validate_rolling_features(
    features: pd.DataFrame,
    *,
    observations: pd.DataFrame | None = None,
    window_cycles: int = ROLLING_WINDOW_CYCLES,
) -> None:
    """Kiểm tra schema, khóa, số hữu hạn và row boundaries của rolling features."""
    validate_columns(features, ROLLING_FEATURE_COLUMNS, _FEATURE_SOURCE)
    if not features["split"].isin(SPLITS).all():
        raise ContractError("SPLIT", _FEATURE_SOURCE, "split phải là train hoặc test, không null")
    for column in ("engine_id", "cycle"):
        validate_integer(features, column, 1, _FEATURE_SOURCE)
    if features.duplicated(list(ROLLING_FEATURE_KEY)).any():
        key = features.loc[features.duplicated(list(ROLLING_FEATURE_KEY), keep=False), list(ROLLING_FEATURE_KEY)].iloc[0].to_dict()
        raise ContractError("ROLLING_KEY", _FEATURE_SOURCE, f"trùng khóa {key}")

    ordered = features.sort_values(list(ROLLING_FEATURE_KEY), kind="mergesort").reset_index(drop=True)
    if not ordered[list(ROLLING_FEATURE_KEY)].equals(features[list(ROLLING_FEATURE_KEY)].reset_index(drop=True)):
        raise ContractError(
            "ROLLING_ORDER",
            _FEATURE_SOURCE,
            "features phải theo thứ tự canonical split, engine_id, cycle",
        )

    values = features.loc[:, list(ROLLING_FEATURE_COLUMNS[3:])]
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ContractError(
            "ROLLING_NUMERIC",
            _FEATURE_SOURCE,
            "mọi feature phải là số hữu hạn, không null",
        )

    if observations is None:
        return

    source = _require_windowable_observations(observations, window_cycles=window_cycles)
    expected_keys: list[tuple[object, int, int]] = []
    for (split, engine_id), group in source.groupby(["split", "engine_id"], sort=True):
        cycles = group["cycle"].to_list()
        expected_keys.extend(
            (split, int(engine_id), int(cycle)) for cycle in cycles[window_cycles - 1 :]
        )
    actual_keys = list(features.loc[:, list(ROLLING_FEATURE_KEY)].itertuples(index=False, name=None))
    if actual_keys != expected_keys:
        raise ContractError(
            "ROLLING_WINDOW",
            _FEATURE_SOURCE,
            "khóa output phải là mọi cycle có đủ trailing window trong cùng engine/split",
        )


def read_silver_observations(path: str | Path) -> pd.DataFrame:
    """Đọc Silver observations Parquet và validate trước feature engineering."""
    input_path = Path(path)
    observations = pd.read_parquet(input_path)
    validate_observations(observations, source=str(input_path))
    return observations


def run_rolling_feature_pipeline(silver_observations: str | Path) -> RollingFeatureResult:
    """Đọc Silver observations và tạo rolling features trong bộ nhớ."""
    input_path = Path(silver_observations)
    observations = read_silver_observations(input_path)
    features = build_rolling_features(observations)
    return RollingFeatureResult(
        features=features,
        input_sha256={input_path.name: _sha256(input_path)},
        source_path=str(input_path),
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Ghi Parquet và báo lỗi dependency có thể xử lý."""
    try:
        frame.to_parquet(path, index=False)
    except ImportError as error:
        raise RuntimeError("Rolling features cần pyarrow để ghi Parquet.") from error


def _json_safe(value: Any) -> Any:
    """Chuyển scalar NumPy/Pandas thành kiểu JSON chuẩn."""
    if hasattr(value, "item"):
        return value.item()
    return value


def write_rolling_feature_artifacts(
    result: RollingFeatureResult,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Ghi Gold rolling features và metadata M3.2."""
    output_path = Path(output_dir).resolve()
    raw_path = Path("data/raw").resolve()
    if output_path == raw_path or raw_path in output_path.parents:
        raise ValueError("Feature output không được ghi vào data/raw để bảo vệ Bronze immutable")

    gold_dir = output_path / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "gold_rolling_features": gold_dir / "rolling_features.parquet",
        "run_info": output_path / "run_info.json",
    }
    _write_parquet(result.features, outputs["gold_rolling_features"])

    feature_columns = list(ROLLING_FEATURE_COLUMNS[3:])
    payload: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "pipeline": "pandas_rolling_features",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": result.source_path,
        "input_sha256": result.input_sha256,
        "window_cycles": ROLLING_WINDOW_CYCLES,
        "row_counts": result.row_counts,
        "engine_counts": {
            split: int(result.features.loc[result.features["split"].eq(split), "engine_id"].nunique())
            for split in SPLITS
        },
        "feature_contract": {
            "key_columns": list(ROLLING_FEATURE_KEY),
            "sensors": list(SENSORS),
            "statistics": list(ROLLING_FEATURE_STATS),
            "feature_column_count": len(feature_columns),
            "feature_column_pattern": "<sensor>__<stat>",
            "std_ddof": ROLLING_FEATURE_STD_DDOF,
            "slope_positions": "1..30 within each trailing window",
            "requires_full_window": True,
            "uses_future_observations": False,
            "joins_test_labels": False,
        },
        "outputs": {
            name: str(path.relative_to(output_path))
            for name, path in outputs.items()
            if name != "run_info"
        },
        "modeling": {
            "status": "not_started",
            "reason": "M3.2 chỉ tạo deterministic rolling features; scaler, feature selection và model fit thuộc mốc sau.",
        },
    }
    outputs["run_info"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_safe) + "\n",
        encoding="utf-8",
    )
    return payload
