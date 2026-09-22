"""I/O helpers cho artifact thống kê sensor FD001."""

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd

from cmapss.data.contracts import ENGINE_SENSOR_PAIR_COLUMNS, SENSOR_INFERENCE_COLUMNS

from .config import SensorInferenceConfig
from .pairs import validate_engine_sensor_pairs
from .validation import validate_sensor_inference

_PAIR_INTEGER_COLUMNS = (
    "engine_id",
    "n_cycles",
    "early_start",
    "early_end",
    "late_start",
    "late_end",
    "n_early",
    "n_late",
)
_PAIR_FLOAT_COLUMNS = ("early_mean", "late_mean", "difference")
_INFERENCE_INTEGER_COLUMNS = ("n_pairs", "n_positive", "n_negative", "n_zero", "bootstrap_resamples")
_INFERENCE_FLOAT_COLUMNS = (
    "mean_difference",
    "median_difference",
    "ci_level",
    "ci_lower",
    "ci_upper",
    "wilcoxon_statistic",
    "raw_p_value",
    "p_value_adjusted",
    "fdr_q",
)
_INFERENCE_STRING_COLUMNS = (
    "sensor",
    "ci_method",
    "bootstrap_status",
    "wilcoxon_method",
    "fdr_method",
    "test_status",
)


def resolve_path(project_root: Path, path: str | Path) -> Path:
    """Trả đường dẫn tuyệt đối hoặc đường dẫn tương đối theo project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def relative_posix(project_root: Path, path: Path) -> str:
    """Biểu diễn path tương đối theo POSIX để ghi vào metadata."""
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    """Tính SHA-256 theo chunk để không đọc toàn bộ file vào bộ nhớ."""
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Ghi JSON UTF-8 ổn định cho run metadata và validation."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def read_engine_sensor_pairs(path: Path) -> pd.DataFrame:
    """Đọc lại CSV cặp engine/sensor và validate round-trip."""
    pairs = pd.read_csv(
        path,
        float_precision="round_trip",
        dtype={
            "split": "string",
            "sensor": "string",
            "pair_status": "string",
            **dict.fromkeys(_PAIR_INTEGER_COLUMNS, "int64"),
            **dict.fromkeys(_PAIR_FLOAT_COLUMNS, "float64"),
        },
    )
    pairs = pairs.loc[:, list(ENGINE_SENSOR_PAIR_COLUMNS)]
    validate_engine_sensor_pairs(pairs)
    return pairs


def read_sensor_inference(path: Path, config: SensorInferenceConfig) -> pd.DataFrame:
    """Đọc lại CSV inference và validate round-trip theo cấu hình M2.0."""
    inference = pd.read_csv(
        path,
        float_precision="round_trip",
        dtype={
            **dict.fromkeys(_INFERENCE_STRING_COLUMNS, "string"),
            **dict.fromkeys(_INFERENCE_INTEGER_COLUMNS, "int64"),
            **dict.fromkeys(_INFERENCE_FLOAT_COLUMNS, "float64"),
            "fdr_reject": "boolean",
        },
    )
    inference = inference.loc[:, list(SENSOR_INFERENCE_COLUMNS)]
    validate_sensor_inference(inference, config)
    return inference
