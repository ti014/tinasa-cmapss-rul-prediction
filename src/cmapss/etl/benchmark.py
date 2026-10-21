"""Benchmark ETL Pandas trên dữ liệu nhân bản 1x/10x/100x."""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.data.validation import validate_observations
from cmapss.etl.pandas_pipeline import build_engine_summary


@dataclass(frozen=True)
class BenchmarkConfig:
    """Cấu hình nhân bản và số lần đo."""

    scales: tuple[int, ...] = (1, 10, 100)
    repetitions: int = 2

    def __post_init__(self) -> None:
        if not self.scales or any(scale < 1 for scale in self.scales):
            raise ValueError("scales phải là các số nguyên dương")
        if self.repetitions < 1:
            raise ValueError("repetitions phải >= 1")


def replicate_observations(
    observations: pd.DataFrame,
    scale: int,
) -> pd.DataFrame:
    """Nhân bản observations `scale` lần, mỗi bản có engine ID mới.

    Bản sao chỉ nằm trong bộ nhớ, không ghi ra artifact.
    """
    if scale < 1:
        raise ValueError("scale phải >= 1")
    validate_observations(observations, source="benchmark seed")
    max_engine = int(observations["engine_id"].max())
    parts: list[pd.DataFrame] = []
    for replica in range(scale):
        part = observations.copy(deep=True)
        part["engine_id"] = part["engine_id"] + replica * (max_engine + 1)
        parts.append(part)
    result = (
        pd.concat(parts, ignore_index=True)
        .sort_values(list(KEY), kind="mergesort")
        .reset_index(drop=True)
    )
    validate_observations(result, source=f"benchmark {scale}x")
    return result


def benchmark_pandas(
    observations: pd.DataFrame,
    config: BenchmarkConfig = BenchmarkConfig(),
) -> dict[str, object]:
    """Đo transform engine summary ở mức 1x/10x/100x."""
    records: list[dict[str, object]] = []
    for scale in config.scales:
        scaled = replicate_observations(observations, scale)
        for repetition in range(config.repetitions):
            started = time.perf_counter()
            summary = build_engine_summary(scaled)
            duration = time.perf_counter() - started
            records.append(
                {
                    "engine": "pandas",
                    "scale": scale,
                    "repetition": repetition + 1,
                    "run_kind": "cold" if repetition == 0 else "warm",
                    "input_rows": int(len(scaled)),
                    "input_engines": int(scaled[["split", "engine_id"]].drop_duplicates().shape[0]),
                    "output_rows": int(len(summary)),
                    "duration_seconds": duration,
                }
            )
        del scaled
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_artifacts_isolated": True,
        "source": {
            "rows": int(len(observations)),
            "engines": int(observations[["split", "engine_id"]].drop_duplicates().shape[0]),
        },
        "config": {
            "scales": list(config.scales),
            "repetitions": config.repetitions,
            "transformation": "build_engine_summary",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pandas": pd.__version__,
        },
        "results": records,
    }
