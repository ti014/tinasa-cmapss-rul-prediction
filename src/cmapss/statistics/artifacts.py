"""Ghi artifact suy luận thống kê sensor từ raw train FD001.

Đọc train, tạo cặp engine--sensor, tính suy luận rồi ghi bảng và metadata đã kiểm tra
round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cmapss.data.readers import read_observations
from cmapss.data.validation import validate_fd001_profile

from .artifact_io import (
    read_engine_sensor_pairs,
    read_sensor_inference,
    resolve_path,
    sha256_file,
    write_json,
)
from .config import SensorInferenceConfig
from .inference import summarize_sensor_changes
from .metadata import run_info, validation_payload
from .pairs import build_engine_sensor_pairs


@dataclass(frozen=True)
class SensorStatisticsArtifacts:
    """Kết quả ghi artifact thống kê FD001."""

    pairs: pd.DataFrame
    inference: pd.DataFrame
    run_info: dict[str, Any]
    validation: dict[str, Any]
    output_dir: Path


def write_sensor_statistics_artifacts(
    *,
    project_root: str | Path = ".",
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "outputs/statistics",
    config: SensorInferenceConfig | None = None,
) -> SensorStatisticsArtifacts:
    """Tạo bảng cặp, bảng suy luận, `run_info.json` và `validation.json`.

    `raw_dir` chứa `train_FD001.txt`. Kết quả ghi vào `output_dir/tables/` và hai file
    JSON ở `output_dir`.
    """
    resolved_config = config or SensorInferenceConfig()
    resolved_config.validate()

    root = Path(project_root).resolve()
    train_path = resolve_path(root, raw_dir) / "train_FD001.txt"
    destination = resolve_path(root, output_dir)
    tables_dir = destination / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    input_sha256_before = sha256_file(train_path)
    observations = read_observations(train_path, split="train")
    validate_fd001_profile(observations, split="train")

    pairs = build_engine_sensor_pairs(observations)
    inference = summarize_sensor_changes(pairs, resolved_config)

    pair_path = tables_dir / "engine_sensor_pairs.csv"
    inference_path = tables_dir / "sensor_inference.csv"
    pairs.to_csv(pair_path, index=False, float_format="%.17g")
    inference.to_csv(inference_path, index=False, float_format="%.17g")

    pairs_roundtrip = read_engine_sensor_pairs(pair_path)
    inference_roundtrip = read_sensor_inference(inference_path, resolved_config)
    input_sha256_after = sha256_file(train_path)

    run_metadata = run_info(
        project_root=root,
        train_path=train_path,
        output_dir=destination,
        input_sha256=input_sha256_before,
        observations=observations,
        pairs=pairs,
        inference=inference,
        config=resolved_config,
    )
    validation = validation_payload(
        pairs=pairs,
        inference=inference,
        pairs_roundtrip=pairs_roundtrip,
        inference_roundtrip=inference_roundtrip,
        input_sha256_before=input_sha256_before,
        input_sha256_after=input_sha256_after,
        config=resolved_config,
    )
    write_json(destination / "run_info.json", run_metadata)
    write_json(destination / "validation.json", validation)

    return SensorStatisticsArtifacts(
        pairs=pairs,
        inference=inference,
        run_info=run_metadata,
        validation=validation,
        output_dir=destination,
    )
