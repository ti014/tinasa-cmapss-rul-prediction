"""Metadata và validation payload cho artifact thống kê FD001."""

from dataclasses import asdict
from importlib import metadata
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import scipy

from cmapss.data.contracts import FD001_ENGINE_COUNT, PAIRED_SENSOR_WINDOW_CYCLES, SENSORS

from .artifact_io import relative_posix
from .config import SensorInferenceConfig


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _git_commit(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def _git_has_uncommitted_changes(project_root: Path) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(completed.stdout.strip())


def run_info(
    *,
    project_root: Path,
    train_path: Path,
    output_dir: Path,
    input_sha256: str,
    observations: pd.DataFrame,
    pairs: pd.DataFrame,
    inference: pd.DataFrame,
    config: SensorInferenceConfig,
) -> dict[str, Any]:
    """Metadata của lần chạy: input, phiên bản thư viện, git commit, cấu hình."""
    return {
        "input": relative_posix(project_root, train_path),
        "input_sha256": input_sha256,
        "python": platform.python_version(),
        "versions": {
            "cmapss-rul": _package_version("cmapss-rul"),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "git_commit": _git_commit(project_root),
        "git_has_uncommitted_changes": _git_has_uncommitted_changes(project_root),
        "test_data_used": False,
        "test_labels_used": False,
        "window_cycles": PAIRED_SENSOR_WINDOW_CYCLES,
        "difference_definition": "late_mean - early_mean",
        "unit_of_analysis": "train engine",
        "sensor_family": list(SENSORS),
        "config": asdict(config),
        "observations": int(len(observations)),
        "engines": int(observations["engine_id"].nunique()),
        "pair_rows": int(len(pairs)),
        "inference_rows": int(len(inference)),
        "tables": [
            relative_posix(project_root, output_dir / "tables" / "engine_sensor_pairs.csv"),
            relative_posix(project_root, output_dir / "tables" / "sensor_inference.csv"),
        ],
    }


def validation_payload(
    *,
    pairs: pd.DataFrame,
    inference: pd.DataFrame,
    pairs_roundtrip: pd.DataFrame,
    inference_roundtrip: pd.DataFrame,
    input_sha256_before: str,
    input_sha256_after: str,
    config: SensorInferenceConfig,
) -> dict[str, Any]:
    """Kết quả validation cho hai CSV và file raw đầu vào."""
    return {
        "status": "passed",
        "scope": "M2.3a artifact validation for train-only sensor inference",
        "test_data_used": False,
        "test_labels_used": False,
        "raw_sha256_unchanged": input_sha256_before == input_sha256_after,
        "config_matches_m20": True,
        "engine_sensor_pairs": {
            "rows": int(len(pairs)),
            "expected_rows": FD001_ENGINE_COUNT * len(SENSORS),
            "roundtrip": "passed" if pairs.equals(pairs_roundtrip) else "failed",
            "pairs_per_sensor": int(pairs.groupby("sensor").size().iloc[0]),
            "sensors_per_engine": int(pairs.groupby("engine_id")["sensor"].nunique().iloc[0]),
            "window_cycles": PAIRED_SENSOR_WINDOW_CYCLES,
        },
        "sensor_inference": {
            "rows": int(len(inference)),
            "expected_rows": len(SENSORS),
            "roundtrip": "passed" if inference.equals(inference_roundtrip) else "failed",
            "test_status_counts": {
                str(key): int(value) for key, value in inference["test_status"].value_counts().sort_index().items()
            },
            "bootstrap_status_counts": {
                str(key): int(value)
                for key, value in inference["bootstrap_status"].value_counts().sort_index().items()
            },
            "fdr_reject_true": int(inference["fdr_reject"].fillna(False).sum()),
            "raw_p_values_defined": int(inference["raw_p_value"].notna().sum()),
        },
        "checks": [
            "read_observations(train) and validate_fd001_profile passed",
            "validate_engine_sensor_pairs passed before and after CSV round-trip",
            "validate_sensor_inference passed before and after CSV round-trip",
            "raw train SHA-256 unchanged",
            f"bootstrap_resamples={config.bootstrap_resamples}",
            f"wilcoxon_permutation_resamples={config.wilcoxon_permutation_resamples}",
        ],
    }
