"""Positive/negative proof cho ETL Silver/Gold và parity."""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from cmapss.etl.benchmark import BenchmarkConfig, benchmark_pandas, replicate_observations
from cmapss.etl.contracts import GOLD_ENGINE_COLUMNS
from cmapss.etl.pandas_pipeline import (
    build_engine_summary,
    run_pandas_etl,
    write_pandas_artifacts,
)
from cmapss.etl.parity import ParityError, compare_frames


def test_engine_summary_is_canonical_and_counts(observations: pd.DataFrame) -> None:
    summary = build_engine_summary(observations)
    assert tuple(summary.columns) == GOLD_ENGINE_COLUMNS
    assert summary.to_dict("records") == [
        {"split": "train", "engine_id": 1, "n_cycles": 4, "first_cycle": 1, "last_cycle": 4},
        {"split": "train", "engine_id": 2, "n_cycles": 2, "first_cycle": 1, "last_cycle": 2},
    ]


def test_parity_accepts_reordered_rows_and_rejects_value_change(
    observations: pd.DataFrame,
) -> None:
    candidate = observations.sample(frac=1, random_state=42).reset_index(drop=True)
    result = compare_frames(
        observations,
        candidate,
        name="observations",
        key_columns=("split", "engine_id", "cycle"),
        numeric_columns=tuple(observations.columns[2:]),
    )
    assert result.rows == len(observations)
    changed = candidate.copy()
    changed.loc[0, "sensor_1"] += 1.0
    with pytest.raises(ParityError, match="sensor_1"):
        compare_frames(
            observations,
            changed,
            name="observations",
            key_columns=("split", "engine_id", "cycle"),
            numeric_columns=tuple(observations.columns[2:]),
        )


def test_replicated_benchmark_remaps_engine_keys(observations: pd.DataFrame) -> None:
    scaled = replicate_observations(observations, 3)
    assert len(scaled) == 3 * len(observations)
    assert scaled[["split", "engine_id", "cycle"]].duplicated().sum() == 0
    assert scaled[["split", "engine_id"]].drop_duplicates().shape[0] == 6


def test_benchmark_records_isolated_results(observations: pd.DataFrame) -> None:
    report = benchmark_pandas(observations, BenchmarkConfig(scales=(1,), repetitions=1))
    assert report["scientific_artifacts_isolated"] is True
    assert report["results"][0]["input_rows"] == len(observations)
    assert report["results"][0]["output_rows"] == 2


def test_fd001_pandas_etl_writes_round_trip(raw_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    result = run_pandas_etl(raw_dir)
    payload = write_pandas_artifacts(result, tmp_path / "etl")
    output_dir = tmp_path / "etl"
    observations = pd.read_parquet(output_dir / "silver" / "observations.parquet")
    targets = pd.read_parquet(output_dir / "silver" / "train_targets.parquet")
    summary = pd.read_parquet(output_dir / "gold" / "engine_summary.parquet")
    assert len(observations) == 33_727
    assert len(targets) == 20_631
    assert len(summary) == 200
    assert payload["rolling_features"]["status"] == "separate_artifact"
    with sqlite3.connect(output_dir / "gold" / "etl.sqlite") as connection:
        sqlite_rows = connection.execute("SELECT COUNT(*) FROM engine_summary").fetchone()[0]
    assert sqlite_rows == 200
    assert json.loads((output_dir / "run_info.json").read_text(encoding="utf-8"))["row_counts"] == {
        "observations": 33_727,
        "train_targets": 20_631,
        "engine_summary": 200,
    }
