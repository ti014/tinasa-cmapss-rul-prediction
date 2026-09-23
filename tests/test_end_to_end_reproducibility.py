"""Focused checks for the isolated FD001 regeneration command."""

from pathlib import Path

import pandas as pd
import pytest

from cmapss.reproducibility import ReproducibilityError, compare_table, run_end_to_end


def test_compare_table_accepts_same_content_with_different_file_bytes(tmp_path: Path) -> None:
    frame = pd.DataFrame({"engine_id": [1, 2], "rul_pred": [12.25, 19.5]})
    actual = tmp_path / "actual.parquet"
    reference = tmp_path / "reference.parquet"
    frame.to_parquet(actual, index=False, compression="snappy")
    frame.to_parquet(reference, index=False, compression=None)

    result = compare_table(actual, reference, "parquet")

    assert result["content_equal"] is True
    assert result["rows"] == 2
    assert result["actual_sha256"] != result["reference_sha256"]


def test_compare_table_rejects_prediction_drift(tmp_path: Path) -> None:
    actual = tmp_path / "actual.csv"
    reference = tmp_path / "reference.csv"
    pd.DataFrame({"engine_id": [1], "rul_pred": [12.5]}).to_csv(actual, index=False)
    pd.DataFrame({"engine_id": [1], "rul_pred": [12.0]}).to_csv(reference, index=False)

    with pytest.raises(ReproducibilityError, match="Artifact content differs"):
        compare_table(actual, reference, "csv")


def test_end_to_end_requires_fresh_output_root(tmp_path: Path) -> None:
    existing = tmp_path / "already-present"
    existing.mkdir()

    with pytest.raises(ReproducibilityError, match="must not exist"):
        run_end_to_end(existing, repo_root=Path(__file__).resolve().parents[1])
