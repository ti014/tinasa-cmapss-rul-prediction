"""Regression checks for historical integrity and notebook delivery metadata."""

import hashlib
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from cmapss.etl.scalability import verify_parity
from cmapss.project import AUTHOR_NAME, STUDENT_ID
from cmapss.provenance import verify_source_snapshot


ROOT = Path(__file__).resolve().parents[1]


def test_historical_snapshot_survives_current_edits_but_rejects_tampering(tmp_path):
    run = tmp_path / "run"
    archive = run / "source_snapshot"
    archive.mkdir(parents=True)
    expected = {}
    for name in ("module.py", "uv.lock"):
        (archive / name).write_bytes(b"original")
        (tmp_path / name).write_bytes(b"changed")
        expected[name] = hashlib.sha256(b"original").hexdigest()
    manifest = {"identity": {"source_sha256": {"module.py": expected["module.py"]}, "dependencies_sha256": expected["uv.lock"]}}
    proof = verify_source_snapshot(run, manifest, tmp_path)
    assert proof["recorded_source_verified"] and not proof["working_tree_matches_recorded_source"]
    (archive / "module.py").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="(?i)checksum.*không khớp"):
        verify_source_snapshot(run, manifest, tmp_path)


def test_all_delivered_notebooks_identify_author():
    notebooks = list((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 5
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert AUTHOR_NAME in notebook.cells[0].source, path.name
        assert STUDENT_ID in notebook.cells[0].source, path.name


def test_parity_rejects_changed_feature_even_when_row_count_matches(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    feature = pd.DataFrame({"split": ["train"], "engine_id": [1], "cycle": [30], "sensor_1__mean": [15.5]})
    summary = pd.DataFrame({"split": ["train"], "engine_id": [1], "n_cycles": [30]})
    for folder in (left, right):
        feature.to_parquet(folder / "features.parquet")
        summary.to_parquet(folder / "summary.parquet")
    assert verify_parity(left, right)["features"]["all_cells_match"]
    feature["sensor_1__mean"] = 16.5
    feature.to_parquet(right / "features.parquet")
    with pytest.raises(AssertionError):
        verify_parity(left, right)
