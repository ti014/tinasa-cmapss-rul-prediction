"""Provenance and evaluation-order checks before expensive nested training."""

from dataclasses import replace
import io
import json
import sys

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import MEASUREMENTS, OBSERVATION_COLUMNS
from cmapss.data.targets import make_train_targets
from cmapss.modeling.tree_comparison import runner
from cmapss.modeling.tree_comparison.contracts import FEATURES, StudyConfig


@pytest.fixture
def input_repository(tmp_path, monkeypatch):
    rows = [[split, engine, cycle, *[float(engine + cycle / (offset + 1))
                                    for offset in range(len(MEASUREMENTS))]]
            for split, engines, length in (("train", range(1, 7), 64), ("test", range(1, 3), 44))
            for engine in engines for cycle in range(1, length + 1)]
    observations = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).astype(
        {"split": "string", "engine_id": "int64", "cycle": "int64",
         **dict.fromkeys(MEASUREMENTS, "float64")})
    targets = make_train_targets(observations[observations.split.eq("train")].copy())
    observations_path, targets_path = tmp_path / "observations.parquet", tmp_path / "targets.parquet"
    observations.to_parquet(observations_path, index=False)
    targets.to_parquet(targets_path, index=False)
    source = tmp_path / "src/study.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen source\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_source_files", lambda root: ("src/study.py", "pyproject.toml"))
    output = tmp_path / "outputs/experiments/tree_comparison/smoke"
    config = StudyConfig(outer_splits=2, inner_splits=2, bootstrap_resamples=20,
                         hgb_max_iter=20, xgb_n_estimators=20)
    return tmp_path, output, observations_path, targets_path, config


def test_frozen_matrix_resume_and_tamper_guard(input_repository, monkeypatch):
    args = input_repository
    manifest, identity, train, test, folds = runner._freeze_inputs(*args)
    assert train.columns.is_unique and test.columns.is_unique
    assert train.loc[:, list(FEATURES)].shape == (6 * 35, 169)
    assert "rul" not in test and test.split.eq("test").all()
    assert train.groupby("engine_id").rul.last().eq(0).all()
    assert set(folds.engine_id) == set(train.engine_id)
    assert not args[4].scientific

    def forbid_recompute(*unused):
        raise AssertionError("compatible resume must use frozen bytes")

    monkeypatch.setattr(runner.audit, "build_study_features", forbid_recompute)
    resumed = runner._freeze_inputs(*args)
    assert resumed[0] == runner.audit._json_value(manifest) and resumed[1] == identity
    pd.testing.assert_frame_equal(resumed[2], train, check_exact=True)
    with pytest.raises(ValueError, match="Resume không khớp"):
        runner._freeze_inputs(*args[:4], replace(args[4], seed=43))
    matrix_path = args[1] / "frozen_train_features.parquet"
    changed = pd.read_parquet(matrix_path)
    changed.loc[0, FEATURES[0]] += 1
    changed.to_parquet(matrix_path, index=False)
    with pytest.raises(ValueError, match="Artifact features đã cố định bị đổi"):
        runner._freeze_inputs(*args)


def test_source_change_and_frozen_selection_rejected(input_repository):
    args = input_repository
    runner._freeze_inputs(*args)
    (args[0] / "src/study.py").write_text("# changed source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Resume không khớp"):
        runner._freeze_inputs(*args)
    selection = args[1] / "selection_fixture.json"
    runner._freeze_json(selection, {"candidate_id": "before"})
    with pytest.raises(ValueError, match="Biên nhận đã cố định bị đổi"):
        runner._freeze_json(selection, {"candidate_id": "after"})


def test_missing_final_artifacts_prevent_test_label_access(tmp_path, monkeypatch):
    touched = []

    def labels(path):
        touched.append(path)
        raise AssertionError("test labels were accessed before final freeze")

    monkeypatch.setattr(runner, "read_test_labels", labels)
    with pytest.raises(ValueError, match="trước|thiếu"):
        runner._evaluate_final_test(pd.DataFrame(), tmp_path / "RUL.txt", StudyConfig(), tmp_path)
    assert touched == []


def test_test_prediction_checksum_rejected_before_labels(tmp_path, monkeypatch):
    def labels(path):
        raise AssertionError("test labels must not be read after a checksum failure")

    monkeypatch.setattr(runner, "read_test_labels", labels)
    predictions = pd.DataFrame({"split": ["test"], "engine_id": [1], "cycle": [40], "y_pred": [20.0]})
    predictions.to_parquet(tmp_path / "final_predictions.parquet", index=False)
    runner._write_json(tmp_path / "selection.json", {
        "test_labels_read_for_selection": False, "source_identity": "fixture"})
    runner._write_json(tmp_path / "final_predictions_receipt.json", {
        "source_identity": "fixture", "test_labels_read": False,
        "saved_model_replay_exact": True, "predictions_sha256": "wrong"})
    for arm in runner.ARMS:
        folder = tmp_path / "models" / arm
        folder.mkdir(parents=True)
        (folder / "model.joblib").write_bytes(b"fixture")
        runner._write_json(folder / "model.json", {})
    with pytest.raises(ValueError, match="(?i)checksum"):
        runner._evaluate_final_test(predictions, tmp_path / "RUL.txt", StudyConfig(), tmp_path)


def test_progress_supports_windows_console_and_keeps_utf8_artifact(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    console = io.TextIOWrapper(buffer, encoding="cp1252", write_through=True)
    monkeypatch.setattr(sys, "stdout", console)
    runner._progress(tmp_path, "completed", author=runner.AUTHOR)
    printed = json.loads(buffer.getvalue().decode("cp1252"))
    stored = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert printed["author"] == stored["author"] == runner.AUTHOR
