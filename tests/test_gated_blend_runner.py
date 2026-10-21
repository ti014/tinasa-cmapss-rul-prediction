"""Guard derived studies against leakage, artifact changes and unsafe resume."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import KEY
from cmapss.modeling.gated_blend import runner
from cmapss.modeling.gated_blend.config import StudyConfig, specs
from cmapss.modeling.metrics import add_prediction_errors


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


@pytest.fixture
def guarded_source(tmp_path, monkeypatch):
    root, source = tmp_path, tmp_path / "source"
    files = {
        "src/protected.py": b"protected source\n", "input.parquet": b"train input\n",
        "folds.csv": b"fold keys\n", "uv.lock": b"lock\n", "pyproject.toml": b"project\n",
        "scripts/verify_lightgbm_study.py": b"verification source\n",
    }
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    source.mkdir()
    for name in ("frozen_train_features.parquet", "frozen_test_features.parquet", "outer_assignments.csv"):
        (source / name).write_bytes(name.encode())
    for name, snapshot in (("src/protected.py", "source_snapshot"), ("input.parquet", "input_snapshot"),
                            ("uv.lock", "source_snapshot"), ("pyproject.toml", "source_snapshot")):
        destination = source / snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    config = asdict(StudyConfig())
    candidates = [{"candidate_id": arm, "family": arm, "model_kind": "hgb", "target_cap": cap,
                   "max_depth": None, "learning_rate": .05, "reg_lambda_nominal": 0., "num_leaves": 31}
                  for arm, cap in (("hgb_uncapped", None), ("hgb_cap125", 125))]
    identity = {
        "config": json.loads(json.dumps(config)), "candidates": candidates,
        "source_sha256": {"src/protected.py": runner._sha(root / "src/protected.py")},
        "inputs_sha256": {"input.parquet": runner._sha(root / "input.parquet")},
        "outer_fold_source_sha256": {"folds.csv": runner._sha(root / "folds.csv")},
        "dependency_lock_sha256": runner._sha(root / "uv.lock"),
        "project_sha256": runner._sha(root / "pyproject.toml"),
        "frozen_features_sha256": {name: runner._sha(source / name)
                                    for name in ("frozen_train_features.parquet", "frozen_test_features.parquet")},
        "outer_assignments_sha256": runner._sha(source / "outer_assignments.csv"),
        "feature_columns": [f"feature_{index}" for index in range(169)],
        "versions": {"fixture_runtime": "1"},
    }
    source_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = {"identity": identity, "source_identity": source_id}
    _json(source / "manifest.json", manifest)
    _json(source / "verification.json", {
        "status": "passed", "source_identity": source_id, "checks": {"all_checks_passed": True},
        "model_fit_performed": False, "raw_test_labels_read": False,
        "source_script_sha256": runner._sha(root / "scripts/verify_lightgbm_study.py"),
    })
    _json(source / "run_info.json", {"status": "completed", "source_identity": source_id})
    monkeypatch.setattr(runner, "_versions", lambda: {"fixture_runtime": "1"})
    return root, source, manifest


def test_source_guard_accepts_complete_matching_provenance_without_reading_predictions(guarded_source, monkeypatch):
    root, source, manifest = guarded_source
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: pytest.fail("guard must not read prediction values"))
    assert runner._guard_source(root, source) == manifest


@pytest.mark.parametrize("relative", [
    "src/protected.py", "input.parquet", "uv.lock", "folds.csv",
    "source/source_snapshot/src/protected.py", "source/frozen_train_features.parquet",
])
def test_source_changes_fail_before_selection(guarded_source, relative):
    root, source, _ = guarded_source
    (root / relative).write_bytes(b"changed after verification")
    with pytest.raises(ValueError, match="(?i)checksum"):
        runner._guard_source(root, source)


def test_missing_source_artifact_and_failed_verification_are_rejected(guarded_source):
    root, source, _ = guarded_source
    verification = json.loads((source / "verification.json").read_text())
    verification["checks"]["all_checks_passed"] = False
    _json(source / "verification.json", verification)
    with pytest.raises(ValueError, match="Xác minh"):
        runner._guard_source(root, source)
    (source / "frozen_test_features.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        runner._guard_source(root, source)


def test_existing_output_fails_without_loading_source(tmp_path, monkeypatch):
    output = tmp_path / "outputs/experiments/gated_blend_comparison/old"
    output.mkdir(parents=True)
    witness = output / "witness.txt"
    witness.write_text("retain")
    monkeypatch.setattr(runner, "_guard_source", lambda *args: pytest.fail("existing run must fail first"))
    with pytest.raises(ValueError, match="đã tồn tại"):
        runner.run_gated_blend_study(repo_root=tmp_path, output_dir=output, evaluate_test=False)
    assert witness.read_text() == "retain"


@pytest.fixture
def cached_vector(tmp_path):
    train = pd.DataFrame({"split": ["train"] * 4, "engine_id": [1, 1, 2, 2],
                          "cycle": [30, 31, 30, 31], "rul": [180, 179, 80, 79]})
    selected = train.loc[train.engine_id.eq(2), list(KEY)].reset_index(drop=True)
    selected["y_true"] = [80., 79.]
    selected["y_pred"] = [78., 78.]
    selected["candidate_id"] = "hgb_uncapped"
    prediction = tmp_path / "prediction.parquet"
    add_prediction_errors(selected).to_parquet(prediction, index=False)
    metadata = tmp_path / "prediction.json"
    _json(metadata, {"prediction_sha256": runner._sha(prediction)})
    record = {"prediction_path": "prediction.parquet", "metadata_path": "prediction.json",
              "validation_engine_ids": [2], "candidate_id": "hgb_uncapped"}
    return tmp_path, train, prediction, metadata, record


def test_cache_requires_exact_raw_truth_and_full_row_coverage(cached_vector):
    root, train, prediction, metadata, record = cached_vector
    assert len(runner._load_cached_prediction(root, record, train)) == 2
    broken = pd.read_parquet(prediction)
    broken.loc[0, "y_true"] = 125.
    broken.to_parquet(prediction, index=False)
    _json(metadata, {"prediction_sha256": runner._sha(prediction)})
    with pytest.raises(ValueError, match="nhãn gốc"):
        runner._load_cached_prediction(root, record, train)
    broken.iloc[:1].to_parquet(prediction, index=False)
    _json(metadata, {"prediction_sha256": runner._sha(prediction)})
    with pytest.raises(ValueError, match="khóa validation"):
        runner._load_cached_prediction(root, record, train)


def test_cache_byte_tampering_is_rejected_before_parquet_read(cached_vector, monkeypatch):
    root, train, prediction, _, record = cached_vector
    prediction.write_bytes(b"tampered")
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: pytest.fail("tampered vector must not load"))
    with pytest.raises(ValueError, match="(?i)checksum"):
        runner._load_cached_prediction(root, record, train)


def test_outer_prediction_values_cannot_be_read_before_frozen_choice(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(runner, "_load_cached_prediction", lambda *args: called.append(True))
    with pytest.raises(FileNotFoundError):
        runner._outer_fold(tmp_path, tmp_path, {"cv_records": []}, pd.DataFrame(), {}, {}, 1, StudyConfig())
    assert called == []


def test_full_grid_and_engineering_mode_are_distinct():
    scientific = StudyConfig()
    assert scientific.scientific and len(specs(scientific)) == 16
    smoke = replace(scientific, smoke=True, bootstrap_resamples=20,
                    lower_ruls=(60.,), upper_ruls=(150.,), max_capped_weights=(.25, .5))
    assert not smoke.scientific and len(specs(smoke)) == 2
    assert smoke.outer_splits == 5 and smoke.inner_splits == 3 and smoke.hgb_max_iter == 200
    with pytest.raises(ValueError, match="gốc"):
        replace(scientific, hgb_max_iter=20)


def test_checksum_rejection_precedes_raw_test_label_access(tmp_path, monkeypatch):
    output = tmp_path
    for name in ("final_predictions.parquet", "selection.json", "inference.json", "outer_decision_before_test.json"):
        (output / name).write_text("placeholder")
    _json(output / "final_predictions_receipt.json", {
        "predictions_sha256": "invalid", "selection_sha256": "invalid",
        "inference_sha256": "invalid", "outer_decision_sha256": "invalid",
    })
    monkeypatch.setattr(runner, "read_test_labels", lambda *args: pytest.fail("test labels must remain closed"))
    with pytest.raises(ValueError, match="(?i)checksum"):
        runner._evaluate_final_test(output, pd.DataFrame(), output / "raw_labels.txt")


def test_snapshot_guard_checks_own_protocol_and_source_artifacts(tmp_path):
    root, output = tmp_path, tmp_path / "output"
    mappings = {"source_sha256": {"protocol.md": ""}, "source_artifacts_sha256": {"source/vector": ""}}
    mappings["source_artifact_snapshot_paths"] = runner._artifact_snapshot_paths(mappings["source_artifacts_sha256"])
    for field, snapshot in (("source_sha256", "source_snapshot"),
                            ("source_artifacts_sha256", "source_artifact_snapshot")):
        for name in mappings[field]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            snapshot_name = mappings["source_artifact_snapshot_paths"][name] if field == "source_artifacts_sha256" else name
            destination = output / snapshot / snapshot_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            mappings[field][name] = runner._sha(path)
    for name, field in (("uv.lock", "dependency_lock_sha256"), ("pyproject.toml", "project_sha256")):
        (root / name).write_bytes(name.encode())
        shutil.copy2(root / name, output / "source_snapshot" / name)
        mappings[field] = runner._sha(root / name)
    manifest = {"identity": mappings}
    runner._assert_immutable(root, output, manifest)
    (output / "source_artifact_snapshot" / mappings["source_artifact_snapshot_paths"]["source/vector"]).write_bytes(b"corrupt copied vector")
    with pytest.raises(ValueError, match="bất biến"):
        runner._assert_immutable(root, output, manifest)


def test_long_original_artifact_paths_have_short_deterministic_snapshot_destinations(tmp_path):
    original = ("outputs/experiments/lightgbm_comparison/20261005_study01/source_snapshot/"
                "src/cmapss/modeling/lightgbm_comparison/runner.py")
    representative_output = Path(
        "C:/Users/timewo/.codex/worktrees/windows-linux-rul-replay/Final_Project/"
        "outputs/experiments/gated_blend_comparison/20261005_smoke02")
    assert len(str(representative_output / "source_artifact_snapshot" / original)) > 260
    mapping = runner._artifact_snapshot_paths((original,))
    flattened = representative_output / "source_artifact_snapshot" / mapping[original]
    assert len(str(flattened)) < 260
    assert mapping == runner._artifact_snapshot_paths((original,))
    assert Path(mapping[original]).name == mapping[original]
    assert mapping[original].endswith(".py")
    source_file = tmp_path / "source.py"
    source_file.write_bytes(b"immutable snapshot witness\n")
    destination = tmp_path / "snap" / mapping[original]
    destination.parent.mkdir()
    shutil.copy2(source_file, destination)
    assert runner._sha(destination) == runner._sha(source_file)


def test_snapshot_filename_collisions_fail_before_any_copy(monkeypatch):
    class SameDigest:
        def hexdigest(self):
            return "0" * 64

    monkeypatch.setattr(runner.hashlib, "sha256", lambda *args: SameDigest())
    with pytest.raises(ValueError, match="duy nhất|trùng"):
        runner._artifact_snapshot_paths(("source/first.parquet", "source/second.parquet"))


def test_atomic_json_retries_transient_windows_denial_without_partial_publication(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    _json(path, {"stage": "old"})
    real_replace = runner.os.replace
    calls, delays = [], []

    def transient_replace(source, destination):
        calls.append((source, destination))
        if len(calls) <= 2:
            assert json.loads(path.read_text()) == {"stage": "old"}
            error = PermissionError(13, "temporary reader handle")
            error.winerror = 5
            raise error
        return real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", transient_replace)
    monkeypatch.setattr(runner.time, "sleep", delays.append)
    runner._write_json(path, {"stage": "new", "fresh_fits": 0})
    assert len(calls) == 3 and delays == [.02, .04]
    assert json.loads(path.read_text()) == {"stage": "new", "fresh_fits": 0}
    assert not path.with_suffix(".tmp.json").exists()


def test_atomic_json_propagates_persistent_windows_denial_and_preserves_old_receipt(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    _json(path, {"stage": "old"})
    calls, delays = [], []
    error = PermissionError(13, "persistent denial witness")
    error.winerror = 5

    def denied_replace(source, destination):
        calls.append((source, destination))
        raise error

    monkeypatch.setattr(runner.os, "replace", denied_replace)
    monkeypatch.setattr(runner.time, "sleep", delays.append)
    with pytest.raises(PermissionError, match="persistent denial") as caught:
        runner._write_json(path, {"stage": "new"})
    assert caught.value is error
    assert len(calls) == 8 and len(delays) == 7 and sum(delays) == pytest.approx(1.62)
    assert json.loads(path.read_text()) == {"stage": "old"}
    assert json.loads(path.with_suffix(".tmp.json").read_text()) == {"stage": "new"}


def test_atomic_json_does_not_retry_unrelated_permission_failure(tmp_path, monkeypatch):
    calls, delays = [], []

    def unrelated_replace(source, destination):
        calls.append((source, destination))
        raise PermissionError(13, "ordinary permission failure")

    monkeypatch.setattr(runner.os, "replace", unrelated_replace)
    monkeypatch.setattr(runner.time, "sleep", delays.append)
    with pytest.raises(PermissionError, match="ordinary permission"):
        runner._write_json(tmp_path / "progress.json", {"stage": "new"})
    assert len(calls) == 1 and delays == []

