"""Kiểm chứng Bronze boundary và phục hồi artifact sau lỗi ghi/publish."""

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from cmapss import artifacts
from cmapss.data.targets import make_train_targets
from cmapss.etl import pandas_pipeline as etl
from cmapss.etl import cli as etl_cli
from cmapss.features import rolling


@pytest.fixture
def etl_result(observations: pd.DataFrame, tmp_path: Path) -> etl.PandasETLResult:
    raw = tmp_path / "custom-bronze"
    raw.mkdir()
    (raw / "source.txt").write_text("immutable raw input\n", encoding="utf-8")
    return etl.PandasETLResult(
        observations=observations,
        train_targets=make_train_targets(observations),
        engine_summary=etl.build_engine_summary(observations),
        input_sha256={"source.txt": "old-input-hash"},
        raw_dir=str(raw.resolve()),
    )


def _hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*") if path.is_file()
    }


def _updated(result: etl.PandasETLResult) -> etl.PandasETLResult:
    changed = result.observations.copy()
    changed["sensor_1"] += 100
    return replace(result, observations=changed, input_sha256={"source.txt": "new-input-hash"})


def _feature_result(result: etl.PandasETLResult) -> rolling.RollingFeatureResult:
    return rolling.RollingFeatureResult(
        features=rolling.build_rolling_features(result.observations, window_cycles=2),
        input_sha256={"observations.parquet": "old-silver-hash"},
        window_cycles=2,
        raw_dir=result.raw_dir,
    )


@pytest.mark.parametrize("writer", ["etl", "rolling"])
@pytest.mark.parametrize("child", ["", "generated"])
def test_custom_bronze_is_protected_after_changing_cwd(
    etl_result: etl.PandasETLResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: str,
    child: str,
) -> None:
    raw = Path(etl_result.raw_dir)
    before = _hashes(raw)
    elsewhere = tmp_path / "another-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    output = raw / child
    with pytest.raises(ValueError, match="raw"):
        if writer == "etl":
            etl.write_pandas_artifacts(etl_result, output)
        else:
            rolling.write_rolling_feature_artifacts(_feature_result(etl_result), output)
    assert _hashes(raw) == before
    assert not (raw / "generated").exists()


def test_explicit_raw_override_cannot_disable_original_boundary(
    etl_result: etl.PandasETLResult, tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="raw"):
        etl.write_pandas_artifacts(
            etl_result, Path(etl_result.raw_dir) / "generated", raw_dir=tmp_path / "other-raw",
        )


def test_default_project_bronze_is_protected_from_another_cwd(
    etl_result: etl.PandasETLResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="raw"):
        etl.write_pandas_artifacts(etl_result, artifacts.default_raw_dir() / "generated")


def test_etl_resolves_raw_before_reading(
    etl_result: etl.PandasETLResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    received = []

    def load(raw: Path, dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
        received.append(raw)
        return etl_result.observations, etl_result.train_targets, etl_result.input_sha256

    monkeypatch.setattr(etl, "load_dataset", load)
    result = etl.run_pandas_etl("custom-bronze")
    assert received == [Path(etl_result.raw_dir)]
    assert result.raw_dir == etl_result.raw_dir


def test_rolling_inherits_raw_boundary_and_protects_silver_manifest(
    etl_result: etl.PandasETLResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "etl"
    etl.write_pandas_artifacts(etl_result, output)
    silver = output / "silver" / "observations.parquet"
    result = rolling.run_rolling_feature_pipeline(silver, window_cycles=2)
    assert result.raw_dir == etl_result.raw_dir
    assert result.source_path == str(silver.resolve())
    assert result.source_manifest_path == str((output / "run_info.json").resolve())
    before = _hashes(output)
    monkeypatch.chdir(Path(etl_result.raw_dir))
    with pytest.raises(ValueError, match="raw"):
        rolling.write_rolling_feature_artifacts(result, Path(etl_result.raw_dir) / "features")
    with pytest.raises(ValueError, match="input"):
        rolling.write_rolling_feature_artifacts(result, output)
    assert _hashes(output) == before


def test_legacy_silver_accepts_explicit_custom_raw(
    etl_result: etl.PandasETLResult, tmp_path: Path,
) -> None:
    silver = tmp_path / "legacy-silver.parquet"
    etl_result.observations.to_parquet(silver, index=False)
    result = rolling.run_rolling_feature_pipeline(
        silver, window_cycles=2, raw_dir=etl_result.raw_dir,
    )
    assert result.raw_dir == etl_result.raw_dir
    assert result.source_manifest_path is None
    with pytest.raises(ValueError, match="raw"):
        rolling.write_rolling_feature_artifacts(result, Path(etl_result.raw_dir))


def test_rolling_raw_override_preserves_manifest_boundary(
    etl_result: etl.PandasETLResult, tmp_path: Path,
) -> None:
    output = tmp_path / "etl"
    etl.write_pandas_artifacts(etl_result, output)
    override = tmp_path / "override-raw"
    override.mkdir()
    result = rolling.run_rolling_feature_pipeline(
        output / "silver" / "observations.parquet", window_cycles=2, raw_dir=override,
    )
    assert result.raw_dir == etl_result.raw_dir
    for raw in (Path(etl_result.raw_dir), override):
        before = _hashes(raw)
        with pytest.raises(ValueError, match="raw"):
            rolling.write_rolling_feature_artifacts(result, raw / "generated")
        assert _hashes(raw) == before


def test_installed_legacy_pipeline_captures_default_raw_before_cwd_changes(
    etl_result: etl.PandasETLResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel_file = tmp_path / "venv" / "Lib" / "site-packages" / "cmapss" / "artifacts.py"
    monkeypatch.setattr(artifacts, "__file__", str(wheel_file))
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    silver = tmp_path / "legacy.parquet"
    etl_result.observations.to_parquet(silver, index=False)
    result = rolling.run_rolling_feature_pipeline(silver, window_cycles=2)
    assert result.raw_dir == str(raw.resolve())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(ValueError, match="raw"):
        rolling.write_rolling_feature_artifacts(result, raw / "generated")
    assert not (raw / "generated").exists()


@pytest.mark.parametrize("failure", ["parquet", "sqlite", "metadata"])
def test_etl_stage_failures_leave_existing_artifacts_unchanged(
    etl_result: etl.PandasETLResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    output = tmp_path / "etl"
    etl.write_pandas_artifacts(etl_result, output)
    before = _hashes(output)
    parquet = etl._write_parquet
    write_text = Path.write_text
    calls = 0

    def fail_parquet(frame: pd.DataFrame, path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected parquet error")
        parquet(frame, path)

    def fail_sqlite(result: etl.PandasETLResult, path: Path) -> None:
        raise OSError("injected sqlite error")

    def fail_metadata(path: Path, *args: object, **kwargs: object) -> int:
        if path.name == "run_info.json":
            raise OSError("injected metadata error")
        return write_text(path, *args, **kwargs)

    if failure == "parquet":
        monkeypatch.setattr(etl, "_write_parquet", fail_parquet)
    elif failure == "sqlite":
        monkeypatch.setattr(etl, "_write_sqlite", fail_sqlite)
    else:
        monkeypatch.setattr(Path, "write_text", fail_metadata)
    with pytest.raises(OSError, match="injected"):
        etl.write_pandas_artifacts(_updated(etl_result), output)
    assert _hashes(output) == before
    assert not list(tmp_path.glob(".cmapss-write-*"))


def test_successful_rerun_updates_full_bundle_and_keeps_other_files(
    etl_result: etl.PandasETLResult, tmp_path: Path,
) -> None:
    output = tmp_path / "etl"
    etl.write_pandas_artifacts(etl_result, output)
    (output / "README.md").write_text("keep this file\n", encoding="utf-8")
    (output / "benchmark.json").write_text("historical benchmark\n", encoding="utf-8")
    changed = _updated(etl_result)
    payload = etl.write_pandas_artifacts(changed, output)
    pd.testing.assert_frame_equal(
        pd.read_parquet(output / "silver" / "observations.parquet"), changed.observations,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(output / "silver" / "train_targets.parquet"), changed.train_targets,
    )
    assert json.loads((output / "run_info.json").read_text(encoding="utf-8")) == payload
    assert payload["input_sha256"] == changed.input_sha256
    assert (output / "README.md").read_text(encoding="utf-8") == "keep this file\n"
    assert (output / "benchmark.json").read_text(encoding="utf-8") == "historical benchmark\n"


@pytest.mark.parametrize("existing", [True, False])
@pytest.mark.parametrize("fail_at", [1, 2, 3, 4, 5])
def test_etl_publish_failure_restores_old_bundle_or_removes_new_bundle(
    etl_result: etl.PandasETLResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
    fail_at: int,
) -> None:
    output = tmp_path / "etl"
    if existing:
        etl.write_pandas_artifacts(etl_result, output)
        (output / "README.md").write_text("keep", encoding="utf-8")
    before = _hashes(output)
    replace_file = os.replace
    calls = 0

    def fail_publish(source: Path, destination: Path) -> None:
        nonlocal calls
        if "new" in Path(source).parts:
            calls += 1
            if calls == fail_at:
                raise OSError("injected publish error")
        replace_file(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", fail_publish)
    with pytest.raises(OSError, match="injected publish"):
        etl.write_pandas_artifacts(_updated(etl_result), output)
    assert _hashes(output) == before
    assert output.exists() == existing
    assert not list(tmp_path.glob(".cmapss-write-*"))


def test_rolling_metadata_stage_failure_keeps_old_bundle(
    etl_result: etl.PandasETLResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "features"
    baseline = _feature_result(etl_result)
    rolling.write_rolling_feature_artifacts(baseline, output)
    before = _hashes(output)

    def fail_metadata(path: Path, *args: object, **kwargs: object) -> int:
        raise OSError("injected metadata error")

    monkeypatch.setattr(Path, "write_text", fail_metadata)
    with pytest.raises(OSError, match="injected metadata"):
        rolling.write_rolling_feature_artifacts(_feature_result(_updated(etl_result)), output)
    assert _hashes(output) == before


@pytest.mark.parametrize("fail_at", [1, 2])
def test_rolling_publish_failure_keeps_old_bundle(
    etl_result: etl.PandasETLResult, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, fail_at: int,
) -> None:
    output = tmp_path / "features"
    rolling.write_rolling_feature_artifacts(_feature_result(etl_result), output)
    before = _hashes(output)
    replace_file = os.replace
    calls = 0

    def fail_publish(source: Path, destination: Path) -> None:
        nonlocal calls
        if "new" in Path(source).parts:
            calls += 1
            if calls == fail_at:
                raise OSError("injected rolling publish error")
        replace_file(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", fail_publish)
    with pytest.raises(OSError, match="injected rolling publish"):
        rolling.write_rolling_feature_artifacts(_feature_result(_updated(etl_result)), output)
    assert _hashes(output) == before


def test_rollback_failure_preserves_backup_and_reports_recovery_path(
    etl_result: etl.PandasETLResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "etl"
    etl.write_pandas_artifacts(etl_result, output)
    observations_before = (output / "silver" / "observations.parquet").read_bytes()
    replace_file = os.replace
    calls = 0

    def fail_publish_and_rollback(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError("injected publish and rollback error")
        replace_file(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", fail_publish_and_rollback)
    with pytest.raises(artifacts.ArtifactRollbackError) as raised:
        etl.write_pandas_artifacts(_updated(etl_result), output)
    recovery = raised.value.recovery_dir
    assert recovery.parent == tmp_path
    assert str(recovery) in str(raised.value)
    assert (recovery / "backup" / "silver" / "observations.parquet").read_bytes() == observations_before


def test_missing_staged_file_is_rejected_before_publication(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="Staging"):
        with artifacts.staged_artifact_bundle(output, ["data", "run_info.json"]) as stage:
            (stage / "data").write_text("complete data", encoding="utf-8")
    assert not output.exists()


def test_cleanup_error_does_not_hide_primary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_cleanup(workspace: Path, parent: Path) -> None:
        raise OSError("injected cleanup error")

    monkeypatch.setattr(artifacts, "_remove_workspace", fail_cleanup)
    with pytest.raises(ValueError, match="primary stage error") as raised:
        with artifacts.staged_artifact_bundle(tmp_path / "bundle", ["data"]) as stage:
            raise ValueError("primary stage error")
    assert "injected cleanup error" in raised.value.__notes__[0]
    assert not (tmp_path / "bundle").exists()


def test_cleanup_warning_distinguishes_successful_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_cleanup(workspace: Path, parent: Path) -> None:
        raise OSError("injected cleanup error")

    monkeypatch.setattr(artifacts, "_remove_workspace", fail_cleanup)
    with pytest.warns(artifacts.ArtifactCleanupWarning, match="publish thành công"):
        with artifacts.staged_artifact_bundle(tmp_path / "bundle", ["data"]) as stage:
            (stage / "data").write_text("new data", encoding="utf-8")
    assert (tmp_path / "bundle" / "data").read_text(encoding="utf-8") == "new data"


@pytest.mark.parametrize("failure", ["benchmark", "parity"])
def test_cli_keeps_entire_old_run_if_auxiliary_stage_fails(
    etl_result: etl.PandasETLResult, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    output = tmp_path / "etl"
    etl.write_pandas_artifacts(etl_result, output)
    for name in ("benchmark.json", "parity.json"):
        (output / name).write_text(json.dumps({"run": "old"}), encoding="utf-8")
    before = _hashes(output)
    monkeypatch.setattr(etl_cli, "run_pandas_etl", lambda *args: _updated(etl_result))

    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError(f"injected {failure} failure")

    if failure == "benchmark":
        monkeypatch.setattr(etl_cli, "benchmark_pandas", fail)
    else:
        monkeypatch.setattr(etl_cli, "_run_parity", fail)
    args = ["--raw-dir", etl_result.raw_dir, "--output-dir", str(output),
            "--benchmark-scales", "1", "--benchmark-repetitions", "1"]
    if failure == "parity":
        args.append("--with-spark")
    assert etl_cli.main(args) == 1
    assert _hashes(output) == before


def test_cli_receipts_are_linked_to_current_run_and_old_parity_is_not_claimed(
    etl_result: etl.PandasETLResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "etl"
    output.mkdir()
    (output / "parity.json").write_text(json.dumps({"status": "old"}), encoding="utf-8")
    monkeypatch.setattr(etl_cli, "run_pandas_etl", lambda *args: etl_result)
    args = ["--raw-dir", etl_result.raw_dir, "--output-dir", str(output),
            "--benchmark-scales", "1", "--benchmark-repetitions", "1"]
    assert etl_cli.main(args) == 0
    manifest = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
    benchmark = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    assert manifest["auxiliary_checks"]["parity"]["status"] == "not_requested"
    assert manifest["auxiliary_checks"]["benchmark"]["status"] == "passed"
    assert benchmark["input_sha256"] == manifest["input_sha256"]
    assert benchmark["etl_generated_at_utc"] == manifest["generated_at_utc"]
    assert json.loads((output / "parity.json").read_text(encoding="utf-8"))["status"] == "old"


def test_cli_parity_receipt_is_linked_to_current_run(
    etl_result: etl.PandasETLResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "etl"
    monkeypatch.setattr(etl_cli, "run_pandas_etl", lambda *args: etl_result)
    monkeypatch.setattr(etl_cli, "_run_parity", lambda *args: {"status": "passed", "checks": []})
    args = ["--raw-dir", etl_result.raw_dir, "--output-dir", str(output), "--with-spark",
            "--benchmark-scales", "1", "--benchmark-repetitions", "1"]
    assert etl_cli.main(args) == 0
    manifest = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
    parity = json.loads((output / "parity.json").read_text(encoding="utf-8"))
    assert manifest["auxiliary_checks"]["parity"]["status"] == "passed"
    assert parity["dataset_id"] == manifest["dataset_id"]
    assert parity["input_sha256"] == manifest["input_sha256"]
    assert parity["etl_generated_at_utc"] == manifest["generated_at_utc"]


def test_silver_junction_into_raw_is_rejected_before_writing(
    etl_result: etl.PandasETLResult, tmp_path: Path,
) -> None:
    output = tmp_path / "etl"
    output.mkdir()
    raw = Path(etl_result.raw_dir)
    before = _hashes(raw)
    junction = output / "silver"
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction), str(raw)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    else:
        junction.symlink_to(raw, target_is_directory=True)
    try:
        with pytest.raises((ValueError, OSError)) as raised:
            etl.write_pandas_artifacts(etl_result, output)
        if isinstance(raised.value, OSError):
            # Windows có thể từ chối junction không tin cậy ngay lúc resolve.
            assert getattr(raised.value, "winerror", None) == 448
        else:
            assert "raw" in str(raised.value)
        assert _hashes(raw) == before
        assert not (output / "gold").exists()
    finally:
        if os.name == "nt":
            junction.rmdir()
        else:
            junction.unlink()
