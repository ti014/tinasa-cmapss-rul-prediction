"""Regression checks for package boundaries, legacy artifacts and source snapshots."""

import importlib
import json
import pickle
from pathlib import Path
import shutil

import pandas as pd
import pytest

from cmapss.data.contracts import MEASUREMENTS, OBSERVATION_COLUMNS
from cmapss.data.targets import make_train_targets
from cmapss.features.enhanced import FeatureRecipe
from cmapss.modeling.audit_study import AuditStudyConfig, run_audit_study
from cmapss.modeling.audit_study.provenance import BASELINE_FILES, SOURCE_FILES
from cmapss.modeling.followup.artifact_io import freeze_manifest
from cmapss.modeling.followup.config import StudyConfig
from cmapss.provenance import local_source_files


@pytest.mark.parametrize("module,name", [
    ("pipeline", "WeightedMedianRegressor"),
    ("pipeline", "ModelingConfig"),
    ("pipeline", "ModelingResult"),
    ("audit_study", "CandidateSpec"),
    ("audit_study", "AuditStudyConfig"),
    ("audit_study", "WarningPolicy"),
    ("audit_study", "FittedCandidate"),
    ("followup", "StudyConfig"),
    ("followup", "RecipeModel"),
    ("improvement", "ImprovementConfig"),
])
def test_legacy_pickle_class_paths_resolve(module, name, monkeypatch):
    """Old model/config references still resolve to the canonical class."""
    legacy = f"cmapss.modeling.{module}"
    cls = getattr(importlib.import_module(legacy), name)
    with monkeypatch.context() as patch:
        patch.setattr(cls, "__module__", legacy)
        serialized = pickle.dumps(cls)
    assert pickle.loads(serialized) is cls


def test_source_closure_follows_nested_relative_imports(tmp_path):
    """Snapshot dependencies include nested modules, not unrelated siblings."""
    sources = {
        "src/cmapss/__init__.py": "",
        "src/cmapss/modeling/__init__.py": "",
        "src/cmapss/modeling/study/__init__.py": "from .runner import run\n",
        "src/cmapss/modeling/study/runner.py": "from .. import metrics\ndef run(): pass\n",
        "src/cmapss/modeling/metrics.py": "",
    }
    for name, content in {**sources, "src/cmapss/modeling/unrelated.py": ""}.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    entry = "src/cmapss/modeling/study/__init__.py"
    assert set(local_source_files(tmp_path, (entry,))) == set(sources)
    with pytest.raises(FileNotFoundError, match="Thiếu mã nguồn"):
        local_source_files(tmp_path, ("src/cmapss/missing.py",))


def test_followup_snapshots_nested_modules_and_rejects_changed_sources(tmp_path):
    """A refactored estimator cannot silently reuse an older fit cache."""
    module = tmp_path / "src/cmapss/modeling/followup/estimators.py"
    module.parent.mkdir(parents=True)
    module.write_text("# original estimator\n", encoding="utf-8")
    inputs = [tmp_path / name for name in ("observations.parquet", "targets.parquet", "uv.lock")]
    for path in inputs:
        path.write_bytes(b"fixture")
    output = tmp_path / "run"
    args = (tmp_path, output, "FD001", *inputs[:2], (FeatureRecipe("baseline"),), StudyConfig(outer_splits=2))

    identity = freeze_manifest(*args)
    assert freeze_manifest(*args) == identity
    snapshot = output / "source_snapshot" / module.relative_to(tmp_path)
    assert snapshot.read_bytes() == module.read_bytes()
    module.write_text("# changed estimator\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Resume không khớp"):
        freeze_manifest(*args)


def test_audit_package_runs_with_complete_source_snapshot(tmp_path):
    """Exercise moved fit/selection/reporting modules and frozen source identity."""
    repo = Path(__file__).resolve().parents[1]
    for name in local_source_files(repo, SOURCE_FILES):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, destination)
    for name in (*BASELINE_FILES, "uv.lock"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    rows = [
        [split, engine, cycle, *[float(engine + cycle / (i + 1)) for i in range(len(MEASUREMENTS))]]
        for split, count, length in (("train", 6, 80), ("test", 2, 40))
        for engine in range(1, count + 1) for cycle in range(1, length + 1)
    ]
    observations = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).astype({
        "split": "string", "engine_id": "int64", "cycle": "int64",
        **dict.fromkeys(MEASUREMENTS, "float64"),
    })
    observations.to_parquet(tmp_path / "observations.parquet", index=False)
    make_train_targets(observations[observations.split.eq("train")]).to_parquet(
        tmp_path / "targets.parquet", index=False,
    )
    output = tmp_path / "outputs/experiments/audit_remediation/smoke"
    result = run_audit_study(
        repo_root=tmp_path, observations_path="observations.parquet", targets_path="targets.parquet",
        output_dir=output, evaluate_test=False,
        config=AuditStudyConfig(outer_splits=2, inner_splits=2, hgb_max_iter=2,
                                candidate_ids=("hgb_baseline",), thresholds=(30,), persistences=(1,),
                                bootstrap_resamples=10, fit_threads=1),
    )

    assert result["status"] == "completed"
    assert result["current_used_source_equivalent_to_snapshot"] is True
    assert result["test_evaluation_completed"] is False
    assert len(pd.read_parquet(output / "final_predictions.parquet")) == 2
    sources = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["identity"]["source_sha256"]
    for module in ("config", "features", "estimators", "training", "evaluation", "artifact_io", "runner"):
        assert f"src/cmapss/modeling/audit_study/{module}.py" in sources
