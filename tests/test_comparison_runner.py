"""Exercise shared lifecycle artifacts with cheap deterministic estimators."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import MEASUREMENTS, OBSERVATION_COLUMNS
from cmapss.data.readers import read_test_labels
from cmapss.data.targets import make_train_targets
from cmapss.modeling import comparison_runner as common
from cmapss.modeling.comparison_runner import runner as lifecycle
from cmapss.modeling.lightgbm_comparison import cache as lgbm_cache
from cmapss.modeling.lightgbm_comparison import runner as lgbm
from cmapss.modeling.tree_comparison import cache as xgb_cache
from cmapss.modeling.tree_comparison import runner as xgb


@dataclass
class DeterministicPredictor:
    """Replace fitting only; real folds, cache, serialization and reports run."""

    spec: Any
    terminal_cycle: int
    feature_columns: tuple[str, ...]
    effective_parameters: dict[str, Any]
    weight_metadata: dict[str, Any]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Keep distinct model families and capped arms visible in outputs."""
        offset = {"hgb": 0.0, "xgboost": 2.0, "lightgbm": 3.0}[self.spec.model_kind]
        return (self.terminal_cycle - frame.cycle.to_numpy(dtype="float64")
                + offset + float(self.spec.target_cap is not None))


def deterministic_fit(frame: pd.DataFrame, spec: Any, config: Any) -> DeterministicPredictor:
    """Expose the exact supplied spec and seed in persisted model metadata."""
    return DeterministicPredictor(
        spec, int(frame.cycle.max()), xgb.FEATURES,
        {"model_kind": spec.model_kind, "target_cap": spec.target_cap, "seed": config.seed},
        {"fixture": True},
    )


def thread_pool(**kwargs: Any) -> ThreadPoolExecutor:
    """Retain the submit/future protocol without starting training processes."""
    return ThreadPoolExecutor(max_workers=kwargs["max_workers"])


def create_inputs(root: Path) -> None:
    """Create six complete train engines and two independent test endpoints."""
    rows = [[split, engine, cycle, *[float(engine + cycle / (offset + 1))
                                    for offset in range(len(MEASUREMENTS))]]
            for split, engines, length in (("train", range(1, 7), 164),
                                           ("test", range(1, 3), 44))
            for engine in engines for cycle in range(1, length + 1)]
    observations = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).astype(
        {"split": "string", "engine_id": "int64", "cycle": "int64",
         **dict.fromkeys(MEASUREMENTS, "float64")})
    targets = make_train_targets(observations[observations.split.eq("train")].copy())
    observations.to_parquet(root / "observations.parquet", index=False)
    targets.to_parquet(root / "targets.parquet", index=False)
    (root / "labels.txt").write_text("120\n120\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    source = root / "src/study.py"
    source.parent.mkdir()
    source.write_text("# fixed fixture source\n", encoding="utf-8")


def test_studies_share_real_folds_receipts_and_reporting(tmp_path, monkeypatch):
    """Both adapters retain model differences through the entire lifecycle."""
    create_inputs(tmp_path)
    monkeypatch.setattr(lifecycle, "ProcessPoolExecutor", thread_pool)
    outputs = []
    for runner, cache, study, iterations in (
        (xgb, xgb_cache, "tree_comparison", {"xgb_n_estimators": 20}),
        (lgbm, lgbm_cache, "lightgbm_comparison", {"lgbm_n_estimators": 20}),
    ):
        output = tmp_path / "outputs/experiments" / study / "lifecycle"
        monkeypatch.setattr(runner, "_source_files", lambda root: ("src/study.py",))
        monkeypatch.setattr(runner, "_versions", lambda: {"python": "fixture"})
        monkeypatch.setattr(runner, "fit_predictor", deterministic_fit)
        monkeypatch.setattr(cache, "fit_predictor", deterministic_fit)
        reads = []

        def guarded_labels(path):
            receipt = common._read_json(output / "final_predictions_receipt.json")
            selection = common._read_json(output / "selection.json")
            assert receipt["test_labels_read"] is False
            assert selection["test_labels_read_for_selection"] is False
            assert receipt["saved_model_replay_exact"] is True
            assert receipt["predictions_sha256"] == common._sha(output / "final_predictions.parquet")
            assert set(receipt["models_sha256"]) == set(runner.ARMS)
            reads.append(path)
            return read_test_labels(path)

        monkeypatch.setattr(runner, "read_test_labels", guarded_labels)
        config = runner.StudyConfig(outer_splits=2, inner_splits=2, fit_threads=1,
                                   bootstrap_resamples=20, hgb_max_iter=20, **iterations)
        run = xgb.run_tree_comparison if runner is xgb else lgbm.run_lightgbm_comparison
        result = run(repo_root=tmp_path, observations_path="observations.parquet",
                     targets_path="targets.parquet", test_labels_path="labels.txt",
                     output_dir=output, config=config)
        assert result["status"] == "completed" and reads == [tmp_path / "labels.txt"]
        assert result["selected_procedure"] == "hgb_uncapped"
        outer = pd.read_parquet(output / "outer_predictions.parquet")
        assert set(outer.procedure) == {*runner.ARMS, "selected_pipeline"}
        assert outer.groupby("procedure").size().eq(6 * 135).all()
        assert outer.y_true.max() == 134  # Cap125 never replaces evaluation truth.
        metrics = pd.read_csv(output / "test_metrics.csv").set_index("procedure")
        expected_rmse = dict(zip(runner.ARMS, (0.0, 1.0, 2.0 if runner is xgb else 3.0,
                                             3.0 if runner is xgb else 4.0), strict=True))
        for arm, expected in expected_rmse.items():
            assert metrics.loc[arm, "rmse"] == pytest.approx(expected)
            metadata = common._read_json(output / "models" / arm / "model.json")
            assert metadata["effective_parameters"]["seed"] == 42
            assert metadata["fit_engine_ids"] == list(range(1, 7))
            assert metadata["saved_model_replay_max_abs_delta"] == 0.0
        folds = pd.read_csv(output / "outer_assignments.csv")
        for outer_fold in (1, 2):
            inner = pd.read_csv(output / "stages" / f"outer_{outer_fold}" / "inner_assignments.csv")
            held_out = set(folds.loc[folds.outer_fold.eq(outer_fold), "engine_id"])
            assert held_out.isdisjoint(inner.engine_id)
            assert set(inner.engine_id) == set(range(1, 7)) - held_out
        choices = common._read_json(output / "selection.json")
        assert choices["arms"][runner.ARMS[2]]["numeric_controls_met"] is False
        assert choices["arms"][runner.ARMS[2]]["n_evaluated"] == 2
        before = common._read_json(output / "outer_decision_before_test.json")
        assert before == choices["outer_decision"] == result["decision"]
        outputs.append(output)

        # Compatible replay uses persisted folds, fit cache and model objects.
        def forbid_fit(*unused):
            raise AssertionError("compatible resume must reuse cached models")

        monkeypatch.setattr(runner, "fit_predictor", forbid_fit)
        monkeypatch.setattr(cache, "fit_predictor", forbid_fit)
        replay = run(repo_root=tmp_path, observations_path="observations.parquet",
                     targets_path="targets.parquet", test_labels_path="labels.txt",
                     output_dir=output, config=config, evaluate_test=False)
        assert replay["test_evaluated"] is False and len(reads) == 1
        assert before == common._read_json(output / "outer_decision_before_test.json")

    for name in ("frozen_train_features.parquet", "frozen_test_features.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(outputs[0] / name),
                                      pd.read_parquet(outputs[1] / name), check_exact=True)
    for name in ("outer_assignments.csv", "inner_assignments.csv", "final_inner_assignments.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(outputs[0] / name),
                                      pd.read_csv(outputs[1] / name), check_exact=True)
    for arm in ("hgb_uncapped", "hgb_cap125", "selected_pipeline"):
        frames = [pd.read_parquet(output / "outer_predictions.parquet") for output in outputs]
        pd.testing.assert_frame_equal(
            frames[0].loc[frames[0].procedure.eq(arm)].reset_index(drop=True),
            frames[1].loc[frames[1].procedure.eq(arm)].reset_index(drop=True), check_exact=True)


@pytest.mark.parametrize("runner,study,model", [(xgb, "tree_comparison", "xgboost"),
                                               (lgbm, "lightgbm_comparison", "lightgbm")])
def test_shared_source_is_frozen_and_workers_remain_picklable(runner, study, model):
    root = Path(__file__).resolve().parents[1]
    sources = runner._source_files(root)
    for module in ("__init__", "contracts", "artifact_io", "provenance", "training", "evaluation", "runner"):
        assert f"src/cmapss/modeling/comparison_runner/{module}.py" in sources
    assert f"src/cmapss/modeling/{study}/runner.py" in sources
    other = "lightgbm_comparison" if study == "tree_comparison" else "tree_comparison"
    assert not any(f"/{other}/" in name for name in sources)
    assert pickle.loads(pickle.dumps(runner._final_model_task)) is runner._final_model_task
    assert pickle.loads(pickle.dumps(runner.run_fit_task)) is runner.run_fit_task
    backend = runner._backend()
    assert backend.select_candidate.__module__.endswith(f"{study}.analysis")
    assert {spec.model_kind for spec in backend.active_specs(runner.StudyConfig())} == {"hgb", model}


@pytest.mark.parametrize("runner,name,iterations", [
    (xgb, "run_tree_comparison", "xgb_n_estimators"),
    (lgbm, "run_lightgbm_comparison", "lgbm_n_estimators"),
])
def test_cli_retains_explicit_smoke_protocol(tmp_path, monkeypatch, runner, name, iterations):
    calls = []
    monkeypatch.setattr(runner, name, lambda **kwargs: calls.append(kwargs))
    runner.main(["--repo-root", str(tmp_path), "--output-dir", "isolated-run",
                 "--dry-run", "--skip-test-evaluation", "--fit-threads", "1",
                 "--max-workers", "1"])
    assert len(calls) == 1
    assert calls[0]["repo_root"] == tmp_path and calls[0]["output_dir"] == "isolated-run"
    assert calls[0]["evaluate_test"] is False
    config = calls[0]["config"]
    assert (config.outer_splits, config.inner_splits, config.bootstrap_resamples) == (2, 2, 20)
    assert config.fit_threads == config.max_workers == 1
    assert config.hgb_max_iter == getattr(config, iterations) == 20
    assert config.seed == 42 and config.endpoint_fractions == (0.5, 0.7, 0.9)
    candidates = runner._active_specs(config)
    assert len(candidates) == 6 and {spec.family for spec in candidates} == set(runner.ARMS)
