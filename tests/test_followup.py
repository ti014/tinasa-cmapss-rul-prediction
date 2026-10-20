"""Checks for causal features, fold isolation, target capping and saved inference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import MEASUREMENTS, OBSERVATION_COLUMNS, SENSORS
from cmapss.features.enhanced import FeatureRecipe, OperatingConditionNormalizer, build_history_features
from cmapss.modeling.followup import FeatureStore, StudyConfig, fit_pair, run_study


def synthetic_observations() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    rows = []
    for split, engines in (("train", 8), ("test", 2)):
        for engine in range(1, engines + 1):
            for cycle in range(1, (75 if split == "train" else 45) + 1):
                regime = cycle % 6
                measurements = [regime + rng.normal(0, 0.01), regime * 3.0, regime * 10.0]
                measurements.extend(cycle * (sensor + 1) + regime * 100 + rng.normal() for sensor in range(21))
                rows.append([split, engine, cycle, *measurements])
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).astype({"split": "string", "engine_id": "int64", "cycle": "int64", **dict.fromkeys(MEASUREMENTS, "float64")})


def targets_for(frame: pd.DataFrame) -> pd.DataFrame:
    target = frame.loc[frame.split.eq("train"), ["split", "engine_id", "cycle"]].copy()
    target["rul"] = target.groupby("engine_id").cycle.transform("max") - target.cycle
    return target


def test_history_is_causal_and_matches_linear_slope() -> None:
    observations = synthetic_observations()
    observations["sensor_1"] = observations.cycle.astype(float)
    original = build_history_features(observations)
    np.testing.assert_allclose(original.sensor_1__prefix_slope, 1.0)
    np.testing.assert_allclose(original.sensor_1__since_start, original.cycle - 15.5)
    changed = observations.copy()
    changed.loc[changed.cycle.gt(40), list(SENSORS)] += 10000
    perturbed = build_history_features(changed)
    pd.testing.assert_frame_equal(original.loc[original.cycle.le(40)], perturbed.loc[perturbed.cycle.le(40)])


def test_normalizer_uses_only_fit_engines_and_never_updates_on_transform() -> None:
    observations = synthetic_observations()
    train = observations.loc[observations.split.eq("train") & observations.engine_id.le(4)]
    model = OperatingConditionNormalizer().fit(train)
    means = model.means_.copy(deep=True)
    scales = model.scales_.copy(deep=True)
    heldout = observations.loc[observations.split.eq("train") & observations.engine_id.gt(4)].copy()
    heldout.loc[:, list(SENSORS)] += 100000
    assert np.isfinite(model.transform(heldout).loc[:, list(SENSORS)]).all().all()
    assert model.fit_engine_ids_ == (1, 2, 3, 4)
    pd.testing.assert_frame_equal(means, model.means_)
    pd.testing.assert_frame_equal(scales, model.scales_)
    with pytest.raises(ValueError, match="train observations only"):
        OperatingConditionNormalizer().fit(observations)


def test_cap_does_not_change_validation_truth_and_cache_detects_tampering(tmp_path: Path) -> None:
    observations = synthetic_observations()
    store = FeatureStore(observations, targets_for(observations))
    recipe = FeatureRecipe("cap10", target_cap=10)
    config = StudyConfig(max_iter=3, fit_threads=1, bootstrap_resamples=10)
    actual = fit_pair(store, recipe, {1, 2, 3, 4}, {5, 6, 7, 8}, config, tmp_path, "frozen")
    assert actual.y_true.max() > 10
    assert actual.y_pred.max() <= 10 + 1e-10
    again = fit_pair(store, recipe, {1, 2, 3, 4}, {5, 6, 7, 8}, config, tmp_path, "frozen")
    pd.testing.assert_frame_equal(actual, again)
    path = next(tmp_path.glob("*.parquet"))
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="modified"):
        fit_pair(store, recipe, {1, 2, 3, 4}, {5, 6, 7, 8}, config, tmp_path, "frozen")
    with pytest.raises(ValueError, match="overlap"):
        fit_pair(store, recipe, {1, 2}, {2, 3}, config, tmp_path, "frozen")


@pytest.mark.parametrize("module_entry", [False, True])
def test_nested_runner_writes_predictions_before_reading_test_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_entry: bool) -> None:
    from cmapss.modeling.followup import cli, runner
    observations = synthetic_observations()
    silver = tmp_path / "outputs/etl/silver"
    silver.mkdir(parents=True)
    observations.to_parquet(silver / "observations.parquet", index=False)
    targets_for(observations).to_parquet(silver / "train_targets.parquet", index=False)
    (tmp_path / "uv.lock").write_text("test lock", encoding="utf-8")
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    (raw / "RUL_FD001.txt").write_text("30\n35\n", encoding="utf-8")
    output = tmp_path / "study" / "FD001" if module_entry else tmp_path / "study"
    original_evaluate = runner.evaluate_predictions_from_path

    def guarded_evaluate(prediction_path: Path, label_path: Path) -> pd.DataFrame:
        assert prediction_path.is_file()
        assert (output / "selected_model.joblib").is_file()
        assert (output / "final_selection_metrics.csv").is_file()
        assert json.loads((output / "run_info.json").read_text())["test_label_joined"] is False
        return original_evaluate(prediction_path, label_path)

    monkeypatch.setattr(runner, "evaluate_predictions_from_path", guarded_evaluate)
    monkeypatch.setattr(runner, "study_candidates", lambda _: (FeatureRecipe("baseline"), FeatureRecipe("cap125_history", target_cap=125, history=True)))
    config = StudyConfig(outer_splits=2, inner_splits=2, bootstrap_resamples=20, fit_threads=1, max_iter=3)
    if module_entry:
        import runpy
        import subprocess
        import sys

        monkeypatch.setattr(cli, "StudyConfig", lambda: config)
        monkeypatch.setattr(sys, "argv", ["followup", "--datasets", "FD001", "--repo-root", str(tmp_path), "--output-root", "study"])
        runpy.run_module("cmapss.modeling.followup", run_name="__main__")
        # A fresh interpreter cannot resolve __main__.RecipeModel; this catches
        # the real serialization failure specific to python -m execution.
        subprocess.run([sys.executable, "-c", "import joblib,sys; m=joblib.load(sys.argv[1]); assert type(m).__module__ == 'cmapss.modeling.followup.estimators'",
                        str(output / "selected_model.joblib")], check=True, capture_output=True)
        result = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
    else:
        result = run_study("FD001", tmp_path, output, config)
    assert result["test_label_joined"] and result["saved_model_predictions_match"]
    outer = pd.read_parquet(output / "outer_predictions.parquet")
    assert len(outer) == 2 * 8 * 46
    assert outer.groupby(["stage", "engine_id"]).outer_fold.nunique().eq(1).all()
    for path in (output / "fit_cache").glob("*.json"):
        record = json.loads(path.read_text())["identity_record"]
        assert not set(record["train_engines"]) & set(record["validation_engines"])
