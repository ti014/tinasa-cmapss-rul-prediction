"""Scientific contracts: matched features, raw truth, weights and cache integrity."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cmapss.modeling.tree_comparison import cache, estimators
from cmapss.modeling.tree_comparison.contracts import FEATURES, StudyConfig, specs


@pytest.fixture
def fitting_frame() -> pd.DataFrame:
    """Unequal trajectory lengths expose accidental row-equal weighting."""
    engines = np.repeat([1, 2, 3], [6, 8, 10])
    matrix = np.random.default_rng(14).normal(size=(len(engines), len(FEATURES)))
    frame = pd.DataFrame(matrix, columns=FEATURES)
    frame["split"] = "train"
    frame["engine_id"] = engines
    frame["cycle"] = frame.groupby("engine_id").cumcount() + 30
    frame["rul"] = 240 - frame["cycle"]
    return frame


def test_feature_contract_and_engine_weights(fitting_frame):
    assert len(FEATURES) == len(set(FEATURES)) == 169
    assert "rul" not in FEATURES and "y_true" not in FEATURES
    weights = estimators.engine_total_one_weights(fitting_frame)
    totals = pd.Series(weights).groupby(fitting_frame.engine_id).sum()
    np.testing.assert_allclose(totals, np.ones(3), rtol=0, atol=1e-15)
    assert weights[0] != weights[-1]


def test_capping_changes_fitting_target_only_and_preserves_hgb_reference(monkeypatch, fitting_frame):
    capture = {}

    def capture_fit(model, matrix, target, sample_weight):
        capture.update(parameters=model.get_params(), target=target.copy(),
                       weights=sample_weight.copy(), shape=matrix.shape)
        return model

    monkeypatch.setattr(estimators.HistGradientBoostingRegressor, "fit", capture_fit)
    before = fitting_frame.copy(deep=True)
    spec = next(item for item in specs() if item.family == "hgb_cap125")
    fitted = estimators.fit_predictor(fitting_frame, spec, StudyConfig())
    np.testing.assert_array_equal(capture["target"], np.full(len(before), 125.0))
    pd.testing.assert_frame_equal(fitting_frame, before, check_exact=True)
    assert capture["shape"][1] == len(FEATURES)
    parameters = capture["parameters"]
    assert parameters["learning_rate"] == 0.05
    assert parameters["max_depth"] is None
    assert parameters["max_leaf_nodes"] == 31
    assert parameters["min_samples_leaf"] == 20
    assert parameters["l2_regularization"] == 0.0
    assert parameters["early_stopping"] is False
    assert fitted.feature_columns == FEATURES


def test_xgb_penalty_scale_uses_fitting_rows_only(monkeypatch, fitting_frame):
    capture = {}

    class Model:
        def fit(self, values, target, sample_weight):
            capture.update(target=target.copy(), weights=sample_weight.copy(), shape=values.shape)

    def build(spec, config, effective):
        capture["parameters"] = dict(effective)
        return Model()

    monkeypatch.setattr(estimators, "_build_xgb", build)
    fitting = fitting_frame.loc[fitting_frame.engine_id.isin((1, 2))].copy()
    spec = next(item for item in specs() if item.family == "xgb_uncapped"
                and item.reg_lambda_nominal == 10.0)
    fitted = estimators.fit_predictor(fitting, spec, StudyConfig())
    scale = 2 / 14
    assert fitted.effective_parameters["min_child_weight"] == pytest.approx(20 * scale)
    assert fitted.effective_parameters["reg_lambda"] == pytest.approx(10 * scale)
    np.testing.assert_array_equal(capture["target"], fitting.rul)
    assert capture["shape"][1] == 169
    assert fitted.weight_metadata["total_weight"] == pytest.approx(2)


def test_fit_rejects_test_rows_and_nonfinite_features(fitting_frame):
    spec = specs()[0]
    with pytest.raises(ValueError, match="train"):
        estimators.fit_predictor(fitting_frame.assign(split="test"), spec, StudyConfig())
    broken = fitting_frame.copy()
    broken.loc[0, FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="hữu hạn"):
        estimators.fit_predictor(broken, spec, StudyConfig())


def test_cache_shared_matrix_keeps_one_cycle_column_and_reuses_fit(monkeypatch, tmp_path, fitting_frame):
    feature_path = tmp_path / "frozen.parquet"
    fitting_frame.to_parquet(feature_path, index=False)
    loaded = cache.load_frozen_features(feature_path, "sourceA")
    assert loaded.columns.is_unique
    pd.testing.assert_frame_equal(loaded.loc[:, list(FEATURES)], fitting_frame.loc[:, list(FEATURES)],
                                  check_exact=True)
    calls = []

    def fit(train, spec, config):
        calls.append(set(train.engine_id))
        assert set(train.engine_id) == {1, 2}
        assert train.loc[:, list(FEATURES)].shape[1] == 169
        return SimpleNamespace(predict=lambda frame: np.full(len(frame), 110.0),
                               effective_parameters={}, weight_metadata={})

    monkeypatch.setattr(cache, "fit_predictor", fit)
    config, spec = StudyConfig(), specs()[1]
    args = (feature_path, spec, (1, 2), (3,), config, tmp_path / "cache", "sourceA")
    first = Path(cache.run_fit_task(*args))
    second = Path(cache.run_fit_task(*args))
    assert first == second and calls == [{1, 2}]
    predicted = pd.read_parquet(first)
    np.testing.assert_array_equal(predicted.y_true, fitting_frame.loc[fitting_frame.engine_id.eq(3), "rul"])
    assert predicted.y_true.min() > 125
    changed_args = (*args[:4], replace(config, seed=43), *args[5:])
    assert Path(cache.run_fit_task(*changed_args)) != first
    corrupted = pd.read_parquet(first)
    corrupted.loc[0, "y_pred"] += 1
    corrupted.to_parquet(first, index=False)
    with pytest.raises(ValueError, match="(?i)checksum"):
        cache.run_fit_task(*args)


def test_cache_rejects_overlap_and_refreshes_changed_file(tmp_path, fitting_frame):
    feature_path = tmp_path / "frozen.parquet"
    fitting_frame.to_parquet(feature_path, index=False)
    with pytest.raises(ValueError, match="không giao"):
        cache.run_fit_task(feature_path, specs()[0], (1, 2), (2, 3), StudyConfig(),
                           tmp_path / "cache", "sourceB")
    before = cache.load_frozen_features(feature_path, "sourceB")
    changed = fitting_frame.copy()
    changed.loc[0, FEATURES[0]] += 2
    changed.to_parquet(feature_path, index=False)
    after = cache.load_frozen_features(feature_path, "sourceB")
    assert after.loc[0, FEATURES[0]] == before.loc[0, FEATURES[0]] + 2
