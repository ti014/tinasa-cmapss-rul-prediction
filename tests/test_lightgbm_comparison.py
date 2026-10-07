"""Focused contract tests for the isolated LightGBM study."""

import numpy as np
import pandas as pd
import pytest

from cmapss.modeling.lightgbm_comparison.contracts import FEATURES, StudyConfig, specs
from cmapss.modeling.lightgbm_comparison import estimators


def _frame() -> pd.DataFrame:
    engines = np.repeat([1, 2, 3], [6, 8, 10])
    frame = pd.DataFrame(np.random.default_rng(11).normal(size=(len(engines), len(FEATURES))), columns=FEATURES)
    frame["split"] = "train"
    frame["engine_id"] = engines
    frame["cycle"] = frame.groupby("engine_id").cumcount() + 30
    frame["rul"] = 240 - frame["cycle"]
    return frame


def test_lightgbm_grid_and_feature_contract():
    candidates = specs()
    assert len(FEATURES) == 169
    assert [item.family for item in candidates] == ["hgb_uncapped", "hgb_cap125", "lgbm_uncapped", "lgbm_uncapped", "lgbm_cap125", "lgbm_cap125"]
    assert {item.reg_lambda_nominal for item in candidates if item.model_kind == "lightgbm"} == {0.0, 10.0}
    assert all(item.num_leaves == 31 and item.max_depth == -1 for item in candidates if item.model_kind == "lightgbm")


def test_engine_total_one_weights():
    frame = _frame()
    weights = estimators.engine_total_one_weights(frame)
    totals = pd.Series(weights).groupby(frame.engine_id).sum()
    np.testing.assert_allclose(totals.to_numpy(), np.ones(3), rtol=0, atol=1e-15)


def test_lgbm_effective_regularization_and_capping(monkeypatch):
    captured = {}

    class FakeModel:
        def fit(self, matrix, target, sample_weight):
            captured.update(target=target.copy(), shape=matrix.shape, weights=sample_weight.copy())
        def predict(self, matrix):
            return np.zeros(len(matrix))

    def build(spec, config, effective):
        captured["effective"] = effective.copy()
        return FakeModel()

    monkeypatch.setattr(estimators, "_build_lgbm", build)
    frame = _frame()
    spec = next(item for item in specs() if item.family == "lgbm_cap125" and item.reg_lambda_nominal == 10.0)
    fitted = estimators.fit_predictor(frame, spec, StudyConfig())
    assert captured["effective"]["reg_lambda"] == pytest.approx(10.0 * 3 / len(frame))
    assert captured["effective"]["min_child_weight"] == pytest.approx(0.001 * 3 / len(frame))
    assert np.all(captured["target"] <= 125)
    assert fitted.feature_columns == FEATURES
