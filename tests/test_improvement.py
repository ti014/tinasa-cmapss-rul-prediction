"""Focused checks for the RUL improvement runner."""

import numpy as np
import pandas as pd
import pytest

from cmapss.modeling.improvement import (
    ImprovementConfig,
    low_rul_weights,
    make_hgb_candidates,
    _fit_predict, _metrics, _select_candidate, _inner_splits,
)
from cmapss.modeling.improvement.analysis import paired_comparison


def test_low_rul_weights_sum_to_one_per_engine() -> None:
    frame = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2],
            "cycle": [30, 31, 32, 30, 31],
            "rul": [31, 20, 10, 30, 0],
        }
    )
    weights = low_rul_weights(frame, "moderate_low_rul")
    totals = pd.Series(weights).groupby(frame["engine_id"]).sum()
    assert totals.to_dict() == {1: pytest.approx(1.0), 2: pytest.approx(1.0)}
    assert weights[2] > weights[0]


def test_hgb_candidates_are_deterministic_and_unique() -> None:
    config = ImprovementConfig(n_hgb_candidates=24, seed=42)
    first = make_hgb_candidates(config)
    second = make_hgb_candidates(config)
    assert first == second
    assert len(first) == 24
    assert len({tuple(candidate.items()) for candidate in first}) == 24
    assert first[0]["learning_rate"] == 0.05


def test_low_rul_weights_reject_unknown_scheme() -> None:
    frame = pd.DataFrame({"engine_id": [1], "cycle": [30], "rul": [0]})
    with pytest.raises(ValueError, match="Cách gán trọng số chưa biết"):
        low_rul_weights(frame, "unknown")


def test_weights_at_rul_boundaries() -> None:
    frame = pd.DataFrame({"engine_id": [1]*3, "rul": [15, 30, 31]})
    np.testing.assert_allclose(low_rul_weights(frame, "strong_low_rul"), [3/6, 2/6, 1/6])


def test_engine_leakage_is_rejected_before_fit() -> None:
    frame = pd.DataFrame({"engine_id": [1]})
    with pytest.raises(ValueError, match="giao nhau"):
        _fit_predict(frame, frame, {}, "uniform", 42)


def test_inner_folds_keep_engines_separate() -> None:
    frame = pd.DataFrame({"engine_id": np.repeat(np.arange(6), 3), "rul": np.tile([2, 1, 0], 6)})
    seen = []
    for train, validation in _inner_splits(frame, ImprovementConfig()):
        assert not set(frame.iloc[train].engine_id) & set(frame.iloc[validation].engine_id)
        seen.extend(frame.iloc[validation].engine_id.unique())
    assert sorted(seen) == list(range(6))


def test_selection_does_not_trade_warning_burden_for_lower_error() -> None:
    baseline = dict(candidate_id="baseline", mean_engine_mae=10., low_rul_engine_mae=5., late_or_missed_rate=.5, early_alarm_burden=.1)
    risky = dict(baseline, candidate_id="risky", low_rul_engine_mae=1., early_alarm_burden=.11)
    assert _select_candidate([risky, baseline], baseline) == 1


def test_no_warning_and_endpoint_warning_count_as_late() -> None:
    rows = []
    for engine in (1, 2):
        for cycle in range(1, 33):
            rows.append(dict(split="train", engine_id=engine, cycle=cycle, model_name="candidate", y_true=32-cycle,
                             y_pred=0. if engine == 2 and cycle == 32 else 40.))
    metrics = _metrics(pd.DataFrame(rows), 30)
    assert metrics["late_or_missed_count"] == 2
    assert metrics["early_alarm_burden"] == 0.


def test_burden_has_no_denominator_when_all_rows_are_low_rul() -> None:
    frame = pd.DataFrame(dict(split=["train"]*3, engine_id=[1]*3, cycle=[1,2,3], model_name=["candidate"]*3, y_true=[2.,1.,0.], y_pred=[2.,1.,0.]))
    result = _metrics(frame, 30)
    assert result["early_alarm_burden"] is None
    assert result["n_burden_engines"] == 0


def test_paired_bootstrap_preserves_known_constant_difference() -> None:
    baseline = pd.DataFrame({"mae": [2., 4., 6.]}, index=[1, 2, 3])
    result = paired_comparison(baseline, baseline-1, 100, 42).iloc[0]
    assert result.delta == result.ci_lower == result.ci_upper == -1
    with pytest.raises(ValueError, match="cùng ID động cơ"):
        paired_comparison(baseline, baseline.iloc[::-1], 10, 42)


@pytest.mark.parametrize("count", [0, 244])
def test_candidate_budget_bounds(count) -> None:
    with pytest.raises(ValueError, match="n_hgb_candidates"):
        ImprovementConfig(n_hgb_candidates=count)


def test_fit_cache_reuses_predictions_and_invalidates_changed_data(tmp_path, monkeypatch) -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    frame = pd.DataFrame(dict(split=["train"]*6, engine_id=[1]*3+[2]*3, cycle=[1,2,3]*2,
                              rul=[2.,1.,0.]*2, sensor__mean=[3.,2.,1.,4.,3.,2.]))
    params = dict(max_iter=2, min_samples_leaf=1, max_leaf_nodes=3)
    first = _fit_predict(frame.iloc[:3], frame.iloc[3:], params, "uniform", 42, tmp_path, 1)
    def forbidden(*args, **kwargs):
        raise AssertionError("fit called")
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", forbidden)
    second = _fit_predict(frame.iloc[:3], frame.iloc[3:], params, "uniform", 42, tmp_path, 1)
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    changed = frame.copy()
    changed.loc[0, "sensor__mean"] = 9.
    with pytest.raises(AssertionError, match="fit called"):
        _fit_predict(changed.iloc[:3], changed.iloc[3:], params, "uniform", 42, tmp_path, 1)
    assert list(tmp_path.glob("*.failed.json"))


def test_runner_writes_predictions_before_test_labels(tmp_path, monkeypatch) -> None:
    import json
    from cmapss.modeling.improvement import runner
    import cmapss.modeling.improvement.analysis as analysis
    source = tmp_path / "input.bin"
    source.write_bytes(b"fixture")
    output = tmp_path / "run"
    monkeypatch.setattr(runner, "_prepare_run", lambda *a: None)
    monkeypatch.setattr(runner, "load_window_datasets", lambda *a, **kw: {30: pd.DataFrame({"engine_id": [1]})})
    monkeypatch.setattr(runner, "_save_assignments", lambda *a: None)
    monkeypatch.setattr(runner, "_run_outer_stages", lambda *a: (pd.DataFrame(), pd.DataFrame(), {"inner_tables": pd.DataFrame()}))
    monkeypatch.setattr(analysis, "write_outer_comparison", lambda *a: None)
    monkeypatch.setattr(analysis, "write_test_comparison", lambda *a: None)
    def select(*args):
        assert not (output / "final_predictions.parquet").exists()
        return 30, {}, "uniform", pd.DataFrame()
    monkeypatch.setattr(runner, "_select_final_configuration", select)
    monkeypatch.setattr(runner, "_final_test_predictions", lambda *a: (pd.DataFrame({"rul_pred": [7.]}), {"model": "fixture"}))
    calls = []
    def evaluate(prediction_path, label_path):
        assert pd.read_parquet(prediction_path).rul_pred.tolist() == [7.]
        calls.append("read labels")
        return pd.DataFrame()
    monkeypatch.setattr(runner, "evaluate_predictions_from_path", evaluate)
    monkeypatch.setattr(runner, "write_final_evaluation_artifacts", lambda *a, **kw: {"metrics": {}, "outputs": {}})
    runner.run_improvement_experiment(source, source, test_labels_path=source, output_dir=output)
    assert calls == ["read labels"]
    assert json.loads((output / "run_info.json").read_text())["test_label_joined"] is True
