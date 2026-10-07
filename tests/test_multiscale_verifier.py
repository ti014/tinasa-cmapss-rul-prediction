"""Independent arithmetic and tamper checks for the multi-scale verifier."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import MEASUREMENTS, OBSERVATION_COLUMNS
from cmapss.modeling.multiscale_comparison.config import StudyConfig

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("multiscale_independent_verifier", ROOT / "scripts/verify_multiscale_study.py")
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


def _prediction(length: int = 12, *, start: int = 1) -> pd.DataFrame:
    cycles = np.arange(start, length + 1)
    return pd.DataFrame({"split": "train", "engine_id": 1, "cycle": cycles,
                         "y_true": length - cycles, "y_pred": length - cycles})


def _config(**values):
    return {**asdict(StudyConfig()), **values}


def test_score_recomputes_asymmetric_nasa_and_ignores_stale_errors() -> None:
    frame = pd.DataFrame({"y_true": [13.0, 0.0], "y_pred": [0.0, 10.0],
                          "absolute_error": [999.0, 999.0], "squared_error": [1.0, 1.0],
                          "nasa_score_component": [0.0, 0.0]})
    actual = verifier._score(frame)
    np.testing.assert_array_equal(actual.absolute_error, [13.0, 10.0])
    np.testing.assert_array_equal(actual.squared_error, [169.0, 100.0])
    np.testing.assert_allclose(actual.nasa_score_component, np.expm1([1.0, 1.0]))


@pytest.mark.parametrize("truth,prediction", [(-1.0, 0.0), (1.0, np.inf), (np.nan, 0.0)])
def test_score_rejects_capped_invalid_or_nonfinite_raw_inputs(truth, prediction) -> None:
    with pytest.raises(ValueError, match="finite predictions and raw nonnegative"):
        verifier._score(pd.DataFrame({"y_true": [truth], "y_pred": [prediction]}))


def test_endpoint_origins_are_retrospective_and_clamp_to_available_prefix() -> None:
    frame = _prediction(100, start=60)
    panel = verifier._endpoints(frame, (.5, .7, .9))
    assert panel.cycle.tolist() == [60, 70, 90]
    assert panel.requested_cycle.tolist() == [50, 70, 90]
    assert panel.origin_clamped_to_available_prefix.tolist() == [True, False, False]
    broken = frame.copy()
    broken.loc[broken.cycle.eq(70), "y_true"] += 1
    with pytest.raises(ValueError, match="inconsistent raw lifetime"):
        verifier._endpoints(broken, (.5, .7, .9))


def test_warning_persistence_allows_retraction_and_uses_current_and_past_rows() -> None:
    frame = _prediction(8)
    frame["y_pred"] = [8, 4, 3, 8, 2, 1, 0, 0]
    event = verifier._warning_events(frame, {"threshold": 4, "persistence": 2}, 3).iloc[0]
    assert event.first_warning_cycle == 3
    assert event.lead_time_cycles == 5
    assert event.on_time
    assert event.n_retractions == 1
    assert event.n_alarm_episodes == 2
    assert event.n_alarm_rows == 4
    assert event.early_alarm_burden == .25


def test_policy_calibration_is_deterministic_and_uses_inner_target() -> None:
    frame = _prediction(12)
    policy, grid = verifier._policy(frame, _config(thresholds=(2, 3, 4), persistences=(1, 2), lead_requirement=3))
    assert policy == {"threshold": 3, "persistence": 1}
    assert grid.selected.sum() == 1
    assert grid.loc[grid.selected, "early_alarm_burden"].iloc[0] == 0


def _inner() -> pd.DataFrame:
    return pd.DataFrame([
        {"procedure": arm, "trajectory_rmse": rmse, "endpoint_mean_engine_mae": 10.0,
         "endpoint_mean_engine_nasa": 2.0, "on_time_rate": .9, "early_alarm_burden": .2}
        for arm, rmse in zip(verifier.ARMS, (30.0, 29.0, 28.0))
    ])


def test_inner_selection_excludes_better_rmse_when_endpoint_or_warning_worsens() -> None:
    records = _inner()
    records.loc[records.procedure.eq(verifier.ARMS[2]), "endpoint_mean_engine_nasa"] = 2.01
    winner, controls = verifier._choose(records, _config())
    assert winner == verifier.ARMS[1]
    assert not controls[verifier.ARMS[2]]["eligible"]
    records.loc[records.procedure.eq(verifier.ARMS[1]), "on_time_rate"] = .89
    winner, _ = verifier._choose(records, _config())
    assert winner == verifier.ARMS[0]


def test_inner_selection_rejects_nan_before_a_false_fallback() -> None:
    records = _inner()
    records.loc[1, "trajectory_rmse"] = np.nan
    with pytest.raises(ValueError, match="finite inner selection"):
        verifier._choose(records, _config())


def _engines() -> pd.DataFrame:
    frame = pd.DataFrame({"trajectory_sse": [4.0, 90.0, 30.0], "trajectory_n_rows": [1, 5, 10]},
                         index=pd.Index([1, 2, 3], name="engine_id"))
    for column in verifier.ENGINE_METRICS:
        frame[column] = [1.0, 2.0, 3.0]
    return frame


def test_paired_rmse_uses_pooled_sse_counts_and_preserves_sample_multiplicity() -> None:
    before = _engines()
    after = before.copy()
    after["trajectory_sse"] = [3.0, 9.0, 60.0]
    config = _config(bootstrap_resamples=31)
    result = verifier._paired(before, after, config).set_index("metric")
    indices = np.random.default_rng(42).integers(0, 3, size=(31, 3))
    manual = []
    for draw in indices:
        old = sum(before.iloc[int(i)].trajectory_sse for i in draw)
        new = sum(after.iloc[int(i)].trajectory_sse for i in draw)
        n = sum(before.iloc[int(i)].trajectory_n_rows for i in draw)
        manual.append(np.sqrt(new / n) - np.sqrt(old / n))
    row = result.loc["trajectory_rmse"]
    assert row.delta == pytest.approx(np.sqrt(72 / 16) - np.sqrt(124 / 16))
    assert row.ci_lower == pytest.approx(np.quantile(manual, .025))
    assert row.ci_upper == pytest.approx(np.quantile(manual, .975))
    assert row.baseline != pytest.approx(np.sqrt(before.trajectory_sse / before.trajectory_n_rows).mean())
    assert any(len(set(draw)) < len(draw) for draw in indices)


def test_paired_engine_order_is_strict_and_mean_summaries_are_equal_engine() -> None:
    before = _engines()
    after = before.copy()
    after["endpoint_mean_engine_mae"] += [1.0, -1.0, 3.0]
    result = verifier._paired(before, after, _config(bootstrap_resamples=17)).set_index("metric")
    assert result.loc["endpoint_mean_engine_mae", "delta"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="identical ordered engine indices"):
        verifier._paired(before, after.iloc[::-1], _config())


def test_fixed_gate_requires_strict_rmse_improvement_and_all_other_bounds() -> None:
    rows = pd.DataFrame({"metric": ["trajectory_rmse", "endpoint_mean_engine_mae", "endpoint_mean_engine_nasa", "on_time_rate", "early_alarm_burden"],
                         "ci_lower": [-.2, -.1, -.1, 0, -.1], "ci_upper": [-.01, 0, 0, .1, 0]})
    assert all(verifier._gate(rows).values())
    rows.loc[rows.metric.eq("trajectory_rmse"), "ci_upper"] = 0
    assert not verifier._gate(rows)["trajectory_rmse_improved"]
    rows.loc[rows.metric.eq("on_time_rate"), "ci_lower"] = -.0001
    assert not verifier._gate(rows)["warning_on_time_not_worse"]


def _observations() -> pd.DataFrame:
    rows = [["test", 1, cycle, *[float(cycle + i) for i in range(24)]] for cycle in range(1, 33)]
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).astype({
        "split": "string", "engine_id": "int64", "cycle": "int64", **dict.fromkeys(MEASUREMENTS, "float64")})


def test_numpy_causal_reconstruction_has_hand_calculated_values_and_no_future() -> None:
    observations = _observations()
    expected = verifier._causal_expected(observations)
    assert expected.loc[0, "sensor_1__w10__mean"] == 28.5
    assert expected.loc[0, "sensor_1__mean"] == 18.5
    assert expected.loc[0, "sensor_1__from_initial_last"] == 14.5
    assert expected.loc[0, "sensor_1__initial30_std"] == pytest.approx(np.std(np.arange(1., 31.), ddof=1))
    changed = observations.copy()
    changed.loc[changed.cycle.eq(32), "sensor_1"] = 999
    actual = verifier._causal_expected(changed)
    pd.testing.assert_frame_equal(expected.loc[expected.cycle.le(31)].reset_index(drop=True),
                                  actual.loc[actual.cycle.le(31)].reset_index(drop=True), check_exact=True)
    observations["rul"] = 1
    with pytest.raises(ValueError, match="label-free"):
        verifier._causal_expected(observations)


def test_hash_integrity_detects_changed_bytes_and_rejects_escaped_paths(tmp_path: Path) -> None:
    artifact = tmp_path / "source.py"
    artifact.write_text("immutable", encoding="utf-8")
    hashes = {"source.py": verifier._sha(artifact)}
    assert verifier._hashes(tmp_path, hashes)
    artifact.write_text("changed", encoding="utf-8")
    assert not verifier._hashes(tmp_path, hashes)
    with pytest.raises(ValueError, match="escapes repository"):
        verifier._resolve(tmp_path, "../outside.py")


def test_equal_rmse_candidate_cannot_replace_baseline_even_with_better_endpoint() -> None:
    records = _inner()
    records["trajectory_rmse"] = 30.0
    records.loc[records.procedure.eq(verifier.ARMS[2]), "endpoint_mean_engine_mae"] = 1.0
    winner, controls = verifier._choose(records, _config())
    assert winner == verifier.ARMS[0]
    assert not controls[verifier.ARMS[2]]["trajectory_improves"]


def test_cache_integrity_requires_checksum_bound_raw_truth_parameters_and_weights(tmp_path: Path) -> None:
    columns = list(dict.fromkeys((*verifier.KEY,
        *(column for arm in verifier.ARMS for column in verifier.FEATURE_COLUMNS_BY_ARM[arm]), "rul")))
    data = {column: [0.0, 0.0] for column in columns}
    data.update(split=["train", "train"], engine_id=[1, 2], cycle=[30, 30], rul=[1.0, 2.0])
    train = pd.DataFrame(data).loc[:, columns]
    cfg = _config(outer_splits=2, inner_splits=2)
    task = {"source_identity": "source", "source_hash": verifier._frame_hash(train), "arm_id": verifier.ARMS[0],
            "feature_columns": list(verifier.FEATURE_COLUMNS_BY_ARM[verifier.ARMS[0]]), "config": cfg,
            "fit_engine_ids": [1], "validation_engine_ids": [2]}
    key = verifier.hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    folder = tmp_path / "fit_cache"
    folder.mkdir()
    parquet = folder / f"{key}.parquet"
    prediction = verifier._score(pd.DataFrame({"split": ["train"], "engine_id": [2], "cycle": [30],
                                                "candidate_id": [verifier.ARMS[0]], "y_true": [2.0], "y_pred": [1.0]}))
    prediction.to_parquet(parquet, index=False)
    metadata = {"identity": json.loads(json.dumps(task)), "cache_key": key, "prediction_sha256": verifier._sha(parquet),
                "effective_parameters": verifier._params(cfg),
                "weight_metadata": {"scheme": "engine_total_one", "total_weight": 1., "n_rows": 1, "n_engines": 1,
                                    "min_row_weight": 1., "max_row_weight": 1., "fit_threads": cfg["fit_threads"], "target_cap": None}}
    meta_path = parquet.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    checks = {}
    verifier._verify_cache(tmp_path, {}, "source", train, json.loads(json.dumps(cfg)), checks)
    assert checks["checksum_bound_cache_raw_truth_scores_and_fixed_hgb_weights"]
    assert not checks["complete_unique_cv_fit_count"]
    metadata["weight_metadata"]["target_cap"] = 125
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    checks = {}
    verifier._verify_cache(tmp_path, {}, "source", train, json.loads(json.dumps(cfg)), checks)
    assert not checks["checksum_bound_cache_raw_truth_scores_and_fixed_hgb_weights"]
