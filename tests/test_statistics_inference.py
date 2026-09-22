"""Kiểm tra bootstrap, Wilcoxon, FDR và tóm tắt inference sensor FD001."""

import numpy as np
import pandas as pd
import pytest

from cmapss.data.contracts import ContractError, SENSORS
from cmapss.data.readers import read_observations
from cmapss.data.validation import validate_fd001_profile
from cmapss.statistics import (
    SensorInferenceConfig,
    adjust_sensor_pvalues,
    bootstrap_mean_difference,
    build_engine_sensor_pairs,
    paired_wilcoxon,
    summarize_sensor_changes,
    validate_engine_sensor_pairs,
    validate_sensor_inference,
)


def _pairs_from_differences(differences: dict[str, list[float]]) -> pd.DataFrame:
    """Tạo bảng cặp hợp lệ từ differences tính tay cho từng sensor."""
    n_engines = max(len(values) for values in differences.values())
    rows = []
    for engine_id in range(1, n_engines + 1):
        for sensor in SENSORS:
            sensor_differences = differences.get(sensor, [0.0] * n_engines)
            difference = float(sensor_differences[engine_id - 1])
            rows.append(
                {
                    "split": "train",
                    "engine_id": engine_id,
                    "sensor": sensor,
                    "n_cycles": 60,
                    "early_start": 1,
                    "early_end": 30,
                    "late_start": 31,
                    "late_end": 60,
                    "n_early": 30,
                    "n_late": 30,
                    "early_mean": 10.0,
                    "late_mean": 10.0 + difference,
                    "difference": difference,
                    "pair_status": "valid",
                }
            )
    pairs = pd.DataFrame(rows)
    validate_engine_sensor_pairs(pairs)
    return pairs


def test_bootstrap_mean_difference_reports_degenerate_interval() -> None:
    config = SensorInferenceConfig(bootstrap_resamples=99, seed=42)

    result = bootstrap_mean_difference(np.array([2.0, 2.0, 2.0]), config, sensor_index=0)

    assert result["mean_difference"] == 2.0
    assert result["ci_lower"] == 2.0
    assert result["ci_upper"] == 2.0
    assert result["bootstrap_status"] == "degenerate"
    assert result["ci_method"] == "percentile"


def test_paired_wilcoxon_matches_exhaustive_sign_flip_fixture() -> None:
    config = SensorInferenceConfig(wilcoxon_permutation_resamples=100_000, seed=42)

    result = paired_wilcoxon(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), config, sensor_index=0)

    assert result["test_status"] == "ok"
    assert result["wilcoxon_method"] == "permutation"
    assert result["wilcoxon_statistic"] == 0.0
    assert result["raw_p_value"] == 2 / 32


def test_paired_wilcoxon_keeps_all_zero_as_not_testable() -> None:
    config = SensorInferenceConfig(wilcoxon_permutation_resamples=100_000, seed=42)

    result = paired_wilcoxon(np.array([0.0, 0.0, 0.0]), config, sensor_index=0)

    assert result["test_status"] == "all_zero_difference"
    assert np.isnan(result["wilcoxon_statistic"])
    assert np.isnan(result["raw_p_value"])


def test_fdr_uses_full_sensor_family_and_keeps_all_zero_public_values_na() -> None:
    config = SensorInferenceConfig(bootstrap_resamples=99, fdr_q=0.05)
    rows = []
    raw_p_values = {"sensor_1": 0.001, "sensor_2": 0.02, "sensor_3": 0.04}
    for sensor in SENSORS:
        status = "ok" if sensor in raw_p_values else "all_zero_difference"
        rows.append(
            {
                "sensor": sensor,
                "n_pairs": 5,
                "n_positive": 5 if status == "ok" else 0,
                "n_negative": 0,
                "n_zero": 0 if status == "ok" else 5,
                "mean_difference": 1.0 if status == "ok" else 0.0,
                "median_difference": 1.0 if status == "ok" else 0.0,
                "ci_method": "percentile",
                "ci_level": 0.95,
                "ci_lower": 1.0 if status == "ok" else 0.0,
                "ci_upper": 1.0 if status == "ok" else 0.0,
                "bootstrap_resamples": 99,
                "bootstrap_status": "ok",
                "wilcoxon_method": "permutation" if status == "ok" else "not_applicable",
                "wilcoxon_statistic": 0.0 if status == "ok" else np.nan,
                "raw_p_value": raw_p_values.get(sensor, np.nan),
                "p_value_adjusted": np.nan,
                "fdr_method": "bh",
                "fdr_q": 0.05,
                "fdr_reject": pd.NA,
                "test_status": status,
            }
        )
    inference = pd.DataFrame(rows)

    adjusted = adjust_sensor_pvalues(inference, config)

    assert adjusted.loc[adjusted["sensor"].eq("sensor_1"), "p_value_adjusted"].item() == pytest.approx(0.021)
    assert adjusted.loc[adjusted["sensor"].eq("sensor_2"), "p_value_adjusted"].item() == pytest.approx(0.21)
    assert adjusted.loc[adjusted["sensor"].eq("sensor_3"), "p_value_adjusted"].item() == pytest.approx(0.28)
    assert bool(adjusted.loc[adjusted["sensor"].eq("sensor_1"), "fdr_reject"].item()) is True
    assert bool(adjusted.loc[adjusted["sensor"].eq("sensor_2"), "fdr_reject"].item()) is False
    assert pd.isna(adjusted.loc[adjusted["sensor"].eq("sensor_4"), "raw_p_value"].item())
    assert pd.isna(adjusted.loc[adjusted["sensor"].eq("sensor_4"), "p_value_adjusted"].item())
    assert pd.isna(adjusted.loc[adjusted["sensor"].eq("sensor_4"), "fdr_reject"].item())
    validate_sensor_inference(adjusted, config)


def test_summarize_sensor_changes_keeps_all_sensors_and_statuses() -> None:
    config = SensorInferenceConfig(
        bootstrap_resamples=99,
        wilcoxon_permutation_resamples=100_000,
        seed=42,
    )
    pairs = _pairs_from_differences(
        {
            "sensor_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "sensor_2": [0.0, 0.0, 0.0, 0.0, 0.0],
            "sensor_3": [2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )

    inference = summarize_sensor_changes(pairs, config)

    validate_sensor_inference(inference, config)
    assert inference["sensor"].tolist() == list(SENSORS)
    sensor_1 = inference.set_index("sensor").loc["sensor_1"]
    assert sensor_1["mean_difference"] == 3.0
    assert sensor_1["median_difference"] == 3.0
    assert sensor_1["raw_p_value"] == 2 / 32
    sensor_2 = inference.set_index("sensor").loc["sensor_2"]
    assert sensor_2["test_status"] == "all_zero_difference"
    assert pd.isna(sensor_2["raw_p_value"])
    assert pd.isna(sensor_2["p_value_adjusted"])
    sensor_3 = inference.set_index("sensor").loc["sensor_3"]
    assert sensor_3["bootstrap_status"] == "degenerate"
    assert sensor_3["ci_lower"] == 2.0
    assert sensor_3["ci_upper"] == 2.0


def test_sensor_inference_validator_rejects_invalid_artifacts() -> None:
    config = SensorInferenceConfig(bootstrap_resamples=99, wilcoxon_permutation_resamples=128)
    inference = summarize_sensor_changes(
        _pairs_from_differences({"sensor_1": [1.0, 2.0, 3.0]}),
        config,
    )

    with pytest.raises(ContractError, match=r"\[INFERENCE_SENSOR\]"):
        validate_sensor_inference(inference.drop(index=0).reset_index(drop=True), config)

    with pytest.raises(ContractError, match=r"\[INFERENCE_PVALUE\]"):
        validate_sensor_inference(inference.assign(raw_p_value=2.0), config)


def test_real_fd001_summarizes_all_sensors_without_writing_artifacts(raw_dir) -> None:
    observations = read_observations(raw_dir / "train_FD001.txt", split="train")
    validate_fd001_profile(observations, split="train")
    pairs = build_engine_sensor_pairs(observations)
    config = SensorInferenceConfig(
        bootstrap_resamples=128,
        wilcoxon_permutation_resamples=256,
        seed=42,
    )

    inference = summarize_sensor_changes(pairs, config)

    validate_sensor_inference(inference, config)
    assert inference["sensor"].tolist() == list(SENSORS)
    assert inference["n_pairs"].eq(100).all()
    assert set(inference["test_status"]).issubset({"ok", "all_zero_difference"})
    assert np.isfinite(inference[["mean_difference", "median_difference", "ci_lower", "ci_upper"]].to_numpy()).all()
