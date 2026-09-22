"""Kiểm tra xuất và đọc lại artifact thống kê sensor FD001."""

from cmapss.statistics import SensorInferenceConfig, validate_engine_sensor_pairs, validate_sensor_inference
from cmapss.statistics_artifacts import write_sensor_statistics_artifacts


def test_statistics_artifact_writer_exports_roundtrippable_tables(raw_dir, tmp_path) -> None:
    config = SensorInferenceConfig(
        bootstrap_resamples=32,
        wilcoxon_permutation_resamples=64,
        seed=42,
    )

    result = write_sensor_statistics_artifacts(
        raw_dir=raw_dir,
        output_dir=tmp_path,
        config=config,
    )

    pair_path = tmp_path / "tables" / "engine_sensor_pairs.csv"
    inference_path = tmp_path / "tables" / "sensor_inference.csv"
    assert pair_path.exists()
    assert inference_path.exists()
    assert (tmp_path / "run_info.json").exists()
    assert (tmp_path / "validation.json").exists()
    validate_engine_sensor_pairs(result.pairs)
    validate_sensor_inference(result.inference, config)
    assert result.run_info["input_sha256"]
    assert result.run_info["test_data_used"] is False
    assert result.run_info["config"]["bootstrap_resamples"] == 32
    assert result.validation["status"] == "passed"
    assert result.validation["engine_sensor_pairs"]["rows"] == 100 * 21
    assert result.validation["sensor_inference"]["rows"] == 21
    assert result.validation["raw_sha256_unchanged"] is True
