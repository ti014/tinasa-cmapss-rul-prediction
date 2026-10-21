"""Tính effect size, khoảng tin cậy, kiểm định và FDR cho sensor FD001."""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.stats import PermutationMethod, false_discovery_control, wilcoxon

from cmapss.data.contracts import SENSOR_INFERENCE_COLUMNS, SENSORS, ContractError
from cmapss.data.validation import validate_columns

from .config import (
    _BOOTSTRAP_PURPOSE,
    _BOOTSTRAP_STATUS_DEGENERATE,
    _BOOTSTRAP_STATUS_OK,
    _TEST_STATUS_ALL_ZERO,
    _TEST_STATUS_OK,
    _WILCOXON_PURPOSE,
    SensorInferenceConfig,
)
from .pairs import validate_engine_sensor_pairs
from .validation import (
    _validate_raw_pvalues,
    _validate_sensor_order,
    validate_sensor_inference,
)

_INFERENCE_SOURCE = "sensor inference"
_INFERENCE_INTEGER_COLUMNS = ("n_pairs", "n_positive", "n_negative", "n_zero", "bootstrap_resamples")
_INFERENCE_FLOAT_COLUMNS = (
    "mean_difference",
    "median_difference",
    "ci_level",
    "ci_lower",
    "ci_upper",
    "wilcoxon_statistic",
    "raw_p_value",
    "p_value_adjusted",
    "fdr_q",
)


def _inference_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """DataFrame inference với dtype cố định."""
    frame = pd.DataFrame(rows, columns=SENSOR_INFERENCE_COLUMNS)
    return frame.astype(
        {
            "sensor": "string",
            **dict.fromkeys(_INFERENCE_INTEGER_COLUMNS, "int64"),
            **dict.fromkeys(_INFERENCE_FLOAT_COLUMNS, "float64"),
            "ci_method": "string",
            "bootstrap_status": "string",
            "wilcoxon_method": "string",
            "fdr_method": "string",
            "fdr_reject": "boolean",
            "test_status": "string",
        }
    )


def _rng_for(config: SensorInferenceConfig, sensor_index: int, purpose: int) -> np.random.Generator:
    """RNG riêng cho từng (seed, sensor, mục đích) để kết quả tái lập được."""
    return np.random.default_rng(np.random.SeedSequence([config.seed, sensor_index + 1, purpose]))


def _difference_vector(differences: np.ndarray | pd.Series, *, source: str) -> np.ndarray:
    """Ép về float64 1 chiều và chặn giá trị không hữu hạn."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ContractError(
            "INFERENCE_NUMERIC",
            source,
            "cần vector float64 một chiều, không rỗng và hữu hạn",
        )
    return values


def bootstrap_mean_difference(
    differences: np.ndarray | pd.Series,
    config: SensorInferenceConfig,
    *,
    sensor_index: int,
) -> dict[str, float | int | str]:
    """Mean paired difference và percentile bootstrap CI.

    Bootstrap lấy mẫu lại engine (mỗi phần tử là một engine), không lấy mẫu cycle.
    """
    config.validate()
    values = _difference_vector(differences, source="bootstrap differences")
    rng = _rng_for(config, sensor_index, _BOOTSTRAP_PURPOSE)
    indices = rng.integers(0, values.size, size=(config.bootstrap_resamples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    alpha = 1.0 - config.bootstrap_confidence_level
    ci_lower, ci_upper = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method=config.bootstrap_quantile_method,
    )
    status = _BOOTSTRAP_STATUS_DEGENERATE if np.all(values == values[0]) else _BOOTSTRAP_STATUS_OK
    return {
        "mean_difference": float(values.mean()),
        "ci_method": config.bootstrap_method,
        "ci_level": float(config.bootstrap_confidence_level),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "bootstrap_resamples": int(config.bootstrap_resamples),
        "bootstrap_status": status,
    }


def paired_wilcoxon(
    differences: np.ndarray | pd.Series,
    config: SensorInferenceConfig,
    *,
    sensor_index: int,
) -> dict[str, float | str]:
    """Wilcoxon signed-rank dùng permutation; trả NA khi mọi difference bằng 0."""
    config.validate()
    values = _difference_vector(differences, source="wilcoxon differences")
    if np.all(values == 0.0):
        return {
            "wilcoxon_method": "not_applicable",
            "wilcoxon_statistic": np.nan,
            "raw_p_value": np.nan,
            "test_status": _TEST_STATUS_ALL_ZERO,
        }

    method = PermutationMethod(
        n_resamples=config.wilcoxon_permutation_resamples,
        rng=_rng_for(config, sensor_index, _WILCOXON_PURPOSE),
    )
    result = wilcoxon(
        values,
        zero_method=config.wilcoxon_zero_method,
        correction=config.wilcoxon_correction,
        alternative=config.wilcoxon_alternative,
        method=method,
    )
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not np.isfinite(statistic) or not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
        raise ContractError(
            "INFERENCE_PVALUE",
            "wilcoxon",
            "Wilcoxon trả statistic hoặc p-value không hợp lệ",
        )
    return {
        "wilcoxon_method": config.wilcoxon_method,
        "wilcoxon_statistic": statistic,
        "raw_p_value": p_value,
        "test_status": _TEST_STATUS_OK,
    }


def adjust_sensor_pvalues(inference: pd.DataFrame, config: SensorInferenceConfig) -> pd.DataFrame:
    """Hiệu chỉnh BH trên family 21 sensors; sensor không kiểm định được giữ NA."""
    config.validate()
    validate_columns(inference, SENSOR_INFERENCE_COLUMNS, _INFERENCE_SOURCE)
    _validate_sensor_order(inference)
    _validate_raw_pvalues(inference)

    adjusted = inference.copy()
    p_values = adjusted["raw_p_value"].astype("float64").fillna(1.0).to_numpy()
    corrected = false_discovery_control(p_values, method=config.fdr_method)
    ok_mask = adjusted["test_status"].eq(_TEST_STATUS_OK)
    adjusted.loc[ok_mask, "p_value_adjusted"] = corrected[ok_mask.to_numpy()]
    adjusted.loc[~ok_mask, "p_value_adjusted"] = np.nan
    adjusted.loc[ok_mask, "fdr_reject"] = adjusted.loc[ok_mask, "p_value_adjusted"].le(config.fdr_q)
    adjusted.loc[~ok_mask, "fdr_reject"] = pd.NA
    adjusted["fdr_method"] = config.fdr_method
    adjusted["fdr_q"] = config.fdr_q
    adjusted = _inference_frame(adjusted.to_dict("records"))
    validate_sensor_inference(adjusted, config)
    return adjusted


def summarize_sensor_changes(
    pairs: pd.DataFrame,
    config: SensorInferenceConfig | None = None,
) -> pd.DataFrame:
    """Effect size, CI, Wilcoxon và FDR cho 21 sensors."""
    resolved_config = config or SensorInferenceConfig()
    resolved_config.validate()
    validate_engine_sensor_pairs(pairs)

    rows: list[dict[str, object]] = []
    for sensor_index, sensor in enumerate(SENSORS):
        sensor_pairs = pairs.loc[pairs["sensor"].eq(sensor)].sort_values("engine_id")
        differences = sensor_pairs["difference"].to_numpy(dtype=np.float64)
        bootstrap = bootstrap_mean_difference(differences, resolved_config, sensor_index=sensor_index)
        test = paired_wilcoxon(differences, resolved_config, sensor_index=sensor_index)
        rows.append(
            {
                "sensor": sensor,
                "n_pairs": int(differences.size),
                "n_positive": int(np.sum(differences > 0.0)),
                "n_negative": int(np.sum(differences < 0.0)),
                "n_zero": int(np.sum(differences == 0.0)),
                "mean_difference": bootstrap["mean_difference"],
                "median_difference": float(np.median(differences)),
                "ci_method": bootstrap["ci_method"],
                "ci_level": bootstrap["ci_level"],
                "ci_lower": bootstrap["ci_lower"],
                "ci_upper": bootstrap["ci_upper"],
                "bootstrap_resamples": bootstrap["bootstrap_resamples"],
                "bootstrap_status": bootstrap["bootstrap_status"],
                "wilcoxon_method": test["wilcoxon_method"],
                "wilcoxon_statistic": test["wilcoxon_statistic"],
                "raw_p_value": test["raw_p_value"],
                "p_value_adjusted": np.nan,
                "fdr_method": resolved_config.fdr_method,
                "fdr_q": resolved_config.fdr_q,
                "fdr_reject": pd.NA,
                "test_status": test["test_status"],
            }
        )

    inference = _inference_frame(rows)
    return adjust_sensor_pvalues(inference, resolved_config)
