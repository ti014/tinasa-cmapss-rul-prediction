"""Kiểm tra contract của bảng suy luận sensor FD001."""

import numpy as np
import pandas as pd

from cmapss.data.contracts import SENSOR_INFERENCE_COLUMNS, SENSORS, ContractError
from cmapss.data.validation import validate_columns, validate_integer

from .config import (
    _BOOTSTRAP_STATUS_DEGENERATE,
    _BOOTSTRAP_STATUS_OK,
    _TEST_STATUS_ALL_ZERO,
    _TEST_STATUS_OK,
    SensorInferenceConfig,
)

_INFERENCE_SOURCE = "sensor inference"


def _validate_sensor_order(inference: pd.DataFrame) -> None:
    actual = inference["sensor"].tolist()
    if actual != list(SENSORS):
        raise ContractError(
            "INFERENCE_SENSOR",
            _INFERENCE_SOURCE,
            "bảng suy luận phải giữ đủ 21 sensors theo thứ tự sensor_1 đến sensor_21",
        )


def _validate_raw_pvalues(inference: pd.DataFrame) -> None:
    statuses = set(inference["test_status"].dropna())
    if not statuses.issubset({_TEST_STATUS_OK, _TEST_STATUS_ALL_ZERO}):
        raise ContractError("INFERENCE_STATUS", _INFERENCE_SOURCE, "test_status không thuộc tập đã chốt")
    ok_mask = inference["test_status"].eq(_TEST_STATUS_OK)
    raw_p = inference["raw_p_value"].astype("float64")
    ok_raw = raw_p[ok_mask]
    if ok_raw.isna().any() or not ok_raw.between(0.0, 1.0, inclusive="both").all():
        raise ContractError("INFERENCE_PVALUE", _INFERENCE_SOURCE, "raw p-value phải thuộc [0, 1] khi kiểm định được")
    if raw_p[~ok_mask].notna().any():
        raise ContractError("INFERENCE_PVALUE", _INFERENCE_SOURCE, "raw p-value phải là NA khi không kiểm định được")


def validate_sensor_inference(
    inference: pd.DataFrame,
    config: SensorInferenceConfig | None = None,
) -> None:
    """Kiểm tra contract của bảng suy luận theo sensor.

    Validator không suy diễn kết quả thống kê; nó bảo vệ family 21 sensors,
    trạng thái all-zero, khoảng p-value và cấu hình M2.0 đã chốt.
    """
    resolved_config = config or SensorInferenceConfig()
    resolved_config.validate()
    validate_columns(inference, SENSOR_INFERENCE_COLUMNS, _INFERENCE_SOURCE)
    _validate_sensor_order(inference)

    validate_integer(inference, "n_pairs", 1, _INFERENCE_SOURCE)
    for column in ("n_positive", "n_negative", "n_zero"):
        validate_integer(inference, column, 0, _INFERENCE_SOURCE)
    validate_integer(inference, "bootstrap_resamples", 1, _INFERENCE_SOURCE)
    if not (inference["n_positive"] + inference["n_negative"] + inference["n_zero"]).eq(inference["n_pairs"]).all():
        raise ContractError("INFERENCE_COUNTS", _INFERENCE_SOURCE, "n_positive+n_negative+n_zero phải bằng n_pairs")

    finite_columns = ("mean_difference", "median_difference", "ci_level", "ci_lower", "ci_upper", "fdr_q")
    for column in finite_columns:
        values = inference[column]
        if values.dtype != np.dtype("float64") or values.isna().any() or not np.isfinite(values).all():
            raise ContractError("INFERENCE_NUMERIC", f"{_INFERENCE_SOURCE}.{column}", "cần float64 hữu hạn")
    if not inference["ci_lower"].le(inference["ci_upper"]).all():
        raise ContractError("INFERENCE_CI", _INFERENCE_SOURCE, "ci_lower phải <= ci_upper")
    if not inference["ci_method"].eq(resolved_config.bootstrap_method).all():
        raise ContractError("INFERENCE_CONFIG", "ci_method", "không khớp cấu hình M2.0")
    if not inference["ci_level"].eq(resolved_config.bootstrap_confidence_level).all():
        raise ContractError("INFERENCE_CONFIG", "ci_level", "không khớp cấu hình M2.0")
    if not inference["bootstrap_resamples"].eq(resolved_config.bootstrap_resamples).all():
        raise ContractError("INFERENCE_CONFIG", "bootstrap_resamples", "không khớp cấu hình M2.0")
    if not inference["fdr_method"].eq(resolved_config.fdr_method).all():
        raise ContractError("INFERENCE_CONFIG", "fdr_method", "không khớp cấu hình M2.0")
    if not inference["fdr_q"].eq(resolved_config.fdr_q).all():
        raise ContractError("INFERENCE_CONFIG", "fdr_q", "không khớp cấu hình M2.0")

    if not inference["bootstrap_status"].isin({_BOOTSTRAP_STATUS_OK, _BOOTSTRAP_STATUS_DEGENERATE}).all():
        raise ContractError("INFERENCE_STATUS", "bootstrap_status", "trạng thái bootstrap không hợp lệ")
    _validate_raw_pvalues(inference)

    ok_mask = inference["test_status"].eq(_TEST_STATUS_OK)
    p_adjusted = inference["p_value_adjusted"].astype("float64")
    ok_adjusted = p_adjusted[ok_mask]
    if ok_adjusted.isna().any() or not ok_adjusted.between(0.0, 1.0, inclusive="both").all():
        raise ContractError("INFERENCE_PVALUE", _INFERENCE_SOURCE, "adjusted p-value phải thuộc [0, 1] khi kiểm định được")
    if p_adjusted[~ok_mask].notna().any():
        raise ContractError("INFERENCE_PVALUE", _INFERENCE_SOURCE, "adjusted p-value phải là NA khi không kiểm định được")

    statistic = inference["wilcoxon_statistic"].astype("float64")
    if statistic[ok_mask].isna().any() or not np.isfinite(statistic[ok_mask]).all():
        raise ContractError("INFERENCE_NUMERIC", "wilcoxon_statistic", "statistic phải hữu hạn khi kiểm định được")
    if statistic[~ok_mask].notna().any():
        raise ContractError("INFERENCE_NUMERIC", "wilcoxon_statistic", "statistic phải NA khi không kiểm định được")
    expected_method = inference["wilcoxon_method"].where(ok_mask, "not_applicable")
    if not expected_method[ok_mask].eq(resolved_config.wilcoxon_method).all():
        raise ContractError("INFERENCE_CONFIG", "wilcoxon_method", "không khớp cấu hình M2.0")
    if not expected_method[~ok_mask].eq("not_applicable").all():
        raise ContractError("INFERENCE_CONFIG", "wilcoxon_method", "trường hợp không kiểm định được phải ghi not_applicable")

    reject = inference["fdr_reject"]
    if not reject[ok_mask].notna().all() or not reject[~ok_mask].isna().all():
        raise ContractError("INFERENCE_PVALUE", "fdr_reject", "fdr_reject chỉ xác định khi kiểm định được")
