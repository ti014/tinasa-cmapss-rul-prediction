"""Tạo cặp và thống kê sensor đầu/cuối đời trên train FD001.

Mô-đun này hiện chỉ hiện thực hóa đơn vị phân tích đã chốt: mỗi engine train
đóng góp mean của 30 cycles đầu, mean của 30 cycles cuối và chênh lệch cuối trừ
đầu cho từng trong 21 sensors. Cấu hình suy luận M2.0 được cố định trong
`SensorInferenceConfig`; các hàm ở đây không đọc test labels hoặc ghi artifact.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import PermutationMethod, false_discovery_control, wilcoxon

from cmapss.data.contracts import (
    ENGINE_SENSOR_PAIR_COLUMNS,
    ENGINE_SENSOR_PAIR_KEY,
    PAIRED_SENSOR_WINDOW_CYCLES,
    SENSORS,
    SENSOR_INFERENCE_COLUMNS,
    ContractError,
)
from cmapss.data.validation import validate_columns, validate_integer, validate_observations

_PAIR_SOURCE = "engine sensor pairs"
_INFERENCE_SOURCE = "sensor inference"
_FLOAT_PAIR_COLUMNS = ("early_mean", "late_mean", "difference")
_INTEGER_PAIR_COLUMNS = (
    "n_cycles",
    "early_start",
    "early_end",
    "late_start",
    "late_end",
    "n_early",
    "n_late",
)
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
_BOOTSTRAP_PURPOSE = 1
_WILCOXON_PURPOSE = 2
_TEST_STATUS_OK = "ok"
_TEST_STATUS_ALL_ZERO = "all_zero_difference"
_BOOTSTRAP_STATUS_OK = "ok"
_BOOTSTRAP_STATUS_DEGENERATE = "degenerate"


@dataclass(frozen=True)
class SensorInferenceConfig:
    """Cấu hình M2.0 cho suy luận trên differences theo sensor."""

    bootstrap_resamples: int = 10_000
    bootstrap_confidence_level: float = 0.95
    bootstrap_method: str = "percentile"
    bootstrap_quantile_method: str = "linear"
    wilcoxon_alternative: str = "two-sided"
    wilcoxon_zero_method: str = "wilcox"
    wilcoxon_correction: bool = False
    wilcoxon_method: str = "permutation"
    wilcoxon_permutation_resamples: int = 100_000
    fdr_method: str = "bh"
    fdr_q: float = 0.05
    fdr_family_size: int = len(SENSORS)
    seed: int = 42

    def validate(self) -> None:
        """Kiểm tra cấu hình không dựa vào default ngầm của thư viện."""
        if self.bootstrap_method != "percentile":
            raise ContractError("INFERENCE_CONFIG", "bootstrap_method", "M2.0 chỉ chấp nhận percentile")
        if self.bootstrap_resamples < 1:
            raise ContractError("INFERENCE_CONFIG", "bootstrap_resamples", "cần số lần bootstrap dương")
        if not 0.0 < self.bootstrap_confidence_level < 1.0:
            raise ContractError("INFERENCE_CONFIG", "bootstrap_confidence_level", "cần nằm trong (0, 1)")
        if self.wilcoxon_alternative != "two-sided":
            raise ContractError("INFERENCE_CONFIG", "wilcoxon_alternative", "M2.0 chỉ chấp nhận two-sided")
        if self.wilcoxon_zero_method != "wilcox":
            raise ContractError("INFERENCE_CONFIG", "wilcoxon_zero_method", "M2.0 chỉ chấp nhận wilcox")
        if self.wilcoxon_correction:
            raise ContractError("INFERENCE_CONFIG", "wilcoxon_correction", "M2.0 không dùng continuity correction")
        if self.wilcoxon_method != "permutation":
            raise ContractError("INFERENCE_CONFIG", "wilcoxon_method", "M2.0 yêu cầu permutation, không dùng auto")
        if self.wilcoxon_permutation_resamples < 1:
            raise ContractError("INFERENCE_CONFIG", "wilcoxon_permutation_resamples", "cần số lần permutation dương")
        if self.fdr_method != "bh":
            raise ContractError("INFERENCE_CONFIG", "fdr_method", "M2.0 dùng Benjamini-Hochberg")
        if not 0.0 < self.fdr_q < 1.0:
            raise ContractError("INFERENCE_CONFIG", "fdr_q", "cần nằm trong (0, 1)")
        if self.fdr_family_size != len(SENSORS):
            raise ContractError("INFERENCE_CONFIG", "fdr_family_size", "family phải gồm đủ 21 sensors")


def _require_train_pairs_source(observations: pd.DataFrame) -> None:
    """Kiểm tra observations trước khi tạo cặp đầu/cuối."""
    validate_observations(observations, source="paired sensor observations")
    if not observations["split"].eq("train").fillna(False).all():
        raise ContractError(
            "TRAIN_ONLY",
            "paired sensor analysis",
            "chỉ nhận observations train; không dùng test cho paired analysis",
        )

    starts = observations.groupby("engine_id", sort=True)["cycle"].min()
    if not starts.eq(1).all():
        engine_id = int(starts.loc[~starts.eq(1)].index[0])
        raise ContractError(
            "CYCLE_START",
            f"engine_id={engine_id}",
            "mỗi engine phải bắt đầu tại cycle 1 trước khi lấy 30 cycles đầu",
        )

    last_cycles = observations.groupby("engine_id", sort=True)["cycle"].max()
    if not last_cycles.ge(PAIRED_SENSOR_WINDOW_CYCLES * 2).all():
        engine_id = int(last_cycles.loc[~last_cycles.ge(PAIRED_SENSOR_WINDOW_CYCLES * 2)].index[0])
        raise ContractError(
            "PAIR_WINDOW",
            f"engine_id={engine_id}",
            f"cần ít nhất {PAIRED_SENSOR_WINDOW_CYCLES * 2} cycles cho hai đoạn không chồng nhau",
        )


def _pair_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Tạo DataFrame cặp với dtype cố định cho serialization về sau."""
    pairs = pd.DataFrame(rows, columns=ENGINE_SENSOR_PAIR_COLUMNS)
    return pairs.astype(
        {
            "split": "string",
            "engine_id": "int64",
            "sensor": "string",
            **dict.fromkeys(_INTEGER_PAIR_COLUMNS, "int64"),
            **dict.fromkeys(_FLOAT_PAIR_COLUMNS, "float64"),
            "pair_status": "string",
        }
    )


def _inference_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Tạo DataFrame inference với dtype ổn định cho xuất CSV/JSON sau này."""
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
    """Tạo RNG ổn định theo seed, sensor và mục đích tính toán."""
    return np.random.default_rng(np.random.SeedSequence([config.seed, sensor_index + 1, purpose]))


def _difference_vector(differences: np.ndarray | pd.Series, *, source: str) -> np.ndarray:
    """Chuẩn hóa vector difference và chặn giá trị không hữu hạn."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ContractError(
            "INFERENCE_NUMERIC",
            source,
            "cần vector float64 một chiều, không rỗng và hữu hạn",
        )
    return values


def build_engine_sensor_pairs(observations: pd.DataFrame) -> pd.DataFrame:
    """Tạo một cặp mean đầu/cuối cho mỗi engine train và sensor.

    Hàm cố định window ở 30 cycles theo protocol đã chốt. Cycle đầu/cuối được
    xác định từ thứ tự cycle trong từng engine, không từ vị trí dòng input.

    Args:
        observations: Bảng observations train có schema data contract đầy đủ.

    Returns:
        DataFrame theo `ENGINE_SENSOR_PAIR_COLUMNS`, có thứ tự engine tăng dần
        rồi `SENSORS`; mỗi dòng biểu diễn một `(engine_id, sensor)` hợp lệ.

    Raises:
        ContractError: Khi observations không hợp lệ, không chỉ có train,
            không bắt đầu ở cycle 1, hoặc một engine có dưới 60 cycles.

    Examples:
        Một sensor có giá trị bằng cycle trong engine dài 60 cycles tạo mean
        đầu 15,5, mean cuối 45,5 và difference 30.
    """
    _require_train_pairs_source(observations)
    rows: list[dict[str, object]] = []

    for engine_id, engine in observations.groupby("engine_id", sort=True):
        ordered = engine.sort_values("cycle")
        n_cycles = int(ordered["cycle"].max())
        late_start = n_cycles - PAIRED_SENSOR_WINDOW_CYCLES + 1
        early = ordered.loc[ordered["cycle"].between(1, PAIRED_SENSOR_WINDOW_CYCLES)]
        late = ordered.loc[ordered["cycle"].between(late_start, n_cycles)]

        if len(early) != PAIRED_SENSOR_WINDOW_CYCLES or len(late) != PAIRED_SENSOR_WINDOW_CYCLES:
            raise ContractError(
                "PAIR_WINDOW",
                f"engine_id={int(engine_id)}",
                "không tạo được đúng 30 observations cho mỗi đoạn đầu/cuối",
            )

        for sensor in SENSORS:
            early_mean = float(early[sensor].mean())
            late_mean = float(late[sensor].mean())
            rows.append(
                {
                    "split": "train",
                    "engine_id": int(engine_id),
                    "sensor": sensor,
                    "n_cycles": n_cycles,
                    "early_start": 1,
                    "early_end": PAIRED_SENSOR_WINDOW_CYCLES,
                    "late_start": late_start,
                    "late_end": n_cycles,
                    "n_early": len(early),
                    "n_late": len(late),
                    "early_mean": early_mean,
                    "late_mean": late_mean,
                    "difference": late_mean - early_mean,
                    "pair_status": "valid",
                }
            )

    pairs = _pair_frame(rows)
    validate_engine_sensor_pairs(pairs)
    return pairs


def bootstrap_mean_difference(
    differences: np.ndarray | pd.Series,
    config: SensorInferenceConfig,
    *,
    sensor_index: int,
) -> dict[str, float | int | str]:
    """Tính mean paired difference và percentile bootstrap CI.

    Bootstrap lấy mẫu lại engine, không lấy mẫu cycle, để giữ đúng đơn vị phân
    tích đã chốt. Hàm trả metadata phương pháp để notebook và artifact không
    phải suy ra từ default của SciPy/NumPy.
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
    """Tính Wilcoxon signed-rank theo cấu hình permutation đã chốt."""
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
    """Hiệu chỉnh BH trên đúng family 21 sensors, giữ NA ở output public."""
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
    """Tóm tắt effect size, CI, Wilcoxon và FDR cho đủ 21 sensors."""
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


def validate_engine_sensor_pairs(pairs: pd.DataFrame) -> None:
    """Kiểm tra contract của bảng cặp engine/sensor đã tạo.

    Validator giữ bảng ở dạng canonical để các artifact và bước suy luận tiếp
    theo không ghép nhầm engine, bỏ sensor hoặc thay đổi chiều chênh lệch.

    Args:
        pairs: Bảng cặp có schema `ENGINE_SENSOR_PAIR_COLUMNS`.

    Raises:
        ContractError: Khi schema, khóa, phạm vi train, window, thứ tự, số liệu
            hoặc công thức difference không phù hợp contract.
    """
    validate_columns(pairs, ENGINE_SENSOR_PAIR_COLUMNS, _PAIR_SOURCE)
    if not pairs["split"].eq("train").fillna(False).all():
        raise ContractError("TRAIN_ONLY", _PAIR_SOURCE, "bảng cặp chỉ thuộc split train")
    if not pairs["sensor"].isin(SENSORS).all():
        raise ContractError("PAIR_SENSOR", _PAIR_SOURCE, "sensor phải thuộc đủ 21 sensors đã chốt")
    if pairs.duplicated(list(ENGINE_SENSOR_PAIR_KEY)).any():
        raise ContractError("PAIR_KEY", _PAIR_SOURCE, "trùng khóa (split, engine_id, sensor)")

    validate_integer(pairs, "engine_id", 1, _PAIR_SOURCE)
    for column in _INTEGER_PAIR_COLUMNS:
        validate_integer(pairs, column, 1, _PAIR_SOURCE)
    for column in _FLOAT_PAIR_COLUMNS:
        values = pairs[column]
        if values.dtype != np.dtype("float64") or values.isna().any() or not np.isfinite(values).all():
            raise ContractError(
                "PAIR_NUMERIC",
                f"{_PAIR_SOURCE}.{column}",
                "cần dtype float64 với giá trị hữu hạn, không null",
            )

    for engine_id, engine_pairs in pairs.groupby("engine_id", sort=True):
        if len(engine_pairs) != len(SENSORS) or set(engine_pairs["sensor"]) != set(SENSORS):
            raise ContractError(
                "PAIR_SENSOR",
                f"engine_id={int(engine_id)}",
                "mỗi engine phải có đúng một cặp cho từng trong 21 sensors",
            )
        if engine_pairs["n_cycles"].nunique() != 1:
            raise ContractError(
                "PAIR_WINDOW",
                f"engine_id={int(engine_id)}",
                "n_cycles phải nhất quán giữa các sensors của cùng engine",
            )

    expected_keys = [
        (int(engine_id), sensor)
        for engine_id in sorted(pairs["engine_id"].unique())
        for sensor in SENSORS
    ]
    actual_keys = list(pairs.loc[:, ["engine_id", "sensor"]].itertuples(index=False, name=None))
    if actual_keys != expected_keys:
        raise ContractError(
            "PAIR_ORDER",
            _PAIR_SOURCE,
            "các dòng phải theo engine_id tăng dần rồi sensor_1 đến sensor_21",
        )

    window_is_valid = (
        pairs["n_cycles"].ge(PAIRED_SENSOR_WINDOW_CYCLES * 2)
        & pairs["early_start"].eq(1)
        & pairs["early_end"].eq(PAIRED_SENSOR_WINDOW_CYCLES)
        & pairs["late_start"].eq(pairs["n_cycles"] - PAIRED_SENSOR_WINDOW_CYCLES + 1)
        & pairs["late_end"].eq(pairs["n_cycles"])
        & pairs["n_early"].eq(PAIRED_SENSOR_WINDOW_CYCLES)
        & pairs["n_late"].eq(PAIRED_SENSOR_WINDOW_CYCLES)
        & pairs["early_end"].lt(pairs["late_start"])
    )
    if not window_is_valid.all():
        raise ContractError(
            "PAIR_WINDOW",
            _PAIR_SOURCE,
            "ranh giới hoặc số observations của hai đoạn 30 cycles không hợp lệ",
        )
    expected_difference = pairs["late_mean"] - pairs["early_mean"]
    if not pairs["difference"].eq(expected_difference).all():
        raise ContractError(
            "PAIR_DIFFERENCE",
            _PAIR_SOURCE,
            "difference phải bằng late_mean - early_mean trước khi làm tròn trình bày",
        )
    if not pairs["pair_status"].eq("valid").fillna(False).all():
        raise ContractError("PAIR_STATUS", _PAIR_SOURCE, "mọi dòng đã xuất phải có pair_status=valid")


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
