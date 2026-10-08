"""Tạo và kiểm tra bảng cặp sensor đầu/cuối đời trên train FD001."""

import numpy as np
import pandas as pd

from cmapss.data.contracts import (
    ENGINE_SENSOR_PAIR_COLUMNS,
    ENGINE_SENSOR_PAIR_KEY,
    PAIRED_SENSOR_WINDOW_CYCLES,
    SENSORS,
    ContractError,
)
from cmapss.data.validation import validate_columns, validate_integer, validate_observations

_PAIR_SOURCE = "engine sensor pairs"
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


def _require_train_pairs_source(observations: pd.DataFrame) -> None:
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


def build_engine_sensor_pairs(observations: pd.DataFrame) -> pd.DataFrame:
    """Mean 30 cycle đầu và 30 cycle cuối cho mỗi (engine train, sensor).

    `difference = late_mean - early_mean`. Đầu/cuối xác định theo giá trị
    `cycle`, không theo vị trí dòng. Mỗi engine cần ít nhất 60 cycles.
    Ví dụ: sensor có giá trị bằng cycle, engine dài 60 cycles cho mean đầu
    15,5, mean cuối 45,5, difference 30.
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


def validate_engine_sensor_pairs(pairs: pd.DataFrame) -> None:
    """Kiểm tra schema, khóa, thứ tự, window và công thức difference của bảng cặp."""
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
