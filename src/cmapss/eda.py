"""Phép tính dùng lại trong notebook EDA; chỉ phân tích observations train."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _require_train(data: pd.DataFrame) -> None:
    if data.empty or not data["split"].eq("train").fillna(False).all():
        raise ValueError("EDA chỉ nhận dữ liệu train không rỗng; không dùng test.")


def engine_lifetimes(data: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp cycle đầu/cuối và số quan sát của từng động cơ train."""
    from cmapss.data.contracts import OBSERVATION_COLUMNS
    from cmapss.etl.pandas_pipeline import build_engine_summary

    _require_train(data)
    # Notebook truyền train đã ghép nhãn RUL; engine summary chỉ nhận đúng cột observations.
    return build_engine_summary(data.loc[:, list(OBSERVATION_COLUMNS)]).rename(columns={"n_cycles": "n_observations"}).loc[
        :, ["engine_id", "first_cycle", "last_cycle", "n_observations"]
    ]


def within_engine_variability(data: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tính độ biến thiên từng phép đo và số động cơ có giá trị hằng."""
    _require_train(data)
    measurements = data.melt(id_vars="engine_id", value_vars=list(columns),
                             var_name="measurement", value_name="value")
    within = measurements.groupby(["engine_id", "measurement"], as_index=False).agg(
        n_observations=("value", "size"), n_unique=("value", "nunique"),
        min=("value", "min"), max=("value", "max"), mean=("value", "mean"), std=("value", "std"),
    )
    within["range"] = within["max"] - within["min"]
    within["is_constant"] = within.n_unique.eq(1)
    counts = within.groupby("measurement", as_index=False).agg(
        constant_engines=("is_constant", "sum"), total_engines=("engine_id", "size"),
    )
    return within, counts


def engine_cycle_correlations(data: pd.DataFrame, column: str) -> pd.Series:
    """Tính Spearman rho giữa cycle và một đại lượng trong từng động cơ train."""
    _require_train(data)
    return data.groupby("engine_id").apply(
        lambda group: spearmanr(group.cycle, group[column]).statistic, include_groups=False,
    )


def describe_measurements(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Thống kê mô tả cho từng cột settings/sensors, một dòng mỗi cột.

    std là sample std (ddof=1). Cột hằng xác định bằng n_unique == 1, không dùng ngưỡng std.
    """
    _require_train(data)
    measurements = data[list(columns)]
    summary = measurements.describe().T.rename(
        columns={"count": "n_observations", "25%": "q25", "50%": "median", "75%": "q75"}
    )
    summary["n_unique"] = measurements.nunique()
    summary["missing"] = measurements.isna().sum()
    summary["range"] = summary["max"] - summary["min"]
    summary["is_constant"] = summary["n_unique"].eq(1)
    return summary.rename_axis("measurement").reset_index()


def select_example_engines(lifetimes: pd.DataFrame) -> pd.DataFrame:
    """Chọn engine có tuổi thọ gần Q25/Q50/Q75 nhất; hòa thì lấy ID nhỏ nhất.

    Một engine có thể xuất hiện ở nhiều phân vị nếu tuổi thọ trùng nhau.
    """
    selected = []
    for quantile in (0.25, 0.5, 0.75):
        target = lifetimes["last_cycle"].quantile(quantile, interpolation="linear")
        candidates = lifetimes.assign(distance=(lifetimes["last_cycle"] - target).abs())
        engine = candidates.sort_values(["distance", "engine_id"]).iloc[0]
        selected.append({
            "quantile": quantile,
            "target_lifetime": target,
            "engine_id": int(engine["engine_id"]),
            "last_cycle": int(engine["last_cycle"]),
        })
    return pd.DataFrame(selected)


def sensor_rul_correlations(data: pd.DataFrame, sensors: Sequence[str]) -> pd.DataFrame:
    """Tính Spearman rho giữa sensor và RUL trong từng engine.

    Trả một dòng mỗi engine/sensor; rho là NaN khi sensor hoặc RUL không đổi.
    """
    _require_train(data)
    rows = []
    for engine_id, engine in data.groupby("engine_id", sort=True):
        for sensor in sensors:
            defined = engine[sensor].nunique() > 1 and engine["rul"].nunique() > 1
            rho = float(spearmanr(engine[sensor], engine["rul"]).statistic) if defined else np.nan
            rows.append({
                "engine_id": engine_id,
                "sensor": sensor,
                "n_observations": len(engine),
                "rho": rho,
                "defined": defined,
            })
    return pd.DataFrame(rows)
