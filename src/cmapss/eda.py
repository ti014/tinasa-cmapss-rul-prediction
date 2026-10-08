"""Phép tính dùng lại trong notebook EDA; chỉ phân tích observations train."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _require_train(data: pd.DataFrame) -> None:
    if data.empty or not data["split"].eq("train").fillna(False).all():
        raise ValueError("EDA chỉ nhận dữ liệu train không rỗng; không dùng test.")


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
