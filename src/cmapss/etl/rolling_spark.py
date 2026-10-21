"""Tạo cùng 126 đặc trưng window sensor bằng Spark."""

from __future__ import annotations


from typing import Any

from cmapss.data.contracts import KEY, SENSORS


def build_spark_rolling_features(observations: Any, window_cycles: int = 30) -> Any:
    """
    Tạo features theo động cơ từ Silver đã kiểm tra, chỉ dùng quá khứ; sensor bằng cycle
    có slope 1 và mean 15,5 tại cycle 30.
    """
    from pyspark.sql import Window, functions as F

    if window_cycles < 2:
        raise ValueError("window_cycles phải ít nhất bằng 2")
    window = Window.partitionBy("split", "engine_id").orderBy("cycle").rowsBetween(1 - window_cycles, 0)
    columns = [F.col(key) for key in KEY]
    for sensor in SENSORS:
        columns.extend([
            F.col(sensor).alias(f"{sensor}__last"),
            F.avg(sensor).over(window).alias(f"{sensor}__mean"),
            F.stddev_samp(sensor).over(window).alias(f"{sensor}__std"),
            F.min(sensor).over(window).alias(f"{sensor}__min"),
            F.max(sensor).over(window).alias(f"{sensor}__max"),
            F.regr_slope(F.col(sensor), F.col("cycle")).over(window).alias(f"{sensor}__slope"),
        ])
    return (observations.select(*columns, F.count("cycle").over(window).alias("_window_rows"))
            .where(F.col("_window_rows") == window_cycles).drop("_window_rows"))
