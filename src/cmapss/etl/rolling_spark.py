"""Spark implementation of the same 126 trailing-window sensor features."""

from typing import Any

from cmapss.data.contracts import KEY, SENSORS


def build_spark_rolling_features(observations: Any, window_cycles: int = 30) -> Any:
    """Build causal, engine-local features on already validated Silver data.

    Example: a sensor equal to cycle has slope 1 and mean 15.5 at cycle 30.
    Validation of the input contract is performed before benchmark replication.
    """
    from pyspark.sql import Window, functions as F

    if window_cycles < 2:
        raise ValueError("window_cycles must be at least 2")
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
