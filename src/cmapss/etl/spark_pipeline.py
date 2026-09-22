"""PySpark adapter tái hiện ETL Silver/Gold khi môi trường có Spark và Java."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cmapss.data.contracts import KEY, RAW_COLUMNS
from cmapss.etl.contracts import GOLD_ENGINE_COLUMNS, SILVER_TARGET_COLUMNS


class SparkRuntimeUnavailable(RuntimeError):
    """PySpark hoặc Java runtime chưa sẵn sàng."""


@dataclass
class SparkETLResult:
    """Các DataFrame Spark và session để caller quản lý vòng đời."""

    spark: Any
    observations: Any
    train_targets: Any
    engine_summary: Any

    def to_pandas(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Collect ba boundary về Pandas để parity với reference."""
        observations = self.observations.orderBy(*KEY).toPandas()
        targets = self.train_targets.orderBy(*SILVER_TARGET_COLUMNS[:3]).toPandas()
        summary = self.engine_summary.orderBy("split", "engine_id").toPandas()
        observations["split"] = observations["split"].astype("string")
        targets["split"] = targets["split"].astype("string")
        summary["split"] = summary["split"].astype("string")
        return observations, targets, summary


def _spark_modules() -> tuple[Any, Any, Any]:
    """Import lazily để Pandas ETL không bắt buộc cài Spark."""
    try:
        from pyspark.sql import SparkSession, functions as F, types as T
    except ImportError as error:
        raise SparkRuntimeUnavailable(
            "PySpark chưa được cài; cài optional dependency `etl` để chạy --with-spark."
        ) from error
    return SparkSession, F, T


def create_spark_session(app_name: str = "cmapss-etl") -> Any:
    """Tạo Spark local session hoặc báo lỗi runtime rõ ràng."""
    SparkSession, _, _ = _spark_modules()
    try:
        return (
            SparkSession.builder.master("local[1]")
            .appName(app_name)
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "1")
            .config("spark.default.parallelism", "1")
            .getOrCreate()
        )
    except Exception as error:
        raise SparkRuntimeUnavailable(
            "Không khởi tạo được Spark local; cần Java runtime tương thích."
        ) from error


def _read_raw(spark: Any, path: Path, split: str) -> Any:
    """Đọc file whitespace thành DataFrame Spark với schema raw contract."""
    _, F, T = _spark_modules()
    raw_schema = T.StructType(
        [T.StructField(column, T.StringType(), nullable=False) for column in RAW_COLUMNS]
    )
    lines = spark.read.text(str(path)).where(F.length(F.trim("value")) > 0)
    tokens = F.split(F.trim(F.col("value")), r"\s+")
    selected = [
        tokens.getItem(index).cast(
            T.LongType() if column in {"engine_id", "cycle"} else T.DoubleType()
        ).alias(column)
        for index, column in enumerate(RAW_COLUMNS)
    ]
    return lines.select(*selected).withColumn("split", F.lit(split)).select(
        "split", *RAW_COLUMNS
    )


def run_spark_etl(
    raw_dir: str | Path,
    *,
    spark: Any | None = None,
) -> SparkETLResult:
    """Chạy transform tương đương Pandas bằng Spark DataFrame."""
    _, F, _ = _spark_modules()
    owns_session = spark is None
    session = spark or create_spark_session()
    raw_path = Path(raw_dir)
    try:
        train = _read_raw(session, raw_path / "train_FD001.txt", "train")
        test = _read_raw(session, raw_path / "test_FD001.txt", "test")
        observations = train.unionByName(test).orderBy(*KEY)
        max_cycles = train.groupBy("engine_id").agg(F.max("cycle").alias("max_cycle"))
        train_targets = (
            train.join(max_cycles, on="engine_id", how="inner")
            .withColumn("rul", F.col("max_cycle") - F.col("cycle"))
            .select(*SILVER_TARGET_COLUMNS)
            .orderBy(*SILVER_TARGET_COLUMNS[:3])
        )
        engine_summary = (
            observations.groupBy("split", "engine_id")
            .agg(
                F.count("cycle").cast("long").alias("n_cycles"),
                F.min("cycle").cast("long").alias("first_cycle"),
                F.max("cycle").cast("long").alias("last_cycle"),
            )
            .select(*GOLD_ENGINE_COLUMNS)
            .orderBy("split", "engine_id")
        )
        return SparkETLResult(
            spark=session,
            observations=observations,
            train_targets=train_targets,
            engine_summary=engine_summary,
        )
    except Exception:
        if owns_session:
            session.stop()
        raise
