"""Isolated-process local Pandas/Spark benchmark with materialized stages and RSS."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import psutil

from cmapss.data.contracts import KEY
from cmapss.etl.benchmark import replicate_observations
from cmapss.etl.pandas_pipeline import build_engine_summary
from cmapss.features.rolling import build_rolling_features


class MemorySampler:
    """Sample resident memory of Python and all descendants, including the JVM."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            total = 0
            for process in [self.process, *self.process.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak = max(self.peak, total)
            self.stop_event.wait(0.05)

    def __enter__(self) -> "MemorySampler":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop_event.set()
        self.thread.join()


def _stop_spark(spark) -> None:
    """Do not mask an original JVM failure with a second cleanup exception."""
    try:
        spark.stop()
    except Exception as error:
        warnings.warn(f"Spark cleanup after JVM shutdown failed: {type(error).__name__}", RuntimeWarning)


def worker(engine: str, source: Path, destination: Path) -> dict:
    """Execute one fresh-process repetition; every timed stage is materialized."""
    destination.mkdir(parents=True, exist_ok=True)
    times = {}
    spark = None
    driver_heap = None
    with MemorySampler() as memory, ExitStack() as resources:
        started = time.perf_counter()
        if engine == "spark":
            from pyspark.sql import SparkSession, functions as F
            from cmapss.etl.rolling_spark import build_spark_rolling_features

            spark = (SparkSession.builder.master("local[4]").appName("cmapss-stage-benchmark")
                     .config("spark.ui.enabled", "false")
                     .config("spark.driver.memory", "4g")
                     .config("spark.executor.memory", "4g")
                     .config("spark.driver.host", "127.0.0.1")
                     .config("spark.driver.bindAddress", "127.0.0.1")
                     .config("spark.sql.shuffle.partitions", "4")
                     .config("spark.default.parallelism", "4").getOrCreate())
            spark.sparkContext.setLogLevel("ERROR")
            resources.callback(_stop_spark, spark)
            driver_heap = int(spark._jvm.java.lang.Runtime.getRuntime().maxMemory())
            if driver_heap < 3 * 1024 ** 3:
                raise RuntimeError("Spark did not apply the declared 4 GiB driver heap")
        times["startup_seconds"] = time.perf_counter() - started
        pipeline_start = time.perf_counter()
        stage = pipeline_start
        if engine == "pandas":
            observations = pd.read_parquet(source)
            input_rows = len(observations)
        else:
            observations = spark.read.parquet(str(source)).repartition(4, "split", "engine_id").cache()
            input_rows = observations.count()
        times["read_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        if engine == "pandas":
            summary = build_engine_summary(observations)
            summary_rows = len(summary)
        else:
            summary = observations.groupBy("split", "engine_id").agg(
                F.count("cycle").alias("n_cycles"), F.min("cycle").alias("first_cycle"),
                F.max("cycle").alias("last_cycle")).cache()
            summary_rows = summary.count()
        times["summary_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        if engine == "pandas":
            features = build_rolling_features(observations, window_cycles=30)
            feature_rows = len(features)
        else:
            features = build_spark_rolling_features(observations, 30).cache()
            feature_rows = features.count()
        times["rolling_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        if engine == "pandas":
            features.to_parquet(destination / "features.parquet", index=False)
            summary.to_parquet(destination / "summary.parquet", index=False)
        else:
            import pyarrow.parquet as pq

            # Windows lacks Hadoop native write support. Export through Arrow;
            # this timed stage explicitly includes collection to the driver.
            pq.write_table(features.toArrow(), destination / "features.parquet")
            pq.write_table(summary.toArrow(), destination / "summary.parquet")
        times["write_seconds"] = time.perf_counter() - stage
        times["pipeline_seconds"] = time.perf_counter() - pipeline_start
        times["total_seconds"] = times["startup_seconds"] + times["pipeline_seconds"]
    return {"engine": engine, "input_rows": input_rows, "summary_rows": summary_rows,
            "spark_driver_heap_bytes": driver_heap,
            "feature_rows": feature_rows, "peak_process_tree_rss_bytes": memory.peak, **times}


def verify_parity(left_dir: Path, right_dir: Path) -> dict:
    """Compare every feature and summary cell, sorted by stable keys."""
    result = {}
    for table in ("features", "summary"):
        keys = list(KEY) if table == "features" else ["split", "engine_id"]
        left = pd.read_parquet(left_dir / f"{table}.parquet").sort_values(keys).reset_index(drop=True)
        right = pd.read_parquet(right_dir / f"{table}.parquet").sort_values(keys).reset_index(drop=True)
        right = right[left.columns]
        left["split"] = left["split"].astype("string")
        right["split"] = right["split"].astype("string")
        pd.testing.assert_frame_equal(left, right, check_dtype=False, atol=1e-7, rtol=1e-7)
        result[table] = {"rows": len(left), "columns": len(left.columns), "all_cells_match": True}
    return result


def main() -> None:
    """Run one worker, or run every scale and repetition in fresh subprocesses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("pandas", "spark"))
    parser.add_argument("--source", type=Path, default=Path("outputs/etl/silver/observations.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/20260929"))
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/scalability-20260929"))
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        result = worker(args.worker, args.source.resolve(), args.output.resolve())
        (args.output / "measurement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    if args.repetitions < 1 or any(scale < 1 for scale in args.scales):
        raise ValueError("scales and repetitions must be positive")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    observations = pd.read_parquet(args.source)
    records, parity = [], {}
    for scale in args.scales:
        source = (args.work_dir / f"input_{scale}x.parquet").resolve()
        replicate_observations(observations, scale).to_parquet(source, index=False)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        for repetition in range(1, args.repetitions + 1):
            # Alternate execution order to reduce systematic order bias.
            engines = ["pandas", "spark"] if repetition % 2 else ["spark", "pandas"]
            destinations = {}
            for engine in engines:
                destination = (args.work_dir / f"{engine}_{scale}x_{repetition}").resolve()
                destinations[engine] = destination
                if (destination / "measurement.json").exists():
                    raise FileExistsError(f"Choose a fresh --work-dir: {destination}")
                print(f"scale={scale} repetition={repetition} engine={engine}", flush=True)
                subprocess.run([sys.executable, "-m", "cmapss.etl.scalability", "--worker", engine,
                                "--source", str(source), "--output", str(destination)], check=True)
                record = json.loads((destination / "measurement.json").read_text(encoding="utf-8"))
                record.update(scale=scale, repetition=repetition, input_sha256=digest)
                records.append(record)
                pd.DataFrame(records).to_csv(args.output / "measurements.csv", index=False)
            if scale == 1:
                parity[str(repetition)] = verify_parity(destinations["pandas"], destinations["spark"])
            expected_rows = len(observations) * scale
            expected_features = (len(observations) - 29 * observations.groupby(["split", "engine_id"]).ngroups) * scale
            for record in records[-2:]:
                assert record["input_rows"] == expected_rows
                assert record["feature_rows"] == expected_features
    frame = pd.DataFrame(records)
    metrics = [column for column in frame if column.endswith("_seconds") or column.endswith("_bytes")]
    frame.groupby(["engine", "scale"])[metrics].agg(["median", "min", "max"]).to_csv(args.output / "summary.csv")
    metadata = {"status": "complete", "author": "Phan Nguyễn Mai Phương", "student_id": "25002711",
                "scales": args.scales, "repetitions": args.repetitions, "parity": parity,
                "python": platform.python_version(), "platform": platform.platform(),
                "cpu_logical": os.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
                "spark_master": "local[4]", "spark_partitions": 4, "window": 30,
                "spark_driver_memory": "4g", "spark_executor_memory": "4g",
                "rss_sampling_seconds": 0.05, "os_cache_cleared": False,
                "parquet_writer": "PyArrow on driver; Spark write_seconds includes toArrow collection",
                "scope": "local replicated data; no distributed-cluster throughput claim",
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "versions": {name: version(name) for name in ("pandas", "numpy", "pyarrow", "pyspark", "psutil")},
                "source_code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                       for name in ("scalability.py", "rolling_spark.py", "benchmark.py", "pandas_pipeline.py")}}
    (args.output / "run_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
