"""Benchmark Pandas/Spark local bằng process riêng, thực thi từng stage và đo RSS."""

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
    """Đo bộ nhớ thường trú của Python và mọi process con, kể cả JVM."""

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
    """Giữ nguyên lỗi JVM gốc nếu bước dọn dẹp cũng phát sinh lỗi."""
    try:
        spark.stop()
    except Exception as error:
        warnings.warn(f"Spark cleanup after JVM shutdown failed: {type(error).__name__}", RuntimeWarning)


def worker(engine: str, source: Path, destination: Path, driver_memory: str = "4g",
           write: bool = True) -> dict:
    """Chạy một lượt trong process mới; mỗi stage đo thời gian đều được thực thi đầy đủ."""
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
                     .config("spark.driver.memory", driver_memory)
                     .config("spark.executor.memory", driver_memory)
                     # Bảng đặc trưng được gom về driver để ghi; bỏ trần 1 GB mặc định của kết quả.
                     .config("spark.driver.maxResultSize", "0")
                     .config("spark.driver.host", "127.0.0.1")
                     .config("spark.driver.bindAddress", "127.0.0.1")
                     .config("spark.sql.shuffle.partitions", "4")
                     .config("spark.default.parallelism", "4").getOrCreate())
            spark.sparkContext.setLogLevel("ERROR")
            resources.callback(_stop_spark, spark)
            driver_heap = int(spark._jvm.java.lang.Runtime.getRuntime().maxMemory())
            if driver_heap < 3 * 1024 ** 3:
                raise RuntimeError("Spark chưa áp dụng heap driver 4 GiB đã khai báo")
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
        if not write:
            pass  # --no-write: chỉ đo tính toán; bước ghi qua driver của Spark trên Windows hết heap ở 100×
        elif engine == "pandas":
            features.to_parquet(destination / "features.parquet", index=False)
            summary.to_parquet(destination / "summary.parquet", index=False)
        else:
            import pyarrow.parquet as pq

            # Windows thiếu Hadoop native (winutils) và Python worker của Spark bị crash, nên
            # gom về driver rồi ghi bằng PyArrow; write_seconds tính cả bước gom này.
            pq.write_table(features.toArrow(), destination / "features.parquet")
            pq.write_table(summary.toArrow(), destination / "summary.parquet")
        times["write_seconds"] = time.perf_counter() - stage
        times["pipeline_seconds"] = time.perf_counter() - pipeline_start
        times["total_seconds"] = times["startup_seconds"] + times["pipeline_seconds"]
    return {"engine": engine, "input_rows": input_rows, "summary_rows": summary_rows,
            "spark_driver_heap_bytes": driver_heap,
            "feature_rows": feature_rows, "peak_process_tree_rss_bytes": memory.peak, **times}


def verify_parity(left_dir: Path, right_dir: Path) -> dict:
    """So sánh từng ô feature và summary sau khi sắp theo khóa ổn định."""
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
    """Chạy một worker hoặc mọi scale/lượt lặp trong các subprocess mới."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("pandas", "spark"))
    parser.add_argument("--source", type=Path, default=Path("outputs/etl/silver/observations.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/20260929"))
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/scalability-20260929"))
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--spark-driver-memory", default="4g")
    parser.add_argument("--engines", nargs="+", choices=("pandas", "spark"), default=["pandas", "spark"])
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.worker:
        result = worker(args.worker, args.source.resolve(), args.output.resolve(), args.spark_driver_memory,
                        write=not args.no_write)
        (args.output / "measurement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    if args.repetitions < 1 or any(scale < 1 for scale in args.scales):
        raise ValueError("scale và số lượt lặp phải dương")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    observations = pd.read_parquet(args.source)
    records, parity = [], {}
    for scale in args.scales:
        source = (args.work_dir / f"input_{scale}x.parquet").resolve()
        replicate_observations(observations, scale).to_parquet(source, index=False)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        for repetition in range(1, args.repetitions + 1):
            # Đảo thứ tự thực thi để giảm sai lệch hệ thống theo thứ tự chạy.
            order = ["pandas", "spark"] if repetition % 2 else ["spark", "pandas"]
            engines = [engine for engine in order if engine in args.engines]
            destinations = {}
            for engine in engines:
                destination = (args.work_dir / f"{engine}_{scale}x_{repetition}").resolve()
                destinations[engine] = destination
                if (destination / "measurement.json").exists():
                    raise FileExistsError(f"Choose a fresh --work-dir: {destination}")
                print(f"scale={scale} repetition={repetition} engine={engine}", flush=True)
                started = time.perf_counter()
                try:
                    subprocess.run([sys.executable, "-m", "cmapss.etl.scalability", "--worker", engine,
                                    "--source", str(source), "--output", str(destination),
                                    "--spark-driver-memory", args.spark_driver_memory,
                                    *(["--no-write"] if args.no_write else [])],
                                   check=True, timeout=args.timeout_seconds)
                    record = json.loads((destination / "measurement.json").read_text(encoding="utf-8"))
                    record["status"] = "ok"
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    # Lượt hỏng (hết RAM, quá thời gian) là kết quả cần ghi, không dừng cả benchmark.
                    status = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "failed"
                    record = {"engine": engine, "status": status,
                              "elapsed_before_stop_seconds": time.perf_counter() - started}
                record.update(scale=scale, repetition=repetition, input_sha256=digest)
                records.append(record)
                pd.DataFrame(records).to_csv(args.output / "measurements.csv", index=False)
            completed = [record for record in records[-len(engines):] if record["status"] == "ok"]
            if scale == 1 and len(completed) == 2:
                parity[str(repetition)] = verify_parity(destinations["pandas"], destinations["spark"])
            expected_rows = len(observations) * scale
            expected_features = (len(observations) - 29 * observations.groupby(["split", "engine_id"]).ngroups) * scale
            for record in completed:
                assert record["input_rows"] == expected_rows
                assert record["feature_rows"] == expected_features
    frame = pd.DataFrame(records)
    ok = frame.loc[frame["status"].eq("ok")]
    metrics = [column for column in ok if column.endswith("_seconds") or column.endswith("_bytes")]
    ok.groupby(["engine", "scale"])[metrics].agg(["median", "min", "max"]).to_csv(args.output / "summary.csv")
    frame.groupby(["engine", "scale"])["status"].value_counts().unstack(fill_value=0).to_csv(
        args.output / "status.csv")
    metadata = {"status": "complete", "author": "Phan Nguyễn Mai Phương", "student_id": "25002711",
                "scales": args.scales, "repetitions": args.repetitions, "parity": parity,
                "python": platform.python_version(), "platform": platform.platform(),
                "cpu_logical": os.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
                "spark_master": "local[4]", "spark_partitions": 4, "window": 30,
                "spark_driver_memory": args.spark_driver_memory, "spark_executor_memory": args.spark_driver_memory,
                "rss_sampling_seconds": 0.05, "os_cache_cleared": False,
                "parquet_writer": "PyArrow on driver; Spark write_seconds includes toArrow collection",
                "timeout_seconds": args.timeout_seconds, "write_output": not args.no_write,
                "scope": "local replicated data; no distributed-cluster throughput claim",
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "versions": {name: version(name) for name in ("pandas", "numpy", "pyarrow", "pyspark", "psutil")},
                "source_code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                       for name in ("scalability.py", "rolling_spark.py", "benchmark.py", "pandas_pipeline.py")}}
    (args.output / "run_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
