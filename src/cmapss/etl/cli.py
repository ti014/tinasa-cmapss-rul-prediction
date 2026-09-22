"""CLI chạy ETL Pandas, benchmark và parity PySpark tùy chọn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cmapss.etl.benchmark import BenchmarkConfig, benchmark_pandas
from cmapss.etl.pandas_pipeline import run_pandas_etl, write_pandas_artifacts
from cmapss.etl.parity import ParityError, compare_frames
from cmapss.etl.spark_pipeline import SparkRuntimeUnavailable, run_spark_etl


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Ghi JSON UTF-8 với newline cuối."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_scales(value: str) -> tuple[int, ...]:
    """Parse danh sách scale như `1,10,100`."""
    try:
        scales = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("scales phải là số nguyên, ví dụ 1,10,100") from error
    if not scales or any(scale < 1 for scale in scales):
        raise argparse.ArgumentTypeError("scales phải chứa số nguyên dương")
    return scales


def _run_parity(result: Any, raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Collect Spark boundary và so sánh với Pandas."""
    spark_result = run_spark_etl(raw_dir)
    try:
        spark_observations, spark_targets, spark_summary = spark_result.to_pandas()
        checks = [
            compare_frames(
                result.observations,
                spark_observations,
                name="silver_observations",
                key_columns=("split", "engine_id", "cycle"),
                numeric_columns=tuple(result.observations.columns[2:]),
            ).as_dict(),
            compare_frames(
                result.train_targets,
                spark_targets,
                name="silver_train_targets",
                key_columns=("split", "engine_id", "cycle"),
                numeric_columns=("rul",),
            ).as_dict(),
            compare_frames(
                result.engine_summary,
                spark_summary,
                name="gold_engine_summary",
                key_columns=("split", "engine_id"),
                numeric_columns=("n_cycles", "first_cycle", "last_cycle"),
            ).as_dict(),
        ]
        payload = {"status": "passed", "checks": checks}
        _write_json(output_dir / "parity.json", payload)
        return payload
    finally:
        spark_result.spark.stop()


def main(argv: list[str] | None = None) -> int:
    """Chạy ETL; trả 0 khi các boundary được yêu cầu đều đạt."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="ETL Silver/Gold FD001 bằng Pandas và PySpark.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/etl"))
    parser.add_argument("--with-spark", action="store_true", help="Chạy PySpark và parity.")
    parser.add_argument("--benchmark-scales", type=_parse_scales, default=(1, 10, 100))
    parser.add_argument("--benchmark-repetitions", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        result = run_pandas_etl(args.raw_dir)
        write_pandas_artifacts(result, args.output_dir)
        benchmark = benchmark_pandas(
            result.observations,
            BenchmarkConfig(
                scales=tuple(args.benchmark_scales),
                repetitions=args.benchmark_repetitions,
            ),
        )
        _write_json(args.output_dir / "benchmark.json", benchmark)
        if args.with_spark:
            _run_parity(result, args.raw_dir, args.output_dir)
        print(f"ETL Pandas hoàn tất: {args.output_dir}")
        return 0
    except (OSError, ValueError, RuntimeError, ParityError, SparkRuntimeUnavailable) as error:
        print(f"ETL failed: {error}", file=sys.stderr)
        return 1
