"""CLI chạy phân tích PCA--ARIMA trên Health Index train."""

from __future__ import annotations

import argparse
from pathlib import Path

from cmapss.timeseries.pipeline import run_timeseries_pipeline, write_timeseries_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phân tích phụ PCA--ARIMA trên Health Index FD001")
    parser.add_argument(
        "--observations",
        default="outputs/etl/silver/observations.parquet",
        help="Silver observations Parquet từ M3.1",
    )
    parser.add_argument(
        "--fold-assignments",
        default="outputs/modeling/tables/fold_assignments.csv",
        help="Fold assignment theo engine từ M3.3",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/timeseries",
        help="Thư mục ghi artifact PCA--ARIMA",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_timeseries_pipeline(args.observations, args.fold_assignments)
    payload = write_timeseries_artifacts(result, Path(args.output_dir))
    successful = payload["row_counts"]["successful_engines"]
    print(
        f"selected_engines={payload['row_counts']['selected_engines']} "
        f"successful_engines={successful} "
        f"forecasts={payload['row_counts']['forecasts']}"
    )
    return 0
