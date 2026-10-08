"""CLI tạo Gold rolling features từ Silver observations."""

from __future__ import annotations

import argparse
from pathlib import Path

from cmapss.features.rolling import run_rolling_feature_pipeline, write_rolling_feature_artifacts


def build_parser() -> argparse.ArgumentParser:
    """Tạo parser cho `cmapss-features`."""
    parser = argparse.ArgumentParser(description="Tạo rolling features FD001 từ Silver observations")
    parser.add_argument(
        "--silver-observations",
        default="outputs/etl/silver/observations.parquet",
        help="Đường dẫn Silver observations Parquet (outputs/etl/silver)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Bronze thực tế cho Silver cũ chưa lưu raw_dir; mặc định lấy từ metadata ETL",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/features",
        help="Thư mục ghi Gold rolling features và run_info.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Chạy feature pipeline và in tóm tắt row counts."""
    args = build_parser().parse_args(argv)
    result = run_rolling_feature_pipeline(Path(args.silver_observations), raw_dir=args.raw_dir)
    payload = write_rolling_feature_artifacts(result, Path(args.output_dir))
    print(
        "rolling_features rows="
        f"{payload['row_counts']['rolling_features']} "
        f"train={payload['row_counts']['rolling_features_train']} "
        f"test={payload['row_counts']['rolling_features_test']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
