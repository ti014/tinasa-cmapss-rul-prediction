"""CLI huấn luyện và đánh giá mô hình RUL FD001."""

from __future__ import annotations

from pathlib import Path
import argparse

from cmapss.modeling.baseline import (
    ModelingConfig,
    evaluate_predictions_from_path,
    run_modeling_pipeline,
    write_final_evaluation_artifacts,
    write_modeling_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá RUL models FD001")
    parser.add_argument(
        "--rolling-features",
        default="outputs/features/gold/rolling_features.parquet",
        help="Gold rolling features Parquet (outputs/features/gold)",
    )
    parser.add_argument(
        "--train-targets",
        default="outputs/etl/silver/train_targets.parquet",
        help="Silver train targets Parquet (outputs/etl/silver)",
    )
    parser.add_argument(
        "--test-labels",
        default="data/raw/RUL_FD001.txt",
        help="NASA test labels, chỉ đọc sau khi final_predictions đã được ghi",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/modeling",
        help="Thư mục ghi artifact modeling",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_modeling_pipeline(
        args.rolling_features,
        args.train_targets,
        config=ModelingConfig(),
    )
    output_dir = Path(args.output_dir)
    payload = write_modeling_artifacts(result, output_dir)
    final_predictions_path = output_dir / payload["outputs"]["final_predictions"]
    evaluation = evaluate_predictions_from_path(final_predictions_path, args.test_labels)
    final_payload = write_final_evaluation_artifacts(
        evaluation,
        output_dir,
        test_label_path=args.test_labels,
        bootstrap_resamples=result.config.bootstrap_resamples,
        seed=result.config.seed,
    )
    metrics = final_payload["metrics"]
    print(
        "selected_model="
        f"{payload['selected_model']} "
        f"final_engine_mae={metrics['mean_engine_mae']:.6f} "
        f"rmse={metrics['rmse']:.6f} "
        f"nasa_score={metrics['nasa_score']:.6f}"
    )
    return 0
