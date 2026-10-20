"""CLI cho thí nghiệm cải thiện model RUL."""

from __future__ import annotations

import argparse

from cmapss.modeling.improvement.config import ImprovementConfig
from cmapss.modeling.improvement.runner import run_improvement_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chạy thí nghiệm cải thiện RUL FD001")
    parser.add_argument("--observations", default="outputs/etl/silver/observations.parquet")
    parser.add_argument("--targets", default="outputs/etl/silver/train_targets.parquet")
    parser.add_argument("--test-labels", default="data/raw/RUL_FD001.txt")
    parser.add_argument("--output-dir", default="outputs/experiments/rul_improvement/20260927")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--min-cycle", type=int, default=30)
    parser.add_argument("--n-hgb-candidates", type=int, default=24)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--fit-threads", type=int, default=4)
    parser.add_argument("--skip-test-evaluation", action="store_true", help="For dry-runs: never read test labels")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ImprovementConfig(
        seed=args.seed,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        min_cycle=args.min_cycle,
        n_hgb_candidates=args.n_hgb_candidates,
        bootstrap_resamples=args.bootstrap_resamples,
        fit_threads=args.fit_threads,
    )
    result = run_improvement_experiment(
        args.observations,
        args.targets,
        test_labels_path=args.test_labels,
        output_dir=args.output_dir,
        config=config,
        evaluate_test=not args.skip_test_evaluation,
    )
    print("selected_window=" + str(result["selected_final"]["window"]))
    print("selected_weight_scheme=" + str(result["selected_final"]["weight_scheme"]))
    print("output_dir=" + args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
