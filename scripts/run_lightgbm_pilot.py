"""Run a non-promotional LightGBM pilot on the frozen 169-feature matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

AUTHOR = {"full_name": "Phan Nguyễn Mai Phương", "student_id": "25002711"}


def run(*, repo_root: Path, output_dir: Path, estimators: int = 300) -> None:
    train = pd.read_parquet(repo_root / "outputs/experiments/tree_comparison/20261005_full/frozen_train_features.parquet")
    assignments = pd.read_csv(repo_root / "outputs/experiments/tree_comparison/20261005_full/outer_assignments.csv")
    train = train.merge(assignments, on="engine_id", how="left", validate="many_to_one")
    # The frozen tree contract contains cycle plus 168 causal sensor features.
    feature_columns = [column for column in train.columns if column not in {"split", "engine_id", "rul", "outer_fold"}]
    if len(feature_columns) != 169:
        raise ValueError(f"expected 169 frozen features, received {len(feature_columns)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    configs = {
        "lgbm_base": {"num_leaves": 31, "learning_rate": 0.05, "reg_lambda": 0.0},
        "lgbm_reg10": {"num_leaves": 31, "learning_rate": 0.05, "reg_lambda": 10.0},
    }
    all_predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    for fold in sorted(train.outer_fold.unique()):
        fitting = train[train.outer_fold.ne(fold)]
        validation = train[train.outer_fold.eq(fold)]
        counts = fitting.groupby("engine_id")["engine_id"].transform("size").to_numpy(dtype="float64")
        weights = 1.0 / counts
        for name, params in configs.items():
            model = LGBMRegressor(
                n_estimators=estimators, max_depth=-1, min_child_samples=20,
                subsample=1.0, colsample_bytree=1.0, objective="regression",
                random_state=42, n_jobs=4, verbosity=-1, deterministic=True,
                force_col_wise=True, **params,
            )
            model.fit(fitting[feature_columns], fitting.rul, sample_weight=weights)
            prediction = model.predict(validation[feature_columns])
            frame = validation[["split", "engine_id", "cycle", "rul"]].copy()
            frame["y_true"] = frame.pop("rul")
            frame["y_pred"] = prediction
            frame["procedure"] = name
            frame["outer_fold"] = int(fold)
            all_predictions.append(frame)
            metrics.append({
                "outer_fold": int(fold), "procedure": name, "n_rows": len(frame),
                "n_engines": int(frame.engine_id.nunique()),
                "rmse": float(np.sqrt(mean_squared_error(frame.y_true, frame.y_pred))),
                "mae": float(mean_absolute_error(frame.y_true, frame.y_pred)),
            })
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(output_dir / "outer_predictions.parquet", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "outer_metrics.csv", index=False)
    summary = pd.DataFrame(metrics).groupby("procedure", as_index=False).agg(
        fold_rmse_mean=("rmse", "mean"), fold_mae_mean=("mae", "mean"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "README.md").write_text(
        "**Phan Nguyễn Mai Phương — MSHV 25002711**\n\n"
        "# LightGBM pilot — FD001\n\n"
        "Pilot non-promotional trên frozen 169 causal features, cùng outer engine folds với HGB/XGBoost. "
        "Dùng raw RUL và engine-total-one weights; không đọc test labels, không thay registry.\n\n"
        + "```text\n" + summary.to_string(index=False) + "\n```\n\n"
        "Đây là screening result; full nested selection và paired promotion gate chưa chạy.\n",
        encoding="utf-8",
    )
    (output_dir / "run_info.json").write_text(json.dumps({
        "author": AUTHOR, "status": "completed", "promotional": False,
        "estimators": estimators, "features": len(feature_columns),
        "test_labels_read": False, "configs": configs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--estimators", type=int, default=300)
    args = parser.parse_args()
    run(repo_root=args.repo_root.resolve(), output_dir=args.output_dir.resolve(), estimators=args.estimators)


if __name__ == "__main__":
    main()
