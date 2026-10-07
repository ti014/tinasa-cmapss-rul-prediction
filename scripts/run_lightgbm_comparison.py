"""Run an exploratory nested LightGBM comparison on the frozen tree features.

This study keeps the 169-feature matrix, published engine folds and raw-RUL
evaluation from the HGB/XGBoost study.  It is isolated and does not publish a
model; the promotion gate is intentionally not inferred from this lightweight
runner until endpoint and warning panels are reviewed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

AUTHOR = {"full_name": "Phan Nguyễn Mai Phương", "student_id": "25002711"}
FEATURE_EXCLUDE = {"split", "engine_id", "rul", "outer_fold"}


def _fit(frame: pd.DataFrame, features: list[str], target_cap: int | None, params: dict[str, float], threads: int):
    count = frame.groupby("engine_id")["engine_id"].transform("size").to_numpy(dtype="float64")
    model = LGBMRegressor(
        boosting_type="gbdt", n_estimators=200, max_depth=-1, min_child_samples=20,
        min_split_gain=0.0, reg_alpha=0.0, subsample=1.0, colsample_bytree=1.0,
        objective="regression", random_state=42, n_jobs=threads, verbosity=-1,
        deterministic=True, force_col_wise=True, **params,
    )
    target = frame.rul.to_numpy(dtype="float64")
    if target_cap is not None:
        target = np.minimum(target, target_cap)
    model.fit(frame[features], target, sample_weight=1.0 / count)
    return model


def _score(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float | int]:
    truth = frame.rul.to_numpy(dtype="float64")
    endpoint = frame.assign(abs=np.abs(prediction - truth)).sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)
    return {"n_rows": int(len(frame)), "n_engines": int(frame.engine_id.nunique()),
            "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
            "mae": float(mean_absolute_error(truth, prediction)),
            "endpoint_mae": float(endpoint["abs"].mean())}


def run(*, repo_root: Path, output_dir: Path, threads: int = 4) -> None:
    source = repo_root / "outputs/experiments/tree_comparison/20261005_full"
    train = pd.read_parquet(source / "frozen_train_features.parquet")
    assignments = pd.read_csv(source / "outer_assignments.csv")
    train = train.merge(assignments, on="engine_id", how="left", validate="many_to_one")
    features = [column for column in train.columns if column not in FEATURE_EXCLUDE]
    if len(features) != 169:
        raise ValueError(f"expected 169 features including cycle, received {len(features)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates = [
        ("lgbm_uncapped", None, "base", {"learning_rate": 0.05, "num_leaves": 31, "reg_lambda": 0.0}),
        ("lgbm_uncapped", None, "reg10", {"learning_rate": 0.05, "num_leaves": 31, "reg_lambda": 10.0}),
        ("lgbm_cap125", 125, "base", {"learning_rate": 0.05, "num_leaves": 31, "reg_lambda": 0.0}),
        ("lgbm_cap125", 125, "reg10", {"learning_rate": 0.05, "num_leaves": 31, "reg_lambda": 10.0}),
    ]
    selected_rows, outer_rows, predictions = [], [], []
    for fold in range(1, 6):
        fitting = train[train.outer_fold.ne(fold)].reset_index(drop=True)
        validation = train[train.outer_fold.eq(fold)].reset_index(drop=True)
        inner_scores = []
        splitter = GroupKFold(n_splits=3)
        for family, cap, setting, params in candidates:
            parts = []
            for inner_fold, (fit_idx, val_idx) in enumerate(splitter.split(fitting, groups=fitting.engine_id), 1):
                fit_frame, val_frame = fitting.iloc[fit_idx], fitting.iloc[val_idx]
                model = _fit(fit_frame, features, cap, params, threads)
                parts.append(_score(val_frame, model.predict(val_frame[features])))
            inner_scores.append({"family": family, "setting": setting, **{key: float(np.mean([part[key] for part in parts])) for key in ("rmse", "mae", "endpoint_mae")}})
        for family in ("lgbm_uncapped", "lgbm_cap125"):
            options = [row for row in inner_scores if row["family"] == family]
            selected = min(options, key=lambda row: (row["rmse"], row["setting"]))
            selected_rows.append({"outer_fold": fold, **selected})
        for family, cap, setting, params in candidates:
            model = _fit(fitting, features, cap, params, threads)
            pred = model.predict(validation[features])
            score = _score(validation, pred)
            outer_rows.append({"outer_fold": fold, "family": family, "setting": setting, **score})
            piece = validation[["split", "engine_id", "cycle", "rul"]].copy()
            piece["y_true"] = piece.pop("rul"); piece["y_pred"] = pred
            piece["procedure"] = f"{family}_{setting}"; piece["outer_fold"] = fold
            predictions.append(piece)
    outer = pd.DataFrame(outer_rows)
    outer.to_csv(output_dir / "outer_metrics.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output_dir / "inner_selection.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(output_dir / "outer_predictions.parquet", index=False)
    summary = outer.groupby("family", as_index=False).agg(outer_rmse=("rmse", "mean"), outer_mae=("mae", "mean"), outer_endpoint_mae=("endpoint_mae", "mean"))
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "README.md").write_text(
        "**Phan Nguyễn Mai Phương — MSHV 25002711**\n\n# LightGBM nested pilot\n\n"
        "Bốn predeclared arms dùng 169 frozen features, cùng 5 outer × 3 inner engine folds và raw RUL. "
        "Chưa đọc test labels và chưa có promotion gate/warning panel.\n\n```text\n"
        + summary.to_string(index=False) + "\n```\n", encoding="utf-8")
    (output_dir / "run_info.json").write_text(json.dumps({"author": AUTHOR, "status": "completed", "promotional": False, "features": 169, "candidates": [row[0] + "_" + row[2] for row in candidates], "test_labels_read": False}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-threads", type=int, default=4)
    args = parser.parse_args()
    run(repo_root=args.repo_root.resolve(), output_dir=args.output_dir.resolve(), threads=args.fit_threads)


if __name__ == "__main__":
    main()
