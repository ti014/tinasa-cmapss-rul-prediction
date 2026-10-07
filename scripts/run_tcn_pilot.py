"""Run a non-promotional TCN pilot on grouped FD001 outer folds.

This pilot is deliberately separate from the frozen HGB/XGBoost study.  It
checks that causal sequence construction, fold-local scaling and the TCN
forward/training path work on this host.  It does not read test labels and it
does not select or publish a model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from cmapss.modeling.tcn_comparison.contracts import TCNConfig, specs
from cmapss.modeling.tcn_comparison.data import build_sequences
from cmapss.modeling.tcn_comparison.model import fit_model

AUTHOR = {"full_name": "Phan Nguyễn Mai Phương", "student_id": "25002711"}


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_pilot(
    *,
    repo_root: Path,
    output_dir: Path,
    epochs: int = 8,
    outer_folds: int = 5,
    fit_threads: int = 4,
) -> None:
    observations = pd.read_parquet(repo_root / "outputs/etl/silver/observations.parquet")
    targets = pd.read_parquet(repo_root / "outputs/etl/silver/train_targets.parquet")
    train_obs = observations.loc[observations["split"].eq("train")].copy()
    sequences = build_sequences(train_obs, targets, split="train", sequence_length=30)
    assignments = pd.read_csv(repo_root / "outputs/modeling/tables/fold_assignments.csv")
    assignments = assignments.loc[assignments["split"].eq("train"), ["engine_id", "fold"]]
    assignment_map = dict(zip(assignments.engine_id.astype(int), assignments.fold.astype(int), strict=True))
    if len(assignment_map) != 100:
        raise ValueError("published outer assignments must cover 100 train engines")
    config = replace(TCNConfig(), epochs=epochs, outer_splits=outer_folds, fit_threads=fit_threads)
    output_dir.mkdir(parents=True, exist_ok=False)
    _json(output_dir / "pilot_info.json", {
        "author": AUTHOR, "status": "RUNNING", "promotional": False,
        "platform": platform.platform(), "python": platform.python_version(),
        "config": asdict(config), "input_rows": len(train_obs),
        "sequence_rows": len(sequences.metadata), "sequence_channels": int(sequences.values.shape[1]),
        "sequence_length": int(sequences.values.shape[2]),
        "raw_test_labels_read": False,
    })
    all_predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    fold_values = sorted(set(assignment_map.values()))[:outer_folds]
    for fold in fold_values:
        validation_engines = {engine for engine, value in assignment_map.items() if value == fold}
        fitting_engines = set(assignment_map) - validation_engines
        validation_mask = sequences.metadata.engine_id.isin(validation_engines).to_numpy()
        fitting_mask = sequences.metadata.engine_id.isin(fitting_engines).to_numpy()
        for spec in specs():
            fitted = fit_model(sequences.values[fitting_mask], sequences.metadata.loc[fitting_mask].reset_index(drop=True), spec, config, fitting_engines)
            prediction = fitted.predict(sequences.values[validation_mask])
            truth = sequences.metadata.loc[validation_mask, "rul"].to_numpy(dtype="float64")
            frame = sequences.metadata.loc[validation_mask, ["split", "engine_id", "cycle", "rul"]].copy()
            frame["y_pred"] = prediction
            frame["procedure"] = spec.candidate_id
            frame["outer_fold"] = fold
            all_predictions.append(frame)
            metrics.append({
                "outer_fold": fold, "procedure": spec.candidate_id,
                "n_rows": len(truth), "n_engines": int(frame.engine_id.nunique()),
                "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
                "mae": float(mean_absolute_error(truth, prediction)),
            })
        print(json.dumps({"outer_fold": fold, "status": "completed"}), flush=True)
    predictions = pd.concat(all_predictions, ignore_index=True).sort_values(
        ["procedure", "engine_id", "cycle"], kind="mergesort"
    ).reset_index(drop=True)
    predictions.to_parquet(output_dir / "outer_predictions.parquet", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "outer_metrics.csv", index=False)
    _json(output_dir / "pilot_info.json", {
        "author": AUTHOR, "status": "COMPLETED", "promotional": False,
        "platform": platform.platform(), "python": platform.python_version(),
        "config": asdict(config), "input_rows": len(train_obs),
        "sequence_rows": len(sequences.metadata), "sequence_channels": int(sequences.values.shape[1]),
        "sequence_length": int(sequences.values.shape[2]), "outer_folds": fold_values,
        "raw_test_labels_read": False, "test_evaluated": False,
        "note": "Pilot only: no inner selection, no promotion gate, no published model change.",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--fit-threads", type=int, default=4)
    args = parser.parse_args()
    run_pilot(repo_root=args.repo_root.resolve(), output_dir=args.output_dir.resolve(),
              epochs=args.epochs, outer_folds=args.outer_folds, fit_threads=args.fit_threads)


if __name__ == "__main__":
    main()
