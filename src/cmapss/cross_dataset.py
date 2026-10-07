"""Đánh giá RUL trong subset và chuyển miền trên NASA C-MAPSS FD001--FD004."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cmapss.data.contracts import DATASET_IDS, KEY
from cmapss.etl.pandas_pipeline import PandasETLResult, run_pandas_etl, write_pandas_artifacts
from cmapss.features.rolling import run_rolling_feature_pipeline, write_rolling_feature_artifacts
from cmapss.modeling.pipeline import (
    ModelingConfig,
    evaluate_predictions_from_path,
    run_modeling_pipeline,
    write_final_evaluation_artifacts,
    write_modeling_artifacts,
)
from cmapss.modeling.contracts import FINAL_PREDICTION_COLUMNS, MODELING_FEATURE_COLUMNS


@dataclass(frozen=True)
class DatasetArtifacts:
    """Đường dẫn canonical của một subset đã qua ETL và feature engineering."""

    dataset_id: str
    etl_dir: Path
    features_dir: Path

    @property
    def observations_path(self) -> Path:
        return self.etl_dir / "silver" / "observations.parquet"

    @property
    def train_targets_path(self) -> Path:
        return self.etl_dir / "silver" / "train_targets.parquet"

    @property
    def features_path(self) -> Path:
        return self.features_dir / "gold" / "rolling_features.parquet"


def prepare_dataset(
    dataset_id: str,
    *,
    raw_dir: str | Path,
    output_root: str | Path,
    window_cycles: int = 10,
) -> DatasetArtifacts:
    """Chạy ETL và rolling features một lần cho một subset."""
    dataset_id = dataset_id.upper()
    if dataset_id not in DATASET_IDS:
        raise ValueError(f"dataset_id không hợp lệ: {dataset_id}")
    root = Path(output_root) / dataset_id
    etl_dir = root / "etl"
    features_dir = root / "features"
    result: PandasETLResult = run_pandas_etl(raw_dir, dataset_id)
    write_pandas_artifacts(result, etl_dir)
    feature_result = run_rolling_feature_pipeline(
        etl_dir / "silver" / "observations.parquet",
        window_cycles=window_cycles,
        raw_dir=raw_dir,
    )
    write_rolling_feature_artifacts(feature_result, features_dir)
    return DatasetArtifacts(dataset_id, etl_dir, features_dir)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_pair(
    source: DatasetArtifacts,
    target: DatasetArtifacts,
    *,
    output_root: str | Path,
    raw_dir: str | Path = "data/raw",
    transfer: bool,
    config: ModelingConfig | None = None,
) -> dict[str, Any]:
    """Đánh giá source train trên target test, hoặc within-subset nếu cùng dataset."""
    pair_id = f"{source.dataset_id}_to_{target.dataset_id}"
    pair_dir = Path(output_root) / "evaluations" / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    if transfer:
        source_model_dir = Path(output_root) / "evaluations" / f"{source.dataset_id}_to_{source.dataset_id}" / "modeling"
        source_info = json.loads((source_model_dir / "run_info.json").read_text(encoding="utf-8"))
        selected_model = str(source_info["selected_model"])
        estimator = joblib.load(source_model_dir / "models" / "selected_model.joblib")
        target_features = pd.read_parquet(target.features_path)
        last_windows = (
            target_features.loc[target_features["split"].eq("test")]
            .sort_values(list(KEY), kind="mergesort")
            .groupby("engine_id", sort=True)
            .tail(1)
            .reset_index(drop=True)
        )
        final = last_windows.loc[:, list(KEY)].copy()
        final["model_name"] = selected_model
        final["rul_pred"] = np.asarray(
            estimator.predict(last_windows.loc[:, list(MODELING_FEATURE_COLUMNS)]),
            dtype="float64",
        )
        if not np.isfinite(final["rul_pred"]).all():
            raise ValueError(f"{pair_id}: transfer prediction không hữu hạn")
        prediction_path = pair_dir / "modeling" / "predictions" / "final_predictions.parquet"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        final.loc[:, list(FINAL_PREDICTION_COLUMNS)].to_parquet(prediction_path, index=False)
        model_dir = pair_dir / "modeling"
        bootstrap_resamples = (config or ModelingConfig()).bootstrap_resamples
        seed = (config or ModelingConfig()).seed
    else:
        model_dir = pair_dir / "modeling"
        result = run_modeling_pipeline(
            source.features_path,
            source.train_targets_path,
            config=config or ModelingConfig(),
        )
        model_payload = write_modeling_artifacts(result, model_dir)
        prediction_path = model_dir / model_payload["outputs"]["final_predictions"]
        selected_model = str(model_payload["selected_model"])
        bootstrap_resamples = result.config.bootstrap_resamples
        seed = result.config.seed
    labels_path = Path(raw_dir) / f"RUL_{target.dataset_id}.txt"
    evaluation = evaluate_predictions_from_path(
        prediction_path,
        labels_path,
    )
    final_payload = write_final_evaluation_artifacts(
        evaluation,
        model_dir,
        test_label_path=labels_path,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    context = {
        "source_dataset": source.dataset_id,
        "target_dataset": target.dataset_id,
        "evaluation_type": "transfer" if transfer else "within_subset",
        "source_train_features": str(source.features_path),
        "target_test_features": str(target.features_path),
        "target_labels": str(labels_path),
        "selected_model": selected_model,
        "metrics": final_payload["metrics"],
        "bootstrap": final_payload["bootstrap"],
    }
    _write_json(pair_dir / "evaluation_context.json", context)
    return context


def run_matrix(
    *,
    raw_dir: str | Path = "data/raw",
    output_root: str | Path = "outputs/cross_dataset",
    window_cycles: int = 10,
) -> list[dict[str, Any]]:
    """Chạy bốn đánh giá within-subset và ba phép transfer từ FD001."""
    artifacts = {
        dataset_id: prepare_dataset(
            dataset_id,
            raw_dir=raw_dir,
            output_root=output_root,
            window_cycles=window_cycles,
        )
        for dataset_id in DATASET_IDS
    }
    rows: list[dict[str, Any]] = []
    for dataset_id in DATASET_IDS:
        rows.append(evaluate_pair(artifacts[dataset_id], artifacts[dataset_id], raw_dir=raw_dir, output_root=output_root, transfer=False))
    for dataset_id in ("FD002", "FD003", "FD004"):
        rows.append(evaluate_pair(artifacts["FD001"], artifacts[dataset_id], raw_dir=raw_dir, output_root=output_root, transfer=True))
    _write_json(Path(output_root) / "matrix.json", {"window_cycles": window_cycles, "evaluations": rows})
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI chạy đầy đủ evaluation matrix."""
    parser = argparse.ArgumentParser(description="Đánh giá C-MAPSS within-subset và transfer")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cross_dataset"))
    parser.add_argument("--window-cycles", type=int, default=10)
    args = parser.parse_args(argv)
    rows = run_matrix(
        raw_dir=args.raw_dir,
        output_root=args.output_root,
        window_cycles=args.window_cycles,
    )
    for row in rows:
        metrics = row["metrics"]
        print(
            f"{row['source_dataset']}->{row['target_dataset']} "
            f"type={row['evaluation_type']} "
            f"mae={metrics['mean_engine_mae']:.6f} "
            f"rmse={metrics['rmse']:.6f} "
            f"nasa={metrics['nasa_score']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
