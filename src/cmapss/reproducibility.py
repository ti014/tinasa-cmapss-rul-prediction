"""Rebuild the FD001 artifact chain and compare it with the checked-in run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd


TABLES = (
    ("etl/silver/observations.parquet", "parquet"),
    ("etl/silver/train_targets.parquet", "parquet"),
    ("etl/gold/engine_summary.parquet", "parquet"),
    ("features/gold/rolling_features.parquet", "parquet"),
    ("modeling/tables/fold_assignments.csv", "csv"),
    ("modeling/tables/fold_metrics.csv", "csv"),
    ("modeling/tables/model_metrics.csv", "csv"),
    ("modeling/predictions/oof_predictions.parquet", "parquet"),
    ("modeling/predictions/final_predictions.parquet", "parquet"),
    ("modeling/predictions/final_evaluation.parquet", "parquet"),
    ("timeseries/tables/selected_engines.csv", "csv"),
    ("timeseries/tables/health_index.csv", "csv"),
    ("timeseries/tables/forecast_metrics.csv", "csv"),
    ("timeseries/tables/diagnostics.csv", "csv"),
    ("timeseries/predictions/arima_forecasts.parquet", "parquet"),
)

FIGURES = (
    "modeling/figures/final_true_vs_predicted.png",
    "modeling/figures/final_residuals_by_rul.png",
    "timeseries/figures/health_index_selected_engines.png",
    "timeseries/figures/arima_forecasts.png",
)


class ReproducibilityError(RuntimeError):
    """A stage failed or regenerated data did not match the reference run."""


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_table(path: Path, kind: str) -> pd.DataFrame:
    if not path.is_file():
        raise ReproducibilityError(f"Missing artifact: {path}")
    return pd.read_parquet(path) if kind == "parquet" else pd.read_csv(path)


def compare_table(actual: Path, reference: Path, kind: str) -> dict[str, Any]:
    """Compare table content, including keys and numeric values, in row order."""

    left = _read_table(actual, kind)
    right = _read_table(reference, kind)
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )
    except AssertionError as exc:
        raise ReproducibilityError(f"Artifact content differs: {actual}: {exc}") from exc
    return {
        "rows": len(left),
        "columns": len(left.columns),
        "actual_sha256": _hash(actual),
        "reference_sha256": _hash(reference),
        "content_equal": True,
    }


def _compare_nested_numbers(actual: Any, reference: Any, name: str) -> None:
    if isinstance(reference, dict):
        if not isinstance(actual, dict) or actual.keys() != reference.keys():
            raise ReproducibilityError(f"Metric structure differs: {name}")
        for key, value in reference.items():
            _compare_nested_numbers(actual[key], value, f"{name}.{key}")
    elif isinstance(reference, (int, float)) and not isinstance(reference, bool):
        if not isinstance(actual, (int, float)) or not np.isclose(
            actual, reference, rtol=1e-9, atol=1e-9, equal_nan=True
        ):
            raise ReproducibilityError(f"Metric differs: {name}: {actual} != {reference}")
    elif actual != reference:
        raise ReproducibilityError(f"Metric differs: {name}: {actual} != {reference}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReproducibilityError(f"Missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_stage(name: str, arguments: list[str], cwd: Path) -> None:
    command = [sys.executable, "-m", name, *arguments]
    print(f"RUN {name}", flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise ReproducibilityError(f"Stage {name} exited with code {completed.returncode}")


def validate_rebuilt_outputs(output_root: Path, reference_root: Path, raw_dir: Path) -> dict[str, Any]:
    """Validate data, evaluation metrics, figures and dashboard loading."""

    output_dir = output_root / "outputs"
    reference_dir = reference_root / "outputs"
    table_results = {
        name: compare_table(output_dir / name, reference_dir / name, kind)
        for name, kind in TABLES
    }
    metrics = _read_json(output_dir / "modeling/tables/final_metrics.json")
    reference_metrics = _read_json(reference_dir / "modeling/tables/final_metrics.json")
    _compare_nested_numbers(metrics["metrics"], reference_metrics["metrics"], "final.metrics")
    _compare_nested_numbers(
        metrics["bootstrap"]["intervals"],
        reference_metrics["bootstrap"]["intervals"],
        "final.bootstrap.intervals",
    )
    model_run = _read_json(output_dir / "modeling/run_info.json")
    reference_model_run = _read_json(reference_dir / "modeling/run_info.json")
    if model_run["selected_model"] != reference_model_run["selected_model"]:
        raise ReproducibilityError("Selected RUL model differs from reference")
    if model_run["test_label_join"]["status"] != "completed_after_final_predictions":
        raise ReproducibilityError("Final label join was not recorded after predictions")
    time_run = _read_json(output_dir / "timeseries/run_info.json")
    if time_run["isolation"]["uses_test_labels"] or time_run["isolation"]["uses_test_observations"]:
        raise ReproducibilityError("PCA--ARIMA isolation contract failed")
    figure_results = {}
    for name in FIGURES:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ReproducibilityError(f"Missing or empty figure: {path}")
        reference_path = reference_dir / name
        if not reference_path.is_file() or _hash(path) != _hash(reference_path):
            raise ReproducibilityError(f"Figure bytes differ: {path}")
        figure_results[name] = {
            "bytes": path.stat().st_size,
            "sha256": _hash(path),
            "reference_sha256": _hash(reference_path),
            "byte_identical": True,
        }

    from app import load_dashboard_data

    dashboard = load_dashboard_data(output_root)
    if len(dashboard.model_evaluation) != 100 or len(dashboard.forecasts) != 704:
        raise ReproducibilityError("Dashboard did not load the expected engine/forecast counts")

    raw_hashes = {name: _hash(raw_dir / name) for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")}
    reference_etl = _read_json(reference_dir / "etl/run_info.json")
    for name in ("train_FD001.txt", "test_FD001.txt"):
        if raw_hashes[name] != reference_etl["input_sha256"][name]:
            raise ReproducibilityError(f"Raw file hash differs from reference: {name}")
    if raw_hashes["RUL_FD001.txt"] != reference_metrics["test_label_sha256"]:
        raise ReproducibilityError("Raw test label hash differs from reference")

    return {
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "raw FD001 -> Pandas ETL -> rolling features -> grouped RUL modeling -> PCA--ARIMA -> dashboard loader",
        "output_root": str(output_root),
        "reference_root": str(reference_root),
        "raw_sha256": raw_hashes,
        "tables": table_results,
        "final_metrics": metrics["metrics"],
        "selected_model": model_run["selected_model"],
        "figures": figure_results,
        "dashboard": {"test_engines": len(dashboard.model_evaluation), "forecast_rows": len(dashboard.forecasts)},
        "comparison": "semantic table equality, numeric rtol=1e-9 and atol=1e-9; timestamps and file bytes excluded",
        "limitations": [
            "PySpark parity and notebooks were validated separately; this command runs the Pandas scientific path.",
            "The dashboard loader was checked against regenerated outputs; the web UI was smoke-tested separately.",
        ],
    }


def run_end_to_end(
    output_root: str | Path,
    *,
    repo_root: str | Path,
    evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the FD001 scientific path in a fresh, isolated output directory."""

    repository = Path(repo_root).resolve()
    target = Path(output_root).resolve()
    raw = repository / "data/raw"
    if target.exists():
        raise ReproducibilityError(f"Output root must not exist for a fresh run: {target}")
    if target == repository or target == raw or raw in target.parents:
        raise ReproducibilityError("Output root must be outside the repository raw data directory")
    for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"):
        if not (raw / name).is_file():
            raise ReproducibilityError(f"Missing raw input: {raw / name}")
    before_hashes = {name: _hash(raw / name) for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")}
    target.mkdir(parents=True)
    outputs = target / "outputs"
    _run_stage(
        "cmapss.etl",
        ["--raw-dir", str(raw), "--output-dir", str(outputs / "etl"), "--benchmark-scales", "1", "--benchmark-repetitions", "1"],
        repository,
    )
    _run_stage(
        "cmapss.features",
        ["--silver-observations", str(outputs / "etl/silver/observations.parquet"), "--output-dir", str(outputs / "features")],
        repository,
    )
    _run_stage(
        "cmapss.modeling",
        [
            "--rolling-features", str(outputs / "features/gold/rolling_features.parquet"),
            "--train-targets", str(outputs / "etl/silver/train_targets.parquet"),
            "--test-labels", str(raw / "RUL_FD001.txt"),
            "--output-dir", str(outputs / "modeling"),
        ],
        repository,
    )
    _run_stage(
        "cmapss.timeseries",
        [
            "--observations", str(outputs / "etl/silver/observations.parquet"),
            "--fold-assignments", str(outputs / "modeling/tables/fold_assignments.csv"),
            "--output-dir", str(outputs / "timeseries"),
        ],
        repository,
    )
    result = validate_rebuilt_outputs(target, repository, raw)
    after_hashes = {name: _hash(raw / name) for name in before_hashes}
    if before_hashes != after_hashes:
        raise ReproducibilityError("Raw FD001 inputs changed during end-to-end run")
    result["raw_immutable"] = True
    validation_path = target / "validation.json"
    validation_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if evidence_path is not None:
        evidence = Path(evidence_path).resolve()
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate and validate FD001 artifacts from raw inputs")
    parser.add_argument("--output-root", required=True, type=Path, help="Fresh directory for isolated outputs")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--evidence-path", type=Path, help="Optional persistent validation JSON")
    args = parser.parse_args(argv)
    try:
        result = run_end_to_end(args.output_root, repo_root=args.repo_root, evidence_path=args.evidence_path)
    except ReproducibilityError as exc:
        print(f"End-to-end validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"End-to-end validation {result['status']}: {len(result['tables'])} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
