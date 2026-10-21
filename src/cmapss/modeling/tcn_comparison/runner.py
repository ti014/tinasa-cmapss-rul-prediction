"""
So sánh TCN nested theo động cơ với hai nhánh không cap/cap125; đọc nhãn test sau khi
ghi dự đoán cuối.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cmapss.data.contracts import KEY, SENSORS
from cmapss.data.readers import read_test_labels
from cmapss.modeling.metrics import add_prediction_errors, bootstrap_metric_intervals

from .contracts import TCNConfig, TCNSpec, specs
from .data import SequenceSet, build_sequences
from .model import FittedTCN, fit_model, build_model

AUTHOR = {"full_name": "Phan Nguyễn Mai Phương", "student_id": "25002711"}
ARMS = tuple(item.family for item in specs())


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               default=lambda item: item.item() if hasattr(item, "item") else str(item)) + "\n",
                    encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    scored = add_prediction_errors(frame)
    endpoint = scored.sort_values(["engine_id", "cycle"], kind="mergesort").groupby(
        "engine_id", sort=True, as_index=False).tail(1)
    return {
        "n_rows": int(len(scored)),
        "n_engines": int(scored.engine_id.nunique()),
        "trajectory_rmse": float(np.sqrt(scored.squared_error.mean())),
        "trajectory_mae": float(scored.absolute_error.mean()),
        "endpoint_mae": float(endpoint.absolute_error.mean()),
        "endpoint_nasa_score": float(endpoint.nasa_score_component.mean()),
        "trajectory_nasa_score": float(scored.nasa_score_component.sum()),
    }


def _engine_folds(metadata: pd.DataFrame, *, root: Path, n_splits: int) -> dict[int, int]:
    """Dùng phân công năm fold công bố nếu có, ngược lại GroupKFold."""
    path = root / "outputs/modeling/tables/fold_assignments.csv"
    engines = sorted(map(int, metadata.engine_id.unique()))
    if n_splits == 5 and path.is_file():
        assignments = pd.read_csv(path)
        assignments = assignments[assignments.split.eq("train")]
        mapping = dict(zip(assignments.engine_id.astype(int), assignments.fold.astype(int)))
        if set(mapping) == set(engines) and set(mapping.values()) == set(range(1, 6)):
            return mapping
    splitter = GroupKFold(n_splits=n_splits)
    groups = metadata.engine_id.to_numpy()
    rows = np.arange(len(metadata))
    mapping: dict[int, int] = {}
    for fold, (_, validation) in enumerate(splitter.split(rows, groups=groups), start=1):
        mapping.update({int(engine): fold for engine in metadata.iloc[validation].engine_id.unique()})
    if set(mapping) != set(engines):
        raise ValueError("Phân công fold không phủ mọi động cơ")
    return mapping


def _inner_select(
    train: SequenceSet,
    fit_ids: set[int],
    config: TCNConfig,
    candidate_specs: tuple[TCNSpec, ...],
) -> tuple[str, list[dict[str, object]]]:
    """Chọn nhánh bằng RMSE quỹ đạo inner theo động cơ."""
    fitting = train.metadata[train.metadata.engine_id.isin(fit_ids)].reset_index(drop=True)
    if fitting.engine_id.nunique() < config.inner_splits:
        raise ValueError("Không đủ động cơ fit cho inner GroupKFold")
    inner = GroupKFold(n_splits=config.inner_splits)
    values = train.values[train.metadata.engine_id.isin(fit_ids).to_numpy()]
    rows: list[dict[str, object]] = []
    for arm in candidate_specs:
        pieces: list[pd.DataFrame] = []
        groups = fitting.engine_id.to_numpy()
        indices = np.arange(len(fitting))
        for inner_fold, (fit_rows, val_rows) in enumerate(inner.split(indices, groups=groups), start=1):
            inner_fit_ids = set(map(int, fitting.iloc[fit_rows].engine_id.unique()))
            inner_val = fitting.iloc[val_rows].reset_index(drop=True)
            model = fit_model(train.values[train.metadata.engine_id.isin(fit_ids).to_numpy()],
                              fitting, arm, config, inner_fit_ids)
            predictions = model.predict(values[val_rows])
            piece = inner_val.loc[:, ["split", "engine_id", "cycle", "rul"]].copy()
            piece["y_true"] = piece.pop("rul")
            piece["y_pred"] = predictions
            piece["inner_fold"] = inner_fold
            pieces.append(piece)
        combined = pd.concat(pieces, ignore_index=True)
        metric = _metric_frame(combined)
        rows.append({"candidate_id": arm.candidate_id, **metric})
    rows.sort(key=lambda row: (float(row["trajectory_rmse"]), str(row["candidate_id"])))
    return str(rows[0]["candidate_id"]), rows


def _save_model(path: Path, fitted: FittedTCN) -> None:
    torch = __import__("torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": fitted.state_dict,
        "scaler_mean": fitted.scaler.mean,
        "scaler_scale": fitted.scaler.scale,
        "spec": asdict(fitted.spec),
        "config": asdict(fitted.config),
        "architecture": {
            "input_sensors": len(SENSORS), "sequence_length": fitted.config.sequence_length,
            "channels": fitted.config.channels, "kernel_size": fitted.config.kernel_size,
            "dilations": list(fitted.config.dilations), "blocks": len(fitted.config.dilations),
            "head": "last_timestep_linear",
        },
    }, path)


def run_tcn_comparison(
    *,
    repo_root: str | Path | None = None,
    observations_path: str | Path = "outputs/etl/silver/observations.parquet",
    targets_path: str | Path = "outputs/etl/silver/train_targets.parquet",
    test_labels_path: str | Path = "data/raw/RUL_FD001.txt",
    output_dir: str | Path = "outputs/experiments/tcn_comparison/20261005",
    config: TCNConfig | None = None,
    evaluate_test: bool = True,
) -> dict[str, object]:
    """Chạy nested cross-validation, fit mô hình cuối và báo cáo metric test cho hai nhánh."""
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[4]
    config = config or TCNConfig()
    output = Path(output_dir)
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    allowed = root / "outputs/experiments/tcn_comparison"
    if not output.is_relative_to(allowed):
        raise ValueError("Output phải nằm dưới outputs/experiments/tcn_comparison")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Từ chối ghi đè output không rỗng: {output}")
    output.mkdir(parents=True, exist_ok=True)
    observations_path = (root / observations_path).resolve() if not Path(observations_path).is_absolute() else Path(observations_path).resolve()
    targets_path = (root / targets_path).resolve() if not Path(targets_path).is_absolute() else Path(targets_path).resolve()
    test_labels_path = (root / test_labels_path).resolve() if not Path(test_labels_path).is_absolute() else Path(test_labels_path).resolve()
    observations = pd.read_parquet(observations_path)
    targets = pd.read_parquet(targets_path)
    train = build_sequences(observations, targets, split="train", sequence_length=config.sequence_length)
    test = build_sequences(observations, None, split="test", sequence_length=config.sequence_length)
    fold_map = _engine_folds(train.metadata, root=root, n_splits=config.outer_splits)
    _json(output / "protocol.json", {
        "author": AUTHOR, "config": asdict(config), "candidates": [asdict(item) for item in specs()],
        "architecture": {"inputs": list(SENSORS), "sequence_length": config.sequence_length,
                          "channels": config.channels, "kernel_size": config.kernel_size,
                          "dilations": list(config.dilations), "residual_blocks": 4,
                          "head": "last_timestep_linear"},
        "target_policy": "cap125 affects fitting labels only; raw RUL is evaluation truth",
        "grouping": "engine_id", "scientific": config.scientific,
        "input_sha256": {"observations": _sha(observations_path), "targets": _sha(targets_path)},
        "versions": {"python": platform.python_version(), "numpy": version("numpy"),
                     "pandas": version("pandas"), "torch": version("torch")},
    })
    pd.DataFrame([{"engine_id": engine, "outer_fold": fold} for engine, fold in sorted(fold_map.items())]).to_csv(
        output / "outer_assignments.csv", index=False)
    started = time.perf_counter()
    candidate_specs = specs()
    outer_pieces: list[pd.DataFrame] = []
    inner_rows: list[dict[str, object]] = []
    outer_choices: list[dict[str, object]] = []
    for fold in range(1, config.outer_splits + 1):
        validation_ids = {engine for engine, assigned in fold_map.items() if assigned == fold}
        fitting_ids = set(fold_map) - validation_ids
        selected, inner_metrics = _inner_select(train, fitting_ids, config, candidate_specs)
        outer_choices.append({"outer_fold": fold, "selected_candidate_id": selected,
                              "fit_engine_ids": sorted(fitting_ids), "validation_engine_ids": sorted(validation_ids)})
        inner_rows.extend({"outer_fold": fold, **row} for row in inner_metrics)
        fit_mask = train.metadata.engine_id.isin(fitting_ids).to_numpy()
        val_mask = train.metadata.engine_id.isin(validation_ids).to_numpy()
        for arm in candidate_specs:
            model = fit_model(train.values[fit_mask], train.metadata.loc[fit_mask].reset_index(drop=True),
                              arm, config, fitting_ids)
            predictions = model.predict(train.values[val_mask])
            piece = train.metadata.loc[val_mask, ["split", "engine_id", "cycle", "rul"]].copy().reset_index(drop=True)
            piece["y_true"] = piece.pop("rul"); piece["y_pred"] = predictions
            piece["candidate_id"] = arm.candidate_id; piece["procedure"] = arm.family; piece["outer_fold"] = fold
            outer_pieces.append(piece)
        pd.DataFrame(inner_metrics).to_csv(output / f"inner_metrics_outer_{fold}.csv", index=False)
    outer = pd.concat(outer_pieces, ignore_index=True)
    metrics_rows = [{"procedure": arm, **_metric_frame(outer[outer.procedure.eq(arm)])} for arm in ARMS]
    pd.DataFrame(metrics_rows).to_csv(output / "outer_metrics.csv", index=False)
    outer.to_parquet(output / "outer_predictions.parquet", index=False)
    final_selected, final_inner = _inner_select(train, set(fold_map), config, candidate_specs)
    pd.DataFrame(final_inner).to_csv(output / "final_inner_metrics.csv", index=False)
    _json(output / "selection.json", {"author": AUTHOR, "selected_candidate_id": final_selected,
                                      "outer_choices": outer_choices, "test_labels_read_for_selection": False})
    test_endpoints = test.metadata.sort_values(["engine_id", "cycle"], kind="mergesort").groupby(
        "engine_id", sort=True, as_index=False).tail(1).reset_index(drop=True)
    test_mask = test.metadata.index.isin(test_endpoints.index)
    # groupby.tail giữ chỉ mục nguồn, nên ghép khóa tường minh để tạo mask.
    endpoint_keys = set(zip(test_endpoints.engine_id.astype(int), test_endpoints.cycle.astype(int)))
    test_mask = np.array([(int(engine), int(cycle)) in endpoint_keys
                          for engine, cycle in zip(test.metadata.engine_id, test.metadata.cycle)])
    final_predictions: list[pd.DataFrame] = []
    final_models: dict[str, FittedTCN] = {}
    full_ids = set(fold_map)
    for arm in candidate_specs:
        model = fit_model(train.values, train.metadata, arm, config, full_ids)
        final_models[arm.family] = model
        _save_model(output / "models" / arm.family / "model.pt", model)
        predictions = model.predict(test.values)
        piece = test.metadata.loc[test_mask, ["split", "engine_id", "cycle"]].copy().reset_index(drop=True)
        endpoint_values = predictions[test_mask]
        piece["candidate_id"] = arm.candidate_id; piece["procedure"] = arm.family; piece["y_pred"] = endpoint_values
        final_predictions.append(piece)
    final = pd.concat(final_predictions, ignore_index=True)
    final.to_parquet(output / "final_predictions.parquet", index=False)
    test_result: dict[str, object] | None = None
    if evaluate_test:
        labels = read_test_labels(test_labels_path).rename(columns={"rul": "y_true"})
        evaluations = []
        for arm in ARMS:
            frame = final[final.procedure.eq(arm)].merge(labels, on=["split", "engine_id"], how="inner", validate="one_to_one")
            scored = add_prediction_errors(frame)
            evaluations.append({"procedure": arm, **_metric_frame(frame),
                                "bootstrap_intervals": bootstrap_metric_intervals(scored, n_resamples=config.bootstrap_resamples, seed=config.seed)})
        _json(output / "test_metrics.json", {"reporting_only": True, "selection_used_test_labels": False,
                                             "procedures": evaluations})
        test_result = {"reporting_only": True, "procedures": evaluations}
    result = {"status": "completed", "author": AUTHOR, "selected_candidate_id": final_selected,
              "output_dir": output.relative_to(root).as_posix(), "scientific": config.scientific,
              "test_evaluated": evaluate_test, "seconds": time.perf_counter() - started}
    _json(output / "run_info.json", result)
    return {**result, "test": test_result}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fit-threads", type=int, default=4)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--skip-test-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = TCNConfig(epochs=args.epochs, fit_threads=args.fit_threads,
                       outer_splits=args.outer_splits, inner_splits=args.inner_splits)
    if args.dry_run:
        config = replace(config, epochs=min(args.epochs, 2), outer_splits=2, inner_splits=2,
                         bootstrap_resamples=20)
    run_tcn_comparison(repo_root=args.repo_root, output_dir=args.output_dir,
                       config=config, evaluate_test=not args.skip_test_evaluation)


if __name__ == "__main__":
    main()
