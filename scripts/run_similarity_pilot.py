"""Run the bounded, non-promotional similarity RUL pilot on FD001."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cmapss.data.contracts import KEY
from cmapss.modeling.audit_study import warning_events, _warning_readout
from cmapss.modeling.multiscale_comparison.analysis import candidate_readout, calibrate_policy
from cmapss.modeling.multiscale_comparison.config import StudyConfig
from cmapss.modeling.similarity import fit_similarity

AUTHOR = {"full_name": "Phan Nguyễn Mai Phương", "student_id": "25002711"}
RUN_ID = "20261007_pilot01"
BASELINE_CACHE = {
    1: "7b8927b945a6417bba10bcdd503a60c5572ce6a96ca5a26eb6f453c67da90c18",
    2: "f7f52b3b6e459759456808579072834d9fffa6baad9d8d26627edce478d473f8",
    3: "29d8d5b7bb411b778190da1fb2d50c9ddf956a12759cd8883e390dab1c5d3eba",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame(keys: pd.DataFrame, target_map: pd.Series, prediction: np.ndarray, procedure: str, fold: int) -> pd.DataFrame:
    result = keys.loc[:, list(KEY)].copy()
    result["y_true"] = [float(target_map.loc[tuple(k)]) for k in keys.itertuples(index=False, name=None)]
    result["y_pred"] = np.asarray(prediction, dtype="float64")
    result["procedure"] = procedure
    result["inner_fold"] = int(fold)
    return result


def _load_reference(root: Path, fold: int, expected: pd.DataFrame) -> pd.DataFrame:
    base = root / "outputs/experiments/multiscale_comparison/20261006_study02/fit_cache"
    path = base / f"{BASELINE_CACHE[fold]}.parquet"
    meta = json.loads((base / f"{BASELINE_CACHE[fold]}.json").read_text(encoding="utf-8"))
    if meta["identity"]["arm_id"] != "hgb_baseline169":
        raise ValueError(f"fold {fold} reference arm mismatch")
    if meta["identity"].get("source_identity") != "87452cb02c68dbead9a98b76eb98bcdd05ac1518b9c8d7ba9214d8357f1f7942":
        raise ValueError(f"fold {fold} reference source identity mismatch")
    if meta.get("prediction_sha256") != sha256(path):
        raise ValueError(f"fold {fold} reference prediction checksum mismatch")
    frame = pd.read_parquet(path).sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
    expected = expected.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(frame.loc[:, list(KEY) + ["y_true"]], expected.loc[:, list(KEY) + ["y_true"]], check_exact=True)
    if not np.isfinite(frame.y_pred).all():
        raise ValueError("reference predictions contain nonfinite values")
    frame["procedure"] = "hgb_baseline169"
    frame["inner_fold"] = int(fold)
    return frame


def _fold_metrics(frames: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows = []
    for (procedure, fold), group in frames.groupby(["procedure", "inner_fold"], sort=True):
        rows.append({"procedure": procedure, "inner_fold": int(fold), **candidate_readout(group, config)})
    return pd.DataFrame(rows)


def run(*, repo_root: Path, output_dir: Path, smoke: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be new: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    obs_path = repo_root / "outputs/etl/silver/observations.parquet"
    target_path = repo_root / "outputs/etl/silver/train_targets.parquet"
    observations = pd.read_parquet(obs_path)
    targets = pd.read_parquet(target_path)
    train_obs = observations.loc[observations.split.eq("train")].copy()
    common = train_obs.loc[train_obs.cycle.ge(30), list(KEY)].merge(
        targets, on=list(KEY), how="left", validate="one_to_one"
    ).sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
    if len(common) != 17_731 or common.engine_id.nunique() != 100:
        raise ValueError(f"unexpected common panel shape {common.shape}")
    assignments = pd.read_csv(repo_root / "outputs/experiments/multiscale_comparison/20261006_study02/stages/final_inner/inner_assignments.csv")
    assignments = assignments.loc[assignments.outer_fold.eq(0), ["inner_fold", "engine_id"]]
    if set(assignments.inner_fold) != {1, 2, 3}:
        raise ValueError("final-inner assignments do not contain exactly three folds")
    config = StudyConfig(outer_splits=3, inner_splits=3, bootstrap_resamples=200)
    identity = {
        "author": AUTHOR, "run_id": output_dir.name, "smoke": smoke,
        "protocol": "experiments/SIMILARITY_PILOT_PROTOCOL.md",
        "window": 30, "block_size": 5, "neighbors": [5, 10],
        "batch_size": 128, "fit_threads": 4, "test_labels_read": False,
        "common_rows": len(common), "common_engines": int(common.engine_id.nunique()),
        "observations_sha256": sha256(obs_path), "targets_sha256": sha256(target_path),
    }
    (output_dir / "run_info.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    target_map = common.set_index(list(KEY))["rul"]
    all_frames: list[pd.DataFrame] = []
    match_frames: list[pd.DataFrame] = []
    fold_rows = []
    fold_ids = [1, 2, 3] if not smoke else [1]
    for fold in fold_ids:
        val_ids = tuple(sorted(assignments.loc[assignments.inner_fold.eq(fold), "engine_id"].astype(int)))
        fit_ids = tuple(sorted(set(train_obs.engine_id.unique()) - set(val_ids)))
        fitting = train_obs.loc[train_obs.engine_id.isin(fit_ids)].copy()
        validation = train_obs.loc[train_obs.engine_id.isin(val_ids)].copy()
        model = fit_similarity(fitting, targets.loc[targets.engine_id.isin(fit_ids)].copy(), batch_size=128, fit_threads=4)
        keys, predictions, matches = model.predict_with_matches(validation, neighbors=(5, 10))
        expected = common.loc[common.engine_id.isin(val_ids), list(KEY)].sort_values(list(KEY), kind="mergesort").reset_index(drop=True)
        keys = keys.astype({"split": "string", "engine_id": "int64", "cycle": "int64"})
        expected = expected.astype({"split": "string", "engine_id": "int64", "cycle": "int64"})
        if not keys.equals(expected):
            raise ValueError(f"similarity fold {fold} keys do not match common panel")
        for k in (5, 10):
            all_frames.append(_frame(keys, target_map, predictions[k], f"similarity_k{k}", fold))
        matches["inner_fold"] = int(fold)
        match_frames.append(matches)
        reference_expected = expected.copy()
        reference_expected["y_true"] = [float(target_map.loc[tuple(k)]) for k in expected.itertuples(index=False, name=None)]
        all_frames.append(_load_reference(repo_root, fold, reference_expected))
        fold_rows.append({"inner_fold": fold, "fit_engines": len(fit_ids), "validation_engines": len(val_ids), "library_rows": len(model.library_keys), "validation_rows": len(keys)})
        if smoke:
            break
    predictions = pd.concat(all_frames, ignore_index=True)
    predictions.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    if match_frames:
        pd.concat(match_frames, ignore_index=True).to_parquet(output_dir / "donor_matches.parquet", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_runtime.csv", index=False)
    fold_metrics = _fold_metrics(predictions, config)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    pooled_rows = []
    policy_rows = []
    event_frames = []
    for procedure, frame in predictions.groupby("procedure", sort=True):
        readout = candidate_readout(frame, config)
        policy, grid = calibrate_policy(frame, config)
        events = warning_events(frame, policy, lead_requirement=config.lead_requirement)
        events["procedure"] = procedure
        event_frames.append(events)
        pooled_rows.append({"procedure": procedure, **readout, **_warning_readout(events), "policy_threshold": policy.threshold, "policy_persistence": policy.persistence})
        policy_rows.append(grid.assign(procedure=procedure))
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(output_dir / "pooled_metrics.csv", index=False)
    pd.concat(policy_rows, ignore_index=True).to_csv(output_dir / "warning_policy_grid.csv", index=False)
    pd.concat(event_frames, ignore_index=True).to_csv(output_dir / "warning_events.csv", index=False)
    baseline = pooled.loc[pooled.procedure.eq("hgb_baseline169")].iloc[0]
    decisions = []
    for procedure in ("similarity_k5", "similarity_k10"):
        row = pooled.loc[pooled.procedure.eq(procedure)].iloc[0]
        fold_rmse = fold_metrics.loc[fold_metrics.procedure.eq(procedure), "trajectory_rmse"].to_numpy()
        base_fold = fold_metrics.loc[fold_metrics.procedure.eq("hgb_baseline169"), "trajectory_rmse"].to_numpy()
        decisions.append({"procedure": procedure, "pooled_rmse_lower": bool(row.trajectory_rmse < baseline.trajectory_rmse), "fold_rmse_wins": int(np.sum(fold_rmse < base_fold)), "fold_rmse_wins_2_of_3": bool(np.sum(fold_rmse < base_fold) >= 2), "endpoint_mae_not_worse": bool(row.endpoint_mean_engine_mae <= baseline.endpoint_mean_engine_mae), "endpoint_nasa_not_worse": bool(row.endpoint_mean_engine_nasa <= baseline.endpoint_mean_engine_nasa), "on_time_not_worse": bool(row.on_time_rate >= baseline.on_time_rate), "burden_not_worse": bool(row.early_alarm_burden <= baseline.early_alarm_burden)})
    decision_frame = pd.DataFrame(decisions)
    decision_frame["pilot_continue_candidate"] = decision_frame[["pooled_rmse_lower", "fold_rmse_wins_2_of_3", "endpoint_mae_not_worse", "endpoint_nasa_not_worse", "on_time_not_worse", "burden_not_worse"]].all(axis=1)
    decision_frame.to_csv(output_dir / "decision.csv", index=False)
    status = "completed" if not smoke else "smoke_completed"
    summary = {"status": status, "elapsed_seconds": time.perf_counter() - started, "test_labels_read": False, "pilot_continue": bool(decision_frame.pilot_continue_candidate.any()) if not smoke else False}
    (output_dir / "RESULTS.md").write_text("**Phan Nguyễn Mai Phương — MSHV 25002711**\n\n# Similarity pilot FD001\n\n" + pooled.to_string(index=False) + "\n\nDecision:\n\n" + decision_frame.to_string(index=False) + "\n\nPilot status: " + json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/similarity_pilot") / RUN_ID)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(repo_root=args.repo_root.resolve(), output_dir=args.output_dir.resolve(), smoke=args.smoke), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
