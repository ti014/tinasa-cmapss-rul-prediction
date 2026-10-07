"""Check the stored nested experiment, without selecting or fitting a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cmapss.features.rolling import build_rolling_features, read_silver_observations
from cmapss.modeling.improvement import _metrics
from cmapss.modeling.improvement_analysis import engine_metrics, paired_comparison
from cmapss.project import IMPROVEMENT_RUN
from cmapss.provenance import verify_source_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=IMPROVEMENT_RUN.as_posix())
    args = parser.parse_args()
    output = Path(args.run)
    manifest = json.loads((output / "search_manifest.json").read_text(encoding="utf-8"))
    run_info = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
    assert run_info["test_label_joined"]
    source_proof = verify_source_snapshot(output, manifest, Path.cwd())
    for name, digest in manifest["baseline_sha256"].items():
        assert hashlib.sha256((Path("outputs/modeling") / name).read_bytes()).hexdigest() == digest, name
        assert hashlib.sha256((output / "baseline_snapshot" / name).read_bytes()).hexdigest() == digest, name
    for name, digest in manifest["identity"]["inputs"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    folds = pd.read_csv(output / "outer_assignments.csv")
    inner = pd.read_csv(output / "inner_assignments.csv")
    assert len(folds) == folds.engine_id.nunique() == 100
    original = pd.read_csv(output / "baseline_snapshot/tables/fold_assignments.csv")
    pd.testing.assert_frame_equal(folds, original[["engine_id", "fold"]].rename(columns={"fold": "outer_fold"}))
    all_ids = set(folds.engine_id)
    valid_fit_pairs = set()
    for fold in range(1, 6):
        held_out = set(folds.loc[folds.outer_fold.eq(fold), "engine_id"])
        valid_fit_pairs.add((frozenset(all_ids-held_out), frozenset(held_out)))
    for outer, group in inner.groupby("outer_fold"):
        excluded = set(folds.loc[folds.outer_fold.eq(outer), "engine_id"]) if outer else set()
        assert set(group.engine_id) == all_ids - excluded
        assert not group.engine_id.duplicated().any()
        for _, validation in group.groupby("inner_fold"):
            validation_ids = set(validation.engine_id)
            valid_fit_pairs.add((frozenset(all_ids-excluded-validation_ids), frozenset(validation_ids)))
    cache_records = list((output / "fit_cache").glob("*.json"))
    for path in cache_records:
        record = json.loads(path.read_text())
        assert record["status"] == "completed", path
        pair = (frozenset(record["train_engines"]), frozenset(record["validation_engines"]))
        assert not pair[0] & pair[1]
        assert pair in valid_fit_pairs, path
    outer = pd.read_parquet(output / "outer_predictions.parquet")
    ablation = pd.read_csv(output / "ablation_metrics.csv").set_index("stage")
    paired = pd.read_csv(output / "paired_comparison.csv")
    reference_keys = None
    baseline = None
    for stage in ("baseline", "windows", "parameters", "weights"):
        frame = outer.loc[outer.stage.eq(stage)].sort_values(["engine_id", "cycle"]).reset_index(drop=True)
        assert len(frame) == 17731 and frame.engine_id.nunique() == 100
        assert not frame.duplicated(["engine_id", "cycle"]).any()
        assert frame.cycle.min() == 30
        expected = frame.engine_id.map(folds.set_index("engine_id").outer_fold)
        np.testing.assert_array_equal(frame.outer_fold, expected)
        keys = frame[["engine_id", "cycle", "y_true"]]
        if reference_keys is None:
            reference_keys = keys
        else:
            pd.testing.assert_frame_equal(keys, reference_keys)
        calculated = _metrics(frame, 30)
        for key in ("mean_engine_mae", "rmse", "nasa_score", "low_rul_engine_mae", "late_or_missed_rate", "early_alarm_burden"):
            np.testing.assert_allclose(calculated[key], ablation.loc[stage, key], rtol=1e-10)
        per_engine = engine_metrics(frame)
        if stage == "baseline":
            baseline = per_engine
        else:
            actual = paired_comparison(baseline, per_engine, 10000, 42)
            expected_paired = paired.loc[paired.stage.eq(stage)].drop(columns="stage").reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected_paired, check_exact=False, rtol=1e-10)
    saved_baseline = pd.read_parquet(output / "baseline_snapshot/predictions/oof_predictions.parquet")
    saved_baseline = saved_baseline.loc[saved_baseline.model_name.eq("hist_gradient_boosting")].sort_values(["engine_id", "cycle"])
    actual_baseline = outer.loc[outer.stage.eq("baseline")].sort_values(["engine_id", "cycle"])
    np.testing.assert_allclose(actual_baseline.y_pred, saved_baseline.y_pred, atol=1e-8, rtol=1e-8)
    final = pd.read_parquet(output / "final_predictions.parquet").sort_values("engine_id")
    assert len(final) == final.engine_id.nunique() == 100
    model = joblib.load(output / "selected_model.joblib")
    observations = read_silver_observations("outputs/etl/silver/observations.parquet")
    features = build_rolling_features(observations, window_cycles=int(run_info["selected_final"]["window"]))
    test = features.loc[features.split.eq("test")].sort_values(["engine_id", "cycle"]).groupby("engine_id").tail(1)
    with threadpool_limits(limits=4):
        predicted = model.predict(test.loc[:, list(model.feature_names_in_)])
    np.testing.assert_allclose(predicted, final.rul_pred, atol=1e-10)
    proof = {"status": "passed", "baseline_unchanged": True, "source_and_inputs_match": True,
             "outer_engines": 100, "rows_per_stage": 17731, "stages": 4,
             "validated_fit_records": len(cache_records), "paired_ci_recomputed": True,
             "saved_model_predictions_match": True, "test_predictions": 100,
             "source_verification": source_proof}
    (output / "validation.json").write_text(json.dumps(proof, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
