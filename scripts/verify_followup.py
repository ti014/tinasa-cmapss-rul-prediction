"""Independently verify the completed follow-up artifacts without refitting."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cmapss.modeling.improvement import _metrics
from cmapss.modeling.improvement_analysis import engine_metrics, paired_comparison
from cmapss.project import FOLLOWUP_RUN
from cmapss.provenance import verify_source_snapshot


def verify_dataset(root: Path, dataset: str, run_root: Path = FOLLOWUP_RUN) -> dict:
    """Verify hashes, fold isolation, metrics and saved-model endpoint inference."""
    run = root / run_root / dataset
    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "search_manifest.json").read_text(encoding="utf-8"))
    identity = manifest["identity"]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    assert digest == info["identity"]
    source_proof = verify_source_snapshot(run, manifest, root)
    for name, expected in identity["inputs"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    assignments = pd.read_csv(run / "outer_assignments.csv").set_index("engine_id").fold
    assert assignments.index.is_unique and assignments.nunique() == 5
    outer = pd.read_parquet(run / "outer_predictions.parquet")
    assert set(outer.stage) == {"baseline", "selected"}
    stored = pd.read_csv(run / "outer_metrics.csv").set_index("stage")
    scores = {}
    reference_keys = None
    for stage in ("baseline", "selected"):
        prediction = outer.loc[outer.stage.eq(stage)].sort_values(["engine_id", "cycle"]).reset_index(drop=True)
        assert not prediction.duplicated(["engine_id", "cycle"]).any()
        np.testing.assert_array_equal(prediction.outer_fold, prediction.engine_id.map(assignments))
        keys = prediction[["engine_id", "cycle", "y_true"]]
        if reference_keys is None:
            reference_keys = keys
        else:
            pd.testing.assert_frame_equal(reference_keys, keys)
        measured = _metrics(prediction, 30)
        for metric in ("mean_engine_mae", "low_rul_engine_mae", "rmse", "nasa_score", "late_or_missed_rate", "early_alarm_burden"):
            np.testing.assert_allclose(measured[metric], stored.loc[stage, metric], rtol=1e-10)
        scores[stage] = engine_metrics(prediction)
    actual = paired_comparison(scores["baseline"], scores["selected"], 10000, 42)
    pd.testing.assert_frame_equal(actual, pd.read_csv(run / "paired_comparison.csv"), check_exact=False, rtol=1e-10)
    fit_records = list((run / "fit_cache").glob("*.json"))
    assert fit_records
    unique_fit_seconds = 0.0
    for path in fit_records:
        record = json.loads(path.read_text(encoding="utf-8"))
        unique_fit_seconds += record["seconds"]
        key = record["identity_record"]
        assert key["identity"] == digest
        assert not set(key["train_engines"]) & set(key["validation_engines"])
        if key["recipe"]["normalize_conditions"]:
            assert sorted(record["normalizer_fit_engines"]) == key["train_engines"]
        assert hashlib.sha256(path.with_suffix(".parquet").read_bytes()).hexdigest() == record["prediction_sha256"]
    source = "outputs/etl" if dataset == "FD001" else f"outputs/cross_dataset/{dataset}/etl"
    observations = pd.read_parquet(root / source / "silver/observations.parquet")
    with threadpool_limits(limits=4):
        predicted = joblib.load(run / "selected_model.joblib").predict(observations.loc[observations.split.eq("test")])
    predicted = predicted.groupby("engine_id").tail(1).reset_index(drop=True)
    pd.testing.assert_frame_equal(predicted, pd.read_parquet(run / "final_predictions.parquet"))
    proof = {"status": "passed", "dataset": dataset, "fit_records": len(fit_records),
             "unique_fit_seconds": unique_fit_seconds,
             "outer_engines": len(assignments), "outer_rows_per_stage": len(reference_keys),
             "test_engines": len(predicted), "paired_ci_recomputed": True,
             "saved_model_predictions_match": True, "source_verification": source_proof}
    (run / "validation.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=FOLLOWUP_RUN)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for dataset in ("FD001", "FD002", "FD004"):
        print(json.dumps(verify_dataset(root, dataset, args.run), ensure_ascii=False), flush=True)
