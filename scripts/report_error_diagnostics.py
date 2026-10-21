"""Compute read-only error diagnostics cited in the report discussion (Section 6).

Reads existing modeling, warning and cross-dataset artifacts plus raw C-MAPSS
files; trains nothing and never modifies the source artifacts. Writes one JSON
file to outputs/diagnostics/error_diagnostics.json.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from cmapss.modeling.contracts import RUL_BINS

ROOT = Path(__file__).resolve().parents[1]
RAW_COLUMNS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]
# Sensors with a within-engine trend in FD001 EDA; constant and near-constant
# sensors are excluded so the range check reflects informative channels.
RANGE_SENSORS = [f"sensor_{i}" for i in (2, 3, 4, 7, 11, 12, 15, 20, 21)]


def read_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/raw" / name, sep=r"\s+", header=None, names=RAW_COLUMNS)


def oof_by_true_rul() -> list[dict]:
    oof = pd.read_parquet(ROOT / "outputs/modeling/predictions/oof_predictions.parquet")
    oof = oof[oof["model_name"] == "hist_gradient_boosting"].copy()
    oof["error"] = oof["y_pred"] - oof["y_true"]
    bins = pd.cut(oof["y_true"], RUL_BINS[:-1])
    table = oof.groupby(bins, observed=True).agg(
        n_rows=("error", "size"),
        mae=("absolute_error", "mean"),
        mean_signed_error=("error", "mean"),
        mean_prediction=("y_pred", "mean"),
    )
    return [
        {"true_rul_bin": str(interval), **{k: float(v) for k, v in row.items()}}
        for interval, row in table.iterrows()
    ]


def test_error_share() -> dict:
    final = pd.read_parquet(ROOT / "outputs/modeling/predictions/final_evaluation.parquet")
    final = final.sort_values("absolute_error", ascending=False)
    total = final["nasa_score_component"].sum()
    bins = pd.cut(final["y_true"], (*RUL_BINS[:4], RUL_BINS[-1]))
    by_bin = final.groupby(bins, observed=True)["absolute_error"].agg(["mean", "size"])
    return {
        "top10_share_of_nasa_score": float(final["nasa_score_component"].head(10).sum() / total),
        "mae_by_true_rul_bin": {
            str(k): {"mae": float(v["mean"]), "n_engines": int(v["size"])} for k, v in by_bin.iterrows()
        },
    }


def warning_by_lifetime() -> dict:
    events = pd.read_csv(ROOT / "outputs/modeling/tables/warning_events.csv")
    grouped = events.groupby("warning_timing")["failure_cycle"].agg(["size", "mean", "median"])
    return {
        "failure_cycle_by_timing": {
            k: {"n_engines": int(v["size"]), "mean": float(v["mean"]), "median": float(v["median"])}
            for k, v in grouped.iterrows()
        },
        "pearson_failure_cycle_vs_threshold_delay": float(
            np.corrcoef(events["failure_cycle"], events["threshold_delay_cycles"])[0, 1]
        ),
    }


def transfer_diagnostics() -> dict:
    reference = read_raw("train_FD001.txt")
    result = {}
    for source, target in [("FD001", t) for t in ("FD001", "FD002", "FD003", "FD004")] + [
        (d, d) for d in ("FD002", "FD003", "FD004")
    ]:
        path = ROOT / f"outputs/cross_dataset/evaluations/{source}_to_{target}/modeling/predictions/final_evaluation.parquet"
        frame = pd.read_parquet(path)
        error = frame["y_pred"] - frame["y_true"]
        result[f"{source}_to_{target}"] = {
            "mean_signed_error": float(error.mean()),
            "share_overpredicted": float((error > 0).mean()),
            "prediction_mean": float(frame["y_pred"].mean()),
            "prediction_std": float(frame["y_pred"].std()),
            "true_mean": float(frame["y_true"].mean()),
        }
    for target in ("FD002", "FD003", "FD004"):
        test = read_raw(f"test_{target}.txt")
        inside = [
            float(((test[c] >= reference[c].min()) & (test[c] <= reference[c].max())).mean())
            for c in RANGE_SENSORS
        ]
        result[f"FD001_to_{target}"]["mean_share_test_values_inside_fd001_train_range"] = float(np.mean(inside))
    return result


def main() -> None:
    output = {
        "description": "Read-only diagnostics for report Section 6; no model is trained.",
        "range_sensors": RANGE_SENSORS,
        "oof_hgb_by_true_rul": oof_by_true_rul(),
        "test_errors": test_error_share(),
        "warning_by_lifetime": warning_by_lifetime(),
        "transfer": transfer_diagnostics(),
    }
    target = ROOT / "outputs/diagnostics/error_diagnostics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
