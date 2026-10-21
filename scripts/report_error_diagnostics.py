"""Compute read-only error diagnostics cited in the report discussion (Section 7).

Reads existing modeling and warning artifacts; trains nothing and never modifies the source artifacts. Writes one JSON
file to outputs/diagnostics/error_diagnostics.json.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from cmapss.modeling.contracts import RUL_BINS

ROOT = Path(__file__).resolve().parents[1]


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


def main() -> None:
    output = {
        "description": "Read-only diagnostics for report Section 7; no model is trained.",
        "oof_hgb_by_true_rul": oof_by_true_rul(),
        "test_errors": test_error_share(),
        "warning_by_lifetime": warning_by_lifetime(),
    }
    target = ROOT / "outputs/diagnostics/error_diagnostics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
