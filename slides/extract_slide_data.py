"""Gom số liệu dùng trên slide từ artifact đã nghiệm thu trong outputs/.

Chạy: uv run --frozen python slides/extract_slide_data.py
Đầu ra: slides/slide_data.json. Script chỉ đọc artifact, không huấn luyện lại.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
IMPROVE = OUT / "experiments" / "rul_improvement" / "20260927_full"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def subsets() -> list[dict]:
    rows = []
    for name in ("FD001", "FD002", "FD003", "FD004"):
        info = read_json(OUT / "cross_dataset" / name / "etl" / "run_info.json")
        rows.append(
            {
                "name": name,
                "train_engines": info["engine_counts"]["train"],
                "test_engines": info["engine_counts"]["test"],
                "observations": info["row_counts"]["observations"],
            }
        )
    return rows


def eda() -> dict:
    life = pd.read_csv(OUT / "eda" / "tables" / "lifetime_summary.csv").set_index("statistic")["cycles"]
    const = pd.read_csv(OUT / "eda" / "tables" / "constant_engine_counts.csv")
    always = const[const["constant_engines"] == const["total_engines"]]["measurement"].tolist()
    partial = const[(const["constant_engines"] > 0) & (const["constant_engines"] < const["total_engines"])]
    return {
        "lifetime": {k: float(life[k]) for k in ("min", "50%", "max", "mean")},
        "always_constant": sorted(always, key=lambda s: (s.split("_")[0], int(s.split("_")[1]))),
        "partially_constant": partial[["measurement", "constant_engines"]].to_dict("records"),
    }


def statistics() -> dict:
    inf = pd.read_csv(OUT / "statistics" / "tables" / "sensor_inference.csv")
    reject = inf["fdr_reject"].astype(str).str.lower() == "true"
    return {
        "n_sensors": int(len(inf)),
        "n_reject": int(reject.sum()),
        "n_degenerate": int((inf["test_status"] == "all_zero_difference").sum()),
        "resamples": int(inf["bootstrap_resamples"].iloc[0]),
    }


def timeseries() -> dict:
    arima = pd.read_csv(OUT / "timeseries" / "tables" / "forecast_metrics.csv")
    base = pd.read_csv(OUT / "timeseries" / "tables" / "baseline_metrics.csv")
    mae = {"ARIMA": float(arima["mae"].mean())}
    for name, frame in base.groupby("baseline_name"):
        mae[name] = float(frame["mae"].mean())
    return {
        "n_engines": int(len(arima)),
        "mae": mae,
        "coverage_min": float(arima["coverage_95"].min()),
        "coverage_max": float(arima["coverage_95"].max()),
    }


def modeling() -> dict:
    final = read_json(OUT / "modeling" / "tables" / "final_metrics.json")
    models = pd.read_csv(OUT / "modeling" / "tables" / "model_metrics.csv")
    evaluation = pd.read_parquet(OUT / "modeling" / "predictions" / "final_evaluation.parquet")
    residual = evaluation["y_pred"] - evaluation["y_true"]
    worst = evaluation.loc[evaluation["absolute_error"].idxmax()]
    warning = read_json(OUT / "modeling" / "tables" / "warning_summary.json")
    return {
        "final": final["metrics"],
        "ci_mae": final["bootstrap"]["intervals"]["mean_engine_mae"],
        "bootstrap_resamples": final["bootstrap"]["resamples"],
        "cv": models[["model_name", "mean_engine_mae", "std_engine_mae", "selected"]].to_dict("records"),
        "n_over": int((residual > 0).sum()),
        "n_under": int((residual < 0).sum()),
        "worst": {
            "engine_id": int(worst["engine_id"]),
            "y_true": float(worst["y_true"]),
            "y_pred": float(worst["y_pred"]),
            "abs_error": float(worst["absolute_error"]),
        },
        "warning": {
            "threshold": warning["rul_threshold"],
            "timing": warning["timing_counts"],
            "lead": warning["lead_time_distribution"],
        },
    }


def pipeline() -> dict:
    feat = read_json(OUT / "features" / "run_info.json")
    bench = read_json(OUT / "etl" / "benchmark.json")
    parity = read_json(OUT / "etl" / "parity.json")
    cold = [r for r in bench["results"] if r["run_kind"] == "cold"]
    return {
        "window": feat["window_cycles"],
        "feature_rows": feat["row_counts"],
        "feature_columns": feat["feature_contract"]["feature_column_count"],
        "benchmark": [
            {"scale": r["scale"], "rows": r["input_rows"], "seconds": r["duration_seconds"]} for r in cold
        ],
        "parity": [
            {"name": c["name"], "rows": c["rows"], "max_abs_diff": c["max_abs_diff"], "status": c["status"]}
            for c in parity["checks"]
        ],
    }


def improvement() -> dict:
    ablation = pd.read_csv(IMPROVE / "ablation_metrics.csv").set_index("stage")
    paired = pd.read_csv(IMPROVE / "paired_comparison.csv")
    low = paired[(paired["stage"] == "weights") & (paired["metric"] == "low_rul_engine_mae")].iloc[0]
    manifest = read_json(IMPROVE / "search_manifest.json")["identity"]["config"]
    inner = pd.read_csv(IMPROVE / "final_selection_metrics.csv")
    pick = {
        "baseline": inner[(inner["candidate_id"] == "baseline") & (inner["stage"] == "parameters")].iloc[0],
        "hgb_08": inner[inner["candidate_id"] == "hgb_08"].iloc[0],
        "strong_low_rul": inner[inner["candidate_id"] == "strong_low_rul"].iloc[0],
    }
    return {
        "outer_splits": manifest["outer_splits"],
        "inner_splits": manifest["inner_splits"],
        "n_hgb_candidates": manifest["n_hgb_candidates"],
        "outer": {
            stage: {
                "mae": float(ablation.loc[stage, "mean_engine_mae"]),
                "low_mae": float(ablation.loc[stage, "low_rul_engine_mae"]),
                "late": int(ablation.loc[stage, "late_or_missed_count"]),
                "rows": int(ablation.loc[stage, "n_rows"]),
            }
            for stage in ("baseline", "weights")
        },
        "low_delta_ci": [float(low["delta"]), float(low["ci_lower"]), float(low["ci_upper"])],
        "inner": {
            key: {
                "mae": float(row["mean_engine_mae"]),
                "low_mae": float(row["low_rul_engine_mae"]),
                "late": int(row["late_or_missed_count"]),
            }
            for key, row in pick.items()
        },
        "decision": read_json(IMPROVE / "decision.json")["default_model"],
    }


def transfer() -> dict:
    matrix = read_json(OUT / "cross_dataset" / "matrix.json")
    return {
        "window": matrix["window_cycles"],
        "rows": [
            {
                "source": e["source_dataset"],
                "target": e["target_dataset"],
                "type": e["evaluation_type"],
                "mae": e["metrics"]["mean_engine_mae"],
                "n_engines": e["metrics"]["n_engines"],
            }
            for e in matrix["evaluations"]
        ],
    }


def diagnostics() -> dict:
    diag = read_json(OUT / "diagnostics" / "error_diagnostics.json")
    oof = {r["true_rul_bin"]: r["mean_prediction"] for r in diag["oof_hgb_by_true_rul"]}
    life = diag["warning_by_lifetime"]
    return {
        "top10_nasa_share": diag["test_errors"]["top10_share_of_nasa_score"],
        "oof_mean_pred_150_200": oof["(150, 200]"],
        "oof_mean_pred_200_400": oof["(200, 400]"],
        "late_mean_life": life["failure_cycle_by_timing"]["after_threshold"]["mean"],
        "early_mean_life": life["failure_cycle_by_timing"]["before_threshold"]["mean"],
        "life_delay_r": life["pearson_failure_cycle_vs_threshold_delay"],
        "transfer": {
            k.split("_to_")[1]: {
                "inside_range": v.get("mean_share_test_values_inside_fd001_train_range"),
                "pred_std": v["prediction_std"],
                "mean_signed_error": v["mean_signed_error"],
            }
            for k, v in diag["transfer"].items()
            if k.startswith("FD001_to_")
        },
    }


def main() -> None:
    data = {
        "diagnostics": diagnostics(),
        "subsets": subsets(),
        "eda": eda(),
        "statistics": statistics(),
        "timeseries": timeseries(),
        "modeling": modeling(),
        "pipeline": pipeline(),
        "improvement": improvement(),
        "transfer": transfer(),
    }
    target = ROOT / "slides" / "slide_data.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
