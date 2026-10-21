"""Gom số liệu dùng trên slide từ artifact đã nghiệm thu trong outputs/.

Chạy: uv run --frozen python slides/extract_slide_data.py
Đầu ra: slides/slide_data.json. Script chỉ đọc artifact, không huấn luyện lại.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cmapss.project import IMPROVEMENT_RUN
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
IMPROVE = ROOT / IMPROVEMENT_RUN
ALIGN = OUT / "course_alignment" / "20261007"


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
        "same_direction": inf.loc[inf["n_positive"].eq(100) | inf["n_negative"].eq(100), "sensor"].tolist(),
    }


def timeseries() -> dict:
    arima = pd.read_csv(OUT / "timeseries" / "tables" / "forecast_metrics.csv")
    base = pd.read_csv(OUT / "timeseries" / "tables" / "baseline_metrics.csv")
    mae = {"ARIMA": float(arima["mae"].mean())}
    for name, frame in base.groupby("baseline_name"):
        mae[name] = float(frame["mae"].mean())
    drift = base.loc[base["baseline_name"].eq("drift")].set_index("engine_id")["mae"]
    return {
        "n_engines": int(len(arima)),
        "mae": mae,
        "n_better_than_drift": int((arima.set_index("engine_id")["mae"] < drift).sum()),
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
    low_rul = evaluation.loc[evaluation["y_true"].le(30)]
    high_rul = evaluation.loc[evaluation["y_true"].gt(60)]
    cv = models.set_index("model_name")
    warning_event = pd.read_csv(OUT / "modeling/tables/warning_events.csv").set_index("engine_id").loc[1]
    return {
        "final": final["metrics"],
        "ci_mae": final["bootstrap"]["intervals"]["mean_engine_mae"],
        "bootstrap_resamples": final["bootstrap"]["resamples"],
        "cv": models[["model_name", "mean_engine_mae", "std_engine_mae", "selected"]].to_dict("records"),
        "n_over": int((residual > 0).sum()),
        "n_under": int((residual < 0).sum()),
        "low_rul": {"n_engines": int(len(low_rul)), "mae": float(low_rul["absolute_error"].mean())},
        "high_rul_mae": float(high_rul["absolute_error"].mean()),
        "test_rul_min": float(evaluation["y_true"].min()),
        "test_rul_max": float(evaluation["y_true"].max()),
        "ridge_minus_hgb_mae": float(cv.loc["ridge_alpha_1", "mean_engine_mae"] - cv.loc["hist_gradient_boosting", "mean_engine_mae"]),
        "example_warning": {"engine_id": 1, "first_warning_cycle": int(warning_event["first_warning_cycle"]),
                            "failure_cycle": int(warning_event["failure_cycle"])},
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


def maintenance() -> dict:
    """Gom lịch và số liệu đối chiếu từ artifact bảo trì đã lưu."""
    summary = read_json(OUT / "maintenance/summary.json")
    schedule = pd.read_csv(OUT / "maintenance/tables/maintenance_schedule.csv")
    evaluation = pd.read_parquet(OUT / "modeling/predictions/final_evaluation.parquet")
    checked = schedule.merge(evaluation[["engine_id", "y_true"]], on="engine_id", validate="one_to_one")
    threshold = summary["rul_threshold"]
    missed = checked.loc[checked["y_true"].le(threshold) & checked["status"].ne("Cần bảo trì")]
    late = checked.assign(delay=checked["maintenance_cycle"] - checked["current_cycle"] - checked["y_true"])
    late = late.loc[late["delay"].gt(0)]
    watch = checked.loc[checked["status"].eq("Theo dõi tăng cường")]
    normal = checked.loc[checked["status"].eq("Bình thường")]
    return {
        **summary,
        # NaN không hợp lệ trong JSON; động cơ chưa có bất thường ghi anomaly_cycle = null.
        "schedule": schedule.astype(object).where(schedule.notna(), None).to_dict("records"),
        "status_check": pd.read_csv(OUT / "maintenance/tables/status_vs_actual.csv").astype(str).to_dict("split"),
        "watch_above_60": int(watch["y_true"].gt(60).sum()),
        "normal_min_actual_rul": float(normal["y_true"].min()),
        "missed": {"n_engines": int(len(missed)),
                   "rul_pred_min": float(missed["rul_pred"].min()), "rul_pred_max": float(missed["rul_pred"].max()),
                   "due_min": int(missed["cycles_to_maintenance"].min()), "due_max": int(missed["cycles_to_maintenance"].max())},
        "late": {"n_engines": int(len(late)), "min_actual_rul": float(late["y_true"].min()),
                 "delay_min": int(late["delay"].min()), "delay_max": int(late["delay"].max())},
    }


def alignment() -> dict:
    """Đọc kết quả đối chiếu HGB126/HGB169 đã lưu, không huấn luyện lại."""
    summary = read_json(ALIGN / "summary.json")
    metrics = pd.read_csv(ALIGN / "model_metrics.csv")
    ci = pd.read_csv(ALIGN / "paired_bootstrap_ci.csv")
    warning = pd.read_csv(ALIGN / "warning_fixed_policies.csv")

    def row(model: str, panel: str) -> dict:
        r = metrics[(metrics["model"] == model) & (metrics["panel"] == panel)].iloc[0]
        return {
            "mean_engine_mae": float(r["mean_engine_mae"]),
            "rmse": float(r["rmse"]),
            "nasa_score": float(r["nasa_score"]),
        }

    def delta(panel: str, metric: str) -> dict:
        r = ci[(ci["panel"] == panel) & (ci["metric"] == metric)].iloc[0]
        return {
            "value": float(r["delta_right_minus_left"]),
            "ci_lower": float(r["ci_lower"]),
            "ci_upper": float(r["ci_upper"]),
        }

    policies = []
    for _, r in warning.iterrows():
        policies.append(
            {
                "model": str(r["model"]),
                "threshold": int(r["threshold"]),
                "on_time_rate": float(r["on_time_rate"]),
                "late_or_missed_rate": float(r["late_or_missed_rate"]),
                "early_alarm_burden": float(r["early_alarm_burden"]),
                "median_lead_time": float(r["median_lead_time"]),
            }
        )
    return {
        "main_predictor": summary["recommendation"]["main_predictor"],
        "baseline": "hgb126",
        "trajectory": {m: row(m, "trajectory") for m in ("hgb126", "hgb169")},
        "endpoint": {m: row(m, "endpoint_50_70_90") for m in ("hgb126", "hgb169")},
        "delta": {
            "trajectory_mae": delta("trajectory", "mean_engine_mae"),
            "trajectory_rmse": delta("trajectory", "rmse"),
            "endpoint_mae": delta("endpoint_50_70_90", "mean_engine_mae"),
            "endpoint_rmse": delta("endpoint_50_70_90", "rmse"),
        },
        "warning_policies": policies,
        "calibrated169_on_time_rate": float(summary["calibrated169_reference"]["on_time_rate"]),
        "limitations": summary["limitations"],
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
    from cmapss.followup_results import load_followup_results
    from cmapss.project import FOLLOWUP_RUN

    followup, followup_paired = load_followup_results(ROOT)
    benchmark = pd.read_csv(OUT / "benchmarks/20260929/measurements.csv")
    benchmark = benchmark.groupby(["engine", "scale"], as_index=False).median(numeric_only=True)
    # Pandas has no JVM heap; omit that unused nullable field from strict JSON.
    benchmark = benchmark.drop(columns=["spark_driver_heap_bytes"], errors="ignore")
    data = {
        "provenance": {"improvement_run": IMPROVEMENT_RUN.as_posix(), "followup_run": FOLLOWUP_RUN.as_posix()},
        "followup": followup.to_dict(orient="records"),
        "followup_paired": followup_paired.to_dict(orient="records"),
        "stage_benchmark": benchmark.to_dict(orient="records"),
        "diagnostics": diagnostics(),
        "subsets": subsets(),
        "eda": eda(),
        "statistics": statistics(),
        "timeseries": timeseries(),
        "modeling": modeling(),
        "maintenance": maintenance(),
        "degradation": read_json(OUT / "degradation/summary.json"),
        "example_anomaly": pd.read_csv(OUT / "degradation/tables/anomaly_events.csv").query("engine_id == 1").to_dict("records"),
        "noise_summary": pd.read_csv(OUT / "degradation/tables/noise_summary.csv").to_dict("records"),
        "hdfs": read_json(OUT / "hdfs/summary.json"),
        "benchmark_parity": read_json(OUT / "benchmarks/20260929/run_info.json")["parity"],
        "sensitivity": pd.read_csv(OUT / "degradation/tables/anomaly_sensitivity.csv").query("signal == 'ewma'").to_dict("records"),
        "alignment": alignment(),
        "pipeline": pipeline(),
        "improvement": improvement(),
        "transfer": transfer(),
    }
    target = ROOT / "slides" / "slide_data.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
