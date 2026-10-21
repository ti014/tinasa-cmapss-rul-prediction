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
        "lifetime": {k: float(life[k]) for k in ("count", "min", "50%", "max", "mean")},
        # Số động cơ hỏng trước trung vị: chi phí của một lịch bảo trì chung cho cả đội.
        "n_fail_before_median": int(
            (pd.read_csv(OUT / "eda" / "tables" / "engine_lifetimes.csv")["last_cycle"] < life["50%"]).sum()),
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
        "n_engines": int(inf["n_pairs"].iloc[0]),
        "comparison_cycles": read_json(OUT / "degradation/summary.json")["anomaly_config"]["baseline_cycles"],
        "fdr_q": float(inf["fdr_q"].iloc[0]),
        "same_direction": inf.loc[
            inf["n_positive"].eq(inf["n_pairs"]) | inf["n_negative"].eq(inf["n_pairs"]),
            "sensor",
        ].tolist(),
    }


def modeling() -> dict:
    final = read_json(OUT / "modeling" / "tables" / "final_metrics.json")
    models = pd.read_csv(OUT / "modeling" / "tables" / "model_metrics.csv")
    evaluation = pd.read_parquet(OUT / "modeling" / "predictions" / "final_evaluation.parquet")
    residual = evaluation["y_pred"] - evaluation["y_true"]
    worst = evaluation.loc[evaluation["absolute_error"].idxmax()]
    warning = read_json(OUT / "modeling" / "tables" / "warning_summary.json")
    low_rul = evaluation.loc[evaluation["y_true"].le(warning["rul_threshold"])]
    high_rul = evaluation.loc[evaluation["y_true"].gt(60)]
    cv = models.set_index("model_name")
    info = read_json(OUT / "modeling/run_info.json")
    events = pd.read_csv(OUT / "modeling/tables/warning_events.csv")
    example_engine = read_json(OUT / "degradation/summary.json")["example_engine"]
    warning_event = events.loc[
        events["engine_id"].eq(example_engine) & events["model_name"].eq(info["selected_model"])
    ].iloc[0]
    return {
        "final": final["metrics"],
        "n_test_engines": int(evaluation["engine_id"].nunique()),
        "cv_splits": info["config"]["n_splits"],
        "n_train_engines": warning["n_engines"],
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
        "example_warning": {"engine_id": int(warning_event["engine_id"]), "first_warning_cycle": int(warning_event["first_warning_cycle"]),
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
        "example": schedule.loc[schedule["engine_id"].eq(90)].iloc[0].to_dict(),
        "status_check": pd.read_csv(OUT / "maintenance/tables/status_vs_actual.csv").astype(str).to_dict("split"),
        "watch_above_60": int(watch["y_true"].gt(60).sum()),
        "normal_min_actual_rul": float(normal["y_true"].min()),
        "missed": {"n_engines": int(len(missed)),
                   "rul_pred_min": float(missed["rul_pred"].min()), "rul_pred_max": float(missed["rul_pred"].max()),
                   "due_min": int(missed["cycles_to_maintenance"].min()), "due_max": int(missed["cycles_to_maintenance"].max())},
        "late": {"n_engines": int(len(late)), "min_actual_rul": float(late["y_true"].min()),
                 "delay_min": int(late["delay"].min()), "delay_max": int(late["delay"].max())},
        "missed_engines": missed[["engine_id", "rul_pred", "y_true"]].to_dict("records"),
        # Kiểm chứng trên test ngưỡng 40 đã chọn từ train OOF; không dùng test để chọn ngưỡng.
        "threshold_40_test": _threshold_check(evaluation, 40, threshold),
    }


def _threshold_check(evaluation: pd.DataFrame, flag_at: float, need_at: float) -> dict:
    """Số động cơ test bị báo, báo đúng, bắt được và báo sớm khi phát cảnh báo mức hai ở `flag_at`."""
    flagged = evaluation["y_pred"].le(flag_at)
    need = evaluation["y_true"].le(need_at)
    return {"flagged": int(flagged.sum()), "correct": int((flagged & need).sum()),
            "actual": int(need.sum()), "early": int((flagged & ~need).sum()),
            "early_true_rul": sorted(evaluation.loc[flagged & ~need, "y_true"].astype(int).tolist())}


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
        "n_sensors": len(feat["feature_contract"]["sensors"]),
        "n_statistics": len(feat["feature_contract"]["statistics"]),
        "benchmark": [
            {"scale": r["scale"], "rows": r["input_rows"], "seconds": r["duration_seconds"]} for r in cold
        ],
        "parity": [
            {"name": c["name"], "rows": c["rows"], "max_abs_diff": c["max_abs_diff"], "status": c["status"]}
            for c in parity["checks"]
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
    }


def benchmark_scaling() -> dict:
    """Đọc cả lượt thành công lẫn lượt dừng, giữ dữ liệu đang đo ở trạng thái chờ."""
    folder = OUT / "benchmarks/20261021"
    source, info = folder / "compute_only.csv", folder / "run_info.json"
    spark_run = folder / "spark_compute_only/measurements.csv"
    metadata = read_json(info) if info.exists() else {"status": "pending"}
    records = []
    summary = {"largest_common_scale": None, "pandas_over_spark_ratio": None,
               "first_pandas_stop_scale": None}
    if (folder / "measurements.csv").exists() and spark_run.exists():
        # So phần tính toán: Pandas lấy lượt đầy đủ trừ bước ghi; Spark lấy lượt --no-write,
        # vì trên Windows bước ghi của Spark phải gom về driver và hết heap từ 100×.
        pandas_rows = pd.read_csv(folder / "measurements.csv").query("engine == 'pandas'").copy()
        pandas_rows["pipeline_seconds"] -= pandas_rows["write_seconds"]
        frame = pd.concat([pandas_rows, pd.read_csv(spark_run)], ignore_index=True)
        frame.to_csv(source, index=False)
        required = {"engine", "scale", "status", "pipeline_seconds", "rolling_seconds",
                    "peak_process_tree_rss_bytes", "elapsed_before_stop_seconds"}
        if missing := required - set(frame.columns):
            raise ValueError(f"Benchmark thiếu cột: {sorted(missing)}")
        if not frame["status"].isin(("ok", "failed", "timeout")).all():
            raise ValueError("Benchmark có trạng thái không hợp lệ")
        frame = frame.replace([float("inf"), -float("inf")], float("nan"))
        records = frame.astype(object).where(frame.notna(), None).to_dict("records")
        ok = frame.loc[frame["status"].eq("ok") & frame["pipeline_seconds"].gt(0)]
        times = ok.groupby(["scale", "engine"])["pipeline_seconds"].median().unstack()
        if {"pandas", "spark"} <= set(times.columns):
            common = times[["pandas", "spark"]].dropna().sort_index()
            if not common.empty:
                scale = common.index[-1]
                summary["largest_common_scale"] = int(scale)
                summary["pandas_over_spark_ratio"] = float(
                    common.loc[scale, "pandas"] / common.loc[scale, "spark"]
                )
        stopped = frame.loc[frame["engine"].eq("pandas") &
                            frame["status"].isin(("failed", "timeout")), "scale"]
        if not stopped.empty:
            summary["first_pandas_stop_scale"] = int(stopped.min())
            if "spark" in times.columns and stopped.min() in times.index:
                summary["spark_seconds_at_stop"] = float(times.loc[stopped.min(), "spark"])
    return {"records": records, "metadata": metadata, "summary": summary,
            "metadata_pending": not info.exists(),
            "source": source.relative_to(ROOT).as_posix()}


SQL_EXAMPLE = """SELECT s.engine_id, s.n_cycles, t.cycle, t.rul
FROM engine_summary AS s
JOIN train_targets AS t
  ON t.split = s.split AND t.engine_id = s.engine_id
WHERE s.split = 'train' AND s.engine_id = 1
  AND t.cycle IN (1, 100, 192);"""


def data_platform() -> dict:
    """Số liệu phần dữ liệu và hạ tầng: thu thập, chất lượng, các lớp lưu trữ, HDFS, truy vấn."""
    import re
    import sqlite3

    import pyarrow.parquet as pq

    from cmapss.data.readers import read_observations

    manifest = read_json(OUT / "acquisition/manifest.json")
    obs = pd.read_parquet(OUT / "etl/silver/observations.parquet")
    steps = obs.groupby(["split", "engine_id"])["cycle"].agg(
        gaps=lambda c: int((c.diff().dropna() != 1).sum()), first="min")
    shapes = {}
    for name, rel in (("observations", "etl/silver/observations.parquet"),
                      ("train_targets", "etl/silver/train_targets.parquet"),
                      ("engine_summary", "etl/gold/engine_summary.parquet"),
                      ("rolling_features", "features/gold/rolling_features.parquet")):
        meta = pq.ParquetFile(OUT / rel).metadata
        shapes[name] = [meta.num_rows, meta.num_columns]
    raw_columns = ["engine_id", "cycle", "setting_1", "setting_2", "sensor_2", "sensor_4", "sensor_11", "sensor_21"]
    raw = read_observations(ROOT / "data/raw/train_FD001.txt", split="train")
    with sqlite3.connect(OUT / "etl/gold/etl.sqlite") as connection:
        sql_rows = connection.execute(SQL_EXAMPLE).fetchall()
    engine1 = obs.query("split == 'train' and engine_id == 1").set_index("cycle")["sensor_11"]
    job = (OUT / "hdfs/mapreduce_job.txt").read_text(encoding="utf-8")
    fsck = (OUT / "hdfs/hdfs_fsck.txt").read_text(encoding="utf-8")
    counter = lambda label: int(re.search(rf"{label}=(\d+)", job).group(1))  # noqa: E731
    listing = [line.split()[-1] for line in (OUT / "hdfs/hdfs_listing.txt").read_text(encoding="utf-8").splitlines()]
    return {
        "archive_bytes": manifest["archive_bytes"],
        "n_files": len(manifest["files"]),
        "retrieved_at_utc": manifest["retrieved_at_utc"],
        "source_url": manifest["source_url"],
        "quality": {
            "rows": len(obs),
            "missing_values": int(obs.isna().sum().sum()),
            "duplicate_keys": int(obs.duplicated(["split", "engine_id", "cycle"]).sum()),
            "cycle_gaps": int(steps["gaps"].sum()),
            "engines_not_starting_at_1": int(steps["first"].ne(1).sum()),
        },
        "shapes": shapes,
        "raw_sample": {"columns": raw_columns, "rows": raw[raw_columns].head(4).values.tolist(),
                       "n_columns": raw.shape[1] - 1},
        "sql": {"query": SQL_EXAMPLE, "rows": [list(row) for row in sql_rows]},
        "rolling_mean_sensor_11": {
            str(cycle): float(engine1.rolling(30).mean().loc[cycle]) for cycle in (30, 100)},
        "mapreduce": {"map_tasks": counter("Launched map tasks"), "reduce_tasks": counter("Launched reduce tasks"),
                      "input_records": counter("Map input records"),
                      "output_records": counter("Reduce output records")},
        "fsck": {"status": re.search(r"Status: (\w+)", fsck).group(1),
                 "blocks": int(re.search(r"Total blocks \(validated\):\s*(\d+)", fsck).group(1))},
        "hdfs_listing": [path for path in listing if path.startswith("/cmapss/") and "." in path.rsplit("/", 1)[-1]],
    }


def main() -> None:
    benchmark = pd.read_csv(OUT / "benchmarks/20260929/measurements.csv")
    benchmark = benchmark.groupby(["engine", "scale"], as_index=False).median(numeric_only=True)
    # Pandas không có JVM heap; bỏ trường không dùng để JSON không chứa NaN.
    benchmark = benchmark.drop(columns=["spark_driver_heap_bytes"], errors="ignore")
    data = {
        "stage_benchmark": benchmark.to_dict(orient="records"),
        "benchmark_scaling": benchmark_scaling(),
        "diagnostics": diagnostics(),
        "subsets": subsets(),
        "eda": eda(),
        "statistics": statistics(),
        "modeling": modeling(),
        "maintenance": maintenance(),
        "degradation": read_json(OUT / "degradation/summary.json"),
        "example_anomaly": pd.read_csv(OUT / "degradation/tables/anomaly_events.csv").loc[
            lambda frame: frame["engine_id"].eq(
                read_json(OUT / "degradation/summary.json")["example_engine"]
            )
        ].to_dict("records"),
        "noise_summary": pd.read_csv(OUT / "degradation/tables/noise_summary.csv").to_dict("records"),
        "hdfs": read_json(OUT / "hdfs/summary.json"),
        "data_platform": data_platform(),
        "benchmark_parity": read_json(OUT / "benchmarks/20260929/run_info.json")["parity"],
        "sensitivity": pd.read_csv(OUT / "degradation/tables/anomaly_sensitivity.csv").query("signal == 'ewma'").to_dict("records"),
        "alignment": alignment(),
        "pipeline": pipeline(),
    }
    target = ROOT / "slides" / "slide_data.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
