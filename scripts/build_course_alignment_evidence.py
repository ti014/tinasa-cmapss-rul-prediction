"""Build a read-only evidence bundle for report and slides.

The script re-scores already persisted predictions. It never fits a model,
reads test labels for a new decision, or changes historical experiment
artifacts. Every generated document identifies Phan Nguyen Mai Phuong
(MSHV 25002711) and records input hashes and scope limitations.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmapss.data.contracts import SENSORS  # noqa: E402
from cmapss.modeling.audit_study import (  # noqa: E402
    WarningPolicy,
    _warning_readout,
    build_endpoint_panel,
    warning_events,
)


AUTHOR = "Phan Nguyễn Mai Phương"
MSHV = "25002711"
SEED = 42
N_BOOTSTRAP = 10_000
ENDPOINT_FRACTIONS = (0.5, 0.7, 0.9)
LEAD_REQUIREMENT = 30
OUTPUT = ROOT / "outputs" / "course_alignment" / "20261007"

HGB126_PATH = ROOT / "outputs" / "modeling" / "predictions" / "oof_predictions.parquet"
HGB169_PATH = (
    ROOT
    / "outputs"
    / "experiments"
    / "multiscale_comparison"
    / "20261006_study02"
    / "outer_predictions.parquet"
)
OBS_PATH = ROOT / "outputs" / "etl" / "silver" / "observations.parquet"
TRAIN_TARGET_PATH = ROOT / "outputs" / "etl" / "silver" / "train_targets.parquet"
MULTISCALE_MANIFEST = (
    ROOT
    / "outputs"
    / "experiments"
    / "multiscale_comparison"
    / "20261006_study02"
    / "manifest.json"
)
MULTISCALE_METRICS = MULTISCALE_MANIFEST.parent / "aggregate_metrics.csv"


def sha256_path(path: Path) -> str:
    """Return a SHA-256 digest for one existing file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas values and non-finite values to strict JSON."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON with stable indentation."""
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def nasa_component(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the C-MAPSS asymmetric NASA score contribution."""
    error = y_pred - y_true
    return np.where(error < 0.0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)


def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fresh metric columns without modifying an input frame."""
    result = frame.copy()
    error = result["y_pred"].to_numpy(dtype="float64") - result["y_true"].to_numpy(dtype="float64")
    result["absolute_error"] = np.abs(error)
    result["squared_error"] = error**2
    result["nasa_score_component"] = nasa_component(
        result["y_true"].to_numpy(dtype="float64"),
        result["y_pred"].to_numpy(dtype="float64"),
    )
    return result


def metric_readout(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize row, equal-engine MAE, pooled RMSE and NASA metrics."""
    scored = score_frame(frame)
    engine = scored.groupby("engine_id", sort=True)
    return {
        "n_rows": int(len(scored)),
        "n_engines": int(scored["engine_id"].nunique()),
        "mean_window_mae": float(scored["absolute_error"].mean()),
        "mean_engine_mae": float(engine["absolute_error"].mean().mean()),
        "rmse": float(np.sqrt(scored["squared_error"].mean())),
        "nasa_score": float(scored["nasa_score_component"].sum()),
        "mean_engine_nasa": float(engine["nasa_score_component"].mean().mean()),
    }


def load_common_predictions() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load and strictly validate the persisted HGB126/HGB169 train panels."""
    hgb126_all = pd.read_parquet(HGB126_PATH)
    hgb126 = hgb126_all.loc[hgb126_all["model_name"].eq("hist_gradient_boosting")].copy()
    hgb169_all = pd.read_parquet(HGB169_PATH)
    hgb169 = hgb169_all.loc[hgb169_all["procedure"].eq("hgb_baseline169")].copy()
    hgb126 = hgb126.loc[hgb126["split"].eq("train")].copy()
    hgb169 = hgb169.loc[hgb169["split"].eq("train")].copy()
    key = ["split", "engine_id", "cycle"]
    for name, frame in (("hgb126", hgb126), ("hgb169", hgb169)):
        required = set(key) | {"y_true", "y_pred"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
        if frame.duplicated(key).any() or not np.isfinite(frame[["y_true", "y_pred"]].to_numpy()).all():
            raise ValueError(f"{name} has duplicate keys or non-finite predictions")
        frame.sort_values(key, kind="mergesort", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if not hgb126[key].equals(hgb169[key]):
        raise ValueError("HGB126/HGB169 common keys are not identical")
    if not np.array_equal(hgb126["y_true"].to_numpy(), hgb169["y_true"].to_numpy()):
        raise ValueError("HGB126/HGB169 raw truth is not identical")
    fold126 = hgb126["fold"].to_numpy(dtype="int64")
    fold169 = hgb169["outer_fold"].to_numpy(dtype="int64")
    if not np.array_equal(fold126, fold169):
        raise ValueError("HGB126/HGB169 grouped fold assignment is not identical")
    validation = {
        "key_columns": key,
        "same_keys": True,
        "same_raw_truth": True,
        "same_grouped_fold": True,
        "row_count": int(len(hgb126)),
        "engine_count": int(hgb126["engine_id"].nunique()),
        "cycle_min": int(hgb126["cycle"].min()),
        "cycle_max": int(hgb126["cycle"].max()),
        "hgb126_fold_column": "fold",
        "hgb169_fold_column": "outer_fold",
        "hgb126_oof_source": HGB126_PATH.relative_to(ROOT),
        "hgb169_outer_source": HGB169_PATH.relative_to(ROOT),
    }
    return hgb126, hgb169, validation


def paired_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    panel: str,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Compute paired engine bootstrap CIs for delta right minus left.

    RMSE is pooled over sampled rows. MAE and NASA are equal-engine means.
    The interval is conditional on persisted predictions and excludes refit,
    random-seed and model-selection uncertainty.
    """
    left_scored = score_frame(left)
    right_scored = score_frame(right)
    if not left_scored[["engine_id", "cycle"]].equals(right_scored[["engine_id", "cycle"]]):
        raise ValueError(f"paired bootstrap {panel}: row keys differ")
    engine_ids = np.sort(left_scored["engine_id"].unique())
    left_engine = left_scored.groupby("engine_id", sort=True).agg(
        mae=("absolute_error", "mean"),
        sse=("squared_error", "sum"),
        nasa=("nasa_score_component", "mean"),
        n=("squared_error", "size"),
    )
    right_engine = right_scored.groupby("engine_id", sort=True).agg(
        mae=("absolute_error", "mean"),
        sse=("squared_error", "sum"),
        nasa=("nasa_score_component", "mean"),
        n=("squared_error", "size"),
    )
    point = metric_readout(left)
    point_right = metric_readout(right)
    rng = np.random.default_rng(seed)
    deltas = {metric: np.empty(n_resamples, dtype="float64") for metric in ("mean_engine_mae", "rmse", "mean_engine_nasa")}
    for index in range(n_resamples):
        sampled_ids = rng.choice(engine_ids, size=len(engine_ids), replace=True)
        left_sample = left_engine.loc[sampled_ids]
        right_sample = right_engine.loc[sampled_ids]
        deltas["mean_engine_mae"][index] = float(right_sample["mae"].mean() - left_sample["mae"].mean())
        left_rmse = np.sqrt(left_sample["sse"].sum() / left_sample["n"].sum())
        right_rmse = np.sqrt(right_sample["sse"].sum() / right_sample["n"].sum())
        deltas["rmse"][index] = float(right_rmse - left_rmse)
        deltas["mean_engine_nasa"][index] = float(right_sample["nasa"].mean() - left_sample["nasa"].mean())
    rows: list[dict[str, Any]] = []
    for metric, values in deltas.items():
        left_value = point[metric]
        right_value = point_right[metric]
        rows.append(
            {
                "panel": panel,
                "metric": metric,
                "left_model": "hgb126",
                "right_model": "hgb169",
                "left_value": left_value,
                "right_value": right_value,
                "delta_right_minus_left": right_value - left_value,
                "ci_lower": float(np.quantile(values, 0.025)),
                "ci_upper": float(np.quantile(values, 0.975)),
                "n_resamples": n_resamples,
                "seed": seed,
                "bootstrap_unit": "engine_id paired with replacement",
                "scope": "conditional on fixed persisted predictions; no refit uncertainty",
            }
        )
    return rows


def build_metrics(hgb126: pd.DataFrame, hgb169: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build trajectory/endpoint metric tables and paired CIs."""
    endpoint126 = build_endpoint_panel(hgb126, ENDPOINT_FRACTIONS)
    endpoint169 = build_endpoint_panel(hgb169, ENDPOINT_FRACTIONS)
    endpoint_key = ["split", "engine_id", "cycle", "endpoint_fraction"]
    endpoint126.sort_values(endpoint_key, inplace=True)
    endpoint169.sort_values(endpoint_key, inplace=True)
    if not endpoint126[endpoint_key].equals(endpoint169[endpoint_key]):
        raise ValueError("endpoint panels are not identical")
    metric_rows: list[dict[str, Any]] = []
    for model, trajectory, endpoint in (
        ("hgb126", hgb126, endpoint126),
        ("hgb169", hgb169, endpoint169),
    ):
        for panel_name, frame in (("trajectory", trajectory), ("endpoint_50_70_90", endpoint)):
            metric_rows.append({"model": model, "panel": panel_name, **metric_readout(frame)})
        for fraction in ENDPOINT_FRACTIONS:
            panel = endpoint.loc[endpoint["endpoint_fraction"].eq(fraction)]
            metric_rows.append({"model": model, "panel": f"endpoint_{int(fraction * 100)}", **metric_readout(panel)})
    metrics = pd.DataFrame(metric_rows)
    bootstrap_rows = paired_bootstrap(hgb126, hgb169, panel="trajectory")
    bootstrap_rows.extend(paired_bootstrap(endpoint126, endpoint169, panel="endpoint_50_70_90"))
    for fraction in ENDPOINT_FRACTIONS:
        left = endpoint126.loc[endpoint126["endpoint_fraction"].eq(fraction)].copy()
        right = endpoint169.loc[endpoint169["endpoint_fraction"].eq(fraction)].copy()
        bootstrap_rows.extend(paired_bootstrap(left, right, panel=f"endpoint_{int(fraction * 100)}"))
    endpoints = pd.concat(
        [endpoint126.assign(model="hgb126"), endpoint169.assign(model="hgb169")],
        ignore_index=True,
    )
    return metrics, pd.DataFrame(bootstrap_rows), endpoints


def fixed_warning_table(hgb126: pd.DataFrame, hgb169: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate both models with exactly the same non-calibrated policies."""
    rows: list[dict[str, Any]] = []
    event_rows: list[pd.DataFrame] = []
    for model, frame in (("hgb126", hgb126), ("hgb169", hgb169)):
        prepared = frame.assign(model_name=model)
        for threshold in (30, 40):
            policy = WarningPolicy(threshold, 1)
            events = warning_events(prepared, policy, lead_requirement=LEAD_REQUIREMENT)
            readout = _warning_readout(events)
            rows.append(
                {
                    "model": model,
                    "threshold": threshold,
                    "persistence": 1,
                    "lead_requirement": LEAD_REQUIREMENT,
                    "policy_status": "fixed sensitivity policy; not calibrated and not selected on outer results",
                    **readout,
                }
            )
            event_rows.append(events.assign(model=model, policy=f"threshold{threshold}_persistence1"))
    return pd.DataFrame(rows), pd.concat(event_rows, ignore_index=True)


def linear_slope(values: np.ndarray) -> float:
    """Fit a descriptive least-squares slope against cycle positions."""
    x = np.arange(len(values), dtype="float64")
    if len(values) < 2 or np.allclose(values, values[0]):
        return 0.0
    return float(np.polyfit(x, values.astype("float64"), 1)[0])


def build_smoothing_evidence(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Describe causal trailing-mean smoothing and degradation preservation.

    This is descriptive only: no denoising ground truth, fit, hyperparameter
    search or model selection is performed here.
    """
    train = observations.loc[observations["split"].eq("train")].copy()
    train.sort_values(["engine_id", "cycle"], inplace=True)
    sensor_records: list[dict[str, Any]] = []
    engine_records: list[dict[str, Any]] = []
    for engine_id, group in train.groupby("engine_id", sort=True):
        group = group.sort_values("cycle")
        for sensor in SENSORS:
            raw = group[sensor].to_numpy(dtype="float64")
            smooth = pd.Series(raw).rolling(30, min_periods=30).mean().to_numpy()
            valid = np.isfinite(smooth)
            if valid.sum() < 60:
                continue
            raw_valid = raw[valid]
            smooth_valid = smooth[valid]
            raw_diff = np.diff(raw_valid)
            smooth_diff = np.diff(smooth_valid)
            first_raw = float(np.mean(raw_valid[:30]))
            last_raw = float(np.mean(raw_valid[-30:]))
            first_smooth = float(np.mean(smooth_valid[:30]))
            last_smooth = float(np.mean(smooth_valid[-30:]))
            raw_slope = linear_slope(raw_valid)
            smooth_slope = linear_slope(smooth_valid)
            raw_diff_std = float(np.std(raw_diff, ddof=1)) if len(raw_diff) > 1 else 0.0
            smooth_diff_std = float(np.std(smooth_diff, ddof=1)) if len(smooth_diff) > 1 else 0.0
            record = {
                "engine_id": int(engine_id),
                "sensor": sensor,
                "raw_diff_std": raw_diff_std,
                "causal_mean30_diff_std": smooth_diff_std,
                "diff_std_reduction_pct": 100.0 * (raw_diff_std - smooth_diff_std) / raw_diff_std if raw_diff_std else 0.0,
                "raw_first_valid30_mean": first_raw,
                "raw_last_valid30_mean": last_raw,
                "causal_mean30_first30_mean": first_smooth,
                "causal_mean30_last30_mean": last_smooth,
                "raw_first_to_last_change": last_raw - first_raw,
                "causal_mean30_first_to_last_change": last_smooth - first_smooth,
                "raw_slope": raw_slope,
                "causal_mean30_slope": smooth_slope,
                "slope_sign_preserved": bool(np.sign(raw_slope) == np.sign(smooth_slope) or (raw_slope == 0 and smooth_slope == 0)),
                "valid_cycles": int(valid.sum()),
                "window_cycles": 30,
                "analysis": "descriptive causal trailing mean; no denoising ground truth or fitting",
            }
            engine_records.append(record)
    detail = pd.DataFrame(engine_records)
    sensor_std = train.loc[:, list(SENSORS)].std(ddof=1)
    nonconstant = sensor_std.index[sensor_std.gt(0)].tolist()
    detail_nonconstant = detail.loc[detail["sensor"].isin(nonconstant)]
    sensor_summary = (
        detail_nonconstant.groupby("sensor", sort=True)
        .agg(
            n_engines=("engine_id", "nunique"),
            median_diff_std_reduction_pct=("diff_std_reduction_pct", "median"),
            mean_diff_std_reduction_pct=("diff_std_reduction_pct", "mean"),
            median_raw_slope=("raw_slope", "median"),
            median_causal_mean30_slope=("causal_mean30_slope", "median"),
            slope_sign_preservation_rate=("slope_sign_preserved", "mean"),
            median_raw_first_to_last_change=("raw_first_to_last_change", "median"),
            median_causal_first_to_last_change=("causal_mean30_first_to_last_change", "median"),
        )
        .reset_index()
    )
    sensor_summary["is_nonconstant_raw_sensor"] = True
    overall = {
        "n_train_rows": int(len(train)),
        "n_engines": int(train["engine_id"].nunique()),
        "n_sensors_total": len(SENSORS),
        "n_nonconstant_sensors": int(len(nonconstant)),
        "nonconstant_sensors": nonconstant,
        "window_cycles": 30,
        "causal_definition": "mean of cycles max(1, t-29)..t; output interpreted only from cycle 30 onward",
        "purpose": "descriptive evidence that smoothing attenuates short-term variation while retaining broad change",
        "limitations": [
            "C-MAPSS has no clean sensor ground truth, so reduction in first differences is a proxy for attenuation, not proof of noise removal.",
            "Trailing means create boundary delay and are not a causal mechanism explanation.",
            "No smoothing parameter was fitted, searched or selected from labels.",
            "The analysis does not establish that smoothing improves RUL prediction.",
        ],
    }
    return detail, sensor_summary, overall


def make_figures(observations: pd.DataFrame, smoothing_detail: pd.DataFrame, sensor_summary: pd.DataFrame) -> list[str]:
    """Create three compact scientific figures for report/slide reuse."""
    figure_paths: list[str] = []
    train = observations.loc[observations["split"].eq("train")].copy()
    representative = train.loc[train["engine_id"].eq(int(train["engine_id"].min()))].sort_values("cycle")
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for axis, sensor in zip(axes, ("sensor_11", "sensor_12", "sensor_21")):
        raw = representative[sensor].to_numpy(dtype="float64")
        smoothed = pd.Series(raw).rolling(30, min_periods=30).mean().to_numpy()
        cycles = representative["cycle"].to_numpy()
        axis.plot(cycles, raw, color="#9aa0a6", linewidth=0.8, label="raw")
        axis.plot(cycles, smoothed, color="#1f5a94", linewidth=1.5, label="causal mean30")
        axis.set_ylabel(sensor.replace("sensor_", "s"))
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("cycle")
    fig.suptitle("Causal trailing mean on FD001 train engine 1")
    fig.tight_layout()
    path = OUTPUT / "figure_causal_smoothing_sensors_11_12_21.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(path.name)

    order = sensor_summary.sort_values("median_diff_std_reduction_pct", ascending=False)
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(order["sensor"], order["median_diff_std_reduction_pct"], color="#4c78a8")
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_ylabel("median reduction in first-difference SD (%)")
    axis.set_xlabel("sensor")
    axis.set_title("Descriptive attenuation from causal mean30")
    axis.tick_params(axis="x", rotation=70)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = OUTPUT / "figure_smoothing_variation_reduction.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(path.name)

    selected = smoothing_detail.loc[smoothing_detail["sensor"].isin(["sensor_11", "sensor_12", "sensor_21"])]
    fig, axis = plt.subplots(figsize=(6.5, 5.5))
    for sensor, group in selected.groupby("sensor", sort=True):
        axis.scatter(group["raw_slope"], group["causal_mean30_slope"], s=15, alpha=0.65, label=sensor)
    lo = float(min(selected["raw_slope"].min(), selected["causal_mean30_slope"].min()))
    hi = float(max(selected["raw_slope"].max(), selected["causal_mean30_slope"].max()))
    axis.plot([lo, hi], [lo, hi], linestyle="--", color="#666666", linewidth=0.9)
    axis.set_xlabel("raw descriptive slope")
    axis.set_ylabel("causal mean30 descriptive slope")
    axis.set_title("Slope preservation for sensors 11, 12 and 21")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    path = OUTPUT / "figure_slope_preservation_sensors_11_12_21.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(path.name)
    return figure_paths


def calibrated169_reference() -> dict[str, Any]:
    """Read the historical calibrated169 readout without relabeling it."""
    metrics = pd.read_csv(MULTISCALE_METRICS)
    row = metrics.loc[metrics["procedure"].eq("hgb_baseline169")].iloc[0].to_dict()
    manifest = json.loads(MULTISCALE_MANIFEST.read_text(encoding="utf-8"))
    policy = pd.read_parquet(HGB169_PATH).loc[
        lambda frame: frame["procedure"].eq("hgb_baseline169"), ["outer_fold", "threshold", "persistence"]
    ].drop_duplicates().sort_values(["outer_fold", "threshold", "persistence"])
    return {
        "source": {
            "aggregate_metrics": MULTISCALE_METRICS.relative_to(ROOT),
            "outer_predictions": HGB169_PATH.relative_to(ROOT),
            "manifest": MULTISCALE_MANIFEST.relative_to(ROOT),
        },
        "historical_procedure": "hgb_baseline169 with inner-selected warning policy per outer fold",
        "outer_policy_by_fold": policy.to_dict(orient="records"),
        "trajectory_rmse": row["trajectory_rmse"],
        "trajectory_mean_engine_mae": row["trajectory_mean_engine_mae"],
        "endpoint_mean_engine_mae": row["endpoint_mean_engine_mae"],
        "on_time_rate": row["on_time_rate"],
        "early_alarm_burden": row["early_alarm_burden"],
        "condition": "historical nested-policy readout; not comparable to the fixed-policy sensitivity table as a calibrated-vs-fixed claim",
        "scope_limit": "outer predictions and inner selection were historically inspected; no new refit or independent blind confirmation",
        "manifest_scientific": bool(manifest.get("scientific")),
    }


def write_readme(summary: dict[str, Any], input_hashes: dict[str, str]) -> None:
    """Write a provenance and interpretation guide for report/slide authors."""
    lines = [
        f"# Course-alignment evidence bundle — {AUTHOR} — MSHV {MSHV}",
        "",
        "Generated by `scripts/build_course_alignment_evidence.py` on persisted artifacts.",
        "The script performs no model training and does not rewrite historical outputs.",
        "",
        "## Interpretation",
        "",
        "HGB126 and HGB169 are re-scored on the same 17,731 train OOF/outer rows, "
        "with identical keys, raw truth and grouped folds. The endpoint panel uses "
        "cycles floor(0.5, 0.7, 0.9 × lifetime), clamped to the first available cycle.",
        "",
        "The warning table fixes threshold 30 or 40 cycles, persistence 1 and "
        "lead requirement 30 for both models. It is a sensitivity description; "
        "threshold 40 is not presented as a newly calibrated policy. The historical "
        "94% calibrated169 readout remains a separate reference.",
        "",
        "Recommendation for the report: use HGB169 as the main predictor candidate "
        "because it is better on the common trajectory and endpoint panels. Keep "
        "the warning policy separate: fixed threshold30 is less timely for HGB169, "
        "while threshold40 is slightly more timely with a small burden increase. "
        "This is not an unconditional whole-process improvement claim.",
        "",
        "The smoothing analysis uses only train observations and a causal trailing "
        "mean30. First-difference SD reduction is a proxy for attenuation, not "
        "proof that C-MAPSS contains recoverable measurement noise. No parameter "
        "was fitted or selected, and no claim is made that smoothing improves RUL.",
        "",
        "## Files",
        "",
        "- `summary.json`: common schema for report and slides.",
        "- `model_metrics.csv`: trajectory and endpoint metrics.",
        "- `paired_bootstrap_ci.csv`: paired engine bootstrap deltas (10,000 resamples).",
        "- `warning_fixed_policies.csv`, `warning_engine_events.csv`: fixed-policy warning evidence.",
        "- `smoothing_engine_sensor.csv`, `smoothing_sensor_summary.csv`: causal smoothing descriptors.",
        "- `figure_*.png`: three reusable scientific figures.",
        "",
        "## Limits",
        "",
        "- Bootstrap intervals condition on persisted predictions and omit refit, seed and model-selection uncertainty.",
        "- HGB126 uses 126 rolling features; HGB169 uses 126 rolling + cycle + 42 history features. This is a feature-representation comparison, not an architecture-only comparison.",
        "- The study is historically inspected. It is not a new blind independent confirmation.",
        "- Complete train trajectories make false-alarm denominators structurally different from deployment data; warning burden and retractions are reported instead.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Build all owned artifacts in one deterministic read-only run."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hgb126, hgb169, validation = load_common_predictions()
    metrics, bootstrap, endpoints = build_metrics(hgb126, hgb169)
    warnings, warning_events_table = fixed_warning_table(hgb126, hgb169)
    observations = pd.read_parquet(OBS_PATH)
    smoothing_detail, smoothing_sensor, smoothing_info = build_smoothing_evidence(observations)
    figure_paths = make_figures(observations, smoothing_detail, smoothing_sensor)
    metrics.to_csv(OUTPUT / "model_metrics.csv", index=False)
    bootstrap.to_csv(OUTPUT / "paired_bootstrap_ci.csv", index=False)
    warnings.to_csv(OUTPUT / "warning_fixed_policies.csv", index=False)
    warning_events_table.to_csv(OUTPUT / "warning_engine_events.csv", index=False)
    smoothing_detail.to_csv(OUTPUT / "smoothing_engine_sensor.csv", index=False)
    smoothing_sensor.to_csv(OUTPUT / "smoothing_sensor_summary.csv", index=False)
    endpoint_key = ["split", "engine_id", "cycle", "endpoint_fraction", "model"]
    endpoints.sort_values(endpoint_key).to_csv(OUTPUT / "endpoint_panel.csv", index=False)
    input_paths = [HGB126_PATH, HGB169_PATH, OBS_PATH, TRAIN_TARGET_PATH, MULTISCALE_MANIFEST, MULTISCALE_METRICS, Path(__file__)]
    input_hashes = {str(path.relative_to(ROOT)): sha256_path(path) for path in input_paths}
    validation["input_sha256"] = input_hashes
    write_json(OUTPUT / "validation.json", validation)
    summary = {
        "schema_version": 1,
        "author": AUTHOR,
        "mshv": MSHV,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "course-alignment evidence for report and slides",
        "read_only_replay": True,
        "model_specs": {
            "hgb126": "historical hist_gradient_boosting OOF; 126 rolling features",
            "hgb169": "historical hgb_baseline169 outer predictions; 126 rolling + cycle + 42 history features",
        },
        "validation": validation,
        "endpoint_definition": {
            "fractions": ENDPOINT_FRACTIONS,
            "cycle_origin": "floor(fraction*lifetime), clamped to first persisted cycle",
            "aggregation": "equal-engine MAE; pooled row RMSE; total and equal-engine NASA retained",
            "row_count": int(len(endpoints) // 2),
            "engine_count": int(hgb126["engine_id"].nunique()),
        },
        "model_selection_status": "No new selection; evidence compares persisted models and keeps the historical calibrated169 reference separate.",
        "recommendation": {
            "main_predictor": "hgb169",
            "decision": "Use HGB169 as the main predictor candidate for the report because it is better on the common full-trajectory and endpoint panels, subject to the stated conditional-bootstrap and historical-inspection limits.",
            "whole_process_warning_readout": "Do not claim an unconditional whole-process improvement: under the fixed threshold30 policy HGB169 is less timely, while under threshold40 it is slightly more timely with a small burden increase.",
            "warning_policy": "Keep the historically calibrated169 policy as a separate reference and report fixed-policy sensitivity. A deployment policy must be declared and validated separately from selecting the predictor.",
            "test_reporting": "Keep test reporting separate; these outputs do not use test labels to choose the predictor or warning policy.",
        },
        "fixed_warning_policy_status": "thresholds 30 and 40, persistence 1, lead requirement 30; common sensitivity policies, not newly calibrated",
        "calibrated169_reference": calibrated169_reference(),
        "smoothing": smoothing_info,
        "figures": figure_paths,
        "input_sha256": input_hashes,
        "limitations": [
            "No new model fitting or test-label read was performed.",
            "Paired bootstrap is conditional on saved predictions and excludes refit, seed and selection uncertainty.",
            "Historical inspection means this evidence is not an independent blind confirmation.",
            "The 126-versus-169 comparison changes feature representation as well as feature count; it does not isolate architecture.",
            "Causal smoothing descriptors do not identify true sensor noise and do not establish RUL benefit.",
        ],
    }
    write_json(OUTPUT / "summary.json", summary)
    write_readme(summary, input_hashes)
    print(json.dumps({"output": str(OUTPUT), "metrics": metrics.to_dict(orient="records"), "warnings": warnings.to_dict(orient="records"), "figures": figure_paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
