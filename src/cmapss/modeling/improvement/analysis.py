"""Paired per-engine comparisons between the baseline and the improved RUL model."""

from cmapss.presentation import COLORS, apply_plot_style, style_figure

from pathlib import Path

import numpy as np
import pandas as pd

from cmapss.modeling.early_warning import build_warning_events


def engine_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-engine error and warning metrics."""
    events, _ = build_warning_events(predictions, model_name="candidate", rul_threshold=30)
    events = events.set_index("engine_id")
    rows = []
    for engine, group in predictions.groupby("engine_id", sort=True):
        low = group.loc[group.y_true.le(30)]
        high = group.loc[group.y_true.gt(30)]
        rows.append({
            "engine_id": int(engine),
            "mean_engine_mae": float((group.y_pred - group.y_true).abs().mean()),
            "low_rul_engine_mae": float((low.y_pred - low.y_true).abs().mean()),
            "late_or_missed_rate": float(events.loc[engine, "warning_timing"] in ("after_threshold", "at_failure", "no_warning")),
            "early_alarm_burden": float(high.y_pred.le(30).mean()) if len(high) else np.nan,
        })
    return pd.DataFrame(rows).set_index("engine_id")


def paired_comparison(baseline: pd.DataFrame, candidate: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    """Resample matched engine differences; negative values favor candidate."""
    if not baseline.index.equals(candidate.index):
        raise ValueError("paired comparison requires identical engine IDs")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(baseline), size=(resamples, len(baseline)))
    rows = []
    for metric in baseline.columns:
        values = (candidate[metric] - baseline[metric]).to_numpy()
        valid = np.isfinite(values)
        sampled = values[indices]
        draws = np.nanmean(sampled, axis=1)
        finite = draws[np.isfinite(draws)]
        rows.append({"metric": metric, "baseline": baseline[metric].mean(), "candidate": candidate[metric].mean(),
                     "delta": float(np.nanmean(values)) if valid.any() else None,
                     "ci_lower": float(np.quantile(finite, .025)) if len(finite) else None,
                     "ci_upper": float(np.quantile(finite, .975)) if len(finite) else None,
                     "n_engines": int(valid.sum()), "resamples": resamples, "seed": seed})
    return pd.DataFrame(rows)


def write_outer_comparison(predictions: pd.DataFrame, output: Path, resamples: int, seed: int) -> None:
    from cmapss.modeling.improvement.selection import _metrics
    from cmapss.modeling.improvement.artifact_io import _write_json
    summaries, engine_tables, comparisons = [], {}, []
    reference_keys = None
    for stage in ("baseline", "windows", "parameters", "weights"):
        frame = predictions.loc[predictions.stage.eq(stage)].sort_values(["engine_id", "cycle"])
        keys = frame.loc[:, ["engine_id", "cycle"]].reset_index(drop=True)
        if reference_keys is not None:
            pd.testing.assert_frame_equal(reference_keys, keys)
        reference_keys = keys
        metrics = _metrics(frame, 30)
        summaries.append({"stage": stage, "evaluation": "outer_train_trajectory_cycle_ge_30",
                          **{k: v for k, v in metrics.items() if k != "warning_summary"}})
        events, warning = build_warning_events(frame, model_name="candidate", rul_threshold=30)
        warning["evaluation_limit"] = "Nested outer evaluation; engine excluded from fitting and all inner configuration selection. Intervals condition on fitted predictions."
        events.to_csv(output / f"warning_events_{stage}.csv", index=False)
        _write_json(output / f"warning_summary_{stage}.json", warning)
        engine_tables[stage] = engine_metrics(frame)
        engine_tables[stage].to_csv(output / f"engine_metrics_{stage}.csv")
        if stage != "baseline":
            comparisons.append(paired_comparison(engine_tables["baseline"], engine_tables[stage], resamples, seed).assign(stage=stage))
    pd.DataFrame(summaries).to_csv(output / "ablation_metrics.csv", index=False)
    paired = pd.concat(comparisons, ignore_index=True)
    paired.to_csv(output / "paired_comparison.csv", index=False)
    final = paired.loc[paired.stage.eq("weights")].set_index("metric")
    controls = all(final.loc[key, "delta"] <= 0 for key in ("mean_engine_mae", "late_or_missed_rate", "early_alarm_burden"))
    improved = final.loc["low_rul_engine_mae", "delta"] < 0 and final.loc["late_or_missed_rate", "delta"] < 0 and controls
    _write_json(output / "decision.json", {
        "observed_improvement_meets_plan": bool(improved),
        "both_primary_intervals_below_zero": bool(all(final.loc[key, "ci_upper"] < 0 for key in ("low_rul_engine_mae", "late_or_missed_rate"))),
        "default_model": "retain_baseline_pending_integration" if improved else "retain_baseline",
        "interval_scope": "paired engine resampling conditional on stored outer predictions; excludes retraining variability",
    })
    _write_figures(predictions, engine_tables, output)


def _write_figures(predictions: pd.DataFrame, engine_tables: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    apply_plot_style()
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    delta = (engine_tables["weights"].low_rul_engine_mae - engine_tables["baseline"].low_rul_engine_mae).sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(delta)), delta, color=np.where(delta < 0, COLORS["teal"], COLORS["red"]))
    ax.axhline(0, color="black", lw=.7)
    ax.set(xlabel="Engine rank by error change", ylabel="Low-RUL MAE change (cycles)", title="Selected procedure minus baseline: nested outer predictions")
    fig.tight_layout(); fig.savefig(figures / "engine_error_delta.png", dpi=160); plt.close(fig)
    selected_ids = [int(delta.index[0]), int(delta.index[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, engine in zip(axes, selected_ids):
        for stage, label in (("baseline", "Baseline"), ("weights", "Selected procedure")):
            frame = predictions.loc[predictions.stage.eq(stage) & predictions.engine_id.eq(engine)].sort_values("cycle")
            ax.plot(frame.cycle, frame.y_pred, label=label)
        ax.plot(frame.cycle, frame.y_true, color="black", linestyle="--", label="True RUL")
        ax.axhline(30, color="gray", linestyle=":")
        ax.set(title=f"Engine {engine}: low-RUL MAE change {delta.loc[engine]:+.2f}", xlabel="Cycle", ylabel="RUL (cycles)")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figures / "trajectory_examples.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for stage, label in (("baseline", "Baseline"), ("weights", "Selected procedure")):
        events = pd.read_csv(output / f"warning_events_{stage}.csv")
        ax.hist(events.lead_time_to_failure_cycles.dropna(), bins=np.arange(0, 151, 5), alpha=.5, label=label)
    ax.axvline(30, color="black", linestyle="--", label="Research threshold")
    ax.set(xlabel="Cycles from first warning to endpoint", ylabel="Engine count")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "lead_time_comparison.png", dpi=160); plt.close(fig)


def write_test_comparison(evaluation: pd.DataFrame, output: Path, resamples: int, seed: int) -> None:
    """Compare test endpoint errors of the baseline snapshot and the selected model."""
    from cmapss.modeling.metrics import summarize_prediction_errors
    baseline = pd.read_parquet(output / "baseline_snapshot/predictions/final_evaluation.parquet")
    baseline = baseline.sort_values("engine_id").reset_index(drop=True)
    candidate = evaluation.sort_values("engine_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(baseline[["split", "engine_id", "cycle", "y_true"]], candidate[["split", "engine_id", "cycle", "y_true"]])
    rows, comparisons = [], []
    for stage, frame in (("baseline", baseline), ("selected_final", candidate)):
        low = frame.loc[frame.y_true.le(30)]
        rows.append({"model": stage, "evaluation": "test_final_observed_cycle", **summarize_prediction_errors(frame),
                     "low_rul_engine_mae": float((low.y_pred-low.y_true).abs().mean()), "n_low_rul_engines": int(len(low))})
    for group, mask in (("all", baseline.y_true.ge(0)), ("rul_le_30", baseline.y_true.le(30))):
        left = baseline.loc[mask].set_index("engine_id")[["absolute_error"]].rename(columns={"absolute_error": "mae"})
        right = candidate.loc[mask].set_index("engine_id")[["absolute_error"]].rename(columns={"absolute_error": "mae"})
        comparisons.append(paired_comparison(left, right, resamples, seed).assign(rul_group=group))
    pd.DataFrame(rows).to_csv(output / "test_comparison.csv", index=False)
    pd.concat(comparisons, ignore_index=True).to_csv(output / "test_paired_comparison.csv", index=False)
