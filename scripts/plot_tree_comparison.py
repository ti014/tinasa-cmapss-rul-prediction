"""Export outer metrics and paired confidence intervals from a completed study."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARMS = ("hgb_uncapped", "hgb_cap125", "xgb_uncapped", "xgb_cap125")
LABELS = ("HGB\nuncapped", "HGB\ncap125", "XGBoost\nuncapped", "XGBoost\ncap125")
COLORS = ("#465b70", "#809bb6", "#177e89", "#70b9b5")


def plot_readouts(folder: Path) -> None:
    """Write standalone PNG charts using outer readouts, without model selection.

    Args:
        folder: Completed study with aggregate and paired CSV readouts.
    """
    metrics = pd.read_csv(folder / "aggregate_metrics.csv", float_precision="round_trip").set_index("procedure")
    pairs = pd.read_csv(folder / "paired_comparison.csv", float_precision="round_trip")
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), constrained_layout=True)
    for axis, column, title in zip(
        axes, ("trajectory_rmse", "endpoint_mean_engine_mae"),
        ("RMSE toàn quỹ đạo", "MAE điểm cuối (50/70/90% tuổi thọ)"), strict=True,
    ):
        values = metrics.loc[list(ARMS), column].to_numpy()
        bars = axis.bar(LABELS, values, color=COLORS, width=0.68)
        axis.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
        axis.set_ylim(0, values.max() * 1.18)
        axis.set_ylabel("Chu kỳ")
        axis.set_title(title, fontsize=12)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    figure.suptitle("FD001 — outer grouped CV, 100 động cơ", fontsize=14)
    figure.supxlabel("Phan Nguyễn Mai Phương — MSHV 25002711", fontsize=9)
    figure.savefig(folder / "outer_metric_comparison.png", dpi=180)
    plt.close(figure)

    compared = ("hgb_cap125", "xgb_uncapped", "xgb_cap125", "selected_pipeline")
    labels = ("HGB cap125", "XGBoost uncapped", "XGBoost cap125", "Quy trình chọn bằng inner CV")
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), constrained_layout=True)
    for axis, column, title in zip(
        axes, ("trajectory_rmse", "endpoint_mean_engine_mae"),
        ("Δ RMSE toàn quỹ đạo", "Δ MAE điểm cuối"), strict=True,
    ):
        readout = pairs.loc[pairs.reference.eq("hgb_uncapped") & pairs.metric.eq(column)].set_index("procedure")
        readout = readout.loc[list(compared)]
        positions = np.arange(len(compared))
        axis.hlines(positions, readout.ci_lower, readout.ci_upper, color="#465b70", linewidth=2)
        axis.scatter(readout.delta, positions, color="#177e89", s=38, zorder=3)
        axis.axvline(0, color="#888888", linestyle="--", linewidth=1)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Chu kỳ; delta = candidate − HGB uncapped")
        axis.set_title(title, fontsize=12)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=0.18)
    draws = int(pairs.resamples.iloc[0])
    figure.suptitle(f"Paired engine bootstrap — 95% CI, {draws:,} mẫu", fontsize=14)
    figure.supxlabel("Phan Nguyễn Mai Phương — MSHV 25002711\n"
                    "Conditional trên predictions cố định; không gồm refitting variability", fontsize=9)
    figure.savefig(folder / "paired_confidence_intervals.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    plot_readouts(arguments.output_dir.resolve())
