"""CLI: khử nhiễu EWMA, Health Index và phát hiện bất thường trên train FD001.

Ví dụ chạy từ thư mục gốc repository::

    uv run --frozen python -m cmapss.degradation

Lệnh chỉ đọc ``data/raw/train_FD001.txt``; không đọc test hoặc nhãn RUL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cmapss.data.readers import read_observations
from cmapss.degradation.anomaly import AnomalyConfig, detect_anomalies, sensitivity_grid
from cmapss.degradation.denoise import INFORMATIVE_SENSORS, ewma_smooth, noise_summary
from cmapss.timeseries.health_index import fit_health_index

EWMA_SPAN = 10
EXAMPLE_ENGINE = 1
EXAMPLE_SENSORS = ("sensor_11", "sensor_4")
BLUE, ORANGE, GRAY, INK_2 = "#2a78d6", "#eb6834", "#8a8985", "#52514e"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Khử nhiễu và phát hiện bất thường trên train FD001")
    parser.add_argument("--train", default="data/raw/train_FD001.txt", help="File train observations gốc")
    parser.add_argument("--output-dir", default="outputs/degradation", help="Thư mục ghi artifact")
    parser.add_argument("--span", type=int, default=EWMA_SPAN, help="Span EWMA tính bằng cycle")
    return parser


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.6)


def _plot_denoise(raw: pd.DataFrame, smoothed: pd.DataFrame, span: int, path: Path) -> None:
    fig, axes = plt.subplots(1, len(EXAMPLE_SENSORS), figsize=(8.4, 3.0), constrained_layout=True)
    raw_engine = raw.loc[raw["engine_id"].eq(EXAMPLE_ENGINE)]
    smooth_engine = smoothed.loc[smoothed["engine_id"].eq(EXAMPLE_ENGINE)]
    for ax, sensor in zip(axes, EXAMPLE_SENSORS):
        ax.plot(raw_engine["cycle"], raw_engine[sensor], color=GRAY, linewidth=0.8, label="Tín hiệu gốc")
        ax.plot(smooth_engine["cycle"], smooth_engine[sensor], color=BLUE, linewidth=1.6, label=f"EWMA span {span}")
        ax.set_title(sensor, fontsize=9, color=INK_2)
        ax.set_xlabel("Cycle", fontsize=8)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_anomaly(
    hi_raw: pd.DataFrame, hi_smooth: pd.DataFrame, events: pd.DataFrame, config: AnomalyConfig, path: Path
) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.1), constrained_layout=True,
                                      gridspec_kw={"width_ratios": [1.5, 1]})
    raw_engine = hi_raw.loc[hi_raw["engine_id"].eq(EXAMPLE_ENGINE)]
    smooth_engine = hi_smooth.loc[hi_smooth["engine_id"].eq(EXAMPLE_ENGINE)]
    event = events.loc[events["engine_id"].eq(EXAMPLE_ENGINE)].iloc[0]
    threshold = event["baseline_mean"] - config.z_threshold * event["baseline_std"]
    left.plot(raw_engine["cycle"], raw_engine["health_index"], color=GRAY, linewidth=0.8, label="HI từ tín hiệu gốc")
    left.plot(smooth_engine["cycle"], smooth_engine["health_index"], color=BLUE, linewidth=1.6,
              label="HI sau khử nhiễu")
    left.axvspan(1, config.baseline_cycles, color="#e8e7e3", alpha=0.7, label="Đoạn tham chiếu")
    left.axhline(threshold, color=INK_2, linestyle="--", linewidth=0.9, label=f"Ngưỡng {config.z_threshold:g}σ")
    if bool(event["detected"]):
        left.axvline(int(event["detection_cycle"]), color=ORANGE, linewidth=1.4, label="Phát hiện bất thường")
    left.axvline(int(event["failure_cycle"]), color="black", linewidth=1.0, label="Hỏng")
    left.set_title(f"Động cơ train {EXAMPLE_ENGINE}", fontsize=9, color=INK_2)
    left.set_xlabel("Cycle", fontsize=8)
    left.set_ylabel("Health Index", fontsize=8)
    left.legend(frameon=False, fontsize=7, loc="lower left")
    _style(left)
    leads = events["lead_cycles"].dropna().astype(int)
    right.hist(leads, bins=range(0, int(leads.max()) + 11, 10), color=BLUE, edgecolor="white")
    right.axvline(float(leads.median()), color=ORANGE, linewidth=1.4, label=f"Trung vị {leads.median():g}")
    right.set_title("Số cycle từ phát hiện đến hỏng", fontsize=9, color=INK_2)
    right.set_xlabel("Cycle", fontsize=8)
    right.set_ylabel("Số động cơ", fontsize=8)
    right.legend(frameon=False, fontsize=8)
    _style(right)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _lead_stats(events: pd.DataFrame) -> dict[str, float | int]:
    leads = events["lead_cycles"].dropna().astype(float)
    fraction = (events["detection_cycle"].dropna().astype(float)
                / events.loc[events["detected"], "failure_cycle"].astype(float))
    return {
        "n_engines": int(len(events)),
        "n_detected": int(events["detected"].sum()),
        "lead_min": float(leads.min()),
        "lead_q1": float(leads.quantile(0.25)),
        "lead_median": float(leads.median()),
        "lead_q3": float(leads.quantile(0.75)),
        "lead_max": float(leads.max()),
        "n_lead_below_30": int((leads < 30).sum()),
        "detection_life_fraction_median": float(fraction.median()),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_path = Path(args.train)
    out = Path(args.output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    raw = read_observations(train_path, split="train")
    smoothed = ewma_smooth(raw, INFORMATIVE_SENSORS, span=args.span)
    noise = noise_summary(raw, smoothed, INFORMATIVE_SENSORS)

    config = AnomalyConfig()
    hi_raw = fit_health_index(raw).health_index
    hi_smooth = fit_health_index(smoothed).health_index
    events = detect_anomalies(hi_smooth, config, column="health_index")
    events_raw = detect_anomalies(hi_raw, config, column="health_index")

    noise.to_csv(out / "tables/noise_summary.csv", index=False)
    events.to_csv(out / "tables/anomaly_events.csv", index=False)
    # Độ nhạy của quy tắc: lưới z 2/3/4 x 3/5/10 cycle liên tiếp, trên HI gốc và HI sau EWMA.
    sensitivity = pd.concat(
        [sensitivity_grid(hi, column="health_index").assign(signal=name)
         for name, hi in (("ewma", hi_smooth), ("raw", hi_raw))],
        ignore_index=True,
    )
    sensitivity.to_csv(out / "tables/anomaly_sensitivity.csv", index=False)
    # Tín hiệu theo cycle cho dashboard: sensor gốc/EWMA và Health Index hai phiên bản.
    keys = ["engine_id", "cycle"]
    raw_sorted = raw.sort_values(keys, kind="mergesort").reset_index(drop=True)
    signals = raw_sorted.loc[:, keys].copy()
    for sensor in INFORMATIVE_SENSORS:
        signals[f"{sensor}__raw"] = raw_sorted[sensor].to_numpy()
        signals[f"{sensor}__ewma"] = smoothed[sensor].to_numpy()
    signals = signals.merge(hi_raw.loc[:, keys + ["health_index"]].rename(columns={"health_index": "health_index_raw"}),
                            on=keys, how="left", validate="one_to_one")
    signals = signals.merge(hi_smooth.loc[:, keys + ["health_index"]].rename(columns={"health_index": "health_index_ewma"}),
                            on=keys, how="left", validate="one_to_one")
    signals.to_parquet(out / "tables/signals.parquet", index=False)
    _plot_denoise(raw, smoothed, args.span, out / "figures/denoise_before_after.png")
    _plot_anomaly(hi_raw, hi_smooth, events, config, out / "figures/anomaly_health_index.png")

    summary = {
        "author": "Phan Nguyễn Mai Phương — MSHV 25002711",
        "input": {"path": train_path.as_posix(),
                  "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()},
        "ewma_span": args.span,
        "sensors": list(INFORMATIVE_SENSORS),
        "noise_reduction_pct_median": float(noise["reduction_pct"].median()),
        "anomaly_config": {"baseline_cycles": config.baseline_cycles,
                           "z_threshold": config.z_threshold, "persistence": config.persistence},
        "anomaly_smoothed": _lead_stats(events),
        "anomaly_raw_for_comparison": _lead_stats(events_raw),
        "sensitivity_table": "tables/anomaly_sensitivity.csv",
        "example_engine": EXAMPLE_ENGINE,
        "uses_rul_labels": False,
        "reads_test_files": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
