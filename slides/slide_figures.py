"""Vẽ lại các hình dùng trong deck theo định dạng trình chiếu tiếng Việt.

Học viên: Phan Nguyễn Mai Phương · MSHV 25002711.

Script chỉ đọc dữ liệu raw và artifact đã nghiệm thu trong ``outputs/``; không huấn
luyện lại mô hình và không ghi đè hình gốc của pipeline. Mỗi hình được vẽ đúng kích
thước (inch) của ô chứa nó trên slide, nên cỡ chữ trong hình là cỡ chữ khi chiếu.
Số dùng dấu phẩy thập phân và dấu trừ "−"; nhãn trục và chú giải bằng tiếng Việt.

Chạy từ thư mục gốc repository, trước ``slides/build_presentation.py``:

    uv run --frozen python slides/slide_figures.py
"""

from __future__ import annotations

from cmapss.presentation import COLORS, apply_plot_style, style_figure

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from cmapss.data.readers import read_observations
from cmapss.presentation import format_number_vi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slides/figures"

# Bảng màu của deck (build_presentation.py).
BLUE, GOLD, RED, TEAL = COLORS["blue"], COLORS["gold"], COLORS["red"], COLORS["teal"]
INK, SLATE, LINE, GRAY = COLORS["ink"], COLORS["muted"], COLORS["grid"], COLORS["gray"]
DPI = 220



apply_plot_style()

def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LINE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: format_number_vi(value, None)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: format_number_vi(value, None)))


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style_figure(fig)
    fig.savefig(OUT / name, dpi=DPI, facecolor="white")
    plt.close(fig)


def _train_raw() -> pd.DataFrame:
    return read_observations(ROOT / "data/raw/train_FD001.txt", split="train")


# ----------------------------------------------------------------------------- slide 8
def engine_lifetimes(train: pd.DataFrame) -> None:
    life = train.groupby("engine_id")["cycle"].max()
    fig, (hist, box) = plt.subplots(1, 2, figsize=(7.4, 2.55), constrained_layout=True,
                                    gridspec_kw={"width_ratios": [1.25, 1]})
    hist.hist(life, bins=range(120, 380, 10), color=BLUE, edgecolor="white")
    hist.set_xlabel("Tuổi thọ (cycle)")
    hist.set_ylabel("Số động cơ")
    hist.set_title("Phân bố tuổi thọ 100 động cơ train")
    _style(hist)
    box.boxplot(life, vert=False, widths=0.5, patch_artist=True,
                boxprops={"facecolor": COLORS["light"], "edgecolor": BLUE}, medianprops={"color": RED, "linewidth": 1.6},
                whiskerprops={"color": BLUE}, capprops={"color": BLUE}, flierprops={"markeredgecolor": SLATE})
    box.set_yticks([])
    box.set_xlabel("Tuổi thọ (cycle)")
    box.set_title("Trung vị 199; lệch phải")
    _style(box)
    box.grid(axis="x", color=LINE, linewidth=0.6)
    box.grid(axis="y", visible=False)
    _save(fig, "engine_lifetimes.png")


# ----------------------------------------------------------------------------- slide 9
def denoise(signals: pd.DataFrame) -> None:
    engine = signals.loc[signals["engine_id"].eq(1)]
    fig, axes = plt.subplots(1, 2, figsize=(7.85, 2.85), constrained_layout=True)
    for ax, sensor in zip(axes, ("sensor_11", "sensor_4")):
        ax.plot(engine["cycle"], engine[f"{sensor}__raw"], color=GRAY, linewidth=0.9, label="Tín hiệu gốc")
        ax.plot(engine["cycle"], engine[f"{sensor}__ewma"], color=BLUE, linewidth=1.8, label="Sau EWMA span 10")
        ax.set_title(f"{sensor}, động cơ train 1")
        ax.set_xlabel("cycle")
        _style(ax)
    axes[0].legend(frameon=False, loc="upper left")
    _save(fig, "denoise_before_after.png")


# ----------------------------------------------------------------------------- slide 11
def sensor_trends(train: pd.DataFrame) -> None:
    engines = [(87, TEAL), (26, GOLD), (75, BLUE)]
    fig, axes = plt.subplots(2, 1, figsize=(4.25, 4.0), constrained_layout=True, sharex=True)
    for ax, (sensor, trend) in zip(axes, (("sensor_4", "tăng dần"), ("sensor_7", "giảm dần"))):
        for engine, color in engines:
            rows = train.loc[train["engine_id"].eq(engine)]
            ax.plot(rows["cycle"], rows[sensor], color=color, linewidth=0.9,
                    label=f"Động cơ {engine} ({len(rows)} cycle)")
        ax.set_title(f"{sensor}: {trend} khi gần hỏng")
        ax.set_ylabel("Giá trị gốc")
        _style(ax)
    axes[1].set_xlabel("cycle")
    # Nới trục trên của hình đầu để chú giải không đè lên tín hiệu.
    low, high = axes[0].get_ylim()
    axes[0].set_ylim(low, high + (high - low) * 0.45)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    _save(fig, "sensor_trends.png")


# ----------------------------------------------------------------------------- slide 12
def anomaly(signals: pd.DataFrame) -> None:
    summary = json.loads((ROOT / "outputs/degradation/summary.json").read_text(encoding="utf-8"))
    config = summary["anomaly_config"]
    with (ROOT / "outputs/degradation/tables/anomaly_events.csv").open(encoding="utf-8") as file:
        events = pd.DataFrame(list(csv.DictReader(file)))
    events["lead_cycles"] = pd.to_numeric(events["lead_cycles"], errors="coerce")
    event = events.loc[events["engine_id"].eq("1")].iloc[0]
    threshold = float(event["baseline_mean"]) - config["z_threshold"] * float(event["baseline_std"])
    engine = signals.loc[signals["engine_id"].eq(1)]

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.15, 3.1), constrained_layout=True,
                                      gridspec_kw={"width_ratios": [1.45, 1]})
    left.axvspan(1, config["baseline_cycles"], color=COLORS["light"], label=f"Tham chiếu: {config['baseline_cycles']} cycle đầu")
    left.plot(engine["cycle"], engine["health_index_raw"], color=GRAY, linewidth=0.8, label="HI gốc")
    left.plot(engine["cycle"], engine["health_index_ewma"], color=BLUE, linewidth=1.8, label="HI đã làm mượt")
    left.axhline(threshold, color=INK, linestyle="--", linewidth=1.0, label=f"Ngưỡng {config['z_threshold']:g}σ")
    left.axvline(int(float(event["detection_cycle"])), color=GOLD, linewidth=2.0,
                 label=f"Phát hiện: cycle {int(float(event['detection_cycle']))}")
    left.axvline(int(float(event["failure_cycle"])), color=RED, linewidth=1.4,
                 label=f"Hỏng: cycle {int(float(event['failure_cycle']))}")
    left.set_title("Health Index (HI) của động cơ train 1")
    left.set_xlabel("cycle")
    left.set_ylabel("Health Index")
    left.legend(frameon=False, loc="lower left", fontsize=8.5)
    _style(left)

    leads = events["lead_cycles"].dropna()
    right.hist(leads, bins=range(0, int(leads.max()) + 11, 10), color=BLUE, edgecolor="white")
    right.axvline(float(leads.median()), color=GOLD, linewidth=2.0, label=f"Trung vị {leads.median():g} cycle")
    right.set_title("Báo trước bao nhiêu cycle (100 động cơ)")
    right.set_xlabel("cycle từ lúc phát hiện đến lúc hỏng")
    right.set_ylabel("Số động cơ")
    right.legend(frameon=False, loc="upper right")
    _style(right)
    _save(fig, "anomaly_health_index.png")


# ----------------------------------------------------------------------------- slide 13–14
def rul_figures() -> None:
    final = pd.read_parquet(ROOT / "outputs/modeling/predictions/final_evaluation.parquet")
    fig, ax = plt.subplots(figsize=(5.47, 4.1), constrained_layout=True)
    top = float(max(final["y_true"].max(), final["y_pred"].max())) + 10
    ax.plot([0, top], [0, top], color=INK, linewidth=1.0, label="Dự đoán đúng (đường chéo)")
    ax.scatter(final["y_true"], final["y_pred"], s=22, color=BLUE, alpha=0.85, label="Một động cơ test")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("RUL thực tế (cycle)")
    ax.set_ylabel("RUL dự đoán (cycle)")
    ax.set_title("RUL dự đoán so với thực tế, 100 động cơ test")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    _save(fig, "final_true_vs_predicted.png")

    residual = final["y_pred"] - final["y_true"]
    fig, ax = plt.subplots(figsize=(5.9, 4.1), constrained_layout=True)
    ax.axvspan(0, 30, color=COLORS["light"])
    ax.axhline(0, color=INK, linewidth=1.0)
    ax.scatter(final["y_true"], residual, s=22, color=BLUE, alpha=0.85)
    ax.set_xlim(0, float(final["y_true"].max()) + 8)
    ax.text(15, residual.max(), "Vùng cần\ncảnh báo\n(RUL ≤ 30)", ha="center", va="top", fontsize=9.5, color=COLORS["muted"])
    ax.set_xlabel("RUL thực tế (cycle)")
    ax.set_ylabel("Sai số = dự đoán − thực tế (cycle)")
    ax.set_title("Mỗi điểm là một động cơ test; trên 0 là dự đoán cao hơn thực tế")
    _style(ax)
    _save(fig, "final_residuals_by_rul.png")


# ----------------------------------------------------------------------------- slide 15
def warning_trajectory() -> None:
    oof = pd.read_parquet(ROOT / "outputs/modeling/predictions/oof_predictions.parquet")
    with (ROOT / "outputs/modeling/tables/warning_events.csv").open(encoding="utf-8") as file:
        event = next(row for row in csv.DictReader(file)
                     if row["engine_id"] == "1" and row["model_name"] == "hist_gradient_boosting")
    rows = oof.loc[oof["engine_id"].eq(1) & oof["model_name"].eq("hist_gradient_boosting")].sort_values("cycle")
    threshold, warning, failure = int(event["rul_threshold"]), int(event["first_warning_cycle"]), int(event["failure_cycle"])
    assert warning == 162 and failure == 192
    fig, ax = plt.subplots(figsize=(6.35, 3.6), constrained_layout=True)
    ax.plot(rows["cycle"], rows["y_true"], color=BLUE, linewidth=1.6, label="RUL thực tế")
    ax.plot(rows["cycle"], rows["y_pred"], color=RED, linewidth=1.3, label="RUL dự đoán (OOF)")
    ax.axhline(threshold, color=GOLD, linestyle="--", linewidth=1.6, label=f"Ngưỡng {threshold} cycle")
    ax.axvline(warning, color=TEAL, linestyle=":", linewidth=1.8, label=f"Cảnh báo mức 2: cycle {warning}")
    ax.axvline(failure, color=INK, linestyle="--", linewidth=1.0, label=f"Hỏng: cycle {failure}")
    ax.set_xlabel("cycle")
    ax.set_ylabel("RUL (cycle)")
    ax.set_title("Cảnh báo sớm trên động cơ train 1 (dự đoán OOF)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.25), fontsize=9)
    _style(ax)
    _save(fig, "warning_trajectory.png")


# ----------------------------------------------------------------------------- slide 16
def fleet_status() -> None:
    schedule = pd.read_csv(ROOT / "outputs/maintenance/tables/maintenance_schedule.csv")
    colors = {"Cần bảo trì": RED, "Theo dõi tăng cường": GOLD, "Bình thường": BLUE}
    ordered = schedule.sort_values("rul_pred").reset_index(drop=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.45, 2.15), constrained_layout=True,
                                      gridspec_kw={"width_ratios": [2.1, 1]})
    left.bar(ordered.index, ordered["rul_pred"], color=ordered["status"].map(colors), width=0.85)
    left.axhline(30, color=INK, linestyle="--", linewidth=1.0)
    left.set_xticks([])
    left.set_xlabel("100 động cơ test, xếp theo RUL dự đoán")
    left.set_ylabel("RUL dự đoán (cycle)")
    handles = [Patch(color=color, label=name) for name, color in colors.items()]
    handles.append(Line2D([0], [0], color=INK, linestyle="--", label="Ngưỡng 30 cycle"))
    left.legend(handles=handles, frameon=False, loc="upper left", fontsize=8.5)
    _style(left)
    counts = schedule["status"].value_counts().reindex(list(colors))
    right.barh(list(colors), counts.to_numpy(), color=list(colors.values()))
    for y, value in enumerate(counts.to_numpy()):
        right.text(value + 1, y, str(int(value)), va="center", fontsize=10, color=INK)
    right.set_xlim(0, counts.max() * 1.25)
    right.set_xlabel("Số động cơ")
    right.set_title("Trạng thái hiện tại")
    _style(right)
    # Trục danh mục: đặt lại nhãn chữ sau khi _style gán bộ định dạng số.
    right.set_yticks(range(len(colors)), list(colors), fontsize=9)
    right.invert_yaxis()
    right.grid(axis="y", visible=False)
    _save(fig, "fleet_status.png")


# ----------------------------------------------------------------------------- slide 11
def benchmark_scaling() -> None:
    """Vẽ thời gian và RAM của benchmark quy mô đang chạy."""
    source = ROOT / "outputs/benchmarks/20261021/compute_only.csv"  # do extract_slide_data.py ghép
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    required = {
        "engine", "scale", "status", "pipeline_seconds", "peak_process_tree_rss_bytes",
        "elapsed_before_stop_seconds",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Benchmark thiếu cột: {sorted(missing)}")
    if not frame["status"].isin(("ok", "failed", "timeout")).all():
        raise ValueError("Benchmark có trạng thái không hợp lệ")
    frame["scale"] = pd.to_numeric(frame["scale"], errors="raise")
    frame["pipeline_seconds"] = pd.to_numeric(frame["pipeline_seconds"], errors="coerce")
    frame["peak_process_tree_rss_bytes"] = pd.to_numeric(
        frame["peak_process_tree_rss_bytes"], errors="coerce"
    )
    frame["elapsed_before_stop_seconds"] = pd.to_numeric(
        frame["elapsed_before_stop_seconds"], errors="coerce"
    )
    metadata_path = source.with_name("run_info.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    ram_limit = float(metadata.get("ram_bytes", np.nan)) / 2**30

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    colors = {"pandas": BLUE, "spark": GOLD}
    labels = {"pandas": "Pandas", "spark": "Spark local[4]"}
    for engine in ("pandas", "spark"):
        rows = frame.loc[frame["engine"].eq(engine)].sort_values("scale")
        ok = rows.loc[rows["status"].eq("ok")]
        if not ok.empty:
            axes[0].plot(ok["scale"], ok["pipeline_seconds"], "o-", color=colors[engine],
                         linewidth=1.8, label=labels[engine])
            axes[1].plot(ok["scale"], ok["peak_process_tree_rss_bytes"] / 2**30, "o-",
                         color=colors[engine], linewidth=1.8, label=labels[engine])
        stopped = rows.loc[rows["status"].isin(("failed", "timeout"))]
        if not stopped.empty:
            # Khi worker dừng trước khi ghi số đo, đặt dấu x tại giới hạn máy để vẫn
            # cho thấy scale đã thử và không biến thất bại thành số đo thành công.
            x = stopped["scale"]
            y_time = stopped["elapsed_before_stop_seconds"]
            y_ram = (np.full(len(stopped), ram_limit) if np.isfinite(ram_limit)
                     else np.full(len(stopped), np.nan))
            axes[0].scatter(x, y_time, marker="x", s=55, linewidths=2, color=colors[engine])
            axes[1].scatter(x, y_ram, marker="x", s=55, linewidths=2, color=colors[engine])
            for _, row in stopped.iterrows():
                status = str(row["status"])
                axes[0].annotate(status, (row["scale"], row["elapsed_before_stop_seconds"]),
                                 xytext=(4, 4), textcoords="offset points", fontsize=8,
                                 color=colors[engine])
                if np.isfinite(ram_limit):
                    axes[1].annotate(status, (row["scale"], ram_limit), xytext=(4, 4),
                                     textcoords="offset points", fontsize=8, color=colors[engine])

    axes[0].set_title("Thời gian tính toán (không gồm ghi file)")
    axes[0].set_ylabel("Giây")
    axes[1].set_title("RAM đỉnh của cây process")
    axes[1].set_ylabel("GiB")
    for ax in axes:
        ax.set_xlabel("Quy mô nhân bản (×)")
        ax.set_xscale("log")
        _style(ax)
        ax.legend(frameon=False, fontsize=8,
                  loc="upper left" if ax is axes[0] else "center left")  # tránh dòng RAM máy
    if np.isfinite(ram_limit):
        axes[1].axhline(ram_limit, color=RED, linestyle="--", linewidth=1.0,
                        label=f"RAM máy: {format_number_vi(ram_limit, 1)} GiB")
        axes[1].text(0.02, 0.97, f"RAM máy {format_number_vi(ram_limit, 1)} GiB",
                     transform=axes[1].transAxes, va="top", fontsize=8, color=RED)
    _save(fig, "benchmark_scaling.png")


def main() -> None:
    train = _train_raw()
    signals = pd.read_parquet(ROOT / "outputs/degradation/tables/signals.parquet")
    engine_lifetimes(train)
    denoise(signals)
    sensor_trends(train)
    anomaly(signals)
    rul_figures()
    warning_trajectory()
    fleet_status()
    benchmark_scaling()
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
