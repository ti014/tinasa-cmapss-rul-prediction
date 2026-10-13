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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slides/figures"

# Bảng màu của deck (build_presentation.py).
BLUE, GOLD, RED, TEAL = "#21409A", "#FCB525", "#B23A3A", "#157A7F"
INK, SLATE, LINE, GRAY = "#1B2733", "#5A6B7B", "#C9D2E3", "#9AA5B1"
DPI = 220

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": INK,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "axes.titlecolor": INK,
    "axes.unicode_minus": True,
})


def _vn(value: float, _pos: int | None = None) -> str:
    """Nhãn trục kiểu Việt Nam: phẩy thập phân, dấu trừ U+2212."""
    text = f"{value:g}"
    return text.replace("-", "−").replace(".", ",")


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LINE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(_vn))
    ax.yaxis.set_major_formatter(FuncFormatter(_vn))


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, facecolor="white")
    plt.close(fig)


def _train_raw() -> pd.DataFrame:
    columns = ["engine_id", "cycle", *(f"setting_{i}" for i in range(1, 4)), *(f"sensor_{i}" for i in range(1, 22))]
    return pd.read_csv(ROOT / "data/raw/train_FD001.txt", sep=r"\s+", header=None, names=columns)


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
                boxprops={"facecolor": "#C9D6F0", "edgecolor": BLUE}, medianprops={"color": RED, "linewidth": 1.6},
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
    left.axvspan(1, config["baseline_cycles"], color="#E3E7EE", label=f"Tham chiếu: {config['baseline_cycles']} cycle đầu")
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
    ax.axvspan(0, 30, color="#FFF4DA")
    ax.axhline(0, color=INK, linewidth=1.0)
    ax.scatter(final["y_true"], residual, s=22, color=BLUE, alpha=0.85)
    ax.set_xlim(0, float(final["y_true"].max()) + 8)
    ax.text(15, residual.max(), "Vùng cần\ncảnh báo\n(RUL ≤ 30)", ha="center", va="top", fontsize=9.5, color="#8A6100")
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
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
