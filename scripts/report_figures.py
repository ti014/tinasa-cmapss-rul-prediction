"""Render data-driven figures used in the report (Sections 4.2, 5.7 and 6).

Reads raw C-MAPSS files and existing artifacts only; trains nothing and never
modifies source artifacts. Writes PNG files to outputs/diagnostics/figures/.

Palette: categorical slots 1-3 of the dataviz reference palette (blue, orange,
aqua; validated all-pairs in light mode) plus a neutral gray for context. Aqua is
below 3:1 contrast on white, so every chart that uses it carries a legend.

Usage (from the repository root):
    .venv/Scripts/python.exe scripts/report_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/diagnostics/figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, LIGHT_GRAY = "#8a8985", "#d9d8d4"
INK, INK_2 = "#0b0b0b", "#52514e"
RAW_COLUMNS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": LIGHT_GRAY,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 10,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#ecebe8",
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def vn(value: float, digits: int = 2) -> str:
    """Format a number with Vietnamese decimal comma."""
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def read_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/raw" / name, sep=r"\s+", header=None, names=RAW_COLUMNS)


def window_example() -> Path:
    """Figure: how one feature row and its label are built for train engine 1."""
    obs = pd.read_parquet(ROOT / "outputs/etl/silver/observations.parquet")
    engine = obs.query("split == 'train' and engine_id == 1").sort_values("cycle")
    targets = pd.read_parquet(ROOT / "outputs/etl/silver/train_targets.parquet").query("engine_id == 1")
    feats = pd.read_parquet(ROOT / "outputs/features/gold/rolling_features.parquet")
    row = feats.query("split == 'train' and engine_id == 1 and cycle == 100").iloc[0]
    window = engine[engine["cycle"].between(71, 100)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax1.axvspan(70.5, 100.5, color=BLUE, alpha=0.10, lw=0)
    ax1.plot(engine["cycle"], engine["sensor_11"], color=GRAY, lw=1.2, label="sensor_11 của động cơ train 1")
    ax1.plot(window["cycle"], window["sensor_11"], color=BLUE, lw=2, label="30 cycles trong window (71–100)")
    ax1.hlines(row["sensor_11__mean"], 71, 100, color=INK, lw=1.2, ls="--", label="mean của window")
    k = window["cycle"].to_numpy()
    fit = row["sensor_11__mean"] + row["sensor_11__slope"] * (k - k.mean())
    ax1.plot(k, fit, color=ORANGE, lw=2, label="độ dốc (slope) của window")
    ax1.scatter([100], [row["sensor_11__last"]], s=36, color=BLUE, edgecolor="white", lw=1.5, zorder=5)
    ax1.axvline(100, color=INK_2, lw=0.8, ls=":")
    text = (
        f"Đặc trưng tại cycle 100\n"
        f"last  = {vn(row['sensor_11__last'])}\n"
        f"mean  = {vn(row['sensor_11__mean'], 3)}\n"
        f"std   = {vn(row['sensor_11__std'], 3)}\n"
        f"min   = {vn(row['sensor_11__min'])}\n"
        f"max   = {vn(row['sensor_11__max'])}\n"
        f"slope = {vn(row['sensor_11__slope'], 4)}"
    )
    ax1.text(0.02, 0.97, text, transform=ax1.transAxes, va="top", ha="left", family="DejaVu Sans Mono",
             fontsize=8, color=INK, bbox={"facecolor": "white", "edgecolor": LIGHT_GRAY, "boxstyle": "round,pad=0.4"})
    ax1.set(xlabel="cycle", ylabel="giá trị sensor_11 (đơn vị gốc)",
            title="(a) Một dòng đặc trưng = 6 thống kê trên 30 cycles gần nhất")
    ax1.legend(loc="lower right", fontsize=7.5)

    ax2.axvspan(0.5, 29.5, color=LIGHT_GRAY, alpha=0.5, lw=0)
    ax2.text(15, 20, "cycle 1–29:\nchưa đủ window,\nkhông có dòng\nđặc trưng", ha="center", fontsize=7.5, color=INK_2)
    ax2.plot(targets["cycle"], targets["rul"], color=BLUE, lw=2)
    ax2.scatter([100], [92], s=36, color=ORANGE, edgecolor="white", lw=1.5, zorder=5)
    ax2.annotate("cycle 100: RUL = 192 − 100 = 92", xy=(100, 92), xytext=(108, 140), fontsize=8, color=INK,
                 arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8})
    ax2.set(xlabel="cycle", ylabel="RUL (cycles)", title="(b) Nhãn RUL = cycle cuối − cycle hiện tại", ylim=(0, 200))
    fig.tight_layout()
    path = OUT / "window_example.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def oof_pred_vs_true() -> Path:
    """Figure: OOF prediction versus true RUL, with the mean prediction per bin."""
    oof = pd.read_parquet(ROOT / "outputs/modeling/predictions/oof_predictions.parquet")
    oof = oof[oof["model_name"] == "hist_gradient_boosting"]
    edges = np.arange(0, 370, 10)
    centers = (edges[:-1] + edges[1:]) / 2
    binned = oof.groupby(pd.cut(oof["y_true"], edges), observed=False)["y_pred"].mean().to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    hb = ax.hexbin(oof["y_true"], oof["y_pred"], gridsize=45, cmap="Blues", mincnt=1, bins="log", linewidths=0)
    ax.plot([0, 360], [0, 360], color=INK_2, lw=1, ls="--", label="dự đoán đúng (y = x)")
    ax.plot(centers, binned, color=ORANGE, lw=2, marker="o", ms=3.5, label="dự đoán trung bình theo khoảng RUL")
    ax.axhline(145, color=ORANGE, lw=0.8, ls=":")
    ax.annotate("dự đoán trung bình ≈ 145 cycles\ndù RUL thật lớn hơn nhiều", xy=(290, 136), xytext=(285, 205),
                fontsize=8, color=INK, ha="center", arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8})
    ax.set(xlabel="RUL thật (cycles)", ylabel="RUL dự đoán OOF (cycles)", xlim=(0, 362), ylim=(0, 260),
           title="Dự đoán OOF của HGB trên 17.731 window train")
    ax.grid(axis="both")
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("số window (thang log)", color=INK_2)
    ax.legend(loc="upper left", fontsize=8)
    path = OUT / "oof_pred_vs_true.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def warning_vs_lifetime() -> Path:
    """Figure: warning delay against engine lifetime."""
    events = pd.read_csv(ROOT / "outputs/modeling/tables/warning_events.csv")
    styles = [
        ("before_threshold", "trước ngưỡng", BLUE, "o"),
        ("at_threshold", "đúng ngưỡng", AQUA, "s"),
        ("after_threshold", "sau ngưỡng (trễ)", ORANGE, "^"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(0, color=INK_2, lw=0.8)
    for key, label, color, marker in styles:
        part = events[events["warning_timing"] == key]
        ax.scatter(part["failure_cycle"], part["threshold_delay_cycles"], s=28, color=color, marker=marker,
                   edgecolor="white", lw=0.8, label=f"{label} ({len(part)})", zorder=3)
    r = np.corrcoef(events["failure_cycle"], events["threshold_delay_cycles"])[0, 1]
    ax.text(0.98, 0.95, f"r = {vn(r)}", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
    ax.text(0.98, 0.05, "δ > 0: cảnh báo sau khi RUL thật đã dưới 30", transform=ax.transAxes, ha="right",
            fontsize=7.5, color=INK_2)
    ax.set(xlabel="tuổi thọ của động cơ (cycle hỏng)", ylabel="độ lệch δ so với ngưỡng (cycles)",
           title="Thời điểm cảnh báo đầu tiên theo tuổi thọ, 100 động cơ train")
    ax.grid(axis="both")
    ax.legend(loc="upper center", fontsize=8, ncol=3, bbox_to_anchor=(0.5, -0.18))
    path = OUT / "warning_vs_lifetime.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def transfer_shift() -> Path:
    """Figure: sensor_11 range shift and prediction spread across subsets."""
    ref = read_raw("train_FD001.txt")
    lo, hi = ref["sensor_11"].min(), ref["sensor_11"].max()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})

    bins = np.linspace(35.5, 49, 110)
    ax1.axvspan(lo, hi, color=BLUE, alpha=0.12, lw=0)
    ax1.hist(ref["sensor_11"], bins=bins, density=True, color=BLUE, alpha=0.85, label="FD001 train")
    for name, color in (("FD002", ORANGE), ("FD003", AQUA)):
        test = read_raw(f"test_{name}.txt")
        ax1.hist(test["sensor_11"], bins=bins, density=True, histtype="step", lw=1.6, color=color, label=f"{name} test")
    ax1.text((lo + hi) / 2, ax1.get_ylim()[1] * 0.92, "miền giá trị\ncủa FD001", ha="center", fontsize=7.5, color=INK)
    ax1.set(xlabel="giá trị sensor_11", ylabel="mật độ",
            title="(a) FD002 có nhiều cụm giá trị nằm ngoài miền FD001")
    ax1.legend(loc="upper left", fontsize=8)

    groups, colors = [], []
    for target in ("FD002", "FD003", "FD004"):
        internal = pd.read_parquet(ROOT / f"outputs/cross_dataset/evaluations/{target}_to_{target}/modeling/predictions/final_evaluation.parquet")
        transfer = pd.read_parquet(ROOT / f"outputs/cross_dataset/evaluations/FD001_to_{target}/modeling/predictions/final_evaluation.parquet")
        groups += [internal["y_true"], internal["y_pred"], transfer["y_pred"]]
        colors += [LIGHT_GRAY, BLUE, ORANGE]
    positions = [1, 2, 3, 5, 6, 7, 9, 10, 11]
    parts = ax2.boxplot(groups, positions=positions, widths=0.7, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": INK_2}, capprops={"color": INK_2})
    for box, color in zip(parts["boxes"], colors):
        box.set(facecolor=color, edgecolor="white", lw=1.5)
    ax2.set_xticks([2, 6, 10], ["test FD002", "test FD003", "test FD004"])
    ax2.set(ylabel="RUL (cycles)", title="(b) Mô hình FD001 dự đoán gần như hằng số trên FD002, FD004")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (LIGHT_GRAY, BLUE, ORANGE)]
    ax2.legend(handles, ["RUL thật", "dự đoán, huấn luyện nội bộ", "dự đoán của mô hình FD001"], fontsize=7.5,
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    path = OUT / "transfer_shift.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def cross_dataset_mae() -> Path:
    """Figure: internal versus transfer MAE with bootstrap confidence intervals."""
    import json

    matrix = json.loads((ROOT / "outputs/cross_dataset/matrix.json").read_text(encoding="utf-8"))
    rows = {}
    for item in matrix["evaluations"]:
        ci = item["bootstrap"]["intervals"]["mean_engine_mae"]
        rows[(item["source_dataset"], item["target_dataset"])] = (
            item["metrics"]["mean_engine_mae"], ci["ci_lower"], ci["ci_upper"])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    targets = ["FD001", "FD002", "FD003", "FD004"]
    for i, target in enumerate(targets):
        for key, offset, color, marker, label in (
            ((target, target), -0.12, BLUE, "o", "huấn luyện nội bộ trên chính subset"),
            (("FD001", target), 0.12, ORANGE, "s", "mô hình FD001 áp dụng trực tiếp"),
        ):
            if key not in rows or (key[0] == "FD001" and target == "FD001" and offset > 0):
                continue
            mae, lo, hi = rows[key]
            ax.errorbar(mae, i + offset, xerr=[[mae - lo], [hi - mae]], fmt=marker, color=color, ms=6,
                        mec="white", mew=1, elinewidth=1.6, capsize=0, label=label if i == 1 else None)
            ax.text(hi + 1.2, i + offset, vn(mae), va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(targets)), [f"test {t}" for t in targets])
    ax.invert_yaxis()
    ax.set(xlabel="MAE theo động cơ (cycles), khoảng tin cậy bootstrap 95%", xlim=(0, 75),
           title="Ma trận đánh giá với window 10 cycles")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    path = OUT / "cross_dataset_mae.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (window_example, oof_pred_vs_true, warning_vs_lifetime, transfer_shift, cross_dataset_mae):
        print("wrote", build().relative_to(ROOT))


if __name__ == "__main__":
    main()
