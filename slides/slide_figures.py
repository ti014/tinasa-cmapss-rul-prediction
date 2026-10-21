"""Vẽ hình benchmark quy mô dùng chung cho slide và báo cáo.

Học viên: Phan Nguyễn Mai Phương · MSHV 25002711.

Các hình khác trên slide là chính file hình do module và notebook sinh ra trong
``outputs/``; chỉ benchmark chưa có module vẽ hình nên được vẽ ở đây, từ
``outputs/benchmarks/20261021/compute_only.csv``. Số dùng dấu phẩy thập phân.

Chạy từ thư mục gốc repository, trước ``slides/build_presentation.py``:

    uv run --frozen python slides/slide_figures.py
"""

from __future__ import annotations

from cmapss.presentation import COLORS, apply_plot_style, style_figure

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

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
    benchmark_scaling()
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
