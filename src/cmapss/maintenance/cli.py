"""CLI: lập báo cáo dự đoán thời điểm cần bảo trì cho 100 động cơ test FD001.

Ví dụ chạy từ thư mục gốc repository::

    uv run --frozen python -m cmapss.maintenance

Thứ tự đọc dữ liệu: train (fit Health Index) → test observations và dự đoán RUL
đã lưu → ghi lịch bảo trì → cuối cùng mới đọc ``RUL_FD001.txt`` để đánh giá.
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

from cmapss.data.readers import read_observations, read_test_labels
from cmapss.degradation.anomaly import AnomalyConfig
from cmapss.degradation.cli import EWMA_SPAN
from cmapss.degradation.denoise import INFORMATIVE_SENSORS, ewma_smooth
from cmapss.maintenance.report import (
    NEEDS_MAINTENANCE,
    NORMAL,
    STATUS_ORDER,
    WATCH,
    MaintenanceConfig,
    anomaly_status,
    build_schedule,
    evaluate_schedule,
    transform_health_index,
)
from cmapss.timeseries.health_index import fit_health_index

STATUS_COLORS = {NEEDS_MAINTENANCE: "#eb6834", WATCH: "#e0a21b", NORMAL: "#2a78d6"}
INK_2, GRID = "#52514e", "#e8e7e3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Báo cáo dự đoán thời điểm cần bảo trì FD001")
    parser.add_argument("--train", default="data/raw/train_FD001.txt")
    parser.add_argument("--test", default="data/raw/test_FD001.txt")
    parser.add_argument("--labels", default="data/raw/RUL_FD001.txt")
    parser.add_argument("--predictions", default="outputs/modeling/predictions/final_predictions.parquet")
    parser.add_argument("--output-dir", default="outputs/maintenance")
    return parser


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=8)


def _plot_fleet(schedule: pd.DataFrame, threshold: int, path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.2), constrained_layout=True,
                                      gridspec_kw={"width_ratios": [2.4, 1]})
    ordered = schedule.sort_values("rul_pred", kind="mergesort").reset_index(drop=True)
    left.bar(range(len(ordered)), ordered["rul_pred"], width=0.85,
             color=[STATUS_COLORS[s] for s in ordered["status"]])
    left.axhline(threshold, color="black", linestyle="--", linewidth=0.9)
    left.text(0, threshold + 5, f"Ngưỡng bảo trì {threshold} cycle", ha="left", fontsize=8, color=INK_2)
    left.set_xlabel("100 động cơ test, sắp theo RUL dự đoán", fontsize=8)
    left.set_ylabel("RUL dự đoán (cycle)", fontsize=8)
    left.set_xticks([])
    left.grid(axis="y", color=GRID, linewidth=0.6)
    _style(left)
    counts = schedule["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    right.barh(list(STATUS_ORDER), counts.to_numpy(), color=[STATUS_COLORS[s] for s in STATUS_ORDER])
    for y, value in enumerate(counts.to_numpy()):
        right.text(value + 0.8, y, str(int(value)), va="center", fontsize=9, color="black")
    right.invert_yaxis()
    right.set_xlim(0, counts.max() * 1.25)
    right.set_xlabel("Số động cơ", fontsize=8)
    right.set_title("Trạng thái hiện tại", fontsize=9, color=INK_2)
    _style(right)
    fig.savefig(path, dpi=200)
    plt.close(fig)


PLANNING_BINS = ((0, 0, "Ngay bây giờ"), (1, 25, "1–25"), (26, 50, "26–50"), (51, 100, "51–100"),
                 (101, 10_000, "> 100"))


def planning_counts(schedule: pd.DataFrame) -> pd.DataFrame:
    """Đếm số động cơ đến hạn bảo trì theo khoảng cycle, tách theo trạng thái."""
    rows = []
    for low, high, label in PLANNING_BINS:
        window = schedule["cycles_to_maintenance"].between(low, high)
        rows.append({"window": label, **{s: int((window & schedule["status"].eq(s)).sum()) for s in STATUS_ORDER}})
    return pd.DataFrame(rows).set_index("window")


def _plot_planning(planning: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.2), constrained_layout=True)
    bottom = pd.Series(0, index=planning.index)
    for status in STATUS_ORDER:
        ax.bar(planning.index, planning[status], bottom=bottom, color=STATUS_COLORS[status], label=status, width=0.6)
        bottom = bottom + planning[status]
    for x, total in enumerate(bottom):
        ax.text(x, total + 0.8, str(int(total)), ha="center", fontsize=9)
    ax.set_xlabel("Số cycle còn lại đến lúc cần bảo trì (RUL dự đoán chạm 30)", fontsize=8)
    ax.set_ylabel("Số động cơ", fontsize=8)
    ax.set_ylim(0, bottom.max() * 1.2)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    _style(ax)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    config, anomaly_config = MaintenanceConfig(), AnomalyConfig()

    # 1. Health Index fit chỉ trên train đã khử nhiễu; test chỉ được transform.
    train = ewma_smooth(read_observations(args.train, split="train"), INFORMATIVE_SENSORS, span=EWMA_SPAN)
    fitted = fit_health_index(train)
    test = ewma_smooth(read_observations(args.test, split="test"), INFORMATIVE_SENSORS, span=EWMA_SPAN)
    test_health_index = transform_health_index(fitted, test)
    anomalies = anomaly_status(test_health_index, anomaly_config)
    test_health_index.to_parquet(out / "tables/test_health_index.parquet", index=False)
    anomalies.to_csv(out / "tables/test_anomaly_status.csv", index=False)

    # 2. Lịch bảo trì từ dự đoán RUL đã lưu; ghi ra file trước khi đọc nhãn.
    predictions = pd.read_parquet(args.predictions)
    schedule = build_schedule(predictions, anomalies, config)
    schedule_path = out / "tables/maintenance_schedule.csv"
    schedule.to_csv(schedule_path, index=False, float_format="%.4f")
    _plot_fleet(schedule, config.rul_threshold, out / "figures/fleet_status.png")
    planning = planning_counts(schedule)
    planning.to_csv(out / "tables/maintenance_planning.csv")
    _plot_planning(planning, out / "figures/maintenance_planning.png")

    # 3. Đánh giá muộn bằng nhãn test.
    crosstab, evaluation = evaluate_schedule(schedule, read_test_labels(args.labels), config)
    crosstab.to_csv(out / "tables/status_vs_actual.csv")

    counts = schedule["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    summary = {
        "author": "Phan Nguyễn Mai Phương — MSHV 25002711",
        "inputs": {name: {"path": Path(value).as_posix(), "sha256": _sha256(Path(value))}
                   for name, value in (("train", args.train), ("test", args.test),
                                       ("predictions", args.predictions), ("labels", args.labels))},
        "rul_threshold": config.rul_threshold,
        "ewma_span": EWMA_SPAN,
        "anomaly_config": {"baseline_cycles": anomaly_config.baseline_cycles,
                           "z_threshold": anomaly_config.z_threshold, "persistence": anomaly_config.persistence},
        "status_counts": {status: int(counts[status]) for status in STATUS_ORDER},
        "n_anomaly_not_evaluable": int((~schedule["anomaly_evaluable"]).sum()),
        "planning": {label: int(planning.loc[label].sum()) for label in planning.index},
        "schedule_sha256": _sha256(schedule_path),
        "labels_read_after_schedule_written": True,
        "evaluation": evaluation,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(schedule.head(15).to_string(index=False))
    print(planning)
    print(crosstab)
    return 0
