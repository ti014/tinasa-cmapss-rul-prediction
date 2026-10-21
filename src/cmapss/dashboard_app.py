"""Giao diện Streamlit đọc artifact qua cmapss.dashboard; app.py gọi main(repo_root)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from cmapss.dashboard import (
    DashboardData,
    DashboardDataError,
    DegradationData,
    ExplorationData,
    MaintenanceData,
    engine_oof_trajectory,
    engine_signal_view,
    engine_rul_view,
    format_metric,
    load_dashboard_data,
    load_exploration_data,
    load_degradation_data,
    load_maintenance_data,
    sensor_trajectory,
)
from cmapss.presentation import COLORS, FONT



@alt.theme.register("cmapss", enable=True)
def _figure_theme() -> dict:
    """Dùng chung font và bảng màu cho hình trên dashboard."""
    return {"config": {"font": FONT, "range": {"category": list(COLORS.values())[:5]}}}

ROOT = Path(__file__).resolve().parents[2]  # main(repo_root) thay đường dẫn này khi khởi động.
RUL_THRESHOLD = 30
DEFAULT_SENSORS = ["sensor_4", "sensor_11", "sensor_12"]

MODEL_LABELS = {
    "hist_gradient_boosting": "HistGradientBoosting",
    "ridge_alpha_1": "Ridge (alpha = 1)",
    "median_baseline": "Mốc trung vị",
}
TIMING_LABELS = {
    "before_threshold": "Trước ngưỡng",
    "at_threshold": "Đúng ngưỡng",
    "after_threshold": "Trễ (trước khi hỏng)",
    "at_failure": "Đúng lúc hỏng",
    "no_warning": "Không cảnh báo",
}
PALETTE = {"true": COLORS["ink"], "pred": COLORS["blue"], "alt": COLORS["orange"],
           "muted": COLORS["gray"], "alert": COLORS["red"]}
STATUS_ORDER = ["Cần bảo trì", "Theo dõi tăng cường", "Bình thường"]
STATUS_COLORS = [COLORS["orange"], COLORS["gold"], COLORS["blue"]]
SCHEDULE_LABELS = {
    "priority": "Ưu tiên",
    "engine_id": "Động cơ",
    "current_cycle": "Cycle hiện tại",
    "rul_pred": "RUL dự đoán",
    "cycles_to_maintenance": "Còn (cycle)",
    "maintenance_cycle": "Cycle cần bảo trì",
    "predicted_failure_cycle": "Cycle dự đoán hỏng",
    "anomaly_cycle": "Bất thường từ cycle",
    "status": "Trạng thái",
}

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1380px;}
[data-testid="stSidebar"] {border-right: 1px solid #e2e8f0;}
[data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e2e8f0; padding: 12px 16px; border-radius: 12px;}
[data-testid="stMetricLabel"] {color: #52606d;}
[data-testid="stMetricValue"] {color: #0f243e; font-size: 1.55rem;}
.hero {background: linear-gradient(115deg, #0f243e 0%, #1f7a8c 100%); border-radius: 18px; padding: 26px 32px; color: white; margin-bottom: 22px;}
.hero-kicker {font-size: 0.75rem; letter-spacing: 0.14em; text-transform: uppercase; color: #bfe7ea; font-weight: 700;}
.hero-title {font-size: 2.2rem; line-height: 1.1; font-weight: 750; margin-top: 6px;}
.hero-subtitle {font-size: 1.05rem; color: #e4f5f6; margin-top: 8px; max-width: 880px;}
.pill {display: inline-block; border: 1px solid rgba(255,255,255,0.35); border-radius: 999px; padding: 4px 10px; margin: 14px 8px 0 0; font-size: 0.82rem;}
.card {border-radius: 10px; padding: 14px 18px; margin: 6px 0 14px 0;}
.card-insight {background: #eef7f7; border-left: 4px solid #1f7a8c; color: #16202a;}
.card-limit {background: #fff6e9; border-left: 4px solid #f28e2b; color: #5f3b10;}
.step {background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 12px; height: 100%;}
.step b {color: #1f7a8c;}
</style>
"""


# ---------------------------------------------------------------------------
# Artifact bất biến trong một phiên, nên loader chỉ đọc một lần.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Đang đọc kết quả mô hình...")
def _dashboard_data(root: str) -> DashboardData:
    return load_dashboard_data(root)


@st.cache_data(show_spinner="Đang đọc kết quả khám phá dữ liệu và đặc trưng...")
def _exploration_data(root: str) -> ExplorationData:
    return load_exploration_data(root)


@st.cache_data(show_spinner="Đang đọc kết quả khử nhiễu và bất thường...")
def _degradation_data(root: str) -> DegradationData:
    return load_degradation_data(root)


@st.cache_data(show_spinner="Đang đọc lịch bảo trì...")
def _maintenance_data(root: str) -> MaintenanceData:
    return load_maintenance_data(root)


# ---------------------------------------------------------------------------
# Hàm hỗ trợ hiển thị
# ---------------------------------------------------------------------------


def _card(title: str, body: str, *, kind: str = "insight") -> None:
    st.markdown(f'<div class="card card-{kind}"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)


def _metric_row(items: list[tuple[str, str] | tuple[str, str, str]]) -> None:
    for column, item in zip(st.columns(len(items)), items):
        label, value, *help_text = item
        column.metric(label, value, help=help_text[0] if help_text else None)


def _figure(relative_path: str, caption: str) -> None:
    path = ROOT / relative_path
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(f"Không tìm thấy hình `{relative_path}`.")


def _engine_select(label: str, options: list[int], *, key: str, default: int | None = None) -> int:
    index = options.index(default) if default in options else 0
    return int(st.selectbox(label, options, index=index, key=key, format_func=lambda value: f"Động cơ {int(value):03d}"))


def _model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def _rul_trajectory_chart(frame: pd.DataFrame, series: dict[str, str], *, marker_cycle: float | None = None) -> alt.Chart:
    """Biểu đồ RUL theo cycle với ngưỡng cảnh báo 30 cycle."""
    long = frame[["cycle", *series]].rename(columns=series).melt("cycle", var_name="Đường", value_name="RUL")
    names = list(series.values())
    colors = [PALETTE["true"], PALETTE["pred"], PALETTE["alt"], PALETTE["muted"]][: len(names)]
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y("RUL:Q", title="RUL (cycle)"),
            color=alt.Color("Đường:N", scale=alt.Scale(domain=names, range=colors), legend=alt.Legend(orient="top")),
            strokeDash=alt.condition(alt.datum["Đường"] == names[0], alt.value([5, 3]), alt.value([1, 0])),
            tooltip=["cycle:Q", "Đường:N", alt.Tooltip("RUL:Q", format=".2f")],
        )
    )
    layers = [lines, alt.Chart(pd.DataFrame({"y": [RUL_THRESHOLD]})).mark_rule(color=PALETTE["muted"], strokeDash=[2, 2]).encode(y="y:Q")]
    if marker_cycle is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"x": [marker_cycle]})).mark_rule(color=PALETTE["alert"]).encode(x="x:Q")
        )
    return alt.layer(*layers).properties(height=340).interactive(bind_y=False)


# ---------------------------------------------------------------------------
# Các trang
# ---------------------------------------------------------------------------


def page_overview(data: DashboardData, degradation: DegradationData, maintenance: MaintenanceData) -> None:
    metrics = data.model_metrics.get("metrics", {})
    evaluation = maintenance.summary["evaluation"]
    anomaly = degradation.summary["anomaly_smoothed"]
    st.header("Dự đoán thời điểm cần bảo trì cho 100 động cơ test FD001")
    _metric_row(
        [
            ("Cần bảo trì ngay", f"{maintenance.summary['status_counts']['Cần bảo trì']} động cơ",
             "RUL dự đoán ≤ 30 cycle tại cycle quan sát cuối."),
            ("Báo cần bảo trì đúng",
             f"{evaluation['n_correct_needs_maintenance']}/{evaluation['n_flagged_needs_maintenance']}",
             "Kiểm tra bằng nhãn test sau khi lịch đã được ghi ra file."),
            ("Bất thường báo trước hỏng", f"{format_metric(anomaly['lead_median'], 0)} cycle",
             "Trung vị trên 100 động cơ train; Health Index giảm quá 3σ so với 30 cycle đầu."),
            ("MAE RUL trên test", f"{format_metric(metrics.get('mean_engine_mae', 0), 2)} cycle",
             "Sai số tuyệt đối trung bình tại cycle quan sát cuối của 100 động cơ test."),
        ]
    )

    st.subheader("Pipeline tạo ra các kết quả này")
    steps = [
        ("1. Thu thập", "File text train/test/RUL FD001, kiểm tra schema"),
        ("2. ETL", "Pandas = PySpark → Silver Parquet, Gold SQLite"),
        ("3. Khử nhiễu", "EWMA span 10, chỉ dùng quá khứ"),
        ("4. Window", "30 cycle → 126 đặc trưng"),
        ("5. Bất thường", "Health Index giảm quá 3σ: cảnh báo mức một"),
        ("6. RUL", "HGB, GroupKFold theo động cơ"),
        ("7. Bảo trì", "RUL ≤ 30: cảnh báo mức hai, lịch bảo trì"),
    ]
    for column, (title, body) in zip(st.columns(len(steps)), steps):
        column.markdown(f'<div class="step"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)

    st.write("")
    _card(
        "Đọc kết quả trong một câu",
        f"Tại điểm quan sát cuối, {maintenance.summary['status_counts']['Cần bảo trì']} động cơ cần bảo trì ngay và "
        f"{maintenance.summary['planning']['1–25']} động cơ đến hạn trong 25 cycle tới; mọi động cơ được báo cần bảo trì đều "
        "có RUL thật ≤ 30.",
    )
    with st.expander("Gợi ý thứ tự demo (khoảng 2 phút)"):
        st.markdown(
            "1. **Lịch bảo trì**: bảng 100 động cơ, chọn động cơ 90 để xem chi tiết.\n"
            "2. **Khử nhiễu**: động cơ 1, sensor 11: tín hiệu gốc và sau EWMA.\n"
            "3. **Xu hướng & bất thường**: Health Index của động cơ 1.\n"
            "4. **Cảnh báo sớm**: RUL dự đoán của động cơ 1 chạm ngưỡng 30.\n"
            "5. **Tổng quan**: quay lại trang này để kết demo."
        )


def page_exploration(exploration: ExplorationData) -> None:
    st.header("Dữ liệu train FD001 và các sensor liên quan tới RUL")
    lifetimes = exploration.engine_lifetimes
    _metric_row(
        [
            ("Động cơ train", f"{lifetimes['engine_id'].nunique():,}"),
            ("Tổng số cycle train", f"{int(lifetimes['n_observations'].sum()):,}"),
            ("Tuổi thọ ngắn nhất", f"{int(lifetimes['n_observations'].min())} cycle"),
            ("Tuổi thọ dài nhất", f"{int(lifetimes['n_observations'].max())} cycle"),
        ]
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Phân phối tuổi thọ động cơ")
        histogram = (
            alt.Chart(lifetimes)
            .mark_bar(color=PALETTE["pred"])
            .encode(
                x=alt.X("n_observations:Q", bin=alt.Bin(step=20), title="Tuổi thọ (cycle)"),
                y=alt.Y("count():Q", title="Số động cơ"),
                tooltip=[alt.Tooltip("count():Q", title="Số động cơ")],
            )
            .properties(height=280)
        )
        st.altair_chart(histogram, width="stretch")
    with right:
        st.subheader("Tương quan Spearman giữa sensor và RUL, tính trong từng động cơ")
        summary = exploration.sensor_rul_summary.loc[lambda frame: frame["defined_engines"] > 0]
        ranking = (
            alt.Chart(summary)
            .mark_bar(color=PALETTE["pred"])
            .encode(
                y=alt.Y("sensor:N", sort=alt.EncodingSortField("median", op="min"), title=None, axis=alt.Axis(labelOverlap=False)),
                x=alt.X("median:Q", title="Trung vị Spearman ρ qua các động cơ", scale=alt.Scale(domain=[-1, 1])),
                tooltip=["sensor:N", alt.Tooltip("median:Q", format=".3f"), alt.Tooltip("q25:Q", format=".3f"), alt.Tooltip("q75:Q", format=".3f")],
            )
            .properties(height=alt.Step(20))
        )
        st.altair_chart(ranking, width="stretch")
        excluded = exploration.sensor_rul_summary.loc[lambda frame: frame["defined_engines"] == 0, "sensor"]
        st.caption(f"Không hiển thị sensor hằng số ở mọi động cơ (ρ không xác định): {', '.join(excluded)}.")

    st.subheader("Quỹ đạo sensor của một động cơ train")
    trajectories = exploration.sensor_trajectories
    sensors = sorted({column.split("__")[0] for column in trajectories.columns if "__" in column}, key=lambda name: int(name.split("_")[1]))
    controls = st.columns([1, 2, 1])
    with controls[0]:
        engine_id = _engine_select("Động cơ", sorted(trajectories["engine_id"].unique()), key="eda_engine")
    with controls[1]:
        chosen = st.multiselect("Sensors", sensors, default=[name for name in DEFAULT_SENSORS if name in sensors])
    with controls[2]:
        stat = st.radio("Giá trị", ["mean", "last"], horizontal=True, format_func={"mean": "Rolling mean", "last": "Giá trị đo"}.get)
    if not chosen:
        st.info("Chọn ít nhất một sensor.")
    else:
        long = sensor_trajectory(exploration, engine_id, chosen, stat)
        failure = int(trajectories.loc[trajectories["engine_id"] == engine_id, "cycle"].max())
        chart = (
            alt.Chart(long)
            .mark_line()
            .encode(
                x=alt.X("cycle:Q", title="Cycle"),
                y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
                color=alt.Color("sensor:N", legend=None),
                tooltip=["sensor:N", "cycle:Q", alt.Tooltip("value:Q", format=".4f")],
            )
            .properties(height=150, width=900)
            .facet(row=alt.Row("sensor:N", title=None))
            .resolve_scale(y="independent")
        )
        st.altair_chart(chart, width="stretch")
        st.caption(
            f"Động cơ {engine_id:03d} hỏng ở cycle {failure}. Dữ liệu lấy từ bảng đặc trưng window (lớp Gold) nên bắt đầu ở "
            f"cycle {exploration.window_cycles}; giá trị trung bình tính trên {exploration.window_cycles} cycle gần nhất."
        )

    with st.expander("Hình khám phá dữ liệu đã lưu (outputs/eda/figures)"):
        tabs = st.tabs(["Quỹ đạo sensor", "Sensor–RUL theo engine", "Phân phối sensor", "Tương quan pooled"])
        with tabs[0]:
            for index in (1, 2, 3):
                _figure(f"outputs/eda/figures/sensor_trajectories_{index}.png", f"Quỹ đạo sensor {index}")
        with tabs[1]:
            _figure("outputs/eda/figures/sensor_rul_by_engine.png", "Tương quan Spearman sensor–RUL theo từng động cơ")
        with tabs[2]:
            for index in (1, 2, 3):
                _figure(f"outputs/eda/figures/sensor_distributions_{index}.png", f"Phân phối sensor {index}")
        with tabs[3]:
            _figure("outputs/eda/figures/sensor_spearman_pooled.png", "Tương quan Spearman giữa các sensor (gộp mọi động cơ)")


def page_rul(data: DashboardData) -> None:
    st.header("Dự đoán RUL")
    st.subheader("So sánh ba mô hình bằng cross-validation theo động cơ")
    left, right = st.columns([1.2, 1])
    comparison = data.model_comparison.assign(
        Model=lambda frame: frame["model_name"].map(_model_label),
        lower=lambda frame: frame["mean_engine_mae"] - frame["std_engine_mae"],
        upper=lambda frame: frame["mean_engine_mae"] + frame["std_engine_mae"],
    )
    order = comparison.sort_values("mean_engine_mae")["Model"].tolist()
    with left:
        bars = (
            alt.Chart(comparison)
            .mark_bar(size=34)
            .encode(
                y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("mean_engine_mae:Q", title="MAE trung bình theo động cơ (cycle), 5 fold"),
                color=alt.condition(alt.datum.selected, alt.value(PALETTE["pred"]), alt.value(PALETTE["muted"])),
                tooltip=["Model:N", alt.Tooltip("mean_engine_mae:Q", format=".4f"), alt.Tooltip("std_engine_mae:Q", format=".4f")],
            )
        )
        errors = alt.Chart(comparison).mark_errorbar(color=PALETTE["true"]).encode(
            y=alt.Y("Model:N", sort=order), x=alt.X("lower:Q", title="MAE trung bình theo động cơ (cycle), 5 fold"), x2="upper:Q"
        )
        st.altair_chart((bars + errors).properties(height=200), width="stretch")
        st.caption("Thanh lỗi: ±1 độ lệch chuẩn giữa 5 fold. Cột tô màu là mô hình được chọn.")
    with right:
        _card(
            "Mô hình được chọn",
            f"{_model_label(data.selected_model)} có sai số thấp nhất khi chia 5 fold theo động cơ (GroupKFold). "
            "Dự đoán test được ghi ra file trước, sau đó mới ghép đáp án để chấm.",
        )
        _card(
            "Cách đọc MAE",
            "MAE trên test chỉ tính tại cycle quan sát cuối của mỗi động cơ. MAE trong cross-validation tính trên mọi cycle "
            "của động cơ train từ cycle 30, gồm cả giai đoạn đầu đời khó đoán, nên hai con số không so trực tiếp.",
            kind="limit",
        )

    test_tab, oof_tab = st.tabs(["Động cơ test: điểm quan sát cuối", "Động cơ train: dự đoán theo từng cycle"])

    with test_tab:
        engine_id = _engine_select("Chọn động cơ test", sorted(data.model_evaluation["engine_id"].unique()), key="rul_test_engine")
        row = engine_rul_view(data, engine_id)
        if row.empty:
            st.warning("Không có dòng đánh giá cho động cơ đã chọn.")
            return
        record = row.iloc[0]
        _metric_row(
            [
                ("RUL thật", format_metric(record["y_true"], 0)),
                ("RUL dự đoán", format_metric(record["y_pred"])),
                ("Sai số tuyệt đối", format_metric(record["absolute_error"])),
                ("Cycle cuối", format_metric(record["cycle"], 0)),
            ]
        )
        direction = "cao hơn" if record["y_pred"] > record["y_true"] else "thấp hơn"
        _card(
            f"Động cơ {engine_id:03d}",
            f"Sai số tuyệt đối {format_metric(record['absolute_error'])} cycle; RUL dự đoán {direction} RUL thật. "
            "Mỗi động cơ test chỉ có một dòng đánh giá tại cycle cuối.",
        )
        evaluation = data.model_evaluation.assign(
            chosen=lambda frame: frame["engine_id"] == engine_id,
            residual=lambda frame: frame["y_pred"] - frame["y_true"],
        )
        upper = float(evaluation[["y_true", "y_pred"]].to_numpy().max()) * 1.05
        diagonal = alt.Chart(pd.DataFrame({"v": [0, upper]})).mark_line(color=PALETTE["true"], strokeDash=[4, 4]).encode(x="v:Q", y="v:Q")
        tooltip = ["engine_id:Q", alt.Tooltip("y_true:Q", format=".0f"), alt.Tooltip("y_pred:Q", format=".2f"), alt.Tooltip("absolute_error:Q", format=".2f")]
        points = (
            alt.Chart(evaluation)
            .mark_circle()
            .encode(
                x=alt.X("y_true:Q", title="RUL thật (cycle)", scale=alt.Scale(domain=[0, upper])),
                y=alt.Y("y_pred:Q", title="RUL dự đoán (cycle)", scale=alt.Scale(domain=[0, upper])),
                color=alt.condition(alt.datum.chosen, alt.value(PALETTE["alert"]), alt.value(PALETTE["pred"])),
                size=alt.condition(alt.datum.chosen, alt.value(160), alt.value(45)),
                opacity=alt.condition(alt.datum.chosen, alt.value(1.0), alt.value(0.55)),
                tooltip=tooltip,
            )
        )
        residuals = (
            alt.Chart(evaluation)
            .mark_circle()
            .encode(
                x=alt.X("y_true:Q", title="RUL thật (cycle)"),
                y=alt.Y("residual:Q", title="Dự đoán − thật (cycle)"),
                color=alt.condition(alt.datum.chosen, alt.value(PALETTE["alert"]), alt.value(PALETTE["pred"])),
                size=alt.condition(alt.datum.chosen, alt.value(160), alt.value(45)),
                tooltip=tooltip,
            )
        )
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=PALETTE["true"], strokeDash=[4, 4]).encode(y="y:Q")
        left, right = st.columns(2)
        left.altair_chart((diagonal + points).properties(height=360, title="Dự đoán so với thực tế"), width="stretch")
        right.altair_chart((zero + residuals).properties(height=360, title="Sai số (dự đoán trừ thật) theo RUL thật"), width="stretch")
        st.caption("Điểm đỏ là động cơ đang chọn. Rê chuột để xem giá trị từng động cơ.")

    with oof_tab:
        oof = data.oof_predictions
        models = sorted(oof["model_name"].unique(), key=lambda name: name != data.selected_model)
        controls = st.columns(2)
        with controls[0]:
            engine_id = _engine_select("Chọn động cơ train", sorted(oof["engine_id"].unique()), key="rul_oof_engine")
        with controls[1]:
            model = st.selectbox("Mô hình", models, format_func=_model_label, key="rul_oof_model")
        trajectory = engine_oof_trajectory(data, engine_id, model)
        mae = trajectory["absolute_error"].mean() if "absolute_error" in trajectory else (trajectory["y_pred"] - trajectory["y_true"]).abs().mean()
        _metric_row(
            [
                ("Fold chứa động cơ này", str(int(trajectory["fold"].iloc[0]))),
                ("Số cycle được dự đoán", f"{len(trajectory):,}"),
                ("MAE trên mọi cycle", format_metric(mae, 2)),
            ]
        )
        st.altair_chart(_rul_trajectory_chart(trajectory, {"y_true": "RUL thật", "y_pred": "RUL dự đoán (OOF)"}), width="stretch")
        st.caption(
            "Dự đoán chéo theo fold (OOF): mô hình tạo ra các dự đoán này không học động cơ đang xem. Đường chấm ngang là ngưỡng "
            f"{RUL_THRESHOLD} cycle. Dữ liệu này cũng đã dùng để chọn mô hình, nên đây là đánh giá nội bộ."
        )


def page_warning(data: DashboardData) -> None:
    st.header("Cảnh báo sớm: RUL dự đoán ≤ 30 cycle")
    summary = data.warning_summary
    _metric_row(
        [
            ("Ngưỡng RUL", f"{int(summary['rul_threshold'])} cycle"),
            ("Động cơ có cảnh báo", f"{int(summary['n_warning'])}/{int(summary['n_engines'])}"),
            ("Bị bỏ lỡ", f"{int(summary['n_missed_warning'])}"),
            ("Lead time trung bình", f"{format_metric(summary['mean_lead_time_cycles'], 2)} cycle"),
        ]
    )
    _card(
        "Cách đọc",
        "Dự đoán ở trang này là dự đoán chéo theo fold (OOF) trên 100 động cơ train: mỗi động cơ được dự đoán bởi mô hình "
        "không học động cơ đó. Đây cũng là dữ liệu dùng để chọn mô hình, nên kết quả có phần lạc quan. Mọi động cơ train "
        "đều đi đến hỏng, nên không tính được tỷ lệ báo động giả. Ngưỡng 30 cycle chưa hiệu chuẩn theo chi phí bảo trì.",
        kind="limit",
    )

    left, right = st.columns([1, 1.4])
    with left:
        counts = pd.DataFrame(
            {"Thời điểm": list(TIMING_LABELS.values()), "Số động cơ": [summary["timing_counts"].get(key, 0) for key in TIMING_LABELS]}
        )
        chart = (
            alt.Chart(counts)
            .mark_bar(color=PALETTE["pred"])
            .encode(
                y=alt.Y("Thời điểm:N", sort=list(TIMING_LABELS.values()), title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("Số động cơ:Q"),
                tooltip=["Thời điểm:N", "Số động cơ:Q"],
            )
            .properties(height=220, title="Cảnh báo đầu tiên so với ngưỡng")
        )
        st.altair_chart(chart, width="stretch")
        distribution = summary["lead_time_distribution"]
        st.caption(
            f"Lead time (cycle), n={distribution['n']}: min={distribution['min']}; Q1={distribution['q25']}; "
            f"median={distribution['median']}; Q3={distribution['q75']}; max={distribution['max']}."
        )
    with right:
        events = data.warning_events
        engine_id = _engine_select("Xem quỹ đạo của động cơ", sorted(events["engine_id"].unique()), key="warning_engine")
        event = events.loc[events["engine_id"] == engine_id].iloc[0]
        trajectory = engine_oof_trajectory(data, engine_id)
        marker = None if pd.isna(event["first_warning_cycle"]) else float(event["first_warning_cycle"])
        st.altair_chart(_rul_trajectory_chart(trajectory, {"y_true": "RUL thật", "y_pred": "RUL dự đoán (OOF)"}, marker_cycle=marker), width="stretch")
        st.caption(
            f"Vạch đỏ: cảnh báo đầu tiên tại cycle {event['first_warning_cycle']:.0f}, RUL thật khi đó "
            f"{event['actual_rul_at_first_warning']:.0f} cycle ({TIMING_LABELS.get(event['warning_timing'], event['warning_timing'])})."
        )

    table = events[
        ["engine_id", "first_warning_cycle", "actual_rul_at_first_warning", "lead_time_to_failure_cycles", "threshold_delay_cycles", "warning_timing", "missed_warning"]
    ].assign(warning_timing=lambda frame: frame["warning_timing"].map(TIMING_LABELS))
    with st.expander("Bảng cảnh báo cho 100 động cơ train"):
        st.dataframe(
            table.rename(
                columns={
                    "engine_id": "Động cơ",
                    "first_warning_cycle": "Cycle cảnh báo đầu",
                    "actual_rul_at_first_warning": "RUL thật tại cảnh báo",
                    "lead_time_to_failure_cycles": "Lead time (cycle)",
                    "threshold_delay_cycles": "Độ trễ so với ngưỡng (cycle)",
                    "warning_timing": "Phân loại thời điểm",
                    "missed_warning": "Không từng cảnh báo",
                }
            ),
            width="stretch",
            hide_index=True,
        )


def page_denoise(degradation: DegradationData) -> None:
    st.header("Khử nhiễu tín hiệu cảm biến bằng EWMA")
    summary, noise = degradation.summary, degradation.noise
    _metric_row(
        [
            ("Span EWMA", f"{summary['ewma_span']} cycle", "α = 2 / (span + 1); giá trị tại cycle t chỉ dùng cycle t và trước đó."),
            ("Mức giảm dao động (trung vị)", f"{format_metric(summary['noise_reduction_pct_median'], 1)}%",
             "Độ lệch chuẩn của bước nhảy giữa hai cycle liền kề, trước và sau EWMA."),
            ("Sensor được làm mượt", f"{len(summary['sensors'])}/21", "Bảy sensor còn lại hằng hoặc gần hằng."),
        ]
    )
    controls = st.columns([1, 1, 2])
    with controls[0]:
        engine_id = _engine_select("Động cơ train", sorted(degradation.signals["engine_id"].unique()), key="dn_engine")
    with controls[1]:
        sensors = degradation.sensors
        sensor = st.selectbox("Sensor", sensors, index=sensors.index("sensor_11") if "sensor_11" in sensors else 0, key="dn_sensor")
    long = engine_signal_view(degradation, engine_id, sensor)
    chart = (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y("value:Q", title=sensor, scale=alt.Scale(zero=False)),
            color=alt.Color("Tín hiệu:N", scale=alt.Scale(domain=["Tín hiệu gốc", "Sau EWMA"], range=[PALETTE["muted"], COLORS["blue"]]),
                            legend=alt.Legend(orient="top")),
            strokeWidth=alt.condition(alt.datum["Tín hiệu"] == "Sau EWMA", alt.value(2.5), alt.value(1)),
            tooltip=["cycle:Q", "Tín hiệu:N", alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=340)
        .interactive(bind_y=False)
    )
    st.altair_chart(chart, width="stretch")
    row = noise.loc[noise["sensor"] == sensor].iloc[0]
    st.caption(
        f"{sensor}: độ lệch chuẩn bước nhảy giữa hai cycle giảm từ {format_metric(row['diff_std_raw'], 4)} xuống "
        f"{format_metric(row['diff_std_smoothed'], 4)} ({format_metric(row['reduction_pct'], 1)}%), tính trên 100 động cơ train."
    )
    left, right = st.columns([1.3, 1])
    with left:
        bars = (
            alt.Chart(noise)
            .mark_bar(color=COLORS["blue"])
            .encode(
                y=alt.Y("sensor:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
                x=alt.X("reduction_pct:Q", title="Mức giảm dao động (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=["sensor:N", alt.Tooltip("reduction_pct:Q", format=".1f")],
            )
            .properties(height=alt.Step(18))
        )
        st.altair_chart(bars, width="stretch")
    with right:
        _card("Vai trò trong pipeline", "Tín hiệu EWMA đi vào Health Index và phát hiện bất thường. Mô hình RUL dùng window 30 cycle "
              "trên tín hiệu gốc, vì trung bình và độ dốc trên window cũng làm dịu dao động.")
        _card("Giới hạn", "Dữ liệu không có tín hiệu sạch để đối chiếu, nên đây là mức giảm dao động, chưa chứng minh đã loại đúng "
              "nhiễu đo. Span 10 được cố định trước, không tinh chỉnh theo kết quả.", kind="limit")


def page_anomaly(degradation: DegradationData) -> None:
    st.header("Xu hướng suy giảm và dấu hiệu bất thường")
    summary, events = degradation.summary, degradation.events
    anomaly, raw, config = summary["anomaly_smoothed"], summary["anomaly_raw_for_comparison"], summary["anomaly_config"]
    _metric_row(
        [
            ("Động cơ được phát hiện", f"{anomaly['n_detected']}/{anomaly['n_engines']}", "Phát hiện trước khi hỏng, trên train."),
            ("Báo trước hỏng (trung vị)", f"{format_metric(anomaly['lead_median'], 0)} cycle",
             f"Tứ phân vị {format_metric(anomaly['lead_q1'], 0)}–{format_metric(anomaly['lead_q3'], 0)}."),
            ("Thời điểm có dấu hiệu", f"{anomaly['detection_life_fraction_median'] * 100:.0f}% vòng đời", "Trung vị."),
            ("Nếu không khử nhiễu", f"{format_metric(raw['lead_median'], 1)} cycle", "Cùng quy tắc trên Health Index từ tín hiệu gốc."),
        ]
    )
    engine_id = _engine_select("Động cơ train", sorted(events["engine_id"].unique()), key="an_engine")
    event = events.loc[events["engine_id"] == engine_id].iloc[0]
    engine = degradation.signals.loc[degradation.signals["engine_id"] == engine_id, ["cycle", "health_index_raw", "health_index_ewma"]]
    long = engine.rename(columns={"health_index_raw": "HI từ tín hiệu gốc", "health_index_ewma": "HI sau khử nhiễu"}).melt(
        "cycle", var_name="Đường", value_name="Health Index"
    )
    threshold = float(event["baseline_mean"] - config["z_threshold"] * event["baseline_std"])
    lines = alt.Chart(long).mark_line().encode(
        x=alt.X("cycle:Q", title="Cycle"),
        y=alt.Y("Health Index:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Đường:N", scale=alt.Scale(domain=["HI từ tín hiệu gốc", "HI sau khử nhiễu"], range=[PALETTE["muted"], COLORS["blue"]]),
                        legend=alt.Legend(orient="top")),
        tooltip=["cycle:Q", "Đường:N", alt.Tooltip("Health Index:Q", format=".3f")],
    )
    baseline = alt.Chart(pd.DataFrame({"x": [1], "x2": [config["baseline_cycles"]]})).mark_rect(opacity=0.12, color=PALETTE["muted"]).encode(x="x:Q", x2="x2:Q")
    layers = [baseline, lines, alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(strokeDash=[4, 3], color=PALETTE["true"]).encode(y=alt.Y("y:Q", title="Health Index"))]
    if bool(event["detected"]):
        layers.append(alt.Chart(pd.DataFrame({"x": [float(event["detection_cycle"])]})).mark_rule(color=COLORS["orange"], strokeWidth=2).encode(x="x:Q"))
    layers.append(alt.Chart(pd.DataFrame({"x": [float(event["failure_cycle"])]})).mark_rule(color="black").encode(x="x:Q"))
    st.altair_chart(alt.layer(*layers).properties(height=340).interactive(bind_y=False), width="stretch")
    detection = "chưa phát hiện" if not bool(event["detected"]) else (
        f"phát hiện bất thường ở cycle {int(event['detection_cycle'])}, tức {int(event['lead_cycles'])} cycle trước khi hỏng")
    st.caption(
        f"Động cơ {engine_id:03d}: {detection}; hỏng ở cycle {int(event['failure_cycle'])}. Vùng xám là {config['baseline_cycles']} "
        f"cycle tham chiếu; đường đứt là ngưỡng {config['z_threshold']:g}σ; vạch cam là lúc phát hiện, vạch đen là lúc hỏng."
    )
    left, right = st.columns([1.3, 1])
    with left:
        histogram = (
            alt.Chart(events.dropna(subset=["lead_cycles"]))
            .mark_bar(color=COLORS["blue"])
            .encode(
                x=alt.X("lead_cycles:Q", bin=alt.Bin(step=20), title="Số cycle từ lúc phát hiện đến khi hỏng"),
                y=alt.Y("count():Q", title="Số động cơ"),
            )
            .properties(height=240)
        )
        st.altair_chart(histogram, width="stretch")
    with right:
        _card("Quy tắc", f"So mỗi động cơ với chính nó ở giai đoạn vận hành ổn định: báo khi Health Index giảm quá {config['z_threshold']:g} độ lệch chuẩn "
              f"so với {config['baseline_cycles']} cycle đầu, kéo dài {config['persistence']} cycle liên tiếp. Không dùng nhãn RUL.")
        _card("Giới hạn", "Mọi động cơ train đều đi đến hỏng và không có nhãn thời điểm bắt đầu suy giảm, nên chưa đo được trực tiếp "
              "tỷ lệ báo động giả. Đây là cảnh báo mức một: động cơ đã lệch khỏi trạng thái vận hành ổn định, chưa ước lượng thời gian "
              "còn lại.", kind="limit")
    st.subheader("Độ nhạy của quy tắc theo ngưỡng z và số cycle liên tiếp")
    signal_label = st.radio("Health Index", ["Sau khử nhiễu", "Tín hiệu gốc"], horizontal=True, key="an_sens_signal")
    grid = degradation.sensitivity.loc[
        degradation.sensitivity["signal"].eq("ewma" if signal_label == "Sau khử nhiễu" else "raw")
    ]
    table = pd.DataFrame(
        {
            "Ngưỡng z": grid["z_threshold"].map(lambda value: f"{value:g}σ"),
            "Số cycle liên tiếp": grid["persistence"].astype(int),
            "Phát hiện": grid["n_detected"].astype(int).astype(str) + "/" + grid["n_engines"].astype(int).astype(str),
            "Lead time nhỏ nhất": grid["lead_min"].round(0).astype(int),
            "Lead time trung vị": grid["lead_median"].round(1),
            "% vòng đời lúc phát hiện": (grid["life_fraction_median"] * 100).round(1),
            "Báo sớm (cycle 31–40)": grid["n_early"].astype(int),
        }
    )
    chosen = grid["z_threshold"].eq(config["z_threshold"]) & grid["persistence"].eq(config["persistence"])
    st.dataframe(
        table.style.apply(lambda row: ["font-weight: 700"] * len(row) if chosen.loc[row.name] else [""] * len(row), axis=1)
        .format({"Lead time trung vị": lambda v: format_metric(v, 1), "% vòng đời lúc phát hiện": lambda v: format_metric(v, 1)}),
        hide_index=True, width="stretch",
    )
    st.caption(
        "Dòng in đậm là cấu hình đang dùng. Không có nhãn bất thường, nên cột Báo sớm (số động cơ bị báo ngay sau đoạn tham chiếu) "
        "là chỉ báo gián tiếp cho độ nhạy quá mức. Mọi cấu hình đều phát hiện đủ 100 động cơ; ngưỡng thấp báo sớm hơn nhưng "
        "nhiều trường hợp báo quá sớm hơn, ngưỡng cao thì ngược lại."
    )
    with st.expander("Hình kiểm định đầu/cuối vòng đời (outputs/statistics)"):
        _figure("outputs/statistics/figures/effect_size_ci.png", "Chênh lệch trung bình 30 cycle cuối và đầu, khoảng tin cậy 95%")


def page_maintenance(maintenance: MaintenanceData) -> None:
    st.header("Báo cáo dự đoán thời điểm cần bảo trì")
    summary, schedule = maintenance.summary, maintenance.schedule
    counts, evaluation = summary["status_counts"], summary["evaluation"]
    _metric_row(
        [
            ("Cần bảo trì ngay", str(counts["Cần bảo trì"]), "RUL dự đoán ≤ 30 cycle."),
            ("Theo dõi tăng cường", str(counts["Theo dõi tăng cường"]), "Health Index đã báo bất thường, RUL dự đoán > 30."),
            ("Bình thường", str(counts["Bình thường"])),
            ("Đến hạn trong 25 cycle tới", str(summary["planning"]["1–25"])),
        ]
    )
    chosen = st.multiselect("Lọc theo trạng thái", STATUS_ORDER, default=STATUS_ORDER, key="mt_status")
    view = schedule.loc[schedule["status"].isin(chosen)].copy()
    view["rul_pred"] = view["rul_pred"].round(1)
    view["anomaly_cycle"] = view["anomaly_cycle"].astype("Int64")
    st.dataframe(
        view[list(SCHEDULE_LABELS)].rename(columns=SCHEDULE_LABELS)
        .style.format({"RUL dự đoán": lambda v: format_metric(v, 1)}),
        width="stretch", hide_index=True, height=300,
    )
    st.caption("Cycle cần bảo trì = cycle hiện tại + (RUL dự đoán − 30), làm tròn xuống. Bảng sắp theo mức ưu tiên.")

    left, right = st.columns([1.2, 1])
    with left:
        planning = maintenance.planning.melt("window", var_name="Trạng thái", value_name="Số động cơ")
        windows = maintenance.planning["window"].tolist()
        chart = (
            alt.Chart(planning)
            .mark_bar()
            .encode(
                x=alt.X("window:N", sort=windows, title="Số cycle còn lại đến lúc cần bảo trì", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Số động cơ:Q"),
                color=alt.Color("Trạng thái:N", scale=alt.Scale(domain=STATUS_ORDER, range=STATUS_COLORS), legend=alt.Legend(orient="top")),
                tooltip=["window:N", "Trạng thái:N", "Số động cơ:Q"],
            )
            .properties(height=360, title="Kế hoạch bảo trì theo thời gian")
        )
        st.altair_chart(chart, width="stretch")
    with right:
        engine_id = _engine_select("Xem chi tiết động cơ test", schedule["engine_id"].tolist(), key="mt_engine",
                                   default=int(schedule["engine_id"].iloc[0]))
        row = schedule.loc[schedule["engine_id"] == engine_id].iloc[0]
        _metric_row(
            [
                ("RUL dự đoán", format_metric(row["rul_pred"], 1)),
                ("Cycle cần bảo trì", str(int(row["maintenance_cycle"]))),
            ]
        )
        color = dict(zip(STATUS_ORDER, STATUS_COLORS))[row["status"]]
        st.markdown(
            f'<span style="background:{color};color:white;border-radius:6px;padding:3px 10px;font-weight:600">'
            f'{row["status"]}</span>',
            unsafe_allow_html=True,
        )
        health = maintenance.test_health_index.loc[maintenance.test_health_index["engine_id"] == engine_id]
        status = maintenance.test_anomaly.loc[maintenance.test_anomaly["engine_id"] == engine_id].iloc[0]
        layers = [alt.Chart(health).mark_line(color=COLORS["blue"]).encode(
            x=alt.X("cycle:Q", title="Cycle"), y=alt.Y("health_index:Q", title="Health Index", scale=alt.Scale(zero=False)),
            tooltip=["cycle:Q", alt.Tooltip("health_index:Q", format=".3f")])]
        if bool(status["anomaly_evaluable"]):
            layers.append(alt.Chart(pd.DataFrame({"y": [float(status["threshold"])]})).mark_rule(strokeDash=[4, 3], color=PALETTE["true"]).encode(y=alt.Y("y:Q", title="Health Index")))
        if pd.notna(row["anomaly_cycle"]):
            layers.append(alt.Chart(pd.DataFrame({"x": [float(row["anomaly_cycle"])]})).mark_rule(color=COLORS["orange"], strokeWidth=2).encode(x="x:Q"))
        st.altair_chart(alt.layer(*layers).properties(height=200), width="stretch")
        note = "chưa đủ 35 cycle để đánh giá bất thường" if not bool(status["anomaly_evaluable"]) else (
            f"bất thường từ cycle {int(row['anomaly_cycle'])}" if pd.notna(row["anomaly_cycle"]) else "chưa có dấu hiệu bất thường")
        st.caption(f"Động cơ {engine_id:03d} đang ở cycle {int(row['current_cycle'])}; {note}. Đường đứt là ngưỡng 3σ của động cơ này.")

    st.subheader("Kiểm tra bằng nhãn test (đọc sau khi lịch đã ghi ra file)")
    check_left, check_right = st.columns([1, 1.2])
    with check_left:
        st.dataframe(maintenance.check.rename(columns={"status": "Trạng thái dự đoán"}), width="stretch", hide_index=True)
    with check_right:
        _card("Kết quả", f"{evaluation['n_correct_needs_maintenance']}/{evaluation['n_flagged_needs_maintenance']} động cơ báo cần bảo trì "
              f"có RUL thật ≤ 30. Trong {evaluation['n_actual_needs_maintenance']} động cơ thật sự cần bảo trì, {evaluation['n_actual_needs_maintenance'] - evaluation['n_correct_needs_maintenance']} động cơ còn lại có RUL "
              "dự đoán 31–38 nên đến hạn sau 1–8 cycle và vẫn ở mức theo dõi.")
        _card("Giới hạn", "Mức theo dõi còn rộng: 41/59 động cơ ở mức này có RUL thật trên 60. Lịch có thể lạc quan với từng động cơ, "
              "ví dụ động cơ 23. Ngưỡng 30 chưa được hiệu chuẩn theo chi phí bảo trì.", kind="limit")


def page_provenance(data: DashboardData) -> None:
    st.header("Nguồn gốc dữ liệu và kiểm soát")
    _card(
        "Chỉ đọc",
        "Dashboard chỉ đọc kết quả đã lưu của các bước: khám phá dữ liệu, mô hình, khử nhiễu, bất thường và lịch bảo trì. "
        "Dashboard không mở file đáp án test gốc và không huấn luyện lại mô hình.",
    )
    _metric_row(
        [
            ("Dòng đánh giá test", f"{len(data.model_evaluation):,}"),
            ("Phiên bản schema mô hình", str(data.model_run_info.get("schema_version", "n/a"))),
        ]
    )
    with st.expander("Biên bản lần chạy mô hình (run_info.json)", expanded=True):
        st.json(data.model_run_info)


# ---------------------------------------------------------------------------
# Bố cục
# ---------------------------------------------------------------------------


def _hero(data: DashboardData) -> None:
    n_engines = int(data.model_metrics.get("metrics", {}).get("n_engines", len(data.model_evaluation)))
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">Phân tích dữ liệu lớn · Đề tài 8</div>
          <div class="hero-title">Phân tích dữ liệu IoT và bảo trì dự đoán</div>
          <div class="hero-subtitle">Khử nhiễu, phát hiện bất thường, dự đoán RUL và lịch bảo trì cho động cơ turbofan NASA C-MAPSS FD001, đọc từ kết quả đã lưu.</div>
          <span class="pill">chỉ đọc kết quả</span><span class="pill">{n_engines} động cơ test</span><span class="pill">mô hình: {_model_label(data.selected_model)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _guarded(render: Callable[[], None]) -> None:
    try:
        render()
    except DashboardDataError as exc:
        st.error(str(exc))


def main(repo_root: str | Path | None = None) -> None:
    global ROOT
    if repo_root is not None:
        ROOT = Path(repo_root).resolve()
    st.set_page_config(page_title="Đề tài 8 | Bảo trì dự đoán", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    root = str(ROOT)
    try:
        data = _dashboard_data(root)
    except DashboardDataError as exc:
        st.error(str(exc))
        st.info("Hãy chạy ETL, rolling features, modeling, cmapss.degradation và cmapss.maintenance trước khi mở dashboard.")
        return

    pages: dict[str, Callable[[], None]] = {
        "Tổng quan": lambda: page_overview(data, _degradation_data(root), _maintenance_data(root)),
        "Dữ liệu & EDA": lambda: page_exploration(_exploration_data(root)),
        "Khử nhiễu": lambda: page_denoise(_degradation_data(root)),
        "Xu hướng & bất thường": lambda: page_anomaly(_degradation_data(root)),
        "Dự đoán RUL": lambda: page_rul(data),
        "Cảnh báo sớm": lambda: page_warning(data),
        "Lịch bảo trì": lambda: page_maintenance(_maintenance_data(root)),
        "Provenance": lambda: page_provenance(data),
    }
    names = list(pages)
    requested = st.query_params.get("page")
    st.sidebar.markdown("### Điều hướng")
    page = st.sidebar.radio(
        "Trang", names, index=names.index(requested) if requested in names else 0, label_visibility="collapsed"
    )
    st.query_params["page"] = page  # deep link, e.g. ?page=Cảnh báo sớm
    st.sidebar.divider()
    st.sidebar.markdown("**Lần chạy hiện tại**")
    st.sidebar.caption("Đọc kết quả đã lưu, không huấn luyện lại mô hình.")
    st.sidebar.markdown(f"Mô hình: **{_model_label(data.selected_model)}**")
    st.sidebar.markdown(f"Động cơ test: **{len(data.model_evaluation):,}**")

    _hero(data)
    _guarded(pages[page])
