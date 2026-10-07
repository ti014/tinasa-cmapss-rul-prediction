"""Streamlit UI for the C-MAPSS FD001 RUL study.

Pages only render persisted artifacts loaded through ``cmapss.dashboard``. The
UI never fits a model, reads raw test labels, or changes the evaluation
protocol. Entry point: ``app.py`` at the repository root, which calls
``main(repo_root)``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from cmapss.dashboard import (
    DashboardData,
    DashboardDataError,
    ExplorationData,
    ImprovementData,
    engine_oof_trajectory,
    engine_rul_view,
    engine_timeseries_view,
    forecast_comparison,
    format_metric,
    load_cross_dataset,
    load_dashboard_data,
    load_exploration_data,
    load_improvement_data,
    sensor_trajectory,
)
from cmapss.followup_results import load_followup_results
from cmapss.project import FOLLOWUP_RUN


ROOT = Path(__file__).resolve().parents[2]  # replaced by main(repo_root)
RUL_THRESHOLD = 30
DEFAULT_SENSORS = ["sensor_4", "sensor_11", "sensor_12"]

MODEL_LABELS = {
    "hist_gradient_boosting": "HistGradientBoosting",
    "ridge_alpha_1": "Ridge (alpha = 1)",
    "median_baseline": "Median baseline",
}
STAGE_LABELS = {
    "baseline": "Baseline",
    "windows": "Sau chọn window",
    "parameters": "Sau chỉnh tham số",
    "weights": "Sau chọn trọng số",
}
TIMING_LABELS = {
    "before_threshold": "Trước ngưỡng",
    "at_threshold": "Đúng ngưỡng",
    "after_threshold": "Sau ngưỡng, trước failure",
    "at_failure": "Tại failure",
    "no_warning": "Không cảnh báo",
}
METRIC_LABELS = {
    "mean_engine_mae": "MAE toàn bộ",
    "low_rul_engine_mae": "MAE khi RUL ≤ 30",
    "late_or_missed_rate": "Tỷ lệ cảnh báo trễ hoặc bỏ lỡ",
    "late_or_missed_count": "Cảnh báo trễ hoặc bỏ lỡ",
    "early_alarm_burden": "Tỷ lệ thời gian báo sớm",
}
FORECAST_METHOD_LABELS = {"arima": "ARIMA", "last_value": "Last value", "drift": "Drift"}
PALETTE = {"true": "#16202a", "pred": "#1f7a8c", "alt": "#f28e2b", "muted": "#94a3b8", "alert": "#c0392b"}

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
# Cached loaders: artifacts are immutable during a session, so read them once.
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Đang đọc artifact modeling và time-series...")
def _dashboard_data(root: str) -> DashboardData:
    return load_dashboard_data(root)


@st.cache_data(show_spinner="Đang đọc artifact EDA và rolling features...")
def _exploration_data(root: str) -> ExplorationData:
    return load_exploration_data(root)


@st.cache_data(show_spinner="Đang đọc thử nghiệm cải thiện...")
def _improvement_data(root: str) -> ImprovementData:
    return load_improvement_data(root)


@st.cache_data(show_spinner=False)
def _cross_dataset(root: str) -> pd.DataFrame:
    return load_cross_dataset(root)


# ---------------------------------------------------------------------------
# Small rendering helpers
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
    return int(st.selectbox(label, options, index=index, key=key, format_func=lambda value: f"Engine {int(value):03d}"))


def _model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def _rul_trajectory_chart(frame: pd.DataFrame, series: dict[str, str], *, marker_cycle: float | None = None) -> alt.Chart:
    """Line chart of RUL columns against cycle with the 30-cycle warning threshold."""

    long = frame[["cycle", *series]].rename(columns=series).melt("cycle", var_name="Đường", value_name="RUL")
    names = list(series.values())
    colors = [PALETTE["true"], PALETTE["pred"], PALETTE["alt"], PALETTE["muted"]][: len(names)]
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("cycle:Q", title="Cycle"),
            y=alt.Y("RUL:Q", title="RUL (cycles)"),
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
# Pages
# ---------------------------------------------------------------------------


def page_overview(data: DashboardData) -> None:
    metrics = data.model_metrics.get("metrics", {})
    interval = data.model_metrics.get("bootstrap", {}).get("intervals", {}).get("mean_engine_mae", {})
    st.header("Kết quả cuối trên 100 test engines FD001")
    _metric_row(
        [
            ("Mean engine MAE", format_metric(metrics.get("mean_engine_mae", 0)), "Metric chọn model chính; càng thấp càng tốt."),
            (
                "Khoảng tin cậy 95% của MAE",
                f"{format_metric(interval.get('ci_lower', 0), 2)} – {format_metric(interval.get('ci_upper', 0), 2)}",
                "Bootstrap theo engine, 10.000 lần lấy lại.",
            ),
            ("RMSE", format_metric(metrics.get("rmse", 0)), "Root mean squared error trên 100 test engines."),
            ("NASA score", format_metric(metrics.get("nasa_score", 0), 2), "Hàm điểm bất đối xứng của NASA; dự đoán trễ bị phạt nặng hơn."),
        ]
    )

    st.subheader("Pipeline tạo ra các kết quả này")
    steps = [
        ("1. Raw", "train/test/RUL FD001 dạng text"),
        ("2. ETL", "Pandas và PySpark → Silver Parquet, Gold SQLite"),
        ("3. Đặc trưng", "Rolling window 30 cycles, 126 cột từ 21 sensors"),
        ("4. Modeling", "5-fold GroupKFold theo engine, chọn theo MAE"),
        ("5. Đánh giá", "Dự đoán test trước, join nhãn test sau"),
        ("6. Phân tích phụ", "Cảnh báo RUL ≤ 30, Health Index/ARIMA"),
    ]
    for column, (title, body) in zip(st.columns(len(steps)), steps):
        column.markdown(f'<div class="step"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)

    st.subheader("So sánh model ứng viên trong cross-validation")
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
                x=alt.X("mean_engine_mae:Q", title="Mean engine MAE (cycles), 5 folds"),
                color=alt.condition(alt.datum.selected, alt.value(PALETTE["pred"]), alt.value(PALETTE["muted"])),
                tooltip=["Model:N", alt.Tooltip("mean_engine_mae:Q", format=".4f"), alt.Tooltip("std_engine_mae:Q", format=".4f")],
            )
        )
        errors = alt.Chart(comparison).mark_errorbar(color=PALETTE["true"]).encode(
            y=alt.Y("Model:N", sort=order), x=alt.X("lower:Q", title="Mean engine MAE (cycles), 5 folds"), x2="upper:Q"
        )
        st.altair_chart((bars + errors).properties(height=200), width="stretch")
        st.caption("Thanh lỗi: ±1 độ lệch chuẩn giữa 5 folds. Cột tô màu là model được chọn.")
    with right:
        _card(
            "Model được chọn",
            f"{_model_label(data.selected_model)} có MAE theo engine thấp nhất trong protocol 5-fold GroupKFold. "
            "Predictions test được ghi trước khi join test labels.",
        )
        _card(
            "Cách đọc MAE",
            "MAE trên test chỉ tính tại cycle quan sát cuối của mỗi engine. MAE trong cross-validation tính trên toàn bộ "
            "trajectory train từ cycle 30, nên hai con số không so sánh trực tiếp.",
            kind="limit",
        )

    with st.expander("Gợi ý thứ tự demo"):
        st.markdown(
            "1. **Dữ liệu & EDA**: tuổi thọ engine và quỹ đạo sensor khi tiến gần failure.\n"
            "2. **Đánh giá RUL**: một test engine cụ thể và quỹ đạo OOF trên train engine.\n"
            "3. **Cảnh báo sớm**: ngưỡng RUL 30 cycles và phân loại thời điểm cảnh báo.\n"
            "4. **Cross-dataset**: sai số tăng khi chuyển model FD001 sang FD002–FD004.\n"
            "5. **Thử nghiệm cải thiện**: nested CV chưa đạt tiêu chí nên giữ baseline.\n"
            "6. **Provenance**: run_info và isolation contract."
        )


def page_exploration(exploration: ExplorationData) -> None:
    st.header("Dữ liệu train FD001 và các sensor liên quan tới RUL")
    lifetimes = exploration.engine_lifetimes
    _metric_row(
        [
            ("Train engines", f"{lifetimes['engine_id'].nunique():,}"),
            ("Tổng cycles train", f"{int(lifetimes['n_observations'].sum()):,}"),
            ("Tuổi thọ ngắn nhất", f"{int(lifetimes['n_observations'].min())} cycles"),
            ("Tuổi thọ dài nhất", f"{int(lifetimes['n_observations'].max())} cycles"),
        ]
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Phân phối tuổi thọ engine")
        histogram = (
            alt.Chart(lifetimes)
            .mark_bar(color=PALETTE["pred"])
            .encode(
                x=alt.X("n_observations:Q", bin=alt.Bin(step=20), title="Tuổi thọ (cycles)"),
                y=alt.Y("count():Q", title="Số engine"),
                tooltip=[alt.Tooltip("count():Q", title="Số engine")],
            )
            .properties(height=280)
        )
        st.altair_chart(histogram, width="stretch")
    with right:
        st.subheader("Spearman giữa sensor và RUL, theo engine")
        summary = exploration.sensor_rul_summary.loc[lambda frame: frame["defined_engines"] > 0]
        ranking = (
            alt.Chart(summary)
            .mark_bar(color=PALETTE["pred"])
            .encode(
                y=alt.Y("sensor:N", sort=alt.EncodingSortField("median", op="min"), title=None, axis=alt.Axis(labelOverlap=False)),
                x=alt.X("median:Q", title="Median Spearman ρ qua các engine", scale=alt.Scale(domain=[-1, 1])),
                tooltip=["sensor:N", alt.Tooltip("median:Q", format=".3f"), alt.Tooltip("q25:Q", format=".3f"), alt.Tooltip("q75:Q", format=".3f")],
            )
            .properties(height=alt.Step(20))
        )
        st.altair_chart(ranking, width="stretch")
        excluded = exploration.sensor_rul_summary.loc[lambda frame: frame["defined_engines"] == 0, "sensor"]
        st.caption(f"Không hiển thị sensor hằng số trong mọi engine (ρ không xác định): {', '.join(excluded)}.")

    st.subheader("Quỹ đạo sensor của một train engine")
    trajectories = exploration.sensor_trajectories
    sensors = sorted({column.split("__")[0] for column in trajectories.columns if "__" in column}, key=lambda name: int(name.split("_")[1]))
    controls = st.columns([1, 2, 1])
    with controls[0]:
        engine_id = _engine_select("Engine", sorted(trajectories["engine_id"].unique()), key="eda_engine")
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
            f"Engine {engine_id:03d} chạy tới failure tại cycle {failure}. Dữ liệu lấy từ Gold rolling features nên bắt đầu ở "
            f"cycle {exploration.window_cycles}; rolling mean dùng trailing window {exploration.window_cycles} cycles."
        )

    with st.expander("Hình EDA đã lưu trong outputs/eda/figures"):
        tabs = st.tabs(["Quỹ đạo sensor", "Sensor–RUL theo engine", "Phân phối sensor", "Tương quan pooled"])
        with tabs[0]:
            for index in (1, 2, 3):
                _figure(f"outputs/eda/figures/sensor_trajectories_{index}.png", f"Sensor trajectories {index}")
        with tabs[1]:
            _figure("outputs/eda/figures/sensor_rul_by_engine.png", "Spearman sensor–RUL theo engine")
        with tabs[2]:
            for index in (1, 2, 3):
                _figure(f"outputs/eda/figures/sensor_distributions_{index}.png", f"Sensor distributions {index}")
        with tabs[3]:
            _figure("outputs/eda/figures/sensor_spearman_pooled.png", "Spearman pooled giữa các sensor")


def page_rul(data: DashboardData) -> None:
    st.header("Đánh giá RUL")
    test_tab, oof_tab = st.tabs(["Test engine: điểm quan sát cuối", "Train engine: quỹ đạo OOF"])

    with test_tab:
        engine_id = _engine_select("Chọn test engine", sorted(data.model_evaluation["engine_id"].unique()), key="rul_test_engine")
        row = engine_rul_view(data, engine_id)
        if row.empty:
            st.warning("Không có evaluation row cho engine đã chọn.")
            return
        record = row.iloc[0]
        _metric_row(
            [
                ("RUL thật", format_metric(record["y_true"], 0)),
                ("RUL dự đoán", format_metric(record["y_pred"])),
                ("Absolute error", format_metric(record["absolute_error"])),
                ("Cycle cuối", format_metric(record["cycle"], 0)),
            ]
        )
        direction = "cao hơn" if record["y_pred"] > record["y_true"] else "thấp hơn"
        _card(
            f"Engine {engine_id:03d}",
            f"Sai số tuyệt đối {format_metric(record['absolute_error'])} cycles; RUL dự đoán {direction} RUL thật. "
            "Mỗi test engine chỉ có một dòng đánh giá tại cycle cuối.",
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
                x=alt.X("y_true:Q", title="RUL thật (cycles)", scale=alt.Scale(domain=[0, upper])),
                y=alt.Y("y_pred:Q", title="RUL dự đoán (cycles)", scale=alt.Scale(domain=[0, upper])),
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
                x=alt.X("y_true:Q", title="RUL thật (cycles)"),
                y=alt.Y("residual:Q", title="Dự đoán − thật (cycles)"),
                color=alt.condition(alt.datum.chosen, alt.value(PALETTE["alert"]), alt.value(PALETTE["pred"])),
                size=alt.condition(alt.datum.chosen, alt.value(160), alt.value(45)),
                tooltip=tooltip,
            )
        )
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=PALETTE["true"], strokeDash=[4, 4]).encode(y="y:Q")
        left, right = st.columns(2)
        left.altair_chart((diagonal + points).properties(height=360, title="Dự đoán và thực tế"), width="stretch")
        right.altair_chart((zero + residuals).properties(height=360, title="Residual theo RUL thật"), width="stretch")
        st.caption("Điểm đỏ là engine đang chọn. Rê chuột để xem giá trị từng engine.")

    with oof_tab:
        oof = data.oof_predictions
        models = sorted(oof["model_name"].unique(), key=lambda name: name != data.selected_model)
        controls = st.columns(2)
        with controls[0]:
            engine_id = _engine_select("Chọn train engine", sorted(oof["engine_id"].unique()), key="rul_oof_engine")
        with controls[1]:
            model = st.selectbox("Model", models, format_func=_model_label, key="rul_oof_model")
        trajectory = engine_oof_trajectory(data, engine_id, model)
        mae = trajectory["absolute_error"].mean() if "absolute_error" in trajectory else (trajectory["y_pred"] - trajectory["y_true"]).abs().mean()
        _metric_row(
            [
                ("Fold giữ engine này", str(int(trajectory["fold"].iloc[0]))),
                ("Số cycle được dự đoán", f"{len(trajectory):,}"),
                ("MAE trên trajectory", format_metric(mae, 2)),
            ]
        )
        st.altair_chart(_rul_trajectory_chart(trajectory, {"y_true": "RUL thật", "y_pred": "RUL dự đoán (OOF)"}), width="stretch")
        st.caption(
            "Out-of-fold: engine này không nằm trong dữ liệu fit của model tạo ra các dự đoán. Đường chấm ngang là ngưỡng "
            f"{RUL_THRESHOLD} cycles. OOF cũng đã được dùng để chọn model nên đây là đánh giá nội bộ."
        )


def page_warning(data: DashboardData) -> None:
    st.header("Cảnh báo RUL thấp trên OOF train")
    summary = data.warning_summary
    _metric_row(
        [
            ("Ngưỡng RUL", f"{int(summary['rul_threshold'])} cycles"),
            ("Engine có cảnh báo", f"{int(summary['n_warning'])}/{int(summary['n_engines'])}"),
            ("Không từng cảnh báo", f"{int(summary['n_missed_warning'])}"),
            ("Lead time trung bình", f"{format_metric(summary['mean_lead_time_cycles'], 2)} cycles"),
        ]
    )
    _card(
        "Phạm vi diễn giải",
        "Có cảnh báo không đồng nghĩa cảnh báo đủ sớm. Ngưỡng minh họa là 30 cycles; OOF cũng đã dùng để chọn model nên đây là "
        "đánh giá nội bộ, không độc lập sau model selection. Không có engine âm tính theo định nghĩa event trọn vòng đời: "
        "false-alarm rate không xác định, không phải 0%. Chưa có yêu cầu lead time tối thiểu cho bảo trì.",
        kind="limit",
    )

    left, right = st.columns([1, 1.4])
    with left:
        counts = pd.DataFrame(
            {"Thời điểm": list(TIMING_LABELS.values()), "Số engine": [summary["timing_counts"].get(key, 0) for key in TIMING_LABELS]}
        )
        chart = (
            alt.Chart(counts)
            .mark_bar(color=PALETTE["pred"])
            .encode(
                y=alt.Y("Thời điểm:N", sort=list(TIMING_LABELS.values()), title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("Số engine:Q"),
                tooltip=["Thời điểm:N", "Số engine:Q"],
            )
            .properties(height=220, title="Thời điểm cảnh báo đầu tiên so với ngưỡng thật")
        )
        st.altair_chart(chart, width="stretch")
        distribution = summary["lead_time_distribution"]
        st.caption(
            f"Lead time (cycles), n={distribution['n']}: min={distribution['min']}; Q1={distribution['q25']}; "
            f"median={distribution['median']}; Q3={distribution['q75']}; max={distribution['max']}."
        )
    with right:
        events = data.warning_events
        engine_id = _engine_select("Xem trajectory của engine", sorted(events["engine_id"].unique()), key="warning_engine")
        event = events.loc[events["engine_id"] == engine_id].iloc[0]
        trajectory = engine_oof_trajectory(data, engine_id)
        marker = None if pd.isna(event["first_warning_cycle"]) else float(event["first_warning_cycle"])
        st.altair_chart(_rul_trajectory_chart(trajectory, {"y_true": "RUL thật", "y_pred": "RUL dự đoán (OOF)"}, marker_cycle=marker), width="stretch")
        st.caption(
            f"Vạch đỏ: cảnh báo đầu tiên tại cycle {event['first_warning_cycle']:.0f}, RUL thật khi đó "
            f"{event['actual_rul_at_first_warning']:.0f} cycles ({TIMING_LABELS.get(event['warning_timing'], event['warning_timing'])})."
        )

    table = events[
        ["engine_id", "first_warning_cycle", "actual_rul_at_first_warning", "lead_time_to_failure_cycles", "threshold_delay_cycles", "warning_timing", "missed_warning"]
    ].assign(warning_timing=lambda frame: frame["warning_timing"].map(TIMING_LABELS))
    with st.expander("Bảng cảnh báo cho 100 train engines"):
        st.dataframe(
            table.rename(
                columns={
                    "engine_id": "Engine",
                    "first_warning_cycle": "Cycle cảnh báo đầu",
                    "actual_rul_at_first_warning": "RUL thật tại cảnh báo",
                    "lead_time_to_failure_cycles": "Lead time (cycles)",
                    "threshold_delay_cycles": "Độ trễ so với ngưỡng (cycles)",
                    "warning_timing": "Phân loại thời điểm",
                    "missed_warning": "Không từng cảnh báo",
                }
            ),
            width="stretch",
            hide_index=True,
        )


def page_cross_dataset(matrix: pd.DataFrame) -> None:
    st.header("Chạy cùng pipeline trên FD001–FD004")
    window = matrix.attrs.get("window_cycles")
    _card(
        "Protocol",
        f"Mỗi subset có một evaluation nội bộ (train và test cùng subset). Transfer dùng model fit trên toàn bộ train FD001 để dự đoán "
        f"test của FD002–FD004, không fit lại. Matrix dùng trailing window {window} cycles vì FD004 có test engine ngắn nhất "
        "19 cycles; kết quả không thay thế kết quả FD001 chính thức dùng window 30.",
    )
    labeled = matrix.assign(
        Evaluation=lambda frame: frame["source"] + " → " + frame["target"],
        Loại=lambda frame: frame["evaluation_type"].map({"within_subset": "Nội bộ subset", "transfer": "Transfer từ FD001"}),
    )
    bars = (
        alt.Chart(labeled)
        .mark_bar()
        .encode(
            y=alt.Y("Evaluation:N", sort=list(labeled["Evaluation"]), title=None, axis=alt.Axis(labelLimit=200)),
            x=alt.X("mean_engine_mae:Q", title="Mean engine MAE (cycles)"),
            color=alt.Color("Loại:N", scale=alt.Scale(range=[PALETTE["pred"], PALETTE["alt"]]), legend=alt.Legend(orient="top")),
            tooltip=["Evaluation:N", "Loại:N", "n_engines:Q", alt.Tooltip("mean_engine_mae:Q", format=".4f"), alt.Tooltip("rmse:Q", format=".4f")],
        )
    )
    errors = alt.Chart(labeled).mark_errorbar(color=PALETTE["true"]).encode(
        y=alt.Y("Evaluation:N", sort=list(labeled["Evaluation"])),
        x=alt.X("mae_ci_lower:Q", title="Mean engine MAE (cycles)"),
        x2="mae_ci_upper:Q",
    )
    st.altair_chart((bars + errors).properties(height=alt.Step(34)), width="stretch")
    st.caption("Thanh lỗi: khoảng tin cậy bootstrap 95% theo engine (10.000 lần lấy lại, seed 42).")
    st.dataframe(
        labeled[["Evaluation", "Loại", "n_engines", "mean_engine_mae", "mae_ci_lower", "mae_ci_upper", "rmse", "nasa_score"]].rename(
            columns={
                "n_engines": "Test engines",
                "mean_engine_mae": "Mean engine MAE",
                "mae_ci_lower": "CI 95% dưới",
                "mae_ci_upper": "CI 95% trên",
                "rmse": "RMSE",
                "nasa_score": "NASA score",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={name: st.column_config.NumberColumn(format="%.4f") for name in ("Mean engine MAE", "CI 95% dưới", "CI 95% trên", "RMSE")}
        | {"NASA score": st.column_config.NumberColumn(format="%.2f")},
    )
    _card(
        "Diễn giải",
        "Transfer từ FD001 làm MAE tăng rõ rệt so với evaluation nội bộ trên target subset. Model fit riêng trên FD001 không nên "
        "được dùng trực tiếp cho FD002–FD004 nếu chưa có retraining hoặc domain adaptation.",
        kind="limit",
    )


def page_improvement(experiment: ImprovementData) -> None:
    if (ROOT / FOLLOWUP_RUN / "completion.json").is_file():
        st.subheader("Thí nghiệm bổ sung: nhãn RUL, lịch sử và điều kiện vận hành")
        summary, paired_followup = load_followup_results(ROOT)
        table = summary[["dataset", "selected_final", "baseline_outer_mae", "selected_outer_mae",
                         "test_mae", "outer_gate_passed"]].copy()
        table["selected_final"] = table.selected_final.replace({
            "baseline": "Baseline", "cap_none_history_1": "History features",
            "condition_normalized": "Chuẩn hóa theo điều kiện", "raw_settings": "Thêm settings"})
        st.dataframe(table.rename(columns={"dataset": "Subset", "selected_final": "Phương án cuối",
            "baseline_outer_mae": "Outer MAE baseline", "selected_outer_mae": "Outer MAE mới",
            "test_mae": "Test MAE", "outer_gate_passed": "Đạt tiêu chí"}), hide_index=True, width="stretch")
        st.caption(
            "Outer CV: 5 fold theo engine; chọn cấu hình bằng 3 inner fold. "
            "FD001 dùng window 30, FD002/FD004 dùng window 10 và huấn luyện trong từng subset. "
            "Test lịch sử đã được xem; kết quả này là exploratory follow-up. "
            "Ví dụ: giảm test MAE chưa đủ để thay baseline nếu outer gate không đạt."
        )
        st.caption("Tiêu chí FD002/FD004 chỉ xét overall MAE; cần đọc cùng cảnh báo và NASA score trong bảng chi tiết.")
        with st.expander("Paired bootstrap 95% của thí nghiệm bổ sung"):
            st.dataframe(summary, hide_index=True, width="stretch")
            st.dataframe(paired_followup, hide_index=True, width="stretch")
        st.divider()
    st.header("So sánh quy trình chọn model với baseline")
    st.write(
        "Mỗi động cơ ở vòng đánh giá ngoài được giữ riêng khỏi cả huấn luyện và chọn cấu hình. "
        "Các phương án dùng cùng cycle từ 30 trở đi."
    )
    if not experiment.decision.get("observed_improvement_meets_plan", False):
        _card("Kết luận", "Chưa đạt đầy đủ tiêu chí cải thiện đã chốt. Model mặc định vẫn là baseline.", kind="limit")

    st.subheader("Đánh giá toàn bộ trajectory của 100 train engines")
    columns = ["stage", "mean_engine_mae", "low_rul_engine_mae", "late_or_missed_count", "early_alarm_burden"]
    ablation = experiment.ablation[columns].assign(stage=lambda frame: frame["stage"].map(STAGE_LABELS))
    st.dataframe(ablation.rename(columns={"stage": "Phương án", **METRIC_LABELS}), hide_index=True, width="stretch")
    st.caption(
        "Tỷ lệ thời gian báo sớm: tỷ lệ cycle có cảnh báo khi RUL thật còn trên 30, tính riêng từng engine rồi lấy trung bình. "
        "Ngưỡng nghiên cứu là 30 cycles."
    )

    st.subheader("Chênh lệch so với baseline và khoảng tin cậy 95%")
    paired = experiment.paired.loc[experiment.paired["stage"].eq("weights")].assign(
        Metric=lambda frame: frame["metric"].map(METRIC_LABELS).fillna(frame["metric"])
    )
    st.dataframe(
        paired[["Metric", "baseline", "candidate", "delta", "ci_lower", "ci_upper", "n_engines"]].rename(
            columns={"baseline": "Baseline", "candidate": "Quy trình chọn", "delta": "Chênh lệch", "ci_lower": "CI 95% dưới", "ci_upper": "CI 95% trên", "n_engines": "Engines"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Chênh lệch = quy trình chọn model − baseline. Giá trị âm là giảm. Bootstrap lấy lại cùng engine cho hai phương án; "
        "khoảng tin cậy chưa bao gồm biến thiên do huấn luyện lại."
    )

    st.subheader("RUL theo từng cycle ở vòng đánh giá ngoài")
    predictions = experiment.predictions
    engine_id = _engine_select("Chọn train engine", sorted(predictions["engine_id"].unique()), key="improvement_engine")
    engine = predictions.loc[predictions["engine_id"].eq(engine_id)]
    wide = (
        engine.loc[engine["stage"].isin(["baseline", "weights"])]
        .pivot_table(index="cycle", columns="stage", values="y_pred")
        .join(engine.drop_duplicates("cycle").set_index("cycle")["y_true"])
        .reset_index()
    )
    st.altair_chart(
        _rul_trajectory_chart(wide, {"y_true": "RUL thật", "baseline": STAGE_LABELS["baseline"], "weights": STAGE_LABELS["weights"]}),
        width="stretch",
    )

    st.subheader("Đánh giá tại điểm cuối của 100 test engines")
    test_columns = {
        "model": "Model",
        "mean_engine_mae": "Mean engine MAE",
        "rmse": "RMSE",
        "nasa_score": "NASA score",
        "low_rul_engine_mae": "MAE khi RUL ≤ 30",
        "n_low_rul_engines": "Engines có RUL ≤ 30",
    }
    st.dataframe(
        experiment.test[[name for name in test_columns if name in experiment.test]].rename(columns=test_columns),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Test FD001 đã được xem trong phân tích trước đợt này. Không dùng kết quả test để chọn lại cấu hình. "
        "MAE tại một điểm cuối khác MAE trên toàn trajectory."
    )

    with st.expander("Các đánh đổi khi chọn cấu hình (inner CV)"):
        selection = experiment.selection
        examples = selection.loc[
            (selection["stage"].eq("parameters") & selection["candidate_id"].isin(["baseline", "hgb_08"]))
            | (selection["stage"].eq("weights") & selection["candidate_id"].eq("strong_low_rul"))
        ]
        st.dataframe(
            examples[["stage", "candidate_id", "low_rul_engine_mae", "mean_engine_mae", "late_or_missed_rate", "early_alarm_burden", "selected"]].rename(
                columns={"stage": "Giai đoạn", "candidate_id": "Ứng viên", "selected": "Được chọn", **METRIC_LABELS}
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Ví dụ trong inner CV trên toàn train. Một cấu hình có thể giảm sai số RUL thấp nhưng tăng chỉ số khác. "
            "Đây là kết quả dùng để chọn cấu hình, chưa phải đánh giá ngoài độc lập của các ứng viên này."
        )
    with st.expander("Cấu hình cuối"):
        st.json(experiment.run_info.get("selected_final", {}))


def page_timeseries(data: DashboardData) -> None:
    st.header("Health Index và ARIMA forecast")
    metrics = data.forecast_metrics
    engine_id = _engine_select("Chọn validation engine", sorted(metrics["engine_id"].unique()), key="ts_engine")
    history, forecast = engine_timeseries_view(data, engine_id)
    metric = metrics.loc[metrics["engine_id"] == engine_id]
    diagnostic = data.diagnostics.loc[data.diagnostics["engine_id"] == engine_id]
    if not metric.empty:
        row = metric.iloc[0]
        _metric_row(
            [
                ("ARIMA order", str(row.get("order", "n/a"))),
                ("MAE holdout", format_metric(row["mae"])),
                ("Coverage 95%", f"{float(row['coverage_95']) * 100:.2f}%"),
                ("ADF p-value", format_metric(diagnostic.iloc[0]["adf_pvalue"]) if not diagnostic.empty else "n/a"),
            ]
        )

    base = alt.Chart(history).mark_line(color=PALETTE["muted"]).encode(
        x=alt.X("cycle:Q", title="Cycle"), y=alt.Y("health_index:Q", title="Health Index", scale=alt.Scale(zero=False)),
        tooltip=["cycle:Q", alt.Tooltip("health_index:Q", format=".4f")],
    )
    layers = [base]
    if not forecast.empty:
        band = alt.Chart(forecast).mark_area(opacity=0.18, color=PALETTE["alert"]).encode(x="cycle:Q", y="lower_95:Q", y2="upper_95:Q")
        actual = alt.Chart(forecast).mark_line(color=PALETTE["alt"]).encode(x="cycle:Q", y="health_index_true:Q")
        predicted = alt.Chart(forecast).mark_line(color=PALETTE["alert"], strokeDash=[5, 3]).encode(
            x="cycle:Q", y="health_index_pred:Q", tooltip=["cycle:Q", alt.Tooltip("health_index_pred:Q", format=".4f")]
        )
        layers = [band, base, actual, predicted]
    st.altair_chart(alt.layer(*layers).properties(height=360).interactive(bind_y=False), width="stretch")
    st.caption("Xám: Health Index trên toàn bộ trajectory. Cam: giá trị thật trong holdout. Đỏ nét đứt: ARIMA forecast, vùng tô là khoảng dự đoán 95%.")

    left, right = st.columns([1, 1.2])
    with left:
        comparison = forecast_comparison(data, engine_id).assign(method=lambda frame: frame["method"].map(FORECAST_METHOD_LABELS).fillna(frame["method"]))
        chart = (
            alt.Chart(comparison)
            .mark_bar()
            .encode(
                y=alt.Y("method:N", title=None, sort=list(comparison["method"])),
                x=alt.X("mae:Q", title="MAE trên holdout"),
                color=alt.condition(alt.datum.method == "ARIMA", alt.value(PALETTE["alert"]), alt.value(PALETTE["muted"])),
                tooltip=["method:N", alt.Tooltip("mae:Q", format=".4f"), alt.Tooltip("rmse:Q", format=".4f")],
            )
            .properties(height=alt.Step(36))
        )
        st.markdown("**ARIMA so với naive baseline trên cùng holdout**")
        st.altair_chart(chart, width="stretch")
    with right:
        _card(
            "Giới hạn diễn giải",
            "Health Index forecast là phân tích chuỗi thời gian phụ, không phải dự đoán RUL, và không được dùng để suy ra lịch bảo trì. "
            "Diagnostics được đọc trực tiếp từ artifact.",
            kind="limit",
        )
    with st.expander("Diagnostics của engine"):
        st.dataframe(diagnostic, width="stretch", hide_index=True)


def page_provenance(data: DashboardData) -> None:
    st.header("Provenance và data contract")
    _card(
        "Read-only contract",
        "Dashboard đọc kết quả modeling, timeseries, EDA, cross-dataset và đợt thử nghiệm cải thiện đã hoàn tất. "
        "Nó không mở raw test labels và không huấn luyện lại model.",
    )
    _metric_row(
        [
            ("Evaluation rows", f"{len(data.model_evaluation):,}"),
            ("Forecast rows", f"{len(data.forecasts):,}"),
            ("Model schema", str(data.model_run_info.get("schema_version", "n/a"))),
            ("Timeseries schema", str(data.timeseries_run_info.get("schema_version", "n/a"))),
        ]
    )
    with st.expander("Modeling run_info.json", expanded=True):
        st.json(data.model_run_info)
    with st.expander("Timeseries run_info.json"):
        st.json(data.timeseries_run_info)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _hero(data: DashboardData) -> None:
    n_engines = int(data.model_metrics.get("metrics", {}).get("n_engines", len(data.model_evaluation)))
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">Phân tích dữ liệu lớn · Artifact-driven model review</div>
          <div class="hero-title">NASA C-MAPSS FD001 — Remaining Useful Life</div>
          <div class="hero-subtitle">Dự đoán RUL theo engine, cảnh báo RUL thấp và Health Index forecast từ các artifact đã sinh sẵn.</div>
          <span class="pill">read-only</span><span class="pill">{n_engines} test engines</span><span class="pill">model: {_model_label(data.selected_model)}</span>
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
    st.set_page_config(page_title="C-MAPSS FD001 | RUL dashboard", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    root = str(ROOT)
    try:
        data = _dashboard_data(root)
    except DashboardDataError as exc:
        st.error(str(exc))
        st.info("Hãy chạy ETL, rolling features, modeling và PCA--ARIMA trước khi mở dashboard.")
        return

    pages: dict[str, Callable[[], None]] = {
        "Tổng quan": lambda: page_overview(data),
        "Dữ liệu & EDA": lambda: page_exploration(_exploration_data(root)),
        "Đánh giá RUL": lambda: page_rul(data),
        "Cảnh báo sớm": lambda: page_warning(data),
        "Cross-dataset": lambda: page_cross_dataset(_cross_dataset(root)),
        "Thử nghiệm cải thiện": lambda: page_improvement(_improvement_data(root)),
        "Health Index / ARIMA": lambda: page_timeseries(data),
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
    st.sidebar.markdown("**Run hiện tại**")
    st.sidebar.caption("Artifact đã sinh sẵn. Dashboard không fit lại model.")
    st.sidebar.markdown(f"Model: **{_model_label(data.selected_model)}**")
    st.sidebar.markdown(f"Test engines: **{len(data.model_evaluation):,}**")

    _hero(data)
    _guarded(pages[page])
