"""Artifact-driven Streamlit dashboard for the FD001 RUL study.

The dashboard reads persisted outputs only. It never fits a model, reads raw
test labels, or changes the evaluation protocol when a page is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


class DashboardDataError(RuntimeError):
    """Raised when a dashboard input artifact is missing or malformed."""


@dataclass(frozen=True)
class DashboardData:
    """Loaded, persisted artifacts consumed by the dashboard."""

    model_metrics: dict[str, Any]
    model_comparison: pd.DataFrame
    model_predictions: pd.DataFrame
    model_evaluation: pd.DataFrame
    health_index: pd.DataFrame
    forecast_metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    forecasts: pd.DataFrame
    model_run_info: dict[str, Any]
    timeseries_run_info: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DashboardDataError(f"Missing dashboard artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"Could not read dashboard artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise DashboardDataError(f"Dashboard JSON must contain an object: {path}")
    return payload


def _read_table(path: Path, *, parquet: bool = False) -> pd.DataFrame:
    if not path.is_file():
        raise DashboardDataError(f"Missing dashboard artifact: {path}")
    try:
        frame = pd.read_parquet(path) if parquet else pd.read_csv(path)
    except (OSError, ValueError, ImportError) as exc:
        raise DashboardDataError(f"Could not read dashboard artifact: {path}") from exc
    if frame.empty:
        raise DashboardDataError(f"Dashboard artifact is empty: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise DashboardDataError(f"{source} is missing columns: {', '.join(missing)}")


def load_dashboard_data(repo_root: str | Path = ".") -> DashboardData:
    """Load dashboard inputs from persisted modeling and time-series outputs."""

    root = Path(repo_root).resolve()
    modeling = root / "outputs" / "modeling"
    timeseries = root / "outputs" / "timeseries"
    model_predictions = _read_table(modeling / "predictions" / "final_predictions.parquet", parquet=True)
    model_evaluation = _read_table(modeling / "predictions" / "final_evaluation.parquet", parquet=True)
    model_comparison = _read_table(modeling / "tables" / "model_metrics.csv")
    health_index = _read_table(timeseries / "tables" / "health_index.csv")
    forecast_metrics = _read_table(timeseries / "tables" / "forecast_metrics.csv")
    diagnostics = _read_table(timeseries / "tables" / "diagnostics.csv")
    forecasts = _read_table(timeseries / "predictions" / "arima_forecasts.parquet", parquet=True)

    _require_columns(
        model_predictions,
        {"split", "engine_id", "cycle", "model_name", "rul_pred"},
        "final_predictions.parquet",
    )
    _require_columns(
        model_evaluation,
        {"split", "engine_id", "cycle", "y_true", "y_pred", "absolute_error"},
        "final_evaluation.parquet",
    )
    _require_columns(model_comparison, {"model_name", "mean_engine_mae", "selected"}, "model_metrics.csv")
    _require_columns(health_index, {"split", "engine_id", "cycle", "health_index"}, "health_index.csv")
    _require_columns(forecast_metrics, {"engine_id", "mae", "rmse", "coverage_95", "status"}, "forecast_metrics.csv")
    _require_columns(diagnostics, {"engine_id", "adf_pvalue", "ljung_box_pvalue", "status"}, "diagnostics.csv")
    _require_columns(
        forecasts,
        {"engine_id", "cycle", "health_index_true", "health_index_pred", "lower_95", "upper_95"},
        "arima_forecasts.parquet",
    )

    return DashboardData(
        model_metrics=_read_json(modeling / "tables" / "final_metrics.json"),
        model_comparison=model_comparison,
        model_predictions=model_predictions,
        model_evaluation=model_evaluation,
        health_index=health_index,
        forecast_metrics=forecast_metrics,
        diagnostics=diagnostics,
        forecasts=forecasts,
        model_run_info=_read_json(modeling / "run_info.json"),
        timeseries_run_info=_read_json(timeseries / "run_info.json"),
    )


def format_metric(value: float | int, decimals: int = 4) -> str:
    """Format a numeric value consistently for dashboard metric cards."""

    return f"{float(value):,.{decimals}f}"


def selected_engine_ids(data: DashboardData) -> list[int]:
    """Return deterministic engine choices available in persisted artifacts."""

    ids = pd.concat(
        [data.model_evaluation["engine_id"], data.health_index["engine_id"]],
        ignore_index=True,
    ).dropna()
    return sorted({int(engine_id) for engine_id in ids})


def engine_rul_view(data: DashboardData, engine_id: int) -> pd.DataFrame:
    """Return the final RUL evaluation row for one test engine."""

    view = data.model_evaluation.loc[data.model_evaluation["engine_id"] == engine_id].copy()
    return view.sort_values(["split", "cycle"]).reset_index(drop=True)


def engine_timeseries_view(data: DashboardData, engine_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Health Index history and ARIMA holdout forecast for one engine."""

    history = data.health_index.loc[data.health_index["engine_id"] == engine_id].copy()
    forecast = data.forecasts.loc[data.forecasts["engine_id"] == engine_id].copy()
    return (
        history.sort_values("cycle").reset_index(drop=True),
        forecast.sort_values("cycle").reset_index(drop=True),
    )


def _render_dashboard() -> None:
    import matplotlib.pyplot as plt
    import streamlit as st

    st.set_page_config(
        page_title="C-MAPSS FD001 | RUL dashboard",
        page_icon="R",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1380px;}
        [data-testid="stSidebar"] {border-right: 1px solid #e2e8f0;}
        [data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e2e8f0; padding: 14px 16px; border-radius: 12px;}
        [data-testid="stMetricLabel"] {color: #52606d; font-size: 0.82rem;}
        [data-testid="stMetricValue"] {color: #0f243e; font-size: 1.65rem;}
        .hero {background: linear-gradient(115deg, #0f243e 0%, #1f7a8c 100%); border-radius: 18px; padding: 30px 34px; color: white; margin-bottom: 26px;}
        .hero-kicker {font-size: 0.75rem; letter-spacing: 0.14em; text-transform: uppercase; color: #bfe7ea; font-weight: 700;}
        .hero-title {font-size: 2.35rem; line-height: 1.1; font-weight: 750; margin-top: 8px;}
        .hero-subtitle {font-size: 1.08rem; color: #e4f5f6; margin-top: 10px; max-width: 850px;}
        .hero-meta {margin-top: 18px; font-size: 0.84rem; color: #d8f2f3;}
        .pill {display: inline-block; border: 1px solid rgba(255,255,255,0.35); border-radius: 999px; padding: 5px 10px; margin-right: 8px;}
        .section-kicker {font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; color: #1f7a8c; font-weight: 750; margin-top: 12px;}
        .insight {background: #eef7f7; border-left: 4px solid #1f7a8c; border-radius: 10px; padding: 16px 18px; color: #16202a;}
        .warning-card {background: #fff6e9; border-left: 4px solid #f28e2b; border-radius: 10px; padding: 15px 18px; color: #5f3b10;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        data = load_dashboard_data(Path(__file__).resolve().parent)
    except DashboardDataError as exc:
        st.error(str(exc))
        st.info("Hãy chạy ETL, rolling features, modeling và PCA--ARIMA trước khi mở dashboard.")
        return

    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Big Data · Artifact-driven model review</div>
          <div class="hero-title">C-MAPSS FD001</div>
          <div class="hero-subtitle">RUL prediction theo engine và Health Index forecast trên dữ liệu train-only.</div>
          <div class="hero-meta"><span class="pill">read-only</span><span class="pill">100 test engines</span><span class="pill">model: HistGradientBoosting</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = ["Tổng quan", "Đánh giá RUL", "Health Index / ARIMA", "Provenance"]
    st.sidebar.markdown("### Điều hướng")
    page = st.sidebar.radio("Trang", pages, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.markdown("**Run hiện tại**")
    st.sidebar.caption("Artifact đã sinh sẵn. Dashboard không fit lại model.")
    st.sidebar.metric("Model", data.model_run_info.get("selected_model", "n/a"))
    st.sidebar.metric("RUL engines", f"{len(data.model_evaluation):,}")

    metrics = data.model_metrics.get("metrics", {})
    if page == "Tổng quan":
        st.markdown('<div class="section-kicker">Tổng quan</div>', unsafe_allow_html=True)
        st.header("Kết quả modeling cuối")
        cards = st.columns(4)
        cards[0].metric("Mean engine MAE", format_metric(metrics.get("mean_engine_mae", 0)), help="Primary selection metric; lower is better.")
        cards[1].metric("RMSE", format_metric(metrics.get("rmse", 0)), help="Root mean squared error on 100 test engines.")
        cards[2].metric("NASA score", format_metric(metrics.get("nasa_score", 0)), help="Asymmetric NASA scoring function.")
        cards[3].metric("Test engines", f"{int(metrics.get('n_engines', 0)):,}")

        left, right = st.columns([1.35, 1])
        with left:
            st.image(
                str(Path(__file__).resolve().parent / "outputs" / "modeling" / "figures" / "final_true_vs_predicted.png"),
                caption="Final RUL evaluation: true versus predicted",
                width="stretch",
            )
        with right:
            st.markdown('<div class="insight"><b>Model finding</b><br>HistGradientBoosting đạt MAE theo engine thấp nhất trong protocol 5-fold GroupKFold. Predictions được tạo trước khi join test labels.</div>', unsafe_allow_html=True)
            st.markdown("#### So sánh ứng viên")
            comparison = data.model_comparison[["model_name", "mean_engine_mae", "selected"]].copy()
            comparison["mean_engine_mae"] = comparison["mean_engine_mae"].round(4)
            comparison = comparison.rename(columns={"model_name": "Model", "mean_engine_mae": "Mean engine MAE", "selected": "Selected"})
            st.dataframe(comparison, hide_index=True, width="stretch")

        st.markdown('<div class="section-kicker">PCA--ARIMA train-only</div>', unsafe_allow_html=True)
        st.header("Health Index và forecast phụ")
        ts_cards = st.columns(3)
        ts_cards[0].metric("Health Index rows", f"{len(data.health_index):,}")
        ts_cards[1].metric("Forecast rows", f"{len(data.forecasts):,}")
        ts_cards[2].metric("Engines fit thành công", f"{int((data.forecast_metrics['status'] == 'ok').sum())}/{len(data.forecast_metrics)}")
        st.image(
            str(Path(__file__).resolve().parent / "outputs" / "timeseries" / "figures" / "arima_forecasts.png"),
            caption="Health Index thực, forecast và prediction interval 95%",
            width="stretch",
        )
        st.markdown('<div class="warning-card"><b>Cách đọc</b><br>Health Index forecast là phân tích chuỗi thời gian phụ. Nó không thay thế mô hình RUL và không được dùng để suy ra lịch bảo trì.</div>', unsafe_allow_html=True)
        return

    if page == "Đánh giá RUL":
        st.markdown('<div class="section-kicker">Chi tiết RUL</div>', unsafe_allow_html=True)
        st.header("Final RUL evaluation")
        engine_id = st.selectbox("Chọn test engine", sorted(data.model_evaluation["engine_id"].unique()), format_func=lambda value: f"Engine {int(value):03d}")
        row = engine_rul_view(data, int(engine_id))
        if row.empty:
            st.warning("Không có evaluation row cho engine đã chọn.")
            return
        record = row.iloc[0]
        cards = st.columns(4)
        cards[0].metric("RUL thật", format_metric(record["y_true"], 0))
        cards[1].metric("RUL dự đoán", format_metric(record["y_pred"]))
        cards[2].metric("Absolute error", format_metric(record["absolute_error"]))
        cards[3].metric("Cycle cuối", format_metric(record["cycle"], 0))
        st.markdown(
            f'<div class="insight"><b>Engine {int(engine_id):03d}</b><br>Sai số tuyệt đối là {format_metric(record["absolute_error"])} cycles. Mỗi test engine chỉ có một dòng đánh giá tại cycle cuối.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(row, width="stretch", hide_index=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(data.model_evaluation["y_true"], data.model_evaluation["y_pred"], alpha=0.45, label="All engines")
        ax.scatter([record["y_true"]], [record["y_pred"]], color="tab:red", s=70, label=f"Engine {int(engine_id)}")
        limits = [0, max(data.model_evaluation[["y_true", "y_pred"]].max()) * 1.05]
        ax.plot(limits, limits, linestyle="--", color="black", linewidth=1)
        ax.set(xlabel="True RUL (cycles)", ylabel="Predicted RUL (cycles)", title="Final RUL predictions")
        ax.legend()
        ax.grid(alpha=0.2)
        ax.set_facecolor("#f8fafc")
        st.pyplot(fig, clear_figure=True)
        return

    if page == "Health Index / ARIMA":
        st.markdown('<div class="section-kicker">Time-series phụ</div>', unsafe_allow_html=True)
        st.header("Health Index và ARIMA forecast")
        engine_options = sorted(data.forecast_metrics["engine_id"].unique())
        engine_id = st.selectbox("Chọn validation engine", engine_options, format_func=lambda value: f"Engine {int(value):03d}")
        history, forecast = engine_timeseries_view(data, int(engine_id))
        metric = data.forecast_metrics.loc[data.forecast_metrics["engine_id"] == engine_id]
        diagnostic = data.diagnostics.loc[data.diagnostics["engine_id"] == engine_id]
        cards = st.columns(4)
        if not metric.empty:
            metric_row = metric.iloc[0]
            cards[0].metric("MAE", format_metric(metric_row["mae"]))
            cards[1].metric("RMSE", format_metric(metric_row["rmse"]))
            cards[2].metric("Coverage 95%", f"{float(metric_row['coverage_95']) * 100:.2f}%")
        if not diagnostic.empty:
            cards[3].metric("ADF p-value", format_metric(diagnostic.iloc[0]["adf_pvalue"]))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(history["cycle"], history["health_index"], label="Health Index", color="tab:blue")
        if not forecast.empty:
            ax.plot(forecast["cycle"], forecast["health_index_true"], label="Holdout actual", color="tab:orange")
            ax.plot(forecast["cycle"], forecast["health_index_pred"], label="ARIMA forecast", color="tab:red")
            ax.fill_between(forecast["cycle"], forecast["lower_95"], forecast["upper_95"], color="tab:red", alpha=0.15, label="95% interval")
        ax.set(xlabel="Cycle", ylabel="Health Index", title=f"Engine {int(engine_id)}")
        ax.set_facecolor("#f8fafc")
        ax.grid(alpha=0.2)
        ax.legend()
        st.pyplot(fig, clear_figure=True)
        st.markdown('<div class="warning-card"><b>Giới hạn diễn giải</b><br>Health Index forecast là phân tích chuỗi thời gian phụ, không phải dự đoán RUL. Diagnostics được đọc trực tiếp từ artifact.</div>', unsafe_allow_html=True)
        with st.expander("Diagnostics của engine"):
            st.dataframe(diagnostic, width="stretch", hide_index=True)
        return

    st.markdown('<div class="section-kicker">Traceability</div>', unsafe_allow_html=True)
    st.header("Provenance và data contract")
    st.markdown('<div class="insight"><b>Read-only contract</b><br>Dashboard chỉ đọc artifact trong outputs/modeling và outputs/timeseries. Nó không mở raw test labels và không huấn luyện lại model.</div>', unsafe_allow_html=True)
    validation = st.columns(4)
    validation[0].metric("Evaluation rows", f"{len(data.model_evaluation):,}")
    validation[1].metric("Forecast rows", f"{len(data.forecasts):,}")
    validation[2].metric("Model schema", str(data.model_run_info.get("schema_version", "n/a")))
    validation[3].metric("Timeseries schema", str(data.timeseries_run_info.get("schema_version", "n/a")))
    with st.expander("Modeling run_info.json", expanded=True):
        st.json(data.model_run_info)
    with st.expander("Timeseries run_info.json"):
        st.json(data.timeseries_run_info)


if __name__ == "__main__":
    _render_dashboard()
