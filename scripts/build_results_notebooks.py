"""Build the result-review notebooks 03, 04 and 05.

Each notebook reads persisted artifacts under ``outputs/``, recomputes the
reported numbers with the same package functions the pipeline uses, and asserts
that the recomputation matches the stored artifact. None of them fits a model or
writes to ``outputs/``.

- ``03_fd001_modeling_evaluation.ipynb``: cross-validation, final test metrics,
  bootstrap intervals, error by RUL range and the early-warning rule.
- ``04_fd001_health_index_arima.ipynb``: Health Index trend, ARIMA holdout
  metrics, naive baselines and residual diagnostics.
- ``05_cross_dataset_and_improvement.ipynb``: FD001--FD004 matrix and the nested
  RUL improvement study.

Usage (from the repository root):
    .venv/Scripts/python.exe scripts/build_results_notebooks.py [--execute] [--only 03]
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import nbformat as nbf

from cmapss.project import add_notebook_identity

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


class NotebookBuilder:
    """Collect markdown and code cells in order."""

    def __init__(self) -> None:
        self.cells: list = []

    def md(self, text: str) -> None:
        self.cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(self, text: str) -> None:
        self.cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

    def setup(self, imports: str) -> None:
        """First code cell: shared setup, package imports, then restore inline plots.

        Importing ``cmapss.modeling`` or ``cmapss.timeseries`` runs
        ``matplotlib.use("Agg")`` in the pipeline modules, so the inline backend
        is re-selected after those imports.
        """
        parts = (textwrap.dedent(SETUP).strip(), textwrap.dedent(imports).strip(), "%matplotlib inline")
        self.code("\n\n".join(parts))

    def notebook(self) -> nbf.NotebookNode:
        notebook = nbf.v4.new_notebook(cells=self.cells)
        notebook.metadata["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
        notebook.metadata["language_info"] = {"name": "python"}
        add_notebook_identity(notebook)
        return notebook


SETUP = """
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "outputs").is_dir():
    ROOT = ROOT.parent  # notebook được mở từ thư mục notebooks/
assert (ROOT / "outputs").is_dir(), "Chạy notebook từ thư mục gốc repository"

pd.set_option("display.precision", 4)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))
"""


def build_modeling() -> nbf.NotebookNode:
    nb = NotebookBuilder()
    nb.md("""
    # Đánh giá mô hình RUL và quy tắc cảnh báo sớm trên FD001

    Notebook này rà soát kết quả của pipeline modeling đã lưu trong
    `outputs/modeling/`. Mỗi con số được tính lại từ bảng prediction bằng chính các
    hàm của package `cmapss.modeling` rồi so với artifact bằng `assert`. Notebook
    không fit model và không ghi file.

    Nội dung:

    1. So sánh ba model ứng viên trong 5-fold GroupKFold theo engine.
    2. Tính lại metric test và khoảng tin cậy bootstrap của model được chọn.
    3. Phân tích sai số theo khoảng RUL thật, trên test và trên OOF train.
    4. Tính lại bảng cảnh báo RUL ≤ 30 cycles từ OOF predictions.

    ```text
    rolling_features.parquet ─┐
    train_targets.parquet ────┴─> GroupKFold (OOF) ─> chọn model ─> fit toàn train ─> dự đoán test
                                        │                                                │
                                        └─> oof_predictions.parquet                      └─> final_evaluation.parquet
                                              └─> warning_events.csv                           (join RUL_FD001 sau khi dự đoán)
    ```
    """)
    nb.setup("""
from cmapss.modeling.early_warning import build_warning_events
from cmapss.modeling.metrics import bootstrap_metric_intervals, summarize_prediction_errors

MODELING = ROOT / "outputs" / "modeling"
run_info = read_json("outputs/modeling/run_info.json")
model_metrics = pd.read_csv(MODELING / "tables" / "model_metrics.csv")
fold_metrics = pd.read_csv(MODELING / "tables" / "fold_metrics.csv")
oof = pd.read_parquet(MODELING / "predictions" / "oof_predictions.parquet")
evaluation = pd.read_parquet(MODELING / "predictions" / "final_evaluation.parquet")
final_metrics = read_json("outputs/modeling/tables/final_metrics.json")
selected = run_info["selected_model"]
print("Model được chọn:", selected)
print("OOF rows:", len(oof), "| test evaluation rows:", len(evaluation))
""")

    nb.md("""
    ## 1. So sánh model trong cross-validation

    Mỗi fold giữ riêng 20 train engines. Metric chọn model là `mean_engine_mae`:
    tính MAE trong từng engine rồi lấy trung bình, để engine dài không chiếm trọng số
    lớn hơn engine ngắn. Ô dưới đây tính lại bảng tổng hợp từ OOF predictions và so với
    `model_metrics.csv`.
    """)
    nb.code("""
recomputed = []
for (model_name, fold), group in oof.groupby(["model_name", "fold"], sort=True):
    recomputed.append({"model_name": model_name, "fold": fold, **summarize_prediction_errors(group)})
recomputed = pd.DataFrame(recomputed)

stored = fold_metrics.set_index(["model_name", "fold"]).sort_index()
check = recomputed.set_index(["model_name", "fold"]).sort_index()
for column in ("mean_engine_mae", "rmse", "nasa_score"):
    assert np.allclose(check[column], stored[column], rtol=1e-9), column

summary = recomputed.groupby("model_name")["mean_engine_mae"].agg(["mean", "std"])
assert np.allclose(summary["mean"], model_metrics.set_index("model_name").loc[summary.index, "mean_engine_mae"])
print("Fold metrics khớp với artifact.")
summary.sort_values("mean")
""")
    nb.code("""
fig, ax = plt.subplots(figsize=(7, 3.4))
for model_name, group in recomputed.groupby("model_name"):
    ax.plot(group["fold"], group["mean_engine_mae"], marker="o", label=model_name)
ax.set(xlabel="Fold", ylabel="Mean engine MAE (cycles)", title="MAE theo fold", xticks=range(1, 6))
ax.legend()
plt.show()
""")
    nb.md("""
    `hist_gradient_boosting` có MAE thấp nhất ở cả năm fold, nên kết luận chọn model
    không phụ thuộc vào một fold riêng lẻ. Chênh lệch giữa HistGradientBoosting và
    Ridge (khoảng 3.8 cycles) lớn hơn nhiều so với độ lệch chuẩn giữa các fold
    (khoảng 1 cycle). Đây là quan sát mô tả trên 5 fold, không phải kiểm định thống kê.
    """)

    nb.md("""
    ## 2. Metric test và khoảng tin cậy bootstrap

    Test FD001 có 100 engines. Mỗi engine chỉ được đánh giá tại cycle quan sát cuối,
    nên `mean_window_mae` và `mean_engine_mae` bằng nhau. Khoảng tin cậy 95% được tính
    lại bằng cùng seed và số lần lấy mẫu đã ghi trong `final_metrics.json`.
    """)
    nb.code("""
metrics = summarize_prediction_errors(evaluation)
for name, value in final_metrics["metrics"].items():
    assert np.isclose(metrics[name], value, rtol=1e-12), name

bootstrap = final_metrics["bootstrap"]
intervals = bootstrap_metric_intervals(evaluation, n_resamples=bootstrap["resamples"], seed=bootstrap["seed"])
for name, bounds in bootstrap["intervals"].items():
    assert np.isclose(intervals[name]["ci_lower"], bounds["ci_lower"]) and np.isclose(intervals[name]["ci_upper"], bounds["ci_upper"]), name

print("Metric test và khoảng tin cậy khớp với final_metrics.json.")
pd.DataFrame(intervals).T.assign(point=[metrics[name] for name in intervals])[["point", "ci_lower", "ci_upper"]]
""")
    nb.md("""
    MAE trên test (15.35 cycles) thấp hơn MAE trong cross-validation (20.21 cycles).
    Hai con số không so sánh trực tiếp: CV tính trên toàn bộ trajectory từ cycle 30,
    còn test chỉ tính tại điểm cuối của mỗi engine, nơi phân phối RUL thật khác.
    """)

    nb.md("""
    ## 3. Sai số theo khoảng RUL thật

    Residual được định nghĩa là `y_pred - y_true`. Giá trị dương nghĩa là model dự đoán
    RUL dài hơn thực tế; NASA score phạt nhánh này nặng hơn.
    """)
    nb.code("""
bins = [0, 30, 60, 90, 120, 200]
test_view = evaluation.assign(residual=evaluation["y_pred"] - evaluation["y_true"])
test_view["rul_range"] = pd.cut(test_view["y_true"], bins, include_lowest=True)
test_table = test_view.groupby("rul_range", observed=True).agg(
    engines=("engine_id", "size"),
    mae=("absolute_error", "mean"),
    mean_residual=("residual", "mean"),
    share_over=("residual", lambda values: float((values > 0).mean())),
)
test_table
""")
    nb.code("""
oof_selected = oof.loc[oof["model_name"].eq(selected)].assign(residual=lambda frame: frame["y_pred"] - frame["y_true"])
oof_selected["rul_range"] = pd.cut(oof_selected["y_true"], [-1, 30, 60, 90, 120, 150, 400])
oof_table = oof_selected.groupby("rul_range", observed=True).agg(
    rows=("engine_id", "size"), mae=("absolute_error", "mean"), mean_residual=("residual", "mean")
)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].scatter(test_view["y_true"], test_view["residual"], s=18, alpha=0.7)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set(xlabel="RUL thật (cycles)", ylabel="Dự đoán − thật", title="Test: residual tại cycle cuối")
curve = oof_selected.groupby("y_true")["absolute_error"].mean()
axes[1].plot(curve.index, curve.values, linewidth=1)
axes[1].axvline(30, color="gray", linestyle=":")
axes[1].set(xlabel="RUL thật (cycles)", ylabel="MAE (cycles)", title="OOF train: MAE theo RUL thật")
plt.tight_layout()
plt.show()
oof_table
""")
    nb.md("""
    Trên cả test và OOF, sai số nhỏ nhất ở vùng RUL ≤ 30 cycles (MAE test khoảng
    4.4 cycles, OOF khoảng 3.7 cycles) và tăng khi RUL thật lớn. Ở đầu vòng đời, sensor
    gần như chưa suy giảm nên model không đủ thông tin để phân biệt RUL 150 với RUL 250;
    MAE OOF ở nhóm RUL > 150 vượt 49 cycles. Trên test, residual trung bình dương ở mọi
    khoảng RUL, tức model có xu hướng dự đoán RUL dài hơn thực tế ở điểm cuối.
    """)

    nb.md("""
    ## 4. Quy tắc cảnh báo RUL ≤ 30 cycles

    Cảnh báo đầu tiên là cycle đầu tiên có RUL dự đoán OOF không quá 30. Ô dưới đây
    gọi lại `build_warning_events` và so với `warning_events.csv`. OOF cũng đã được dùng
    để chọn model, nên đây là đánh giá nội bộ.
    """)
    nb.code("""
events, warning_summary = build_warning_events(oof, model_name=selected, rul_threshold=30)
stored_events = pd.read_csv(MODELING / "tables" / "warning_events.csv")
pd.testing.assert_frame_equal(
    events.reset_index(drop=True), stored_events[events.columns].reset_index(drop=True), check_dtype=False
)
stored_summary = read_json("outputs/modeling/tables/warning_summary.json")
for key in ("n_engines", "n_warning", "n_missed_warning", "timing_counts", "lead_time_distribution"):
    assert warning_summary[key] == stored_summary[key], key
print("Bảng cảnh báo khớp với artifact.")
pd.Series(warning_summary["timing_counts"], name="engines")
""")
    nb.code("""
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.hist(events["threshold_delay_cycles"], bins=range(-30, 31, 2), color="#1f7a8c")
ax.axvline(0, color="black", linestyle="--")
ax.set(xlabel="Cycle cảnh báo − cycle RUL thật chạm 30", ylabel="Số engine", title="Độ trễ so với ngưỡng thật")
plt.show()
events[["threshold_delay_cycles", "lead_time_to_failure_cycles"]].describe()
""")
    nb.md("""
    ## Tóm tắt

    - Metric CV, metric test, khoảng tin cậy bootstrap và bảng cảnh báo đều tính lại
      khớp với artifact đã lưu.
    - Model được chọn có sai số thấp ở vùng RUL nhỏ và sai số lớn ở đầu vòng đời.
    - Mọi engine train đều có cảnh báo, nhưng 49 engine cảnh báo sau khi RUL thật đã
      xuống dưới 30. Ngưỡng 30 cycles là ngưỡng minh họa, chưa gắn với chi phí bảo trì.
    - Kết quả chỉ áp dụng cho FD001 với protocol đã nêu.
    """)
    return nb.notebook()


def build_timeseries() -> nbf.NotebookNode:
    nb = NotebookBuilder()
    nb.md("""
    # Health Index và ARIMA forecast trên FD001 train

    Đây là phân tích chuỗi thời gian phụ, tách khỏi mô hình RUL. Health Index là thành
    phần chính thứ nhất (PCA) của các sensor đã chuẩn hóa, được đổi dấu để giảm dần theo
    cycle. Với 10 validation engines, phần đầu chuỗi được dùng để fit ARIMA và phần cuối
    làm holdout. Notebook đọc artifact trong `outputs/timeseries/`, tính lại metric
    holdout của ARIMA và hai naive baseline, rồi so với artifact. Notebook không fit
    ARIMA và không ghi file.

    ```text
    Silver observations (train) -> StandardScaler + PCA(1) -> Health Index (health_index.csv)
                                                           -> 10 validation engines: prefix | holdout
                                                                 prefix -> ARIMA(p,d,q) -> forecast + khoảng dự đoán 95%
                                                                 prefix -> last value, drift
    ```
    """)
    nb.setup("""
from scipy.stats import spearmanr

from cmapss.timeseries.baselines import evaluate_forecast_baselines

TS = ROOT / "outputs" / "timeseries"
health_index = pd.read_csv(TS / "tables" / "health_index.csv")
selected_engines = pd.read_csv(TS / "tables" / "selected_engines.csv")
forecast_metrics = pd.read_csv(TS / "tables" / "forecast_metrics.csv")
baseline_metrics = pd.read_csv(TS / "tables" / "baseline_metrics.csv")
diagnostics = pd.read_csv(TS / "tables" / "diagnostics.csv")
forecasts = pd.read_parquet(TS / "predictions" / "arima_forecasts.parquet")
ts_run = read_json("outputs/timeseries/run_info.json")
print("Health Index rows:", len(health_index), "| engines:", health_index["engine_id"].nunique())
print("Dùng test labels:", ts_run["isolation"]["uses_test_labels"])
""")

    nb.md("""
    ## 1. Health Index có giảm theo cycle không?

    Spearman ρ giữa cycle và Health Index được tính riêng cho từng engine. Giá trị gần
    −1 nghĩa là Health Index giảm gần như đơn điệu theo thời gian.
    """)
    nb.code("""
rho = health_index.groupby("engine_id").apply(
    lambda group: spearmanr(group["cycle"], group["health_index"]).statistic, include_groups=False
)
print(rho.describe().round(4))

lifetime = health_index.groupby("engine_id")["cycle"].transform("max")
normalized = health_index.assign(life_fraction=health_index["cycle"] / lifetime)
normalized["bin"] = (normalized["life_fraction"] * 20).clip(upper=19.999).astype(int) / 20
band = normalized.groupby("bin")["health_index"].quantile([0.1, 0.5, 0.9]).unstack()

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.fill_between(band.index, band[0.1], band[0.9], alpha=0.25, label="10–90% engines")
ax.plot(band.index, band[0.5], label="Median")
ax.set(xlabel="Tỷ lệ vòng đời đã qua (cycle / cycle cuối)", ylabel="Health Index", title="Health Index của 100 train engines")
ax.legend()
plt.show()
""")
    nb.md("""
    Spearman ρ của cả 100 engines đều âm, trung vị khoảng −0.93, nên Health Index giảm
    nhất quán theo cycle. Khi chuẩn hóa theo tỷ lệ vòng đời, median Health Index giảm chậm
    trong khoảng 60% vòng đời đầu rồi giảm nhanh ở phần cuối. Diễn giải: holdout của ARIMA
    nằm ở phần cuối chuỗi, nơi độ dốc thay đổi, nên forecast ngoại suy từ prefix có thể lệch
    khỏi xu hướng thực tế.
    """)

    nb.md("""
    ## 2. Tính lại metric holdout của ARIMA

    `arima_forecasts.parquet` lưu giá trị thật, forecast và khoảng dự đoán 95% trên
    holdout. MAE, RMSE và coverage được tính lại rồi so với `forecast_metrics.csv`.
    Coverage là tỷ lệ điểm holdout nằm trong khoảng dự đoán.
    """)
    nb.code("""
error = forecasts["health_index_pred"] - forecasts["health_index_true"]
inside = forecasts["health_index_true"].between(forecasts["lower_95"], forecasts["upper_95"])
arima_check = (
    forecasts.assign(abs_error=error.abs(), sq_error=error**2, inside=inside)
    .groupby("engine_id")
    .agg(mae=("abs_error", "mean"), rmse=("sq_error", lambda values: float(np.sqrt(values.mean()))), coverage_95=("inside", "mean"))
)
stored = forecast_metrics.set_index("engine_id").loc[arima_check.index]
for column in ("mae", "rmse", "coverage_95"):
    assert np.allclose(arima_check[column], stored[column]), column
print("Metric ARIMA khớp với forecast_metrics.csv.")
stored[["order", "n_train", "n_holdout", "mae", "rmse", "coverage_95"]]
""")

    nb.md("""
    ## 3. ARIMA so với naive baseline

    Hai baseline dùng cùng điểm cắt với ARIMA: `last_value` giữ nguyên giá trị cuối của
    prefix, `drift` ngoại suy độ dốc trung bình từ điểm đầu tới điểm cuối của prefix.
    Baseline được tính lại từ Health Index bằng `evaluate_forecast_baselines`.
    """)
    nb.code("""
rows = []
for record in forecast_metrics.itertuples():
    series = health_index.loc[health_index["engine_id"].eq(record.engine_id)].sort_values("cycle")["health_index"].to_numpy()
    fit, holdout = series[: record.n_train], series[record.n_train : record.n_train + record.n_holdout]
    for name, scores in evaluate_forecast_baselines(fit, holdout).items():
        rows.append({"engine_id": record.engine_id, "baseline_name": name, "mae": scores["mae"], "rmse": scores["rmse"]})
baseline_check = pd.DataFrame(rows).set_index(["engine_id", "baseline_name"]).sort_index()
stored_baselines = baseline_metrics.set_index(["engine_id", "baseline_name"]).sort_index()
assert np.allclose(baseline_check[["mae", "rmse"]], stored_baselines[["mae", "rmse"]])
print("Baseline khớp với baseline_metrics.csv.")

comparison = baseline_check["mae"].unstack().join(stored["mae"].rename("arima"))
comparison["arima_best"] = comparison["arima"] < comparison[["last_value", "drift"]].min(axis=1)
print("ARIMA có MAE thấp nhất ở", int(comparison["arima_best"].sum()), "/", len(comparison), "engines")
comparison.round(4)
""")
    nb.code("""
engines = forecast_metrics["engine_id"].tolist()[:4]
fig, axes = plt.subplots(1, len(engines), figsize=(15, 3.4), sharey=False)
for ax, engine_id in zip(axes, engines):
    history = health_index.loc[health_index["engine_id"].eq(engine_id)]
    forecast = forecasts.loc[forecasts["engine_id"].eq(engine_id)]
    ax.plot(history["cycle"], history["health_index"], color="0.6", linewidth=1)
    ax.plot(forecast["cycle"], forecast["health_index_pred"], color="tab:red", linestyle="--")
    ax.fill_between(forecast["cycle"], forecast["lower_95"], forecast["upper_95"], color="tab:red", alpha=0.15)
    ax.set(title=f"Engine {engine_id}", xlabel="Cycle")
axes[0].set_ylabel("Health Index")
plt.tight_layout()
plt.show()
""")
    nb.md("""
    ARIMA có MAE thấp hơn cả hai baseline ở 7/10 validation engines. Trung bình coverage
    của khoảng dự đoán 95% chỉ khoảng 0.67, thấp hơn mức danh nghĩa 0.95. Coverage thấp nhất
    ở engine 70 (0.21) và 39 (0.36). Ở engine 39 (hình trên), forecast gần như đi ngang trong
    khi holdout giảm mạnh, nên phần lớn điểm holdout nằm ngoài khoảng dự đoán.
    """)

    nb.md("""
    ## 4. Diagnostics của chuỗi fit

    - ADF test: p-value lớn nghĩa là không bác bỏ giả thuyết có unit root; chuỗi prefix
      không dừng, phù hợp với xu hướng giảm.
    - Ljung–Box trên residual: p-value nhỏ (< 0.05) cho thấy residual còn tự tương quan,
      tức model chưa nắm hết cấu trúc chuỗi.
    """)
    nb.code("""
table = diagnostics[["engine_id", "order", "adf_pvalue", "ljung_box_pvalue", "fit_converged"]].copy()
table["residual_autocorrelated_5pct"] = table["ljung_box_pvalue"] < 0.05
table.sort_values("engine_id")
""")
    nb.md("""
    ## Tóm tắt

    - Metric ARIMA và hai baseline tính lại đều khớp với artifact.
    - Health Index giảm nhất quán theo cycle trên toàn bộ 100 train engines.
    - ARIMA thường tốt hơn naive baseline về MAE, nhưng khoảng dự đoán 95% thiếu coverage.
      Residual của engine 69 và 70 còn tự tương quan ở mức 5%.
    - Health Index forecast không phải dự đoán RUL và không được dùng để suy ra lịch bảo trì.
    """)
    return nb.notebook()


def build_cross_improvement() -> nbf.NotebookNode:
    nb = NotebookBuilder()
    nb.md("""
    # Cross-dataset FD001–FD004 và thử nghiệm cải thiện RUL

    Notebook gồm hai phần dùng artifact đã lưu:

    1. `outputs/cross_dataset/`: chạy cùng pipeline trên bốn subset C-MAPSS và chuyển
       model FD001 sang FD002–FD004.
    2. `outputs/experiments/rul_improvement/20260928_rebuild/`: nested cross-validation so
       sánh baseline với quy trình chọn window, tham số HGB và trọng số mẫu.

    Mỗi bảng tổng hợp được tính lại từ prediction bằng hàm của package rồi so với
    artifact. Notebook không fit model và không ghi file.
    """)
    nb.setup("""
from cmapss.modeling.improvement_analysis import engine_metrics, paired_comparison
from cmapss.modeling.metrics import summarize_prediction_errors

CROSS = ROOT / "outputs" / "cross_dataset"
matrix = read_json("outputs/cross_dataset/matrix.json")
print("Window cycles:", matrix["window_cycles"], "| evaluations:", len(matrix["evaluations"]))
""")

    nb.md("""
    ## 1. Ma trận evaluation FD001–FD004

    Evaluation nội bộ dùng train và test của cùng subset. Transfer dùng model fit trên
    toàn bộ train FD001 để dự đoán test của subset khác, không fit lại. Matrix dùng window
    10 cycles vì FD004 có test engine ngắn nhất 19 cycles, nên kết quả không thay thế
    kết quả FD001 chính thức với window 30.
    """)
    nb.code("""
rows = []
for item in matrix["evaluations"]:
    folder = CROSS / "evaluations" / f"{item['source_dataset']}_to_{item['target_dataset']}"
    evaluation = pd.read_parquet(folder / "modeling" / "predictions" / "final_evaluation.parquet")
    recomputed = summarize_prediction_errors(evaluation)
    for name in ("mean_engine_mae", "rmse", "nasa_score"):
        assert np.isclose(recomputed[name], item["metrics"][name]), (folder.name, name)
    interval = item["bootstrap"]["intervals"]["mean_engine_mae"]
    rows.append({
        "source": item["source_dataset"], "target": item["target_dataset"], "type": item["evaluation_type"],
        "engines": recomputed["n_engines"], "mean_engine_mae": recomputed["mean_engine_mae"],
        "ci_lower": interval["ci_lower"], "ci_upper": interval["ci_upper"], "rmse": recomputed["rmse"],
    })
cross = pd.DataFrame(rows)
print("Metric của 7 evaluation khớp với matrix.json.")
cross
""")
    nb.code("""
within = cross.loc[cross["type"].eq("within_subset")].set_index("target")["mean_engine_mae"]
transfer = cross.loc[cross["type"].eq("transfer")].set_index("target")
transfer = transfer.assign(within_mae=within.loc[transfer.index], ratio=lambda frame: frame["mean_engine_mae"] / frame["within_mae"])

fig, ax = plt.subplots(figsize=(8, 3.6))
labels = cross["source"] + " → " + cross["target"]
colors = np.where(cross["type"].eq("transfer"), "#f28e2b", "#1f7a8c")
ax.barh(labels, cross["mean_engine_mae"], color=colors,
        xerr=[cross["mean_engine_mae"] - cross["ci_lower"], cross["ci_upper"] - cross["mean_engine_mae"]])
ax.invert_yaxis()
ax.set(xlabel="Mean engine MAE (cycles), khoảng tin cậy 95%", title="Nội bộ subset (xanh) và transfer từ FD001 (cam)")
plt.show()
transfer[["mean_engine_mae", "within_mae", "ratio"]]
""")
    nb.md("""
    Transfer từ FD001 làm MAE tăng khoảng 1.7 đến 2.7 lần so với evaluation nội bộ trên
    cùng target subset, và khoảng tin cậy của transfer không chồng lên khoảng tin cậy của
    evaluation nội bộ. FD002 và FD004 có sáu điều kiện vận hành, FD003 và FD004 có hai dạng
    lỗi, trong khi FD001 chỉ có một điều kiện và một dạng lỗi. Model fit riêng trên FD001
    không nên được dùng trực tiếp cho FD002–FD004 nếu chưa có retraining hoặc domain
    adaptation.
    """)

    nb.md("""
    ## 2. Thử nghiệm cải thiện bằng nested cross-validation

    Vòng ngoài giữ riêng từng nhóm engine để đánh giá. Vòng trong chỉ dùng các engine còn
    lại để chọn lần lượt window, tham số HGB và trọng số mẫu. Mọi phương án được so trên
    cùng các cycle từ 30 trở đi. Bảng `ablation_metrics.csv` và `paired_comparison.csv`
    được tính lại từ `outer_predictions.parquet`.
    """)
    nb.code("""
from cmapss.project import IMPROVEMENT_RUN
RUN = ROOT / IMPROVEMENT_RUN
outer = pd.read_parquet(RUN / "outer_predictions.parquet")
ablation = pd.read_csv(RUN / "ablation_metrics.csv").set_index("stage")
paired_stored = pd.read_csv(RUN / "paired_comparison.csv")
decision = json.loads((RUN / "decision.json").read_text(encoding="utf-8"))
run_info = json.loads((RUN / "run_info.json").read_text(encoding="utf-8"))
resamples, seed = int(paired_stored["resamples"].iloc[0]), int(paired_stored["seed"].iloc[0])

tables = {}
for stage in ("baseline", "windows", "parameters", "weights"):
    frame = outer.loc[outer["stage"].eq(stage)].sort_values(["engine_id", "cycle"])
    tables[stage] = engine_metrics(frame)
    assert np.isclose(tables[stage]["mean_engine_mae"].mean(), ablation.loc[stage, "mean_engine_mae"]), stage
    assert np.isclose(tables[stage]["low_rul_engine_mae"].mean(), ablation.loc[stage, "low_rul_engine_mae"]), stage

paired = pd.concat(
    [paired_comparison(tables["baseline"], tables[stage], resamples, seed).assign(stage=stage) for stage in ("windows", "parameters", "weights")],
    ignore_index=True,
)
pd.testing.assert_frame_equal(
    paired[["stage", "metric", "delta", "ci_lower", "ci_upper"]],
    paired_stored[["stage", "metric", "delta", "ci_lower", "ci_upper"]],
    check_dtype=False,
)
print("Ablation và paired comparison khớp với artifact.")
ablation[["mean_engine_mae", "low_rul_engine_mae", "late_or_missed_count", "early_alarm_burden"]]
""")
    nb.code("""
final = paired.loc[paired["stage"].eq("weights")].set_index("metric")
fig, ax = plt.subplots(figsize=(7, 3))
ax.errorbar(final["delta"], final.index, xerr=[final["delta"] - final["ci_lower"], final["ci_upper"] - final["delta"]], fmt="o", capsize=4)
ax.axvline(0, color="black", linewidth=0.8)
ax.set(xlabel="Quy trình chọn − baseline (âm là tốt hơn)", title="Chênh lệch theo engine, khoảng tin cậy 95%")
plt.show()
print(json.dumps(decision, indent=2, ensure_ascii=False))
final[["baseline", "candidate", "delta", "ci_lower", "ci_upper"]]
""")
    nb.md("""
    Quy trình chọn cấu hình giữ nguyên window 30 và cuối cùng không cải thiện metric
    chính: MAE toàn bộ và MAE khi RUL ≤ 30 tăng rất nhẹ, khoảng tin cậy chứa 0. Tỷ lệ cảnh
    báo trễ hoặc bỏ lỡ tăng từ 0.49 lên 0.53, với khoảng tin cậy nằm hoàn toàn trên 0.
    Vì vậy `decision.json` ghi `observed_improvement_meets_plan = false` và model mặc định
    vẫn là baseline. Khoảng tin cậy chỉ lấy lại engine trên prediction đã lưu, chưa gồm biến
    thiên do huấn luyện lại.
    """)
    nb.code("""
print("Cấu hình cuối:")
print(json.dumps(run_info.get("selected_final", {}), indent=2, ensure_ascii=False))
pd.read_csv(RUN / "test_comparison.csv")[["model", "mean_engine_mae", "rmse", "nasa_score", "low_rul_engine_mae"]]
""")
    nb.md("""
    ## Tóm tắt

    - Bảy evaluation cross-dataset và toàn bộ bảng của thử nghiệm cải thiện gốc tính lại
      khớp với artifact.
    - Transfer từ FD001 sang subset khác làm sai số tăng rõ rệt.
    - Nested CV không tìm được cấu hình tốt hơn baseline theo tiêu chí đã chốt trước;
      đây là negative result và được giữ nguyên trong báo cáo.
    """)
    nb.md("""
    ## Thí nghiệm bổ sung ngày 29/09/2026

    FD001 kiểm tra capped RUL 110/125/140 và 42 history features; FD002/FD004
    kiểm tra operational settings và chuẩn hóa theo sáu điều kiện vận hành.
    Mỗi normalizer chỉ học từ training fold. Chọn cấu hình bằng 3 inner fold,
    đánh giá bằng 5 outer fold theo engine. Ví dụ, giảm test MAE chưa đủ để thay
    baseline nếu điều kiện cải thiện trên outer CV chưa đạt. Test đã được xem
    trong nghiên cứu trước, nên đây là exploratory follow-up.
    """)
    nb.code("""
from cmapss.followup_results import load_followup_results
from cmapss.project import FOLLOWUP_RUN

followup, paired_followup = load_followup_results(ROOT)
for dataset in followup.dataset:
    run = ROOT / FOLLOWUP_RUN / dataset
    predictions = pd.read_parquet(run / "outer_predictions.parquet")
    scores = {stage: engine_metrics(predictions.loc[predictions.stage.eq(stage)])
              for stage in ("baseline", "selected")}
    recalculated = paired_comparison(scores["baseline"], scores["selected"], 10000, 42)
    pd.testing.assert_frame_equal(recalculated, pd.read_csv(run / "paired_comparison.csv"),
                                  check_exact=False, rtol=1e-10)
print("Paired comparison của cả ba thí nghiệm khớp với dự đoán đã lưu.")
followup
""")
    nb.code("""
paired_followup[["dataset", "metric", "delta", "ci_lower", "ci_upper"]]
""")
    return nb.notebook()


TARGETS = {
    "03": ("03_fd001_modeling_evaluation.ipynb", build_modeling),
    "04": ("04_fd001_health_index_arima.ipynb", build_timeseries),
    "05": ("05_cross_dataset_and_improvement.ipynb", build_cross_improvement),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true", help="execute notebooks with nbclient before saving")
    parser.add_argument("--only", choices=sorted(TARGETS), action="append", help="build only the given notebook number")
    args = parser.parse_args()
    for key in args.only or sorted(TARGETS):
        filename, build = TARGETS[key]
        notebook = build()
        if args.execute:
            from nbclient import NotebookClient

            NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
        path = NOTEBOOKS / filename
        nbf.write(notebook, path)
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
