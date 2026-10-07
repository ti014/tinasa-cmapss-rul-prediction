"""Generate follow-up delivery tables directly from completed experiment artifacts."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from cmapss.followup_results import load_followup_results
from cmapss.project import AUTHOR_NAME, STUDENT_ID, FOLLOWUP_RUN
from cmapss.provenance import verify_source_snapshot


ROOT = Path(__file__).resolve().parents[1]


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact artifact table without an optional tabulate dependency."""
    def cell(value):
        if isinstance(value, float):
            return f"{value:.4e}" if abs(value) >= 1e7 else f"{value:.4f}"
        return str(value)
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                      "| " + " | ".join(["---"] * len(frame.columns)) + " |",
                      *["| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None)]])


def main() -> None:
    summary, paired = load_followup_results(ROOT)
    for dataset in summary.dataset:
        validation = json.loads((ROOT / FOLLOWUP_RUN / dataset / "validation.json").read_text(encoding="utf-8"))
        assert validation["status"] == "passed", dataset
    benchmark_root = ROOT / "outputs/benchmarks/20260929"
    metadata = json.loads((benchmark_root / "run_info.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert len(metadata["parity"]) == 3
    snapshot_manifest = benchmark_root / "source_snapshot_manifest.json"
    if not snapshot_manifest.exists():
        study = json.loads((ROOT / FOLLOWUP_RUN / "FD001/search_manifest.json").read_text(encoding="utf-8"))["identity"]
        sources = {f"src/cmapss/etl/{name}": digest for name, digest in metadata["source_code_sha256"].items()}
        sources.update({name: digest for name, digest in study["source_sha256"].items()
                        if name.startswith(("src/cmapss/data/", "src/cmapss/features/"))})
        payloads = {}
        for name, digest in {**sources, "uv.lock": study["dependencies_sha256"]}.items():
            payload = (ROOT / name).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == digest, name
            payloads[name] = payload
        for name, payload in payloads.items():
            destination = benchmark_root / "source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != payload:
                raise ValueError(f"Refusing to overwrite a different snapshot: {name}")
            destination.write_bytes(payload)
        snapshot_manifest.write_text(json.dumps({"identity": {"source_sha256": sources,
            "dependencies_sha256": study["dependencies_sha256"]},
            "scope": "Four benchmark source hashes recorded by runner; transitive data/features hashes verified against the earlier frozen study; archived after measurement."}, indent=2), encoding="utf-8")
    verify_source_snapshot(benchmark_root, json.loads(snapshot_manifest.read_text(encoding="utf-8")), ROOT)
    measurements = pd.read_csv(benchmark_root / "measurements.csv")
    assert len(measurements) == 12
    timing = measurements.groupby(["engine", "scale"], as_index=False).median(numeric_only=True)
    timing["peak_rss_gib"] = timing.peak_process_tree_rss_bytes / 1024 ** 3
    timing = timing[["engine", "scale", "startup_seconds", "read_seconds", "summary_seconds",
                     "rolling_seconds", "write_seconds", "pipeline_seconds", "peak_rss_gib"]]
    summary.to_csv(ROOT / FOLLOWUP_RUN / "summary.csv", index=False)
    paired.to_csv(ROOT / FOLLOWUP_RUN / "paired_summary.csv", index=False)
    timing.to_csv(benchmark_root / "medians.csv", index=False)
    old = pd.read_parquet(ROOT / "outputs/modeling/predictions/final_evaluation.parquet")
    new = pd.read_parquet(ROOT / FOLLOWUP_RUN / "FD001/predictions/final_evaluation.parquet")
    columns = ["engine_id", "y_pred", "absolute_error", "nasa_score_component"]
    errors = old[["y_true", *columns]].merge(new[columns], on="engine_id", suffixes=("_baseline", "_new"), validate="one_to_one")
    errors.to_csv(ROOT / FOLLOWUP_RUN / "FD001/test_error_comparison.csv", index=False)
    worst = errors.nlargest(1, "nasa_score_component_new").iloc[0]
    improved = int((errors.absolute_error_new < errors.absolute_error_baseline).sum())
    title = f"**Học viên:** {AUTHOR_NAME}\n\n**MSHV:** {STUDENT_ID}\n"
    report = "# Cập nhật thực nghiệm ngày 29/09/2026\n\n" + title + """

Đã chạy đầy đủ ba nghiên cứu nested GroupKFold 5×3 và 12 lượt benchmark local.
Kết quả chính FD001 trước đây được giữ riêng; phương án mới chỉ được xem xét khi
đạt tiêu chí outer CV đã ghi trước khi fit. `outer_gate_passed` không có nghĩa là
đã triển khai model hoặc đã kiểm chứng trên dữ liệu vận hành mới.

## Kết quả modeling

""" + markdown_table(summary[["dataset", "selected_final", "n_features", "baseline_outer_mae", "selected_outer_mae", "outer_gate_passed"]]) + """

### Cảnh báo vòng ngoài và test cuối

""" + markdown_table(summary[["dataset", "baseline_low_mae", "selected_low_mae", "baseline_late", "selected_late"]]) + "\n\n" + markdown_table(summary[["dataset", "baseline_outer_rmse", "selected_outer_rmse", "baseline_outer_nasa", "selected_outer_nasa"]]) + "\n\n" + markdown_table(summary[["dataset", "historical_test_mae", "test_mae", "test_rmse", "historical_test_nasa_score", "test_nasa_score"]]) + """

Outer NASA score cộng trên tất cả window train; test NASA score cộng trên một
endpoint mỗi test engine. Không so trực tiếp hai thang tổng này. Trong cùng
outer evaluation, NASA score FD004 xấu hơn dù MAE giảm. Gate FD002/FD004 chỉ
xét overall MAE, không bảo đảm chất lượng cảnh báo hoặc mọi chỉ số cùng tốt lên.

Ví dụ FD001: history features cải thiện test MAE nhưng NASA score xấu hơn;
quy trình chọn làm tăng cảnh báo trễ. Do đó không thay baseline theo tiêu chí đã chốt.
""" + f"\n{improved}/100 test engines giảm absolute error. Engine {int(worst.engine_id)} có RUL thật {worst.y_true:.0f}, dự đoán tăng từ {worst.y_pred_baseline:.2f} lên {worst.y_pred_new:.2f}; NASA component tăng từ {worst.nasa_score_component_baseline:.2f} lên {worst.nasa_score_component_new:.2f}. Một số dự đoán quá cao làm NASA score tăng dù MAE trung bình giảm.\n" + """

FD001: window 30, tám cấu hình cap {None,110,125,140} × history {off,on}.
History thêm 42 features: độ lệch sensor so với 30 cycles đầu và slope của toàn
bộ prefix đã quan sát. Ví dụ tại cycle 40, feature không dùng cycle 41 trở đi.
Nhãn validation/test giữ nguyên; chỉ nhãn huấn luyện được cắt ngưỡng.

FD002/FD004: window 10, ba phương án baseline, raw settings và condition
normalization + settings. Ba settings được chuẩn hóa rồi phân thành sáu cụm
bằng KMeans. Mean/std sensor theo cụm chỉ học từ training engines của từng fold.
Đây là **within-subset**, có dữ liệu train của chính subset; không phải transfer
từ FD001. Không thể dùng kết quả này để khẳng định đã giải quyết zero-shot transfer.

Mọi model giữ HGB 200 iterations, learning rate 0.05, tối đa 31 lá/cây,
min_samples_leaf 20, l2=0, seed 42, không early stopping; tổng trọng số mỗi engine
bằng 1. FD001 chọn low-RUL MAE với ràng buộc overall MAE, late rate và burden.
FD002/FD004 chọn overall mean-engine MAE. Outer gate FD001 yêu cầu CI delta của
low-RUL MAE và late rate đều dưới 0, overall MAE/burden không tăng; hai subset
còn lại yêu cầu upper CI delta overall MAE < 0.

### Paired bootstrap theo engine

""" + markdown_table(paired[["dataset", "metric", "baseline", "candidate", "delta", "ci_lower", "ci_upper"]]) + """

10.000 resamples, seed 42. CI có điều kiện theo dự đoán outer đã fit, chưa bao gồm
biến thiên do refit. Test lịch sử đã được xem trước khi thiết kế follow-up, nên
không gọi đây là blind confirmation. Test chỉ được đọc sau khi lưu final model
và predictions; kết quả test không quyết định cấu hình hoặc outer gate.

## Benchmark Pandas/PySpark

""" + markdown_table(timing) + """

Đơn vị thời gian: giây; mỗi ô là trung vị ba lượt. `summary.csv` giữ median/min/max
và `measurements.csv` giữ đủ 12 lượt đo. Peak RSS là tổng resident memory lấy mẫu
50 ms của Python và process con, gồm JVM; tổng RSS có thể đếm trùng shared pages.
Pipeline time không gồm startup; `total_seconds` trong raw measurements có gồm.

FD001 1× = 33.727 dòng; 10× = 337.270 dòng. Replication dùng engine ID mới,
không làm tăng số mẫu khoa học độc lập. Cùng input Parquet, 126 features, window
30; Spark local[4], bốn partitions, driver/executor heap 4 GiB; max heap thực tế
được ghi trong mỗi measurement. Mỗi lượt chạy bằng process riêng, đảo thứ tự
engine giữa các repetition; không xóa OS cache. Không chạy đồng thời training.

Kiểm tra đầy đủ 27.927 dòng feature ở scale 1× trong cả ba lượt: tất cả khóa/cột
và giá trị khớp với rtol=atol=1e-7. Scale 10× kiểm tra row counts và input hash.
Do Windows thiếu Hadoop native write support, cả hai engine ghi bằng PyArrow
trên driver; Spark `write_seconds` gồm cả `toArrow()`. Không suy rộng kết quả
sang distributed writer hoặc cluster nhiều máy.

Lượt với heap mặc định thất bại ở Arrow export scale 10×. Partial measurements
và lỗi được giữ tại `outputs/benchmarks/20260929/failed_default_heap/`; toàn bộ
12 lượt cuối chạy lại với heap 4 GiB, không gộp số đo từ lượt thất bại. Đây là
so sánh hai implementation hiện tại của repository, không phải kết luận chung
cho mọi chương trình Pandas và Spark.

## Tái lập

```powershell
uv sync --frozen --extra etl --extra dashboard
uv run --frozen python scripts/run_followup.py --output-root outputs/experiments/followup/new_run
uv run --frozen python scripts/verify_followup.py --run outputs/experiments/followup/new_run
# JAVA_HOME trỏ tới Java 17 trước khi chạy benchmark.
uv run --frozen python -m cmapss.etl.scalability --work-dir tmp/new_benchmark --output outputs/benchmarks/new_run
uv run --frozen python scripts/build_followup_summary.py
uv run --frozen python scripts/build_results_notebooks.py --execute
uv run --frozen python slides/extract_slide_data.py
pnpm --dir slides install --frozen-lockfile
pnpm --dir slides build
```

Mặc định `verify_followup.py` và các delivery builder đọc run bàn giao 20260929;
muốn chuyển sang run mới, cập nhật `FOLLOWUP_RUN` trong `cmapss.project` sau khi
run đó hoàn tất. Chỉ resume một run khi source, input, dependency và config
trùng manifest; nếu khác phải dùng thư mục run mới. Fit cache có hash và được
gitignore; archive source và bảng kết quả được giữ để kiểm chứng lịch sử.
`run_info.seconds` là thời gian invocation cuối, có thể đã resume; không phải
tổng chi phí từ đầu. `validation.json` ghi tổng thời gian của các fit cache riêng.
Sau khi đóng băng source, entry point `python -m` được sửa để lưu tên class
ổn định cho joblib. Các run bàn giao dùng wrapper import package, đã được kiểm
tra reload model trong process khác; verifier ghi rõ khác biệt entry point hiện
tại với snapshot. Logic feature, fold, fit và lựa chọn trong snapshot không đổi.

```text
src/cmapss/features/enhanced.py       # feature nhân quả và normalizer
src/cmapss/modeling/followup.py       # nested selection, fit, test boundary
src/cmapss/etl/rolling_spark.py       # 126 rolling features bằng Spark
src/cmapss/etl/scalability.py         # benchmark theo stage, process-tree RSS
src/cmapss/followup_results.py        # loader dùng chung dashboard/notebook
outputs/experiments/followup/20260929/{FD001,FD002,FD004}/
outputs/benchmarks/20260929/
```

## Nguồn phương pháp

- [scikit-learn: nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- [scikit-learn: tránh leakage](https://scikit-learn.org/stable/common_pitfalls.html#how-to-avoid-data-leakage).
- [Spark 4.0.1: toArrow và giới hạn bộ nhớ driver](https://spark.apache.org/docs/4.0.1/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.toArrow.html).
- [psutil: process memory](https://psutil.readthedocs.io/stable/).
"""
    (ROOT / "experiments/FOLLOWUP_20260929.md").write_text(report, encoding="utf-8")
    (ROOT / FOLLOWUP_RUN / "README.md").write_text("# Thí nghiệm bổ sung\n\n" + title
        + "\nKết quả, protocol và ví dụ: [báo cáo bổ sung](../../../../experiments/FOLLOWUP_20260929.md).\n", encoding="utf-8")
    journal_path = ROOT / "experiments/journal.md"
    marker = "### 2026-09-29 — Kết quả hoàn tất của đợt bổ sung"
    journal = journal_path.read_text(encoding="utf-8")
    if marker not in journal:
        journal += "\n\n" + marker + "\n\n**Status**: COMPLETED.\n\n"
        journal += markdown_table(summary[["dataset", "selected_final", "baseline_outer_mae", "selected_outer_mae", "test_mae", "outer_gate_passed"]])
        journal += "\n\n**Learning**: FD001 giảm low-RUL MAE nhưng tăng late warnings và test NASA score; không thay baseline. "
        journal += "Kết quả operating condition phải đọc theo within-subset, không suy rộng sang zero-shot transfer. "
        journal += "Benchmark đo cả I/O và process-tree RSS trên local, không đại diện cluster.\n\n"
        journal += "**Evidence**: [Bảng kết quả, paired CI và giới hạn](FOLLOWUP_20260929.md); validation.json ở từng run.\n"
        journal_path.write_text(journal, encoding="utf-8")
    (benchmark_root / "README.md").write_text("# Benchmark local Pandas/PySpark\n\n" + title + "\n" + markdown_table(timing)
        + "\n\nProtocol, giới hạn và cách tái lập: [báo cáo bổ sung](../../../experiments/FOLLOWUP_20260929.md).\n", encoding="utf-8")
    tex = [r"% Generated by scripts/build_followup_summary.py; author: Phan Nguyễn Mai Phương, MSHV 25002711.",
           r"\begin{table}[H]\centering\small",
           r"\caption{Follow-up nested CV: MAE vòng ngoài và test (cycles). Tiêu chí được chốt riêng cho từng nghiên cứu.}",
           r"\label{tab:followup}\begin{tabular}{lrrrr}\toprule",
           r"Subset & Outer baseline & Outer chọn & Test chọn & Đạt tiêu chí\\\midrule"]
    number = lambda value: f"{value:.2f}".replace(".", ",")
    for row in summary.itertuples():
        tex.append(f"{row.dataset} & {number(row.baseline_outer_mae)} & {number(row.selected_outer_mae)} & {number(row.test_mae)} & {'Có' if row.outer_gate_passed else 'Không'}" + r"\\")
    tex.extend([r"\bottomrule\end{tabular}\end{table}"])
    fd = summary.iloc[0]
    tex.append(f"FD001: MAE RUL thấp {number(fd.baseline_low_mae)} chuyển thành {number(fd.selected_low_mae)} cycles; cảnh báo trễ/bỏ lỡ {fd.baseline_late} chuyển thành {fd.selected_late}/100. Test NASA score tăng từ {number(fd.historical_test_nasa_score)} lên {number(fd.test_nasa_score)}, dù MAE giảm. Vì vậy giữ baseline mặc định.")
    tex.extend([r"\begin{table}[H]\centering\small",
                r"\caption{Benchmark local, trung vị ba lượt; thời gian giây, RSS gồm Python và JVM. Cột Pipeline không gồm startup; tổng thời gian gồm startup được lưu trong artifact.}",
                r"\label{tab:followup-benchmark}\begin{tabular}{lrrrrrr}\toprule",
                r"Engine & Scale & Đọc & Feature & Ghi & Pipeline & RSS (GiB)\\\midrule"])
    for row in timing.itertuples():
        tex.append(f"{row.engine} & {row.scale} & {number(row.read_seconds)} & {number(row.rolling_seconds)} & {number(row.write_seconds)} & {number(row.pipeline_seconds)} & {number(row.peak_rss_gib)}" + r"\\")
    tex.append(r"\bottomrule\end{tabular}\end{table}")
    (ROOT / "report/sections/followup-tables.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("Generated experiment summary, benchmark medians and report tables.")


if __name__ == "__main__":
    main()
