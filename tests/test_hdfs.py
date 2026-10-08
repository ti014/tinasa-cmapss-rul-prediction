"""Kiểm thử logic của bước HDFS mà không cần Docker.

Mapper/reducer được chạy như Hadoop Streaming chạy chúng (map -> sort -> reduce) trên
file raw thật, rồi so với bảng Gold mà ETL Pandas đã ghi.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from cmapss.hdfs import lifetime_job as job
from cmapss.hdfs.stage import compare_summaries, normalize_summary, parse_mapreduce_output

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
GOLD = ROOT / "outputs" / "etl" / "gold" / "engine_summary.parquet"


def _line(engine: int, cycle: int) -> str:
    return " ".join([str(engine), str(cycle)] + ["0.5"] * 24)


def _streaming(paths: list[Path]) -> str:
    mapped = []
    for path in paths:
        with path.open(encoding="utf-8") as file:
            mapped.extend(job.map_lines(file, f"hdfs://namenode:8020/cmapss/bronze/fd001/{path.name}"))
    shuffled = sorted(line + "\n" for line in mapped)
    return "".join(line + "\n" for line in job.reduce_lines(shuffled))


def test_mapper_pads_engine_ids_and_reads_split_from_path() -> None:
    out = list(job.map_lines([_line(7, 3), _line(12, 1)], "/cmapss/bronze/fd001/test_FD001.txt"))
    assert out == ["test:0007\t3", "test:0012\t1"]
    # Engine 12 phải đứng sau engine 7 khi Hadoop sắp khóa theo thứ tự chuỗi.
    assert sorted(out) == out


def test_mapper_skips_malformed_lines_and_unknown_files(capsys: pytest.CaptureFixture) -> None:
    assert list(job.map_lines(["1 2 3"], "train_FD001.txt")) == []
    assert list(job.map_lines([_line(1, 1)], "RUL_FD001.txt")) == []
    assert capsys.readouterr().err.count("malformed_lines") == 2


def test_reducer_summarises_each_engine() -> None:
    lines = ["train:0001\t1\n", "train:0001\t2\n", "train:0001\t3\n", "train:0002\t1\n"]
    assert list(job.reduce_lines(lines)) == ["train\t1\t3\t1\t3", "train\t2\t1\t1\t1"]
    assert list(job.reduce_lines([])) == []


@pytest.mark.skipif(not (RAW / "train_FD001.txt").is_file() or not GOLD.is_file(), reason="thiếu raw hoặc Gold")
def test_streaming_job_reproduces_the_pandas_gold_table() -> None:
    result = parse_mapreduce_output(_streaming([RAW / "train_FD001.txt", RAW / "test_FD001.txt"]))
    comparison = compare_summaries(pd.read_parquet(GOLD), result)
    assert comparison == {"rows_reference": 200, "rows_candidate": 200, "mismatched_cells": 0, "exact_match": True}


def test_compare_reports_value_and_key_mismatches() -> None:
    base = pd.DataFrame({"split": ["train", "train"], "engine_id": [1, 2], "n_cycles": [3, 4],
                         "first_cycle": [1, 1], "last_cycle": [3, 4]})
    changed = base.assign(last_cycle=[3, 5])
    assert compare_summaries(base, changed)["mismatched_cells"] == 1
    assert compare_summaries(base, base.assign(engine_id=[1, 3]))["exact_match"] is False
    # Thứ tự dòng không ảnh hưởng.
    assert compare_summaries(base, base.iloc[::-1])["exact_match"] is True
    assert normalize_summary(base.iloc[::-1])["engine_id"].tolist() == [1, 2]


def test_stored_hdfs_run_matches_pandas_if_present() -> None:
    summary = ROOT / "outputs" / "hdfs" / "summary.json"
    if not summary.is_file():
        pytest.skip("chưa chạy python -m cmapss.hdfs")
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["mapreduce_vs_pandas"]["exact_match"] and data["spark_vs_pandas"]["exact_match"]
    assert data["spark_silver_rows"] == 33_727 and data["spark_silver_null_values"] == 0
    for name in ("mapreduce_engine_summary.csv", "spark_engine_summary.csv"):
        table = pd.read_csv(summary.parent / "tables" / name)
        assert compare_summaries(pd.read_parquet(GOLD), table)["exact_match"], name
