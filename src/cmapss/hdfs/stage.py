"""Bước lưu trữ phân tán: đưa raw FD001 lên HDFS, chạy MapReduce và Spark trên HDFS.

Ví dụ chạy từ thư mục gốc repository (cần Docker đang chạy):: uv run --frozen python -m
cmapss.hdfs Kết quả của hai job được so với bảng Gold ``engine_summary`` mà ETL Pandas
đã ghi; ba cách tính phải khớp tuyệt đối.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = ROOT / "infra" / "hadoop" / "docker-compose.yml"
PROJECT = "cmapss-hadoop"
HDFS_BASE = "/cmapss"
RAW_FILES = ("train_FD001.txt", "test_FD001.txt")
KEY = ["split", "engine_id"]
SUMMARY_COLUMNS = KEY + ["n_cycles", "first_cycle", "last_cycle"]


class HadoopStageError(RuntimeError):
    """Một lệnh Docker/Hadoop thất bại hoặc kết quả không khớp."""


def _compose(*args: str, project: str = PROJECT, check: bool = True, timeout: float = 900) -> subprocess.CompletedProcess:
    command = ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", project, *args]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    if check and result.returncode != 0:
        raise HadoopStageError(f"{' '.join(args[:3])}... thất bại:\n{result.stderr[-2000:]}")
    return result


def _namenode(command: str, *, project: str = PROJECT, **kwargs) -> str:
    return _compose("exec", "-T", "namenode", "bash", "-lc", command, project=project, **kwargs).stdout


def parse_mapreduce_output(text: str) -> pd.DataFrame:
    """Đọc output dạng TSV của lifetime_job.py thành bảng giống engine_summary."""
    frame = pd.read_csv(io.StringIO(text), sep="\t", header=None, names=SUMMARY_COLUMNS)
    return normalize_summary(frame)


def normalize_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[:, SUMMARY_COLUMNS].astype({"split": str, "engine_id": "int64", "n_cycles": "int64",
                                                  "first_cycle": "int64", "last_cycle": "int64"})
    return frame.sort_values(KEY).reset_index(drop=True)


def compare_summaries(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    """So hai bảng mỗi động cơ một dòng; trả về số dòng và số ô lệch."""
    left, right = normalize_summary(reference), normalize_summary(candidate)
    if left[KEY].equals(right[KEY]):
        mismatched = int((left != right).to_numpy().sum())
    else:
        mismatched = -1
    return {"rows_reference": len(left), "rows_candidate": len(right), "mismatched_cells": mismatched,
            "exact_match": mismatched == 0 and len(left) == len(right)}


def _wait_for_hdfs(timeout: float = 240, *, project: str = PROJECT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _compose("exec", "-T", "namenode", "bash", "-lc",
                          "hdfs dfsadmin -safemode get && hdfs dfsadmin -report | grep -c 'Live datanodes (1)'",
                          project=project, check=False, timeout=60)
        if result.returncode == 0 and "OFF" in result.stdout:
            return
        time.sleep(3)
    raise HadoopStageError("HDFS chưa sẵn sàng (NameNode còn ở safe mode hoặc chưa có DataNode)")


def run_stage(
    raw_dir: Path,
    gold_path: Path,
    output_dir: Path,
    *,
    keep_running: bool = False,
    hdfs_base: str = HDFS_BASE,
    project: str = PROJECT,
) -> dict:
    """Chạy hai job trên HDFS, có thể tách thư mục dữ liệu cho từng lần chạy."""
    if re.fullmatch(r"/cmapss(?:/[A-Za-z0-9_-]+)*", hdfs_base) is None:
        raise ValueError("hdfs_base phải là /cmapss hoặc thư mục con, chỉ gồm chữ, số, _ và -")
    compose = partial(_compose, project=project)
    namenode = partial(_namenode, project=project)
    started = time.perf_counter()
    compose("up", "-d", "namenode", "datanode", "resourcemanager", "nodemanager")
    _wait_for_hdfs(project=project)

    # 1. Bronze trên HDFS: chép raw vào container rồi put lên HDFS.
    namenode("rm -rf /tmp/cmapss && mkdir -p /tmp/cmapss")
    for name in RAW_FILES:
        compose("cp", str(raw_dir / name), f"namenode:/tmp/cmapss/{name}")
    compose("cp", str(Path(__file__).with_name("lifetime_job.py")), "namenode:/tmp/cmapss/lifetime_job.py")
    namenode(f"hdfs dfs -rm -r -f -skipTrash {hdfs_base} && hdfs dfs -mkdir -p {hdfs_base}/bronze/fd001 "
              f"&& hdfs dfs -put -f /tmp/cmapss/*_FD001.txt {hdfs_base}/bronze/fd001/")

    # 2. MapReduce (Hadoop Streaming trên YARN).
    mr_started = time.perf_counter()
    mr_log = namenode(
        "hadoop jar /opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-*.jar "
        "-files /tmp/cmapss/lifetime_job.py "
        "-mapper 'python3 lifetime_job.py map' -reducer 'python3 lifetime_job.py reduce' "
        f"-input {hdfs_base}/bronze/fd001 -output {hdfs_base}/gold/mapreduce_engine_summary 2>&1"
    )
    mr_seconds = time.perf_counter() - mr_started
    mapreduce = parse_mapreduce_output(namenode(f"hdfs dfs -cat {hdfs_base}/gold/mapreduce_engine_summary/part-*"))

    # 3. Spark đọc HDFS, ghi Silver Parquet lên HDFS rồi tính lại bảng Gold.
    spark_started = time.perf_counter()
    spark_run = compose("run", "--rm", "spark", "/opt/spark/bin/spark-submit", "--master", "local[2]",
                         "--conf", "spark.ui.enabled=false",
                         # Spark client không đọc hdfs-site của cụm; một DataNode nên chỉ cần một bản sao.
                         "--conf", "spark.hadoop.dfs.replication=1", "/work/spark_job.py", "hdfs://namenode:8020" + hdfs_base)
    spark_seconds = time.perf_counter() - spark_started
    spark_info = json.loads([line for line in spark_run.stdout.splitlines() if line.startswith("{")][-1])
    spark = normalize_summary(pd.read_csv(io.StringIO(
        namenode(f"hdfs dfs -cat {hdfs_base}/gold/spark_engine_summary/part-*.csv"))))

    listing = namenode(f"hdfs dfs -ls -R {hdfs_base}")
    fsck = namenode(f"hdfs fsck {hdfs_base} -files -blocks -locations 2>/dev/null")
    hadoop_version = namenode("hadoop version | head -1").strip()

    reference = normalize_summary(pd.read_parquet(gold_path))
    result = {
        "author": "Phan Nguyễn Mai Phương — MSHV 25002711",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hadoop_version": hadoop_version,
        "spark_version": spark_info["spark_version"],
        "hdfs_paths": {
            "bronze": f"{hdfs_base}/bronze/fd001",
            "silver": f"{hdfs_base}/silver/observations",
            "gold_mapreduce": f"{hdfs_base}/gold/mapreduce_engine_summary",
            "gold_spark": f"{hdfs_base}/gold/spark_engine_summary",
        },
        "spark_silver_rows": spark_info["silver_rows"],
        "spark_silver_null_values": spark_info["silver_null_values"],
        "mapreduce_vs_pandas": compare_summaries(reference, mapreduce),
        "spark_vs_pandas": compare_summaries(reference, spark),
        "seconds": {"mapreduce_job": round(mr_seconds, 1), "spark_job": round(spark_seconds, 1),
                    "total": round(time.perf_counter() - started, 1)},
    }

    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    mapreduce.to_csv(tables / "mapreduce_engine_summary.csv", index=False)
    spark.to_csv(tables / "spark_engine_summary.csv", index=False)
    (output_dir / "hdfs_listing.txt").write_text(listing, encoding="utf-8")
    (output_dir / "hdfs_fsck.txt").write_text(fsck, encoding="utf-8")
    (output_dir / "mapreduce_job.txt").write_text(mr_log, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    if not keep_running:
        compose("down")
    if not (result["mapreduce_vs_pandas"]["exact_match"] and result["spark_vs_pandas"]["exact_match"]):
        raise HadoopStageError(f"kết quả trên HDFS không khớp Pandas: {output_dir / 'summary.json'}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lưu raw FD001 lên HDFS, chạy MapReduce và Spark trên HDFS")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--gold", default="outputs/etl/gold/engine_summary.parquet")
    parser.add_argument("--output-dir", default="outputs/hdfs")
    parser.add_argument("--keep-running", action="store_true", help="Không tắt cụm Hadoop sau khi chạy")
    args = parser.parse_args(argv)
    result = run_stage(Path(args.raw_dir), Path(args.gold), Path(args.output_dir), keep_running=args.keep_running)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
