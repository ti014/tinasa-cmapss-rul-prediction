"""Job PySpark chạy trong container: đọc raw từ HDFS, ghi Silver Parquet lên HDFS.

Sau đó đọc lại Parquet vừa ghi và tính bảng mỗi động cơ một dòng (số cycle, cycle
đầu, cycle cuối), ghi thành CSV trên HDFS để so với MapReduce và Pandas.

    spark-submit spark_job.py hdfs://namenode:8020/cmapss
"""

import json
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def main(base):
    spark = SparkSession.builder.appName("cmapss-hdfs-fd001").getOrCreate()
    raw = spark.read.text(f"{base}/bronze/fd001/{{train,test}}_FD001.txt").withColumn("path", F.input_file_name())
    fields = F.split(F.trim(F.col("value")), r"\s+")
    observations = raw.select(
        F.when(F.col("path").contains("/train_"), "train").otherwise("test").alias("split"),
        *[
            fields[i].cast("int" if i < 2 else "double").alias(name)
            for i, name in enumerate(COLUMNS)
        ],
    )
    observations.write.mode("overwrite").partitionBy("split").parquet(f"{base}/silver/observations")

    silver = spark.read.parquet(f"{base}/silver/observations")
    lifetime = (
        silver.groupBy("split", "engine_id")
        .agg(
            F.count("*").alias("n_cycles"),
            F.min("cycle").alias("first_cycle"),
            F.max("cycle").alias("last_cycle"),
        )
        .orderBy("split", "engine_id")
    )
    lifetime.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{base}/gold/spark_engine_summary")
    null_counts = silver.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in COLUMNS]).first()
    print(json.dumps({
        "spark_version": spark.version,
        "silver_rows": silver.count(),
        "silver_null_values": int(sum(null_counts)),
        "engines": lifetime.count(),
    }))
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1])
