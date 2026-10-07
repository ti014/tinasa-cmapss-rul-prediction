# Benchmark local Pandas/PySpark

**Học viên:** Phan Nguyễn Mai Phương

**MSHV:** 25002711

| engine | scale | startup_seconds | read_seconds | summary_seconds | rolling_seconds | write_seconds | pipeline_seconds | peak_rss_gib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pandas | 1 | 0.0000 | 0.0459 | 0.0364 | 10.3011 | 0.1526 | 10.5425 | 0.2495 |
| pandas | 10 | 0.0000 | 0.0897 | 0.2267 | 110.3866 | 0.9740 | 111.6811 | 1.7297 |
| spark | 1 | 5.4800 | 4.8828 | 1.0579 | 7.0338 | 1.4340 | 14.5950 | 1.4789 |
| spark | 10 | 5.4301 | 5.2801 | 1.1169 | 22.1198 | 4.0816 | 32.6799 | 3.0521 |

Protocol, giới hạn và cách tái lập: [báo cáo bổ sung](../../../experiments/FOLLOWUP_20260929.md).
