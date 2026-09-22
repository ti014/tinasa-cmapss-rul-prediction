"""Compatibility API cho bộ ghi artifact thống kê FD001.

Implementation hiện thuộc `cmapss.statistics.artifacts`; module này được giữ
để notebook và caller hiện tại không phải đổi import trong cùng một lượt.
"""

from cmapss.statistics.artifacts import SensorStatisticsArtifacts, write_sensor_statistics_artifacts

__all__ = ["SensorStatisticsArtifacts", "write_sensor_statistics_artifacts"]
