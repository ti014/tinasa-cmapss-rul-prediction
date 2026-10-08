"""Khử nhiễu sensor và phát hiện dấu hiệu bất thường trên train FD001.

Chỉ dùng train observations, không dùng nhãn RUL.
"""

from cmapss.degradation.anomaly import AnomalyConfig, detect_anomalies, sensitivity_grid
from cmapss.degradation.denoise import INFORMATIVE_SENSORS, ewma_smooth, noise_summary

__all__ = [
    "INFORMATIVE_SENSORS",
    "AnomalyConfig",
    "detect_anomalies",
    "ewma_smooth",
    "noise_summary",
    "sensitivity_grid",
]
