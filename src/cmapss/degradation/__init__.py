"""Khử nhiễu sensor và phát hiện dấu hiệu bất thường trên train FD001.

Package này trả lời trực tiếp hai yêu cầu của Đề tài 8: xử lý nhiễu tín hiệu
và phát hiện dấu hiệu bất thường. Mọi phép tính chỉ dùng train observations,
không dùng nhãn RUL và không tạo đặc trưng cho mô hình RUL.
"""

from cmapss.degradation.anomaly import AnomalyConfig, detect_anomalies
from cmapss.degradation.denoise import INFORMATIVE_SENSORS, ewma_smooth, noise_summary

__all__ = [
    "INFORMATIVE_SENSORS",
    "AnomalyConfig",
    "detect_anomalies",
    "ewma_smooth",
    "noise_summary",
]
