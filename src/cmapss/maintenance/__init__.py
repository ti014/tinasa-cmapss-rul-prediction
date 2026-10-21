"""Lịch bảo trì cho các động cơ test FD001.

Ghép dự đoán RUL đã lưu với trạng thái bất thường từ Health Index, mỗi động cơ một dòng.
Nhãn test chỉ được đọc sau khi lịch đã được ghi ra file.
"""

from __future__ import annotations


from cmapss.maintenance.report import (
    STATUS_ORDER,
    MaintenanceConfig,
    anomaly_status,
    build_schedule,
    evaluate_schedule,
    transform_health_index,
)

__all__ = [
    "STATUS_ORDER",
    "MaintenanceConfig",
    "anomaly_status",
    "build_schedule",
    "evaluate_schedule",
    "transform_health_index",
]
