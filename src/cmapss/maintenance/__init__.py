"""Báo cáo dự đoán thời điểm cần bảo trì cho đội động cơ test FD001.

Đây là sản phẩm đặc thù của Đề tài 8. Package ghép dự đoán RUL đã có với tín
hiệu bất thường từ Health Index để tạo một dòng cho mỗi động cơ: cycle hiện tại,
RUL dự đoán, cycle dự kiến cần bảo trì và trạng thái cảnh báo. Không huấn luyện
mô hình mới; nhãn test chỉ được đọc sau khi lịch bảo trì đã được ghi ra file.
"""

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
