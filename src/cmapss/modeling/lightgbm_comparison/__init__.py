"""Đối chứng tree-based cho nghiên cứu dự đoán RUL theo engine.

Package này giữ hợp đồng dữ liệu và worker fitting tách khỏi chính sách chọn
mô hình của study.  Các worker chỉ được đọc ma trận train đã đóng băng và
trả prediction của một fold disjoint-engine.
"""

from .contracts import FEATURES, StudyConfig, TreeSpec, specs
from .estimators import FittedTree, engine_total_one_weights, fit_predictor

__all__ = [
    "FEATURES",
    "StudyConfig",
    "TreeSpec",
    "specs",
    "FittedTree",
    "engine_total_one_weights",
    "fit_predictor",
]
