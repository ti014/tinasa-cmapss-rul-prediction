"""So sánh HGB và XGBoost cho bài toán dự đoán RUL theo engine.

Worker chỉ đọc ma trận train đã đóng băng và trả prediction cho một fold có tập engine
tách rời.
"""

from __future__ import annotations


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
