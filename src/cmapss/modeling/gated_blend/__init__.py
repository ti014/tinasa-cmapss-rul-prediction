"""
Cấu hình gate và phối hợp trơn dự đoán HGB có/không RUL cap; runner chịu trách nhiệm fit
và chọn.
"""

from __future__ import annotations


from .contracts import GateSpec, build_specs
from .gating import blend_predictions, blend_vectors, capped_weights

__all__ = ["GateSpec", "build_specs", "blend_predictions", "blend_vectors", "capped_weights"]
