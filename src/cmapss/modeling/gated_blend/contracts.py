"""
Cấu hình gate phối hợp HGB có/không RUL cap, không chứa trạng thái fit hoặc nhãn; chốt
lưới trước đánh giá outer.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from numbers import Real
from typing import Iterable


@dataclass(frozen=True)
class GateSpec:
    """
    Gate theo dự đoán không RUL cap; trọng số cap125 bằng max_capped_weight ở lower_rul,
    bằng 0 ở upper_rul.
    """

    lower_rul: float
    upper_rul: float
    max_capped_weight: float

    def __post_init__(self) -> None:
        for name in ("lower_rul", "upper_rul", "max_capped_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{name} phải là số thực hữu hạn")
            object.__setattr__(self, name, float(value))
        if not 0.0 <= self.lower_rul < self.upper_rul:
            raise ValueError("Biên gate phải thỏa 0 <= lower_rul < upper_rul")
        if not 0.0 < self.max_capped_weight <= 1.0:
            raise ValueError("max_capped_weight phải thuộc (0, 1]")

    @property
    def candidate_id(self) -> str:
        """Nhãn ổn định, ví dụ gate_lo60_hi150_w0.5."""
        return (f"gate_lo{self.lower_rul:.15g}_hi{self.upper_rul:.15g}"
                f"_w{self.max_capped_weight:.15g}")


def build_specs(
    *, lower_ruls: Iterable[float], upper_ruls: Iterable[float],
    max_capped_weights: Iterable[float],
) -> tuple[GateSpec, ...]:
    """
    Tạo tích Descartes cấu hình gate theo thứ tự đầu vào; từ chối tổ hợp không hợp lệ và
    trùng lặp.
    """
    axes = tuple(tuple(values) for values in (lower_ruls, upper_ruls, max_capped_weights))
    if any(not values for values in axes):
        raise ValueError("Mọi trục lưới gate phải không rỗng")
    result = tuple(GateSpec(*values) for values in product(*axes))
    identities = tuple(spec.candidate_id for spec in result)
    if len(set(identities)) != len(identities):
        raise ValueError("Lưới gate có ứng viên trùng hoặc xung đột định danh")
    return result


__all__ = ["GateSpec", "build_specs"]
