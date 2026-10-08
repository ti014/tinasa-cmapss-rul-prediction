"""Gate specs for blending capped and uncapped HGB predictions.

Specs hold no fitted state or targets; the grid is fixed before any outer fold is scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from numbers import Real
from typing import Iterable


@dataclass(frozen=True)
class GateSpec:
    """Smooth gate driven by the uncapped RUL prediction.

    At or below `lower_rul` the cap125 model gets `max_capped_weight`; at or
    above `upper_rul` it gets zero weight.
    """

    lower_rul: float
    upper_rul: float
    max_capped_weight: float

    def __post_init__(self) -> None:
        for name in ("lower_rul", "upper_rul", "max_capped_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
            object.__setattr__(self, name, float(value))
        if not 0.0 <= self.lower_rul < self.upper_rul:
            raise ValueError("gate boundaries must satisfy 0 <= lower_rul < upper_rul")
        if not 0.0 < self.max_capped_weight <= 1.0:
            raise ValueError("max_capped_weight must be inside (0, 1]")

    @property
    def candidate_id(self) -> str:
        """Stable label such as `gate_lo60_hi150_w0.5`."""
        return (f"gate_lo{self.lower_rul:.15g}_hi{self.upper_rul:.15g}"
                f"_w{self.max_capped_weight:.15g}")


def build_specs(
    *, lower_ruls: Iterable[float], upper_ruls: Iterable[float],
    max_capped_weights: Iterable[float],
) -> tuple[GateSpec, ...]:
    """Build the Cartesian grid of gate specs, in input order.

    Every combination must be valid. Duplicates raise instead of silently
    shrinking the search budget.
    """
    axes = tuple(tuple(values) for values in (lower_ruls, upper_ruls, max_capped_weights))
    if any(not values for values in axes):
        raise ValueError("all gate grid axes must be nonempty")
    result = tuple(GateSpec(*values) for values in product(*axes))
    identities = tuple(spec.candidate_id for spec in result)
    if len(set(identities)) != len(identities):
        raise ValueError("gate grid contains duplicate candidates or identity collisions")
    return result


__all__ = ["GateSpec", "build_specs"]
