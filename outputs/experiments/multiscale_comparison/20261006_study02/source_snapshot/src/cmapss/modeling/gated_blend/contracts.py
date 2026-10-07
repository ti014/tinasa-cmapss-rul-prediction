"""Declare prediction-only gates before a nested comparison.

These contracts contain no fitted state or targets. The study runner must
freeze its explicit grid before evaluating outer engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite
from numbers import Real
from typing import Iterable


@dataclass(frozen=True)
class GateSpec:
    """Describe one smooth gate driven by the uncapped RUL prediction.

    Attributes:
        lower_rul: At or below this prediction, use maximum capped weight.
        upper_rul: At or above this prediction, use no capped contribution.
        max_capped_weight: Largest allowed contribution of cap125 HGB.

    Examples:
        >>> GateSpec(60, 150, 0.5).candidate_id
        'gate_lo60_hi150_w0.5'
    """

    lower_rul: float
    upper_rul: float
    max_capped_weight: float

    def __post_init__(self) -> None:
        """Reject malformed gate parameters before evaluating predictions."""
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
        """Return a deterministic label for the declared parameter triple.

        Returns:
            Stable human-readable candidate identity.
        """
        return (f"gate_lo{self.lower_rul:.15g}_hi{self.upper_rul:.15g}"
                f"_w{self.max_capped_weight:.15g}")


def build_specs(
    *, lower_ruls: Iterable[float], upper_ruls: Iterable[float],
    max_capped_weights: Iterable[float],
) -> tuple[GateSpec, ...]:
    """Build an explicit Cartesian grid with no data-dependent defaults.

    All boundary combinations must be valid. Duplicate parameters or identity
    collisions are rejected instead of silently changing the search budget.

    Args:
        lower_ruls: Predeclared lower RUL boundaries.
        upper_ruls: Predeclared upper RUL boundaries.
        max_capped_weights: Predeclared maximum cap125 contributions.

    Returns:
        Candidate specifications in the input Cartesian-product order.

    Raises:
        ValueError: A grid is empty, duplicates candidates, or is invalid.

    Examples:
        >>> len(build_specs(lower_ruls=(30, 60), upper_ruls=(150,),
        ...                 max_capped_weights=(0.25, 0.5)))
        4
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
