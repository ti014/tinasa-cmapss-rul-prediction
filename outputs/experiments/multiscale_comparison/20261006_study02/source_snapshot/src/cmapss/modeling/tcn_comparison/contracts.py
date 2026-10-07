"""Frozen protocol and candidate contracts for the TCN comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cmapss.data.contracts import SENSORS

Family = Literal["tcn_uncapped", "tcn_cap125"]


@dataclass(frozen=True)
class TCNConfig:
    """Configuration for the causal residual TCN study.

    The scientific defaults are deliberately small enough to run on CPU while
    retaining the declared four blocks and 32 hidden channels.  ``epochs`` is
    the only training budget parameter; it is frozen before a run starts.
    """

    sequence_length: int = 30
    channels: int = 32
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.10
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    outer_splits: int = 5
    inner_splits: int = 3
    seed: int = 42
    fit_threads: int = 4
    bootstrap_resamples: int = 10_000
    endpoint_fractions: tuple[float, ...] = (0.5, 0.7, 0.9)

    def __post_init__(self) -> None:
        positive = (self.sequence_length, self.channels, self.kernel_size,
                    self.epochs, self.batch_size, self.outer_splits,
                    self.inner_splits, self.fit_threads, self.bootstrap_resamples)
        if any(isinstance(value, bool) or value < 1 for value in positive):
            raise ValueError("TCN sizes, splits and budgets must be positive")
        if not self.dilations or any(value < 1 for value in self.dilations):
            raise ValueError("dilations must be positive")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer parameters are invalid")
        if (not self.endpoint_fractions
                or tuple(sorted(set(self.endpoint_fractions))) != self.endpoint_fractions
                or any(not 0 < value < 1 for value in self.endpoint_fractions)):
            raise ValueError("endpoint fractions must be unique and inside (0, 1)")

    @property
    def scientific(self) -> bool:
        """Keep this exploratory scaffold ineligible for scientific promotion.

        The current runner lacks the matched endpoint panel, warning-policy
        safeguards and source/model receipts required by the accepted gate.
        More epochs alone cannot make the run a confirmed comparison.
        """
        return False


@dataclass(frozen=True)
class TCNSpec:
    """One declared target-label treatment."""

    candidate_id: str
    family: Family
    target_cap: int | None

    def __post_init__(self) -> None:
        if self.family not in {"tcn_uncapped", "tcn_cap125"}:
            raise ValueError(f"unknown TCN family: {self.family}")
        expected = 125 if self.family.endswith("cap125") else None
        if self.target_cap != expected:
            raise ValueError("family and target cap do not match")


def specs() -> tuple[TCNSpec, ...]:
    """Return the two predeclared TCN arms."""
    return (TCNSpec("tcn_uncapped", "tcn_uncapped", None),
            TCNSpec("tcn_cap125", "tcn_cap125", 125))


__all__ = ["Family", "SENSORS", "TCNConfig", "TCNSpec", "specs"]
