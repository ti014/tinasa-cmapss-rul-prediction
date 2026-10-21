"""Cấu hình và đặc tả ứng viên so sánh TCN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cmapss.data.contracts import SENSORS

Family = Literal["tcn_uncapped", "tcn_cap125"]


@dataclass(frozen=True)
class TCNConfig:
    """Cấu hình TCN residual chỉ dùng quá khứ; mặc định bốn block, 32 kênh đủ nhỏ để chạy CPU."""

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
            raise ValueError("Kích thước, số fold và ngân sách TCN phải dương")
        if not self.dilations or any(value < 1 for value in self.dilations):
            raise ValueError("dilations phải dương")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout phải thuộc [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Tham số optimizer không hợp lệ")
        if (not self.endpoint_fractions
                or tuple(sorted(set(self.endpoint_fractions))) != self.endpoint_fractions
                or any(not 0 < value < 1 for value in self.endpoint_fractions)):
            raise ValueError("Tỷ lệ endpoint phải duy nhất và thuộc (0, 1)")

    @property
    def scientific(self) -> bool:
        """
        Luôn False: TCN chỉ để thăm dò, thiếu endpoint ghép cặp, kiểm tra chính sách và
        biên nhận mô hình để được chấp nhận.
        """
        return False


@dataclass(frozen=True)
class TCNSpec:
    """Một cách xử lý nhãn: không RUL cap hoặc RUL cap 125."""

    candidate_id: str
    family: Family
    target_cap: int | None

    def __post_init__(self) -> None:
        if self.family not in {"tcn_uncapped", "tcn_cap125"}:
            raise ValueError(f"Họ TCN chưa biết: {self.family}")
        expected = 125 if self.family.endswith("cap125") else None
        if self.target_cap != expected:
            raise ValueError("family và target cap không khớp")


def specs() -> tuple[TCNSpec, ...]:
    """Trả hai nhánh TCN."""
    return (TCNSpec("tcn_uncapped", "tcn_uncapped", None),
            TCNSpec("tcn_cap125", "tcn_cap125", 125))


__all__ = ["Family", "SENSORS", "TCNConfig", "TCNSpec", "specs"]
