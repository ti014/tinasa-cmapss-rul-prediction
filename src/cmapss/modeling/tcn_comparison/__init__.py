"""TCN comparison study on raw FD001 sensor sequences.

PyTorch is imported lazily, so the tree pipeline works without it.
"""

from .contracts import TCNConfig, TCNSpec, specs

__all__ = ["TCNConfig", "TCNSpec", "specs"]
