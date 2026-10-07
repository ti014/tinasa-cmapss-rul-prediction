"""TCN comparison study for the FD001 RUL task.

The package is intentionally isolated from the published model registry.  It
uses raw sensor sequences and imports PyTorch lazily, so the existing tree
pipeline remains usable when PyTorch is unavailable.
"""

from .contracts import TCNConfig, TCNSpec, specs

__all__ = ["TCNConfig", "TCNSpec", "specs"]
