"""Smooth gating between capped and uncapped HGB predictions.

Only gate specs and blending live here; fitting and selection are in the runner.
"""

from .contracts import GateSpec, build_specs
from .gating import blend_predictions, blend_vectors, capped_weights

__all__ = ["GateSpec", "build_specs", "blend_predictions", "blend_vectors", "capped_weights"]
