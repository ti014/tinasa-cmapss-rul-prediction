"""Prediction-only smooth gating for capped and uncapped HGB models.

The package exposes pure contracts and blending operations. Model fitting and
selection must remain in the study runner's grouped inner-validation procedure.
"""

from .contracts import GateSpec, build_specs
from .gating import blend_predictions, blend_vectors, capped_weights

__all__ = ["GateSpec", "build_specs", "blend_predictions", "blend_vectors", "capped_weights"]
