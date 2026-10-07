"""API công khai cho phân tích sensor đầu/cuối đời trên FD001."""

from .config import SensorInferenceConfig
from .inference import (
    adjust_sensor_pvalues,
    bootstrap_mean_difference,
    paired_wilcoxon,
    summarize_sensor_changes,
)
from .pairs import build_engine_sensor_pairs, validate_engine_sensor_pairs
from .validation import validate_sensor_inference

__all__ = [
    "SensorInferenceConfig",
    "adjust_sensor_pvalues",
    "bootstrap_mean_difference",
    "build_engine_sensor_pairs",
    "paired_wilcoxon",
    "summarize_sensor_changes",
    "validate_engine_sensor_pairs",
    "validate_sensor_inference",
]
