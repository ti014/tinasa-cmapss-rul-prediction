"""API của improvement; implementation nằm trong các module theo trách nhiệm.

Giữ đường dẫn import của class để nạp artifact đã lưu trước khi tách package.
"""

from cmapss.modeling.improvement.artifact_io import _json_safe, _sha256, _write_json, _prepare_run
from cmapss.modeling.improvement.config import (
    WINDOWS,
    LOW_RUL_THRESHOLD,
    BASELINE_PARAMS,
    WEIGHT_SCHEMES,
    ImprovementConfig,
    make_hgb_candidates,
)
from cmapss.modeling.improvement.dataset import _feature_columns, load_window_datasets
from cmapss.modeling.improvement.runner import _save_assignments, run_improvement_experiment
from cmapss.modeling.improvement.selection import (
    _metrics,
    _aggregate_prediction_metrics,
    _inner_splits,
    _evaluate_candidate,
    _select_candidate,
    _evaluate_stage,
    _outer_assignments,
    _candidate_groups,
)
from cmapss.modeling.improvement.stages import _run_outer_stages, _select_final_configuration
from cmapss.modeling.improvement.training import (
    low_rul_weights,
    _fit_predict,
    _final_test_predictions,
)

__all__ = [
    "WINDOWS",
    "LOW_RUL_THRESHOLD",
    "BASELINE_PARAMS",
    "WEIGHT_SCHEMES",
    "ImprovementConfig",
    "make_hgb_candidates",
    "load_window_datasets",
    "low_rul_weights",
    "run_improvement_experiment",
]
