"""API của audit_study; implementation nằm trong các module theo trách nhiệm.

Giữ đường dẫn import của class để nạp artifact đã lưu trước khi tách package.
"""

from cmapss.modeling.audit_study.artifact_io import (
    _sha256,
    _json_value,
    _write_json,
    _write_parquet,
    _relative,
    _write_comparisons,
    _freeze_selection,
)
from cmapss.modeling.audit_study.cli import main
from cmapss.modeling.audit_study.config import (
    BASE_FEATURES,
    HISTORY_FEATURES,
    CandidateSpec,
    make_candidate_specs,
    AuditStudyConfig,
    WarningPolicy,
)
from cmapss.modeling.audit_study.estimators import (
    FittedCandidate,
    engine_weights,
    _make_model,
    fit_candidate,
)
from cmapss.modeling.audit_study.evaluation import (
    build_endpoint_panel,
    _prediction_metrics,
    _model_readout,
    select_candidate,
    _alarm_flags,
    warning_events,
    _warning_readout,
    choose_policy,
    _policy_grid,
    _engine_readout,
    _paired_comparison,
)
from cmapss.modeling.audit_study.features import build_study_features
from cmapss.modeling.audit_study.provenance import SOURCE_FILES, BASELINE_FILES, _initialize_run
from cmapss.modeling.audit_study.runner import _baseline_replay, _evaluate_test, run_audit_study
from cmapss.modeling.audit_study.training import (
    _fit_predict,
    _inner_evaluation,
    _outer_assignments,
    _final_predictor,
)

__all__ = [
    "BASE_FEATURES",
    "HISTORY_FEATURES",
    "CandidateSpec",
    "make_candidate_specs",
    "AuditStudyConfig",
    "WarningPolicy",
    "build_study_features",
    "FittedCandidate",
    "engine_weights",
    "fit_candidate",
    "build_endpoint_panel",
    "select_candidate",
    "warning_events",
    "choose_policy",
    "SOURCE_FILES",
    "BASELINE_FILES",
    "run_audit_study",
    "main",
]
