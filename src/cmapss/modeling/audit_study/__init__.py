"""API của audit_study; implementation nằm trong các module theo trách nhiệm.

Giữ đường dẫn import của class để nạp artifact đã lưu trước khi tách package.
"""

from __future__ import annotations


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
    fit_candidate,
)
from cmapss.modeling.audit_study.evaluation import (
    build_endpoint_panel,
    select_candidate,
    warning_events,
    choose_policy,
)
from cmapss.modeling.audit_study.features import build_study_features
from cmapss.modeling.audit_study.provenance import SOURCE_FILES, BASELINE_FILES
from cmapss.modeling.audit_study.runner import run_audit_study

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
