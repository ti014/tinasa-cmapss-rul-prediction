"""API của followup; implementation nằm trong các module theo trách nhiệm.

Giữ đường dẫn import của class để nạp artifact đã lưu trước khi tách package.
"""

from __future__ import annotations


from cmapss.modeling.followup.artifact_io import _digest, freeze_manifest
from cmapss.modeling.followup.cli import main
from cmapss.modeling.followup.config import StudyConfig, study_candidates
from cmapss.modeling.followup.estimators import RecipeModel, fit_recipe
from cmapss.modeling.followup.features import _join_targets, FeatureStore
from cmapss.modeling.followup.runner import run_study
from cmapss.modeling.followup.training import fit_pair, select_record, inner_selection

__all__ = [
    "StudyConfig",
    "study_candidates",
    "FeatureStore",
    "RecipeModel",
    "fit_recipe",
    "fit_pair",
    "select_record",
    "inner_selection",
    "freeze_manifest",
    "run_study",
    "main",
]
