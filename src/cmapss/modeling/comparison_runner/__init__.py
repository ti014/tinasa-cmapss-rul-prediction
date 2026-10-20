"""API của comparison_runner; implementation nằm trong các module theo trách nhiệm.

Giữ đường dẫn import của class để nạp artifact đã lưu trước khi tách package.
"""

from cmapss.modeling.comparison_runner.artifact_io import (
    _sha,
    _read_json,
    _write_json,
    _freeze_json,
    _resolve,
    _progress,
)
from cmapss.modeling.comparison_runner.contracts import (
    AUTHOR,
    ComparisonConfig,
    CandidateSpec,
    ComparisonBackend,
)
from cmapss.modeling.comparison_runner.evaluation import evaluate_final_test
from cmapss.modeling.comparison_runner.provenance import (
    source_files,
    installed_versions,
    _assert_immutable,
    freeze_inputs,
)
from cmapss.modeling.comparison_runner.runner import run_comparison
from cmapss.modeling.comparison_runner.training import fit_batch, inner_stage, final_model_task

__all__ = [
    "AUTHOR",
    "ComparisonConfig",
    "CandidateSpec",
    "ComparisonBackend",
    "source_files",
    "installed_versions",
    "freeze_inputs",
    "fit_batch",
    "inner_stage",
    "final_model_task",
    "evaluate_final_test",
    "run_comparison",
]
