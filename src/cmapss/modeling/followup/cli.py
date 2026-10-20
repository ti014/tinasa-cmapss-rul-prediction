"""Nested CV studies for capped targets, history features and operating regimes.

Usage: ``python -m cmapss.modeling.followup --datasets FD001 FD002 FD004``.
Test labels are read only after the final model and test predictions are saved."""

from __future__ import annotations

from pathlib import Path
import argparse

from cmapss.modeling.followup.config import StudyConfig
from cmapss.modeling.followup.runner import run_study
from cmapss.modeling.improvement.artifact_io import _write_json
from cmapss.project import FOLLOWUP_RUN


def main() -> None:
    """Run the study for each requested dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("FD001", "FD002", "FD004"), default=["FD001", "FD002", "FD004"])
    parser.add_argument("--output-root", type=Path, default=FOLLOWUP_RUN)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--skip-test-evaluation", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = (repo / args.output_root).resolve()
    results = {}
    for dataset in args.datasets:
        results[dataset] = run_study(dataset, repo, output / dataset, StudyConfig(), evaluate_test=not args.skip_test_evaluation)
    _write_json(output / "completion.json", {"datasets": results, "status": "complete"})
