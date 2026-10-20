"""Hợp đồng giữa runner dùng chung và backend của từng study."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from cmapss.modeling import audit_study as audit


AUTHOR = {"full_name": "Phan Nguyễn Mai Phương", "student_id": "25002711"}


class ComparisonConfig(Protocol):
    """Settings the lifecycle reads from a study config."""

    outer_splits: int
    inner_splits: int
    seed: int
    fit_threads: int
    max_workers: int
    bootstrap_resamples: int
    endpoint_fractions: tuple[float, ...]
    lead_requirement: int

    def audit_config(self) -> audit.AuditStudyConfig:
        ...


class CandidateSpec(Protocol):
    """Minimal interface of a study's candidate spec."""

    candidate_id: str
    family: str


@dataclass(frozen=True)
class ComparisonBackend:
    """Study-specific functions used by the lifecycle.

    Built in the caller or worker and never pickled into the process pool.
    """

    study_name: str
    arms: tuple[str, ...]
    features: tuple[str, ...]
    specs: Callable[[Any], tuple[Any, ...]]
    active_specs: Callable[[Any], tuple[Any, ...]]
    source_files: Callable[[Path], tuple[str, ...]]
    versions: Callable[[], dict[str, str]]
    fit_task: Callable[..., str]
    final_model_task: Callable[..., str]
    load_frozen_features: Callable[[Path, str], pd.DataFrame]
    fit_predictor: Callable[..., Any]
    candidate_readout: Callable[..., dict[str, Any]]
    select_candidate: Callable[..., dict[str, Any]]
    calibrate_policy: Callable[..., tuple[audit.WarningPolicy, pd.DataFrame]]
    choose_procedure: Callable[..., dict[str, Any]]
    write_readouts: Callable[..., dict[str, Any]]
    read_test_labels: Callable[[Path], pd.DataFrame]
