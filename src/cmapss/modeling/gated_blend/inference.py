"""Reload copied HGB models and a frozen gate for label-free inference.

Inference checks model and metadata digests before deserializing the verified
base estimators. The gate is loaded from JSON rather than retrained.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from cmapss.modeling.lightgbm_comparison.estimators import FittedTree

from .contracts import GateSpec
from .gating import blend_vectors


def _sha(path: Path) -> str:
    """Hash an artifact before deserializing a copied model."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class SavedBlend:
    """Hold the two verified HGB estimators and an already selected gate.

    Attributes:
        models: Copied capped and uncapped fitted estimators.
        gate: Final-inner gate specification, including a diagnostic gate.
        selected_procedure_id: Inner-selected uncapped fallback or gated blend.
        policies: Frozen policies for subsequent warning generation.
        feature_columns: Required causal input columns in their fitted order.
    """

    models: dict[str, FittedTree]
    gate: GateSpec
    selected_procedure_id: str
    policies: dict[str, dict[str, Any]]
    feature_columns: tuple[str, ...]

    def predict(self, frame: pd.DataFrame, procedure: str = "selected_pipeline") -> np.ndarray:
        """Predict without requiring labels or future observations.

        Args:
            frame: Current-observation causal features on the original scale.
            procedure: A base arm, gated_blend, or the frozen selected_pipeline.

        Returns:
            Raw RUL predictions in the input row order.

        Raises:
            ValueError: The procedure is unknown or features fail base validation.
        """
        selected = self.selected_procedure_id if procedure == "selected_pipeline" else procedure
        if selected in self.models:
            return self.models[selected].predict(frame)
        if selected != "gated_blend":
            raise ValueError(f"unknown saved inference procedure: {procedure}")
        return blend_vectors(self.models["hgb_uncapped"].predict(frame),
                             self.models["hgb_cap125"].predict(frame), self.gate)


def load_inference(output_dir: str | Path) -> SavedBlend:
    """Load an inference artifact only after verifying its copied model bytes.

    Args:
        output_dir: Derived-study folder containing inference.json and models.

    Returns:
        Restored label-free inference object with final-inner policies.

    Raises:
        ValueError: Model bytes, feature/schema/spec or procedure are inconsistent.
        FileNotFoundError: A required saved artifact is missing.
    """
    folder = Path(output_dir).resolve()
    metadata = json.loads((folder / "inference.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("test_labels_read") is not False:
        raise ValueError("inference schema must prove an unlabelled frozen selection")
    selected = metadata["selected_procedure_id"]
    if selected not in ("hgb_uncapped", "gated_blend"):
        raise ValueError("selected inference must be uncapped fallback or gated blend")
    columns = tuple(metadata["feature_columns"])
    if len(columns) != 169 or len(set(columns)) != 169 or {"rul", "y_true"} & set(columns):
        raise ValueError("saved inference requires 169 unique label-free features")
    models = {}
    for arm, cap in (("hgb_uncapped", None), ("hgb_cap125", 125)):
        info = metadata["base_models"][arm]
        model_path = folder / "models" / arm / "model.joblib"
        model_metadata = folder / "models" / arm / "model.json"
        if (_sha(model_path) != info["model_sha256"]
                or _sha(model_metadata) != info["metadata_sha256"]):
            raise ValueError(f"saved base-model checksum mismatch: {arm}")
        fitted = joblib.load(model_path)
        if (not isinstance(fitted, FittedTree) or tuple(fitted.feature_columns) != columns
                or fitted.spec.candidate_id != arm or fitted.spec.target_cap != cap
                or fitted.spec.model_kind != "hgb"):
            raise ValueError(f"saved base-model object contract mismatch: {arm}")
        models[arm] = fitted
    gate = GateSpec(**metadata["gate_spec"])
    if gate.candidate_id != metadata["gate_candidate_id"]:
        raise ValueError("saved gate identity mismatch")
    return SavedBlend(models, gate, selected, metadata["policies"], columns)


__all__ = ["SavedBlend", "load_inference"]
