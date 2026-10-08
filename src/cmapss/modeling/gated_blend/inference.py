"""Load the saved HGB models and gate for inference without labels.

Model and metadata SHA-256 digests are checked before unpickling.
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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class SavedBlend:
    """Saved base models, gate and the procedure chosen on inner validation."""

    models: dict[str, FittedTree]
    gate: GateSpec
    selected_procedure_id: str
    policies: dict[str, dict[str, Any]]
    feature_columns: tuple[str, ...]

    def predict(self, frame: pd.DataFrame, procedure: str = "selected_pipeline") -> np.ndarray:
        """Predict RUL in input row order.

        `procedure` is a base arm name, "gated_blend", or "selected_pipeline".
        """
        selected = self.selected_procedure_id if procedure == "selected_pipeline" else procedure
        if selected in self.models:
            return self.models[selected].predict(frame)
        if selected != "gated_blend":
            raise ValueError(f"unknown saved inference procedure: {procedure}")
        return blend_vectors(self.models["hgb_uncapped"].predict(frame),
                             self.models["hgb_cap125"].predict(frame), self.gate)


def load_inference(output_dir: str | Path) -> SavedBlend:
    """Load inference.json and the models under `output_dir`, verifying checksums."""
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
