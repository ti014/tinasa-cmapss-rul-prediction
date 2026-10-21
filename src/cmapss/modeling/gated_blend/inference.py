"""
Đọc mô hình HGB và gate cho suy luận không cần nhãn; kiểm tra SHA-256 mô hình/metadata
trước giải pickle.
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
    """Mô hình nền, gate và quy trình đã chọn trên inner validation được lưu."""

    models: dict[str, FittedTree]
    gate: GateSpec
    selected_procedure_id: str
    policies: dict[str, dict[str, Any]]
    feature_columns: tuple[str, ...]

    def predict(self, frame: pd.DataFrame, procedure: str = "selected_pipeline") -> np.ndarray:
        """
        Dự đoán RUL theo thứ tự dòng đầu vào; procedure là nhánh nền, gated_blend hoặc
        selected_pipeline.
        """
        selected = self.selected_procedure_id if procedure == "selected_pipeline" else procedure
        if selected in self.models:
            return self.models[selected].predict(frame)
        if selected != "gated_blend":
            raise ValueError(f"Quy trình suy luận đã lưu chưa biết: {procedure}")
        return blend_vectors(self.models["hgb_uncapped"].predict(frame),
                             self.models["hgb_cap125"].predict(frame), self.gate)


def load_inference(output_dir: str | Path) -> SavedBlend:
    """Đọc inference.json và mô hình dưới output_dir, đồng thời kiểm tra checksum."""
    folder = Path(output_dir).resolve()
    metadata = json.loads((folder / "inference.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("test_labels_read") is not False:
        raise ValueError("Schema suy luận phải chứng minh lựa chọn đã cố định không dùng nhãn")
    selected = metadata["selected_procedure_id"]
    if selected not in ("hgb_uncapped", "gated_blend"):
        raise ValueError("Suy luận được chọn phải là nhánh không RUL cap hoặc gated blend")
    columns = tuple(metadata["feature_columns"])
    if len(columns) != 169 or len(set(columns)) != 169 or {"rul", "y_true"} & set(columns):
        raise ValueError("Suy luận đã lưu cần 169 features duy nhất, không chứa nhãn")
    models = {}
    for arm, cap in (("hgb_uncapped", None), ("hgb_cap125", 125)):
        info = metadata["base_models"][arm]
        model_path = folder / "models" / arm / "model.joblib"
        model_metadata = folder / "models" / arm / "model.json"
        if (_sha(model_path) != info["model_sha256"]
                or _sha(model_metadata) != info["metadata_sha256"]):
            raise ValueError(f"Checksum mô hình nền đã lưu không khớp: {arm}")
        fitted = joblib.load(model_path)
        if (not isinstance(fitted, FittedTree) or tuple(fitted.feature_columns) != columns
                or fitted.spec.candidate_id != arm or fitted.spec.target_cap != cap
                or fitted.spec.model_kind != "hgb"):
            raise ValueError(f"Contract object mô hình nền đã lưu không khớp: {arm}")
        models[arm] = fitted
    gate = GateSpec(**metadata["gate_spec"])
    if gate.candidate_id != metadata["gate_candidate_id"]:
        raise ValueError("Định danh gate đã lưu không khớp")
    return SavedBlend(models, gate, selected, metadata["policies"], columns)


__all__ = ["SavedBlend", "load_inference"]
