"""Kiểm tra số liệu slide được suy từ artifact thay vì gõ lại trong builder."""

from pathlib import Path
import importlib.util

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _extractor():
    spec = importlib.util.spec_from_file_location("slide_numbers", ROOT / "slides/extract_slide_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modeling_numbers_follow_changed_artifact(monkeypatch):
    module = _extractor()
    original = module.modeling()
    read_parquet = pd.read_parquet
    monkeypatch.setattr(module.pd, "read_parquet", lambda path: read_parquet(path).assign(
        absolute_error=lambda frame: frame.absolute_error + 1
    ))
    changed = module.modeling()
    assert changed["low_rul"]["mae"] == pytest.approx(original["low_rul"]["mae"] + 1)
    assert changed["high_rul_mae"] == pytest.approx(original["high_rul_mae"] + 1)


def test_maintenance_numbers_match_artifact():
    data = _extractor().maintenance()
    assert data["watch_above_60"] == 41
    assert data["missed"]["n_engines"] == 4
    assert (data["missed"]["due_min"], data["missed"]["due_max"]) == (1, 8)
    assert data["late"]["n_engines"] == 12
    assert (data["late"]["delay_min"], data["late"]["delay_max"]) == (4, 27)


def test_builder_does_not_repeat_audited_numbers():
    source = (ROOT / "slides/build_presentation.py").read_text(encoding="utf-8")
    for literal in ("4,38", "46,95", "3,78", "41/59", "31–38", "4–27", "kiểm định chéo", "hiếm khi vượt"):
        assert literal not in source
    assert "dự đoán trung bình dừng quanh" in source
