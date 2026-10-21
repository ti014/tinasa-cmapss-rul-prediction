"""Giữ số liệu nhóm RUL mà report đang trích dẫn."""

from pathlib import Path
import importlib.util

import pandas as pd
import pytest

from cmapss.modeling.contracts import RUL_BINS


def test_report_diagnostic_bins_preserve_existing_values():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("diagnostics", root / "scripts/report_error_diagnostics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = __import__("json").loads((root / "outputs/diagnostics/error_diagnostics.json").read_text())
    assert module.oof_by_true_rul() == expected["oof_hgb_by_true_rul"]
    assert module.test_error_share() == expected["test_errors"]
    groups = pd.cut(pd.Series([60, 90, 91]), (*RUL_BINS[:4], RUL_BINS[-1]))
    assert str(groups.iloc[1]) == "(60, 90]"
    assert groups.iloc[1] != groups.iloc[2]
