"""Định dạng trình bày dùng chung cho notebook, hình và slide."""

from __future__ import annotations


def format_number_vi(value: float, digits: int | None = 2) -> str:
    """Định dạng phẩy thập phân, chấm hàng nghìn; None bỏ số 0 thừa ở trục."""
    text = (f"{float(value):,.10f}".rstrip("0").rstrip(".")
            if digits is None else f"{float(value):,.{digits}f}")
    return text.replace(",", "_").replace(".", ",").replace("_", ".").replace("-", "−")
