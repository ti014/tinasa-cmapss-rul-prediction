"""Kiểm tra quy ước trình bày số dùng chung."""

import pytest

from cmapss.presentation import format_number_vi


@pytest.mark.parametrize("value,digits,expected", [
    (20631, 0, "20.631"), (15.349047, 2, "15,35"),
    (47.395, 4, "47,3950"), (-1234.5, 2, "−1.234,50"),
    (1000.125, None, "1.000,125"), (0, None, "0"),
])
def test_vietnamese_numbers(value, digits, expected):
    assert format_number_vi(value, digits) == expected
