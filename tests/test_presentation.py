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


def test_shared_plot_style_keeps_category_labels():
    from matplotlib import pyplot as plt
    from cmapss.presentation import FONT, apply_plot_style, style_figure

    apply_plot_style()
    fig, ax = plt.subplots()
    ax.bar(["Nhóm A", "Nhóm B"], [1.25, 2.5])
    style_figure(fig)
    assert plt.rcParams["font.family"] == [FONT]
    assert ax.yaxis.get_major_formatter()(1.25, 0) == "1,25"
    assert ax.get_xticklabels()[0].get_text() == "Nhóm A"
    assert ax.patches[0].get_facecolor()[:3] == pytest.approx((33/255, 64/255, 154/255))
    plt.close(fig)
