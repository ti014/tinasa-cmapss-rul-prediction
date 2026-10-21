"""Định dạng trình bày dùng chung cho notebook, hình và slide."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

FONT = "Arial"
COLORS = {
    "blue": "#21409A", "orange": "#EB6834", "teal": "#157A7F",
    "red": "#B23A3A", "gold": "#FCB525", "ink": "#1B2733",
    "muted": "#5A6B7B", "gray": "#9AA5B1", "grid": "#C9D2E3",
    "light": "#EAF0FA", "white": "#FFFFFF",
}
PLOT_RC_PARAMS = {
    "font.family": FONT, "font.size": 10, "figure.dpi": 110,
    "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": COLORS["muted"], "axes.labelcolor": COLORS["ink"],
    "axes.titlecolor": COLORS["ink"], "grid.color": COLORS["grid"],
    "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
    "axes.prop_cycle": f"cycler('color', {list(COLORS.values())[:5]!r})",
}


def format_number_vi(value: float, digits: int | None = 2) -> str:
    """Định dạng phẩy thập phân, chấm hàng nghìn; None bỏ số 0 thừa ở trục."""
    text = (f"{float(value):,.10f}".rstrip("0").rstrip(".")
            if digits is None else f"{float(value):,.{digits}f}")
    return text.replace(",", "_").replace(".", ",").replace("_", ".").replace("-", "−")


def apply_plot_style() -> None:
    """Áp dụng cùng font, bảng màu và rcParams cho mọi nơi tạo hình."""
    import matplotlib

    matplotlib.rcParams.update(PLOT_RC_PARAMS)


def style_figure(figure: Figure) -> None:
    """Đổi trục số tuyến tính sang phẩy thập phân, giữ nguyên nhãn phân loại."""
    from matplotlib.ticker import FuncFormatter, ScalarFormatter

    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis):
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(lambda value, pos: format_number_vi(value, None)))
