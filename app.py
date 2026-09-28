"""Streamlit entry point for the C-MAPSS FD001 RUL dashboard.

Run with ``uv run --extra dashboard streamlit run app.py``.

The UI lives in ``cmapss.dashboard_app`` and the read-only artifact loaders in
``cmapss.dashboard``. The loaders are re-exported here because
``cmapss.reproducibility`` imports ``load_dashboard_data`` from this module;
importing ``app`` therefore does not require Streamlit.
"""

from __future__ import annotations

from pathlib import Path

from cmapss.dashboard import (  # noqa: F401  (re-exported for cmapss.reproducibility)
    DashboardData,
    DashboardDataError,
    format_metric,
    load_dashboard_data,
    load_improvement_data,
)


if __name__ == "__main__":
    from cmapss.dashboard_app import main

    main(Path(__file__).resolve().parent)
