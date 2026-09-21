"""Fixture tổng hợp nhỏ; không sửa các file FD001 nguồn."""

from pathlib import Path

import pandas as pd
import pytest

from cmapss.data.contracts import MEASUREMENTS, OBSERVATION_COLUMNS


@pytest.fixture
def observations() -> pd.DataFrame:
    """Hai train engines có tuổi thọ khác nhau để phát hiện RUL tính sai group."""
    rows = [
        ["train", engine, cycle, *[float(cycle + i) for i in range(24)]]
        for engine, length in ((1, 4), (2, 2))
        for cycle in range(1, length + 1)
    ]
    return pd.DataFrame(rows, columns=OBSERVATION_COLUMNS).astype(
        {"split": "string", "engine_id": "int64", "cycle": "int64",
         **dict.fromkeys(MEASUREMENTS, "float64")}
    )


@pytest.fixture
def raw_dir() -> Path:
    """Đường dẫn dữ liệu thật, độc lập working directory của pytest."""
    return Path(__file__).resolve().parents[1] / "data" / "raw"
