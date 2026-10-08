"""Làm mượt nhân quả bằng EWMA theo từng động cơ."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from cmapss.data.contracts import ContractError

# 14 sensor có xu hướng theo vòng đời trong FD001. Bảy sensor còn lại
# (1, 5, 6, 10, 16, 18, 19) hằng hoặc gần hằng nên không có nhiễu để khử.
INFORMATIVE_SENSORS = tuple(
    f"sensor_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)
)


def ewma_smooth(frame: pd.DataFrame, columns: Sequence[str], *, span: int) -> pd.DataFrame:
    """Làm mượt các cột bằng EWMA, tính riêng từng động cơ theo thứ tự cycle.

    EWMA với ``adjust=False`` chỉ dùng giá trị hiện tại và quá khứ:
    ``s_t = alpha * x_t + (1 - alpha) * s_{t-1}``, ``alpha = 2 / (span + 1)``.
    Vì vậy giá trị tại cycle t không nhìn thấy cycle t+1. ``span`` tính bằng cycle.

    Trả về bản sao đã sắp theo engine/cycle, với ``columns`` được thay bằng giá trị làm mượt.
    """
    if span < 1:
        raise ValueError("span phải lớn hơn hoặc bằng 1")
    missing = [column for column in ("engine_id", "cycle", *columns) if column not in frame.columns]
    if missing:
        raise ContractError("COLUMNS", "ewma_smooth", f"thiếu cột {missing}")
    ordered = frame.sort_values(["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    smoothed = ordered.copy()
    grouped = ordered.groupby("engine_id", sort=False)[list(columns)]
    smoothed[list(columns)] = grouped.transform(lambda s: s.ewm(span=span, adjust=False).mean())
    return smoothed


def noise_summary(raw: pd.DataFrame, smoothed: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Đo mức dao động ngắn hạn trước và sau làm mượt cho từng sensor.

    Chỉ số dao động là độ lệch chuẩn của sai phân bậc một trong từng động cơ
    (``x_t - x_{t-1}``). Xu hướng suy giảm thay đổi chậm nên đóng góp rất ít
    vào sai phân; phần lớn sai phân đến từ dao động giữa các cycle liền kề.
    ``reduction_pct`` là phần trăm dao động giảm được.
    """
    raw = raw.sort_values(["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    smoothed = smoothed.sort_values(["engine_id", "cycle"], kind="mergesort").reset_index(drop=True)
    if not raw[["engine_id", "cycle"]].equals(smoothed[["engine_id", "cycle"]]):
        raise ContractError("KEY", "noise_summary", "raw và smoothed phải có cùng khóa engine/cycle")
    records = []
    for column in columns:
        raw_diff = raw.groupby("engine_id", sort=False)[column].diff().dropna()
        smooth_diff = smoothed.groupby("engine_id", sort=False)[column].diff().dropna()
        raw_std = float(np.std(raw_diff, ddof=1))
        smooth_std = float(np.std(smooth_diff, ddof=1))
        records.append(
            {
                "sensor": column,
                "diff_std_raw": raw_std,
                "diff_std_smoothed": smooth_std,
                "reduction_pct": 100.0 * (1.0 - smooth_std / raw_std) if raw_std > 0 else float("nan"),
            }
        )
    return pd.DataFrame.from_records(records)
