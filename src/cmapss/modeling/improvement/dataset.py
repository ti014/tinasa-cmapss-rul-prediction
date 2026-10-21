"""Đọc, kiểm tra và tạo dataset theo rolling window."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.features.rolling import build_rolling_features, read_silver_observations
from cmapss.modeling.improvement.config import WINDOWS


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if "__" in column]


def load_window_datasets(
    observations_path: str | Path,
    targets_path: str | Path,
    windows: Iterable[int] = WINDOWS,
    *,
    min_cycle: int = 30,
) -> dict[int, pd.DataFrame]:
    """Bảng features train theo window, cùng bắt đầu ở min_cycle để khớp dòng."""
    observations = read_silver_observations(observations_path)
    targets = pd.read_parquet(targets_path)
    datasets: dict[int, pd.DataFrame] = {}
    for window in windows:
        features = build_rolling_features(observations, window_cycles=int(window))
        train = features.loc[
            features["split"].eq("train") & features["cycle"].ge(min_cycle)
        ].copy()
        train = train.merge(
            targets.loc[:, ["split", "engine_id", "cycle", "rul"]],
            on=list(KEY),
            how="left",
            validate="one_to_one",
        )
        if train["rul"].isna().any():
            raise ValueError(f"window {window} thiếu nhãn train")
        datasets[int(window)] = train.reset_index(drop=True)
    keys = [set(frame.loc[:, list(KEY)].itertuples(index=False, name=None)) for frame in datasets.values()]
    if any(key != keys[0] for key in keys[1:]):
        raise ValueError("Dataset theo window không có cùng khóa đánh giá")
    return datasets
