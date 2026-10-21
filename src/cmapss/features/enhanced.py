"""Đặc trưng lịch sử chỉ dùng quá khứ và chuẩn hóa điều kiện vận hành fit trên train."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from cmapss.data.contracts import KEY, SENSORS
from cmapss.data.validation import validate_observations
from cmapss.features.rolling import build_rolling_features

SETTINGS = ("setting_1", "setting_2", "setting_3")


@dataclass(frozen=True)
class FeatureRecipe:
    """Biểu diễn features cố định; chỉ áp dụng RUL cap lúc fit."""

    name: str
    window: int = 30
    target_cap: float | None = None
    history: bool = False
    settings: bool = False
    normalize_conditions: bool = False

    def __post_init__(self) -> None:
        if self.window < 2 or (self.history and self.window < 30):
            raise ValueError("Features lịch sử cần window ít nhất 30 cycle")
        if self.target_cap is not None and self.target_cap <= 0:
            raise ValueError("target_cap phải dương")


class OperatingConditionNormalizer:
    """Fit sáu nhóm điều kiện vận hành và thống kê sensor chỉ trên động cơ train."""

    def __init__(self, n_conditions: int = 6, seed: int = 42) -> None:
        self.n_conditions = n_conditions
        self.seed = seed

    def fit(self, observations: pd.DataFrame) -> "OperatingConditionNormalizer":
        """Học cụm settings và mean/std sensor tổng thể từ train."""
        validate_observations(observations, source="condition normalizer fit")
        if not observations["split"].eq("train").all():
            raise ValueError("Chỉ fit chuẩn hóa điều kiện bằng observations train")
        self.fit_engine_ids_ = tuple(sorted(map(int, observations.engine_id.unique())))
        self.scaler_ = StandardScaler().fit(observations.loc[:, list(SETTINGS)])
        settings = self.scaler_.transform(observations.loc[:, list(SETTINGS)])
        if len(np.unique(settings, axis=0)) < self.n_conditions:
            raise ValueError("Không đủ settings vận hành phân biệt cho số nhóm đã khai báo")
        self.clusterer_ = KMeans(n_clusters=self.n_conditions, n_init=10, random_state=self.seed)
        labels = self.clusterer_.fit_predict(settings)
        values = observations.loc[:, list(SENSORS)].reset_index(drop=True)
        self.means_ = values.groupby(labels, sort=True).mean().reindex(range(self.n_conditions))
        scales = values.groupby(labels, sort=True).std(ddof=0).reindex(range(self.n_conditions))
        self.scales_ = scales.mask(scales.abs().lt(1e-12), 1.0)
        if not np.isfinite(self.means_.to_numpy()).all() or not np.isfinite(self.scales_.to_numpy()).all():
            raise ValueError("Thống kê sensor của mọi điều kiện vận hành phải hữu hạn")
        return self

    def transform(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Áp dụng thống kê train đã cố định, không cập nhật cụm hoặc thang đo."""
        validate_observations(observations, source="condition normalizer transform")
        settings = self.scaler_.transform(observations.loc[:, list(SETTINGS)])
        labels = self.clusterer_.predict(settings)
        values = observations.loc[:, list(SENSORS)].to_numpy(dtype="float64")
        normalized = (values - self.means_.to_numpy()[labels]) / self.scales_.to_numpy()[labels]
        result = observations.copy()
        result.loc[:, list(SENSORS)] = normalized
        return result


def build_history_features(observations: pd.DataFrame, *, minimum_cycle: int = 30) -> pd.DataFrame:
    """
    Tạo độ lệch so với trung bình 30 cycle đầu và slope từ cycle 1 đến hiện tại cho từng
    sensor.
    """
    validate_observations(observations, source="history features")
    if minimum_cycle < 30:
        raise ValueError("Mốc 30 cycle đầu chỉ khả dụng từ cycle 30")
    frames: list[pd.DataFrame] = []
    for _, group in observations.sort_values(list(KEY)).groupby(["split", "engine_id"], sort=True):
        if int(group.cycle.iloc[0]) != 1:
            raise ValueError("Features lịch sử cần prefix quan sát đầy đủ từ cycle 1")
        if len(group) < minimum_cycle:
            raise ValueError("Động cơ không đủ cycle để tạo features lịch sử")
        x = group.cycle.to_numpy(dtype="float64")
        n = np.arange(1, len(group) + 1, dtype="float64")
        sum_x, sum_x2 = np.cumsum(x), np.cumsum(x * x)
        denominator = n * sum_x2 - sum_x * sum_x
        columns: dict[str, object] = {key: group[key].to_numpy() for key in KEY}
        for sensor in SENSORS:
            y = group[sensor].to_numpy(dtype="float64")
            numerator = n * np.cumsum(x * y) - sum_x * np.cumsum(y)
            slope = np.divide(numerator, denominator, out=np.zeros_like(y), where=denominator > 0)
            columns[f"{sensor}__since_start"] = y - y[:30].mean()
            columns[f"{sensor}__prefix_slope"] = slope
        frame = pd.DataFrame(columns)
        frames.append(frame.loc[frame.cycle.ge(minimum_cycle)])
    return pd.concat(frames, ignore_index=True)


def build_recipe_features(
    observations: pd.DataFrame,
    recipe: FeatureRecipe,
    *,
    normalizer: OperatingConditionNormalizer | None = None,
) -> pd.DataFrame:
    """Tạo features chỉ dùng quá khứ bằng bộ chuẩn hóa đã fit."""
    if recipe.normalize_conditions and normalizer is None:
        raise ValueError("Features chuẩn hóa điều kiện cần bộ chuẩn hóa đã fit trên train")
    transformed = normalizer.transform(observations) if recipe.normalize_conditions else observations
    features = build_rolling_features(transformed, window_cycles=recipe.window)
    if recipe.history:
        features = features.merge(build_history_features(transformed), on=list(KEY), validate="one_to_one")
    if recipe.settings:
        frames: list[pd.DataFrame] = []
        for _, group in observations.sort_values(list(KEY)).groupby(["split", "engine_id"], sort=True):
            frame = group.loc[:, list(KEY)].copy()
            for setting in SETTINGS:
                frame[f"{setting}__last"] = group[setting].to_numpy()
                frame[f"{setting}__mean"] = group[setting].rolling(recipe.window, min_periods=recipe.window).mean().to_numpy()
            frames.append(frame.loc[frame.cycle.ge(recipe.window)])
        features = features.merge(pd.concat(frames, ignore_index=True), on=list(KEY), validate="one_to_one")
    values = features.drop(columns=list(KEY)).to_numpy(dtype="float64")
    if features.empty or not np.isfinite(values).all():
        raise ValueError("Features theo công thức phải không rỗng và hữu hạn")
    return features.sort_values(list(KEY)).reset_index(drop=True)
