"""Causal history features and train-fitted operating-condition normalization."""

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
    """Fixed feature representation, with target cap applied only during fit."""

    name: str
    window: int = 30
    target_cap: float | None = None
    history: bool = False
    settings: bool = False
    normalize_conditions: bool = False

    def __post_init__(self) -> None:
        if self.window < 2 or (self.history and self.window < 30):
            raise ValueError("History features require a window of at least 30 cycles")
        if self.target_cap is not None and self.target_cap <= 0:
            raise ValueError("target_cap must be positive")


class OperatingConditionNormalizer:
    """Fit six operating regimes and sensor statistics on training engines only."""

    def __init__(self, n_conditions: int = 6, seed: int = 42) -> None:
        self.n_conditions = n_conditions
        self.seed = seed

    def fit(self, observations: pd.DataFrame) -> "OperatingConditionNormalizer":
        """Learn settings clusters and population sensor mean/std from train."""
        validate_observations(observations, source="condition normalizer fit")
        if not observations["split"].eq("train").all():
            raise ValueError("Condition normalization fit accepts train observations only")
        self.fit_engine_ids_ = tuple(sorted(map(int, observations.engine_id.unique())))
        self.scaler_ = StandardScaler().fit(observations.loc[:, list(SETTINGS)])
        settings = self.scaler_.transform(observations.loc[:, list(SETTINGS)])
        if len(np.unique(settings, axis=0)) < self.n_conditions:
            raise ValueError("Insufficient distinct operating settings for the declared regimes")
        self.clusterer_ = KMeans(n_clusters=self.n_conditions, n_init=10, random_state=self.seed)
        labels = self.clusterer_.fit_predict(settings)
        values = observations.loc[:, list(SENSORS)].reset_index(drop=True)
        self.means_ = values.groupby(labels, sort=True).mean().reindex(range(self.n_conditions))
        scales = values.groupby(labels, sort=True).std(ddof=0).reindex(range(self.n_conditions))
        self.scales_ = scales.mask(scales.abs().lt(1e-12), 1.0)
        if not np.isfinite(self.means_.to_numpy()).all() or not np.isfinite(self.scales_.to_numpy()).all():
            raise ValueError("Every operating regime must have finite sensor statistics")
        return self

    def transform(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Apply frozen training statistics without updating clusters or scales."""
        validate_observations(observations, source="condition normalizer transform")
        settings = self.scaler_.transform(observations.loc[:, list(SETTINGS)])
        labels = self.clusterer_.predict(settings)
        values = observations.loc[:, list(SENSORS)].to_numpy(dtype="float64")
        normalized = (values - self.means_.to_numpy()[labels]) / self.scales_.to_numpy()[labels]
        result = observations.copy()
        result.loc[:, list(SENSORS)] = normalized
        return result


def build_history_features(observations: pd.DataFrame, *, minimum_cycle: int = 30) -> pd.DataFrame:
    """Use the first 30 cycles and the observed prefix; future rows never contribute.

    For each sensor, append deviation from its first-30-cycle mean and the
    least-squares slope from cycle 1 through the current cycle.
    """
    validate_observations(observations, source="history features")
    if minimum_cycle < 30:
        raise ValueError("First-30-cycle reference is only available from cycle 30")
    frames: list[pd.DataFrame] = []
    for _, group in observations.sort_values(list(KEY)).groupby(["split", "engine_id"], sort=True):
        if int(group.cycle.iloc[0]) != 1:
            raise ValueError("History features require a complete observed prefix from cycle 1")
        if len(group) < minimum_cycle:
            raise ValueError("Engine has too few cycles for history features")
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
    """Construct one causal feature representation using an already-fitted normalizer."""
    if recipe.normalize_conditions and normalizer is None:
        raise ValueError("Condition-normalized features require a training-fitted normalizer")
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
        raise ValueError("Recipe features must be nonempty and finite")
    return features.sort_values(list(KEY)).reset_index(drop=True)
