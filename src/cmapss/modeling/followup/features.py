"""Chuẩn bị features và preprocessing theo engine."""

from __future__ import annotations

import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.features.enhanced import (
    FeatureRecipe,
    OperatingConditionNormalizer,
    build_recipe_features,
)


def _join_targets(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    joined = features.merge(targets.loc[:, [*KEY, "rul"]], on=list(KEY), how="left", validate="one_to_one")
    if joined.rul.isna().any():
        raise ValueError("Thiếu nhãn RUL cho dòng features train")
    return joined


class FeatureStore:
    """Cache features gốc; bộ chuẩn hóa điều kiện luôn fit bên trong fold."""

    def __init__(self, observations: pd.DataFrame, targets: pd.DataFrame) -> None:
        self.observations = observations.loc[observations.split.eq("train")].copy()
        self.targets = targets
        self.raw: dict[tuple[int, bool, bool], pd.DataFrame] = {}

    def features(self, recipe: FeatureRecipe) -> pd.DataFrame:
        """Tạo và cache một lần các features chưa qua tiền xử lý có học tham số."""
        if recipe.normalize_conditions:
            raise ValueError("Features chuẩn hóa phải được tạo trong fold")
        key = (recipe.window, recipe.history, recipe.settings)
        if key not in self.raw:
            self.raw[key] = _join_targets(build_recipe_features(self.observations, recipe), self.targets)
        return self.raw[key]

    def fold_features(
        self, recipe: FeatureRecipe, train_ids: set[int], validation_ids: set[int], seed: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, OperatingConditionNormalizer | None]:
        """Fit tiền xử lý trên động cơ train rồi trả features train/validation."""
        if train_ids & validation_ids:
            raise ValueError("Động cơ train và validation giao nhau")
        normalizer = None
        if recipe.normalize_conditions:
            training = self.observations.loc[self.observations.engine_id.isin(train_ids)]
            normalizer = OperatingConditionNormalizer(seed=seed).fit(training)
            pair = self.observations.loc[self.observations.engine_id.isin(train_ids | validation_ids)]
            frame = _join_targets(build_recipe_features(pair, recipe, normalizer=normalizer), self.targets)
        else:
            frame = self.features(recipe)
        train = frame.loc[frame.engine_id.isin(train_ids)].reset_index(drop=True)
        validation = frame.loc[frame.engine_id.isin(validation_ids)].reset_index(drop=True)
        return train, validation, normalizer
