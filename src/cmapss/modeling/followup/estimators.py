"""Fit estimator và inference trên features đã chọn."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY
from cmapss.features.enhanced import (
    FeatureRecipe,
    OperatingConditionNormalizer,
    build_recipe_features,
)
from cmapss.modeling.baseline.dataset import engine_equal_sample_weights
from cmapss.modeling.followup.config import StudyConfig
from cmapss.modeling.improvement.config import BASELINE_PARAMS


@dataclass
class RecipeModel:
    """Feature recipe, fitted normalizer and regressor saved as one object."""

    recipe: FeatureRecipe
    estimator: HistGradientBoostingRegressor
    normalizer: OperatingConditionNormalizer | None
    feature_columns: tuple[str, ...]

    def predict(self, observations: pd.DataFrame) -> pd.DataFrame:
        """One prediction per window, keyed by split/engine_id/cycle."""
        features = build_recipe_features(observations, self.recipe, normalizer=self.normalizer)
        result = features.loc[:, list(KEY)].copy()
        result["model_name"] = self.recipe.name
        result["rul_pred"] = self.estimator.predict(features.loc[:, list(self.feature_columns)])
        if not np.isfinite(result.rul_pred).all():
            raise ValueError("Non-finite prediction")
        return result


def fit_recipe(
    train: pd.DataFrame, recipe: FeatureRecipe, normalizer: OperatingConditionNormalizer | None, config: StudyConfig,
) -> RecipeModel:
    """Fit the regressor; the RUL cap applies to a copy of the labels."""
    columns = tuple(column for column in train if "__" in column)
    target = train.rul.to_numpy(dtype="float64", copy=True)
    if recipe.target_cap is not None:
        target = np.minimum(target, recipe.target_cap)
    params = {**BASELINE_PARAMS, "max_iter": config.max_iter}
    estimator = HistGradientBoostingRegressor(**params, early_stopping=False, random_state=config.seed)
    with threadpool_limits(limits=config.fit_threads):
        estimator.fit(train.loc[:, list(columns)], target, sample_weight=engine_equal_sample_weights(train))
    return RecipeModel(recipe, estimator, normalizer, columns)
