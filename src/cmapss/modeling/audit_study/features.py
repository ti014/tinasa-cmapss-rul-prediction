"""Chuẩn bị features và preprocessing theo engine."""

from __future__ import annotations

import pandas as pd

from cmapss.data.contracts import SENSORS
from cmapss.features.rolling import build_rolling_features


def build_study_features(observations: pd.DataFrame) -> pd.DataFrame:
    """30-cycle rolling features plus sensor values relative to the first 30 cycles.

    The reference is the per-engine mean of cycles 1..30, so it is already known
    at every kept row (cycle >= 30). Adds 42 columns to the 126 rolling features.
    """
    features = build_rolling_features(observations, window_cycles=30)
    initial = observations.loc[observations.cycle.le(30)]
    counts = initial.groupby(["split", "engine_id"]).size()
    if not counts.eq(30).all() or len(counts) != observations.groupby(["split", "engine_id"]).ngroups:
        raise ValueError("causal history requires exactly cycles1..30 per engine/split")
    means = initial.groupby(["split", "engine_id"], sort=True)[list(SENSORS)].mean()
    joined = features.merge(means.add_suffix("__initial_mean"),
                            on=["split", "engine_id"], validate="many_to_one")
    history = {
        f"{sensor}__from_initial_{stat}":
            joined[f"{sensor}__{stat}"] - joined[f"{sensor}__initial_mean"]
        for sensor in SENSORS for stat in ("last", "mean")
    }
    return pd.concat([features.reset_index(drop=True), pd.DataFrame(history)], axis=1)
