"""CPU similarity based RUL estimator for the bounded FD001 pilot.

The estimator stores causal 30-cycle, six-block sensor profiles from the
run-to-failure training engines.  A query is matched to at most one window per
donor engine; the donor RUL values are then averaged with inverse-distance
weights.  This module deliberately contains no experiment orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from cmapss.data.contracts import KEY, OBSERVATION_COLUMNS, SENSORS
from cmapss.data.targets import validate_train_targets
from cmapss.data.validation import validate_observations

WINDOW = 30
BLOCK_SIZE = 5
N_BLOCKS = WINDOW // BLOCK_SIZE
EPS = 1e-8


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)


def _check_prefix(frame: pd.DataFrame, *, name: str) -> None:
    """Require every engine to contain exactly cycles 1..N."""
    for ident, group in frame.groupby(["split", "engine_id"], sort=False):
        cycles = group["cycle"].to_numpy(dtype=np.int64)
        expected = np.arange(1, len(cycles) + 1, dtype=np.int64)
        if not np.array_equal(cycles, expected):
            raise ValueError(f"{name} must contain a complete prefix 1..t for engine {ident}")


def _windows(frame: pd.DataFrame, *, window: int, block_size: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Return endpoint keys and unscaled block-mean representations."""
    if window != 30 or block_size != 5 or window % block_size:
        raise ValueError("pilot requires window=30 and block_size=5")
    rows: list[tuple[str, int, int]] = []
    values: list[np.ndarray] = []
    for (split, engine_id), group in frame.groupby(["split", "engine_id"], sort=True):
        group = group.sort_values("cycle", kind="mergesort")
        raw = group.loc[:, list(SENSORS)].to_numpy(dtype=np.float64, copy=False)
        cycles = group["cycle"].to_numpy(dtype=np.int64)
        for end in range(window - 1, len(group)):
            segment = raw[end - window + 1 : end + 1]
            # Blocks are chronological and flattened block-major.
            profile = segment.reshape(N_BLOCKS, block_size, len(SENSORS)).mean(axis=1)
            rows.append((str(split), int(engine_id), int(cycles[end])))
            values.append(profile.reshape(-1))
    if not values:
        return pd.DataFrame(columns=KEY), np.empty((0, N_BLOCKS * len(SENSORS)), dtype=np.float64)
    return pd.DataFrame(rows, columns=KEY), np.vstack(values).astype(np.float64, copy=False)


def _engine_key(frame: pd.DataFrame) -> pd.Series:
    return frame["split"].astype(str) + ":" + frame["engine_id"].astype(str)


@dataclass
class SimilarityModel:
    """Persisted state for similarity-based RUL prediction."""

    window: int
    block_size: int
    sensors: tuple[str, ...]
    training_engine_ids: tuple[tuple[str, int], ...]
    scaler: StandardScaler
    library_keys: pd.DataFrame
    library_matrix: np.ndarray
    library_rul: np.ndarray
    fit_engine_keys: frozenset[tuple[str, int]]
    kept_sensor_mask: np.ndarray
    batch_size: int = 128
    fit_threads: int = 4

    def __post_init__(self) -> None:
        self.library_matrix = np.asarray(self.library_matrix, dtype=np.float64)
        self.library_rul = np.asarray(self.library_rul, dtype=np.float64)
        self.kept_sensor_mask = np.asarray(self.kept_sensor_mask, dtype=bool)
        if self.kept_sensor_mask.shape != (len(SENSORS),) or not self.kept_sensor_mask.any():
            raise ValueError("kept_sensor_mask must select at least one canonical sensor")
        if self.batch_size < 1 or not 1 <= self.fit_threads <= 4:
            raise ValueError("batch_size must be positive and fit_threads must be in 1..4")
        if len(self.library_keys) != len(self.library_matrix) or len(self.library_rul) != len(self.library_matrix):
            raise ValueError("library keys, matrix and RUL must have equal length")

    @property
    def library_features(self) -> np.ndarray:
        """Alias retained for inspection and validation of persisted models."""
        return self.library_matrix

    @property
    def kept_sensor_indices(self) -> np.ndarray:
        return np.flatnonzero(self.kept_sensor_mask)

    def _query(self, observations: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        _validate_query(observations, self.fit_engine_keys)
        keys, raw = _windows(_ordered(observations), window=self.window, block_size=self.block_size)
        if raw.size == 0:
            return keys, raw
        # Scaler was fit on individual raw sensor rows; apply it to every block.
        transformed = self.scaler.transform(raw.reshape(-1, len(SENSORS))).reshape(raw.shape)
        transformed = transformed[:, self._kept_indices]
        return keys, transformed

    @property
    def _kept_indices(self) -> np.ndarray:
        return np.concatenate(
            [block * len(SENSORS) + self.kept_sensor_indices for block in range(N_BLOCKS)]
        ).astype(np.int64, copy=False)

    def query_keys(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Return canonical query endpoints (cycle >= 30)."""
        return self._query(observations)[0]

    def predict_with_matches(
        self,
        observations: pd.DataFrame,
        *,
        neighbors: Iterable[int] = (5, 10),
    ) -> tuple[pd.DataFrame, dict[int, np.ndarray], pd.DataFrame]:
        """Predict each requested k and return the ten best donor traces."""
        ks = tuple(dict.fromkeys(int(k) for k in neighbors))
        if not ks or any(k < 1 or k > 10 for k in ks):
            raise ValueError("neighbors must contain positive k values no larger than 10")
        if not self.library_matrix.size:
            raise ValueError("similarity library is empty")
        keys, query_matrix = self._query(observations)
        if query_matrix.size == 0:
            return keys, {k: np.empty(0, dtype=np.float64) for k in ks}, pd.DataFrame()

        donor_ids = self.library_keys["split"].astype(str) + ":" + self.library_keys["engine_id"].astype(str)
        donor_unique = donor_ids.drop_duplicates().to_numpy()
        donor_parts = {name: np.flatnonzero(donor_ids.to_numpy() == name) for name in donor_unique}
        donor_engine_numbers = np.array([int(name.split(":", 1)[1]) for name in donor_unique], dtype=np.int64)
        library_norm = np.einsum("ij,ij->i", self.library_matrix, self.library_matrix)
        best_rows: list[tuple[int, str, float, int, float, float]] = []
        outputs = {k: np.empty(len(query_matrix), dtype=np.float64) for k in ks}
        donor_ids_array = donor_unique
        donor_cycles = self.library_keys["cycle"].to_numpy(dtype=np.int64)
        with threadpool_limits(limits=self.fit_threads):
            for start in range(0, len(query_matrix), self.batch_size):
                block = query_matrix[start : start + self.batch_size]
                # Norm identity keeps allocation O(batch * n_reference), not O(... * features).
                qnorm = np.einsum("ij,ij->i", block, block)
                distances = (qnorm[:, None] + library_norm[None, :] - 2.0 * block @ self.library_matrix.T) / self.library_matrix.shape[1]
                distances = np.maximum(distances, 0.0)
                donor_best_dist = np.empty((len(block), len(donor_unique)), dtype=np.float64)
                donor_best_idx = np.empty((len(block), len(donor_unique)), dtype=np.int64)
                for donor_position, donor in enumerate(donor_unique):
                    idx = donor_parts[donor]
                    local_distances = distances[:, idx]
                    donor_best_dist[:, donor_position] = local_distances.min(axis=1)
                    # Library rows are endpoint-cycle ascending per donor, so argmin resolves ties earliest.
                    donor_best_idx[:, donor_position] = idx[np.argmin(local_distances, axis=1)]
                for local in range(len(block)):
                    order = np.lexsort((donor_engine_numbers, donor_best_dist[local]))[:10]
                    picks = donor_best_idx[local, order]
                    distances10 = donor_best_dist[local, order]
                    weights10 = 1.0 / np.maximum(distances10, EPS)
                    weights10 /= weights10.sum()
                    global_row = start + local
                    for rank, (donor_position, pick, distance, weight) in enumerate(
                        zip(order, picks, distances10, weights10), 1
                    ):
                        best_rows.append((global_row, donor_ids_array[donor_position], float(distance), int(donor_cycles[pick]), float(self.library_rul[pick]), float(weight)))
                    for k in ks:
                        weights = 1.0 / np.maximum(distances10[:k], EPS)
                        weights /= weights.sum()
                        outputs[k][global_row] = float(np.dot(weights, self.library_rul[picks[:k]]))

        match_rows = []
        last_query = -1
        rank = 0
        for query_index, donor, distance, donor_cycle, donor_rul, weight10 in best_rows:
            if query_index != last_query:
                rank = 1
                last_query = query_index
            else:
                rank += 1
            split, engine = donor.split(":", 1)
            record = keys.iloc[query_index].to_dict()
            record.update({"rank": rank,
                           "query_index": query_index, "donor_split": split,
                           "donor_engine_id": int(engine), "donor_cycle": donor_cycle,
                           "distance": distance, "donor_rul": donor_rul,
                           "weight_k10": weight10})
            match_rows.append(record)
        matches = pd.DataFrame(match_rows)
        return keys, outputs, matches

    def predict(self, observations: pd.DataFrame, *, neighbors: Iterable[int] = (5, 10)) -> dict[int, np.ndarray]:
        """Return predictions keyed by donor count."""
        _keys, predictions, _matches = self.predict_with_matches(observations, neighbors=neighbors)
        return predictions


def _validate_query(observations: pd.DataFrame, fit_keys: frozenset[tuple[str, int]]) -> None:
    validate_observations(observations, source="similarity query observations")
    _check_prefix(_ordered(observations), name="query observations")
    overlap = {(str(split), int(engine)) for split, engine in observations[["split", "engine_id"]].drop_duplicates().itertuples(index=False)} & set(fit_keys)
    if overlap:
        raise ValueError(f"query engines overlap fitting library: {sorted(overlap)}")


def fit_similarity(
    observations: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    window: int = WINDOW,
    block_size: int = BLOCK_SIZE,
    batch_size: int = 128,
    fit_threads: int = 4,
) -> SimilarityModel:
    """Fit the CPU similarity library from train-only observations and targets."""
    if batch_size < 1 or not 1 <= fit_threads <= 4:
        raise ValueError("batch_size must be positive and fit_threads must be in 1..4")
    validate_observations(observations, source="similarity fit observations")
    if not observations["split"].eq("train").all():
        raise ValueError("similarity fit observations must contain only split=train")
    _check_prefix(_ordered(observations), name="fit observations")
    validate_train_targets(observations, targets)
    if window != WINDOW or block_size != BLOCK_SIZE:
        raise ValueError("pilot requires window=30 and block_size=5")
    train_keys = frozenset((str(split), int(engine)) for split, engine in observations[["split", "engine_id"]].drop_duplicates().itertuples(index=False))
    scaler = StandardScaler()
    raw = observations.loc[:, list(SENSORS)].to_numpy(dtype=np.float64)
    engine_labels = observations["engine_id"].to_numpy()
    counts = pd.Series(engine_labels).value_counts().to_dict()
    sample_weight = np.array([1.0 / counts[int(e)] for e in engine_labels], dtype=np.float64)
    # sklearn's weighted variance implementation can emit a RuntimeWarning
    # while taking sqrt of a tiny negative round-off value for constant
    # sensors. We clamp that value immediately below and keep the warning
    # local to this documented numerical edge case.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in sqrt",
            category=RuntimeWarning,
            module=r"sklearn\.preprocessing\._data",
        )
        scaler.fit(raw, sample_weight=sample_weight)
    # Weighted one-pass variance can be a tiny negative number for a constant
    # sensor because of floating-point cancellation. Clamp only that numeric
    # artifact before deriving the keep mask and transform scale.
    scaler.var_ = np.maximum(np.asarray(scaler.var_, dtype=np.float64), 0.0)
    scaler.scale_ = np.sqrt(scaler.var_)
    keep = np.asarray(scaler.var_ > 1e-16, dtype=bool)
    if not keep.any():
        raise ValueError("all sensors have near-zero fit standard deviation")
    scaler.scale_[~keep] = 1.0
    keys, raw_windows = _windows(_ordered(observations), window=window, block_size=block_size)
    matrix = scaler.transform(raw_windows.reshape(-1, len(SENSORS))).reshape(raw_windows.shape)
    kept = np.flatnonzero(keep)
    kept_indices = np.concatenate(
        [block * len(SENSORS) + kept for block in range(N_BLOCKS)]
    ).astype(np.int64, copy=False)
    matrix = matrix[:, kept_indices]
    target_map = targets.set_index(list(KEY))["rul"]
    rul = np.array([float(target_map.loc[tuple(key)]) for key in keys.itertuples(index=False, name=None)], dtype=np.float64)
    training_ids = tuple(sorted(train_keys))
    return SimilarityModel(window, block_size, tuple(np.asarray(SENSORS)[keep]), training_ids,
                           scaler, keys, matrix, rul, train_keys, keep, batch_size, fit_threads)


__all__ = ["SimilarityModel", "fit_similarity"]
