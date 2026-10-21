"""Tạo chuỗi sensor chỉ dùng quá khứ cho thí nghiệm TCN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cmapss.data.contracts import KEY, SENSORS


@dataclass(frozen=True)
class SequenceSet:
    """Tensor window và metadata quan sát tương ứng."""

    values: np.ndarray  # Dạng tensor: (n_windows, n_sensors, sequence_length).
    metadata: pd.DataFrame  # Các khóa split, engine_id, cycle và nhãn rul tùy chọn.


def build_sequences(
    observations: pd.DataFrame,
    targets: pd.DataFrame | None = None,
    *,
    split: str,
    sequence_length: int = 30,
) -> SequenceSet:
    """
    Tạo window kết thúc tại cycle >=sequence_length, không vượt động cơ/split; cycle 30
    gồm 1–30, cycle 31 gồm 2–31.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length phải dương")
    required = set(KEY) | set(SENSORS)
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"Observations thiếu cột: {missing[:3]}")
    rows: list[np.ndarray] = []
    metadata: list[dict[str, int | str | float]] = []
    source = observations[observations["split"].eq(split)].copy()
    if source.empty:
        raise ValueError(f"Không có observations cho split {split!r}")
    source = source.sort_values(["engine_id", "cycle"], kind="mergesort")
    labels = None
    if targets is not None:
        labels = targets.loc[targets["split"].eq(split), list(KEY) + ["rul"]]
        labels = labels.drop_duplicates(list(KEY)).set_index(list(KEY))["rul"]
    for engine_id, group in source.groupby("engine_id", sort=True):
        group = group.sort_values("cycle", kind="mergesort")
        cycles = group["cycle"].to_numpy(dtype="int64")
        values = group.loc[:, list(SENSORS)].to_numpy(dtype="float32")
        if not np.array_equal(cycles, np.arange(1, len(cycles) + 1, dtype="int64")):
            raise ValueError(f"engine {engine_id} không có cycle liên tục từ 1")
        if len(values) < sequence_length:
            continue
        for end in range(sequence_length, len(values) + 1):
            cycle = int(cycles[end - 1])
            rows.append(values[end - sequence_length:end].T.copy())
            item: dict[str, int | str | float] = {
                "split": split, "engine_id": int(engine_id), "cycle": cycle,
            }
            if labels is not None:
                key = (split, int(engine_id), cycle)
                if key not in labels.index:
                    raise ValueError(f"Thiếu nhãn cho {key}")
                item["rul"] = float(labels.loc[key])
            metadata.append(item)
    if not rows:
        raise ValueError("Không tạo được window từ chuỗi")
    frame = pd.DataFrame(metadata)
    frame = frame.loc[:, ["split", "engine_id", "cycle"] + (["rul"] if labels is not None else [])]
    array = np.stack(rows).astype("float32", copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Đầu vào chuỗi chứa giá trị không hữu hạn")
    return SequenceSet(array, frame.reset_index(drop=True))


@dataclass(frozen=True)
class SensorScaler:
    """Chuẩn hóa từng sensor bằng thống kê chỉ học từ động cơ fit."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype="float32")
        if array.ndim != 3 or array.shape[1] != len(SENSORS):
            raise ValueError("expected (n_windows, 21, sequence_length) input")
        result = (array - self.mean[None, :, None]) / self.scale[None, :, None]
        if not np.isfinite(result).all():
            raise ValueError("Chuỗi đã chuẩn hóa chứa giá trị không hữu hạn")
        return result.astype("float32", copy=False)


def fit_scaler(values: np.ndarray, metadata: pd.DataFrame, engine_ids: set[int]) -> SensorScaler:
    """Fit thống kê sensor chỉ trên các động cơ được chỉ định."""
    mask = metadata["engine_id"].isin(engine_ids).to_numpy()
    if not mask.any():
        raise ValueError("Thiếu động cơ fit scaler")
    selected = np.asarray(values[mask], dtype="float64")
    mean = selected.transpose(1, 0, 2).reshape(len(SENSORS), -1).mean(axis=1)
    scale = selected.transpose(1, 0, 2).reshape(len(SENSORS), -1).std(axis=1)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return SensorScaler(mean.astype("float32"), scale.astype("float32"))


__all__ = ["SequenceSet", "SensorScaler", "build_sequences", "fit_scaler"]
