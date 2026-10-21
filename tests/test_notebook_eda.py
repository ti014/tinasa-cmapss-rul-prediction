"""Đối chiếu các phép tính notebook được chuyển vào package EDA."""

import pandas as pd
import pytest

from cmapss.eda import engine_cycle_correlations, engine_lifetimes, within_engine_variability


def test_variability_and_correlations_keep_engines_separate():
    data = pd.DataFrame({"split": ["train"] * 6, "engine_id": [1]*3 + [2]*3,
                         "cycle": [1, 2, 3]*2, "sensor": [1, 2, 3, 3, 2, 1], "constant": [7]*6})
    within, counts = within_engine_variability(data, ["sensor", "constant"])
    assert within.loc[within.measurement.eq("sensor"), "std"].tolist() == [1, 1]
    assert counts.set_index("measurement").loc["constant", "constant_engines"] == 2
    assert engine_cycle_correlations(data, "sensor").tolist() == pytest.approx([1, -1])
    with pytest.raises(ValueError, match="chỉ nhận dữ liệu train"):
        within_engine_variability(data.assign(split="test"), ["sensor"])


def test_lifetimes_use_the_existing_engine_summary(raw_dir):
    from cmapss.data.readers import read_observations

    data = read_observations(raw_dir / "train_FD001.txt", split="train")
    result = engine_lifetimes(data.assign(rul=0))  # notebook truyền train đã có cột rul
    first = result.set_index("engine_id").loc[1]
    assert (first.first_cycle, first.last_cycle, first.n_observations) == (1, 192, 192)
