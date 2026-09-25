"""Kiểm tra profile subset và window ngắn cho C-MAPSS FD002--FD004."""

from pathlib import Path

import pandas as pd

from cmapss.data.contracts import DATASET_PROFILES
from cmapss.data.readers import read_observations
from cmapss.data.validation import validate_dataset_profile
from cmapss.etl.pandas_pipeline import run_pandas_etl
from cmapss.features.rolling import build_rolling_features


def test_all_cmapss_profiles_match_raw_files(raw_dir: Path) -> None:
    """Mỗi raw subset phải khớp row count và số engine đã công bố."""
    for dataset_id, profile in DATASET_PROFILES.items():
        for split in ("train", "test"):
            frame = read_observations(raw_dir / f"{split}_{dataset_id}.txt", split=split)
            validate_dataset_profile(frame, dataset_id=dataset_id, split=split)
            assert len(frame) == profile[f"{split}_rows"]
            assert frame["engine_id"].nunique() == profile[f"{split}_engines"]


def test_fd002_etl_is_dataset_aware(raw_dir: Path) -> None:
    """ETL FD002 tạo đúng train targets và engine catalog riêng."""
    result = run_pandas_etl(raw_dir, "FD002")
    assert result.dataset_id == "FD002"
    assert result.row_counts == {
        "observations": 87_750,
        "train_targets": 53_759,
        "engine_summary": 519,
    }


def test_ten_cycle_window_supports_short_cmapss_test_engine() -> None:
    """Window 10 tạo feature cho trajectory 19 cycles mà window 30 sẽ loại."""
    rows = []
    for split in ("train", "test"):
        for cycle in range(1, 20):
            rows.append(
                {
                    "split": split,
                    "engine_id": 1,
                    "cycle": cycle,
                    **{f"sensor_{i}": float(cycle) for i in range(1, 22)},
                    "setting_1": 0.0,
                    "setting_2": 0.0,
                    "setting_3": 0.0,
                }
            )
    observations = pd.DataFrame(rows)
    observations = observations.loc[:, ["split", "engine_id", "cycle", "setting_1", "setting_2", "setting_3", *[f"sensor_{i}" for i in range(1, 22)]]]
    features = build_rolling_features(observations, window_cycles=10)
    assert len(features) == 2 * (19 - 10 + 1)
