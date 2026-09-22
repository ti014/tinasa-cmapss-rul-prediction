"""Schema và quy tắc được mô tả trong data/CONTRACTS.md."""

from typing import Literal

Split = Literal["train", "test"]
SPLITS = ("train", "test")
SETTINGS = tuple(f"setting_{i}" for i in range(1, 4))
SENSORS = tuple(f"sensor_{i}" for i in range(1, 22))
MEASUREMENTS = SETTINGS + SENSORS
RAW_COLUMNS = ("engine_id", "cycle") + MEASUREMENTS
KEY = ("split", "engine_id", "cycle")
OBSERVATION_COLUMNS = ("split",) + RAW_COLUMNS
TEST_LABEL_COLUMNS = ("split", "engine_id", "rul")
TRAIN_TARGET_COLUMNS = KEY + ("rul",)
PAIRED_SENSOR_WINDOW_CYCLES = 30
ENGINE_SENSOR_PAIR_KEY = ("split", "engine_id", "sensor")
ENGINE_SENSOR_PAIR_COLUMNS = ENGINE_SENSOR_PAIR_KEY + (
    "n_cycles",
    "early_start",
    "early_end",
    "late_start",
    "late_end",
    "n_early",
    "n_late",
    "early_mean",
    "late_mean",
    "difference",
    "pair_status",
)
SENSOR_INFERENCE_COLUMNS = (
    "sensor",
    "n_pairs",
    "n_positive",
    "n_negative",
    "n_zero",
    "mean_difference",
    "median_difference",
    "ci_method",
    "ci_level",
    "ci_lower",
    "ci_upper",
    "bootstrap_resamples",
    "bootstrap_status",
    "wilcoxon_method",
    "wilcoxon_statistic",
    "raw_p_value",
    "p_value_adjusted",
    "fdr_method",
    "fdr_q",
    "fdr_reject",
    "test_status",
)
FD001_ROWS = {"train": 20_631, "test": 13_096}
FD001_ENGINE_COUNT = 100
RAW_DTYPES = {
    "engine_id": "int64",
    "cycle": "int64",
    **{column: "float64" for column in MEASUREMENTS},
}


class ContractError(ValueError):
    """Lỗi có mã quy tắc, đối tượng vi phạm và hướng xử lý."""

    def __init__(self, rule: str, item: str, detail: str) -> None:
        self.rule = rule
        anchor = rule.lower().replace("_", "-")
        super().__init__(
            f"[{rule}] {item}: {detail}. "
            f"Hành động: đối chiếu đầu vào với data/CONTRACTS.md#{anchor}"
        )
