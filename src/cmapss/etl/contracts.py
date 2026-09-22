"""Schema output cho các lớp ETL Silver và Gold."""

from cmapss.data.contracts import OBSERVATION_COLUMNS, TRAIN_TARGET_COLUMNS

ETL_SCHEMA_VERSION = 1
SILVER_OBSERVATION_COLUMNS = OBSERVATION_COLUMNS
SILVER_TARGET_COLUMNS = TRAIN_TARGET_COLUMNS
GOLD_ENGINE_COLUMNS = (
    "split",
    "engine_id",
    "n_cycles",
    "first_cycle",
    "last_cycle",
)
GOLD_ENGINE_KEY = ("split", "engine_id")
