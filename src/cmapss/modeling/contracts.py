"""Contract cho artifact modeling RUL FD001."""

from cmapss.data.contracts import KEY
from cmapss.features.contracts import ROLLING_FEATURE_VALUE_COLUMNS

# Mốc cuối dành cho nhóm test >90; OOF dùng các mốc chi tiết đến 400.
RUL_BINS = (-1, 30, 60, 90, 125, 150, 200, 400, 10_000)

MODELING_SCHEMA_VERSION = 3
MODELING_SEED = 42
MODELING_N_SPLITS = 5
MODELING_BOOTSTRAP_RESAMPLES = 10_000
MODELING_PRIMARY_METRIC = "mean_engine_mae"
MODELING_WARNING_RUL_THRESHOLD = 30
MODELING_KEY = KEY
MODELING_FEATURE_COLUMNS = ROLLING_FEATURE_VALUE_COLUMNS

FOLD_ASSIGNMENT_COLUMNS = ("split", "engine_id", "fold")
OOF_PREDICTION_COLUMNS = (
    "split",
    "engine_id",
    "cycle",
    "fold",
    "model_name",
    "y_true",
    "y_pred",
    "absolute_error",
    "squared_error",
    "nasa_score_component",
)
FOLD_METRIC_COLUMNS = (
    "model_name",
    "fold",
    "n_rows",
    "n_engines",
    "mean_window_mae",
    "mean_engine_mae",
    "rmse",
    "nasa_score",
)
MODEL_METRIC_COLUMNS = (
    "model_name",
    "n_folds",
    "mean_engine_mae",
    "std_engine_mae",
    "mean_rmse",
    "mean_nasa_score",
    "selected",
)
FINAL_PREDICTION_COLUMNS = ("split", "engine_id", "cycle", "model_name", "rul_pred")
FINAL_EVALUATION_COLUMNS = (
    "split",
    "engine_id",
    "cycle",
    "model_name",
    "y_true",
    "y_pred",
    "absolute_error",
    "squared_error",
    "nasa_score_component",
)

EVALUATION_PROTOCOL_COLUMNS = (
    "protocol_id",
    "target",
    "unit_of_analysis",
    "group_key",
    "split_and_horizon",
    "metrics",
    "selection_or_reporting_role",
    "uncertainty_or_repetition",
    "artifact",
)
