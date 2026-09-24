"""Contract cho artifact PCA--ARIMA trên Health Index train FD001."""

TIMESERIES_SCHEMA_VERSION = 3
TIMESERIES_SEED = 42
TIMESERIES_N_ENGINES = 10
TIMESERIES_TRAIN_FRACTION = 0.70
TIMESERIES_PREDICTION_ALPHA = 0.05
TIMESERIES_MIN_TRAIN_SIZE = 30
TIMESERIES_ARIMA_P_MAX = 2
TIMESERIES_ARIMA_Q_MAX = 2
TIMESERIES_ARIMA_D_VALUES = (0, 1)

HEALTH_INDEX_COLUMNS = ("split", "engine_id", "cycle", "health_index")
SELECTED_ENGINE_COLUMNS = ("split", "engine_id", "fold", "n_cycles", "selection_rank")
FORECAST_COLUMNS = (
    "split",
    "engine_id",
    "cycle",
    "health_index_true",
    "health_index_pred",
    "lower_95",
    "upper_95",
    "is_holdout",
)
FORECAST_METRIC_COLUMNS = (
    "split",
    "engine_id",
    "fold",
    "n_train",
    "n_holdout",
    "order",
    "mae",
    "rmse",
    "coverage_95",
    "fit_converged",
    "fit_warning_count",
    "status",
    "error",
)
BASELINE_METRIC_COLUMNS = (
    "split",
    "engine_id",
    "fold",
    "baseline_name",
    "n_train",
    "n_holdout",
    "mae",
    "rmse",
)
DIAGNOSTIC_COLUMNS = (
    "split",
    "engine_id",
    "fold",
    "n_train",
    "adf_statistic",
    "adf_pvalue",
    "adf_usedlag",
    "adf_nobs",
    "ljung_box_lag",
    "ljung_box_statistic",
    "ljung_box_pvalue",
    "order",
    "model_df",
    "fit_converged",
    "fit_warning_count",
    "status",
    "error",
)
