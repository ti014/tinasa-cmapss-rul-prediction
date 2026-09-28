<a href="README.md"><img src="https://flagcdn.com/20x15/us.png" alt="English" width="20" height="15"> English</a> · <a href="README.vi.md"><img src="https://flagcdn.com/20x15/vn.png" alt="Tiếng Việt" width="20" height="15"> Tiếng Việt</a>

<hr>

# NASA C-MAPSS RUL Prediction Pipeline

A reproducible Big Data course project for analyzing multivariate turbofan sensor data and predicting Remaining Useful Life (RUL) on NASA C-MAPSS FD001--FD004. FD001 is the primary subset for detailed analysis; FD002--FD004 are evaluated in a supplemental cross-dataset matrix. The project emphasizes data provenance, statistically sound analysis, leakage-safe evaluation, scalable ETL, and explainable visual reporting.

## Project Status

This repository provides **FD001 data validation, train RUL targets, an executed train-only EDA notebook, sensor inference artifacts, Pandas/PySpark Silver/Gold ETL, 30-cycle rolling features, grouped RUL evaluation, train-only PCA--ARIMA Health Index analysis, and a supplemental 10-cycle evaluation matrix for FD001--FD004**.

**Improvement study completed (27 September 2026):** nested 5×3 GroupKFold
compared windows 10/20/30, 24 HistGradientBoosting configurations and three weight
schemes. Outer low-RUL MAE changed from 3.6676 to 3.6730 cycles; late warnings
increased from 49 to 53/100 engines. The final selection retained the baseline,
so test MAE remains 15.3490. Some inner-CV candidates improved low-RUL error at
the cost of other metrics. See [results and reproduction](experiments/README.md).

Available now:

- The original NASA C-MAPSS FD001--FD004 files and source documentation; the primary report uses FD001 and the supplemental matrix uses all four subsets.
- Readers, validators, and tests for row counts, engines, schema, keys, cycles, and train RUL.
- A [data contract](data/CONTRACTS.md), JSON validation CLI, and dependency lockfile.
- An [executed data-pipeline walkthrough notebook](notebooks/00_data_pipeline_walkthrough.ipynb)
  that shows, step by step on FD001, how the files are combined, stored,
  queried (SQLite SQL, Spark SQL and Pandas on Parquet), transformed into the
  126 features, and joined; it also checks every table relationship on the data.
- An [executed EDA notebook](notebooks/01_fd001_eda.ipynb), 12 CSV tables,
  11 figures, and [analysis notes and validation evidence](outputs/eda/README.md).
- A curated bibliography and reading guide.
- A [LaTeX report](report/README.md) and [PDF](report/main.pdf) covering data
  validation, EDA, train-only sensor inference, ETL parity, rolling features,
  grouped RUL evaluation, PCA--ARIMA, early warning, and the supplemental
  FD001--FD004 matrix. It includes a data dictionary, glossary, error analysis,
  and reproducibility notes; the built PDF is 49 pages.
- A cited theoretical background and [statistical protocol](report/sections/methodology.tex) (Section 4.6).
  The paired comparison uses each sensor's mean over the first and last 30 cycles of every train engine.
  The inference configuration was fixed before execution.
- A train-only `build_engine_sensor_pairs` helper that creates and validates one
  first/last 30-cycle pair for every engine and all 21 sensors.
- Helpers for mean paired differences, percentile bootstrap confidence intervals,
  Wilcoxon permutation, and Benjamini-Hochberg FDR under that configuration.
- An [executed inference notebook](notebooks/02_fd001_inference.ipynb),
  two validated CSV tables, run metadata, and [validation notes](outputs/statistics/README.md).
  The outputs keep all 21 sensors and do not read test observations or test labels.
- Figures generated from those artifacts:
  [difference distributions](outputs/statistics/figures/difference_distributions.png)
  and [effect sizes with bootstrap CI](outputs/statistics/figures/effect_size_ci.png).
- A Pandas ETL that writes typed Silver observations and train targets to
  Parquet, writes a Gold engine summary to Parquet and SQLite, records raw
  checksums, and runs isolated 1x/10x/100x scalability measurements in
  [outputs/etl](outputs/etl/README.md).
- A lazy PySpark adapter and parity checker for the same Silver/Gold boundaries.
  The current local run used a portable JRE and wrote `outputs/etl/parity.json`;
  all three Pandas/PySpark boundaries matched with `max_abs_diff=0.0`.
- A deterministic rolling feature pipeline in [outputs/features](outputs/features/README.md).
  It writes 27,927 Gold rows from 30-cycle trailing windows, keeps all 21 sensors,
  creates 126 feature columns, and does not join test labels.
- A grouped modeling pipeline in [outputs/modeling](outputs/modeling/README.md).
  It compares a median baseline, Ridge, and HistGradientBoosting with five-fold
  GroupKFold by engine, writes out-of-fold predictions, writes exactly 100 final
  test predictions before reading labels, and evaluates the selected model on
  NASA test labels.
- A train-only [PCA--ARIMA Health Index analysis](outputs/timeseries/README.md)
  on 10 selected validation engines. StandardScaler/PCA are fit on the other
  90 train engines before the selected engines are transformed. It writes
  20,631 Health Index rows, 704 holdout forecasts, ARIMA and same-horizon
  last-value/drift baseline MAE/RMSE, 95% prediction-interval coverage, ADF and
  adjusted Ljung--Box diagnostics without reading test labels.
- An [artifact-driven Streamlit dashboard](app.py) with eight pages: overview,
  data and EDA, RUL evaluation (test engine and OOF trajectory), early warning,
  cross-dataset, improvement study, Health Index/ARIMA and provenance. Charts are
  interactive Altair charts. It does not fit models or read raw test labels when
  opened.
- Three executed result-review notebooks:
  [modeling and early warning](notebooks/03_fd001_modeling_evaluation.ipynb),
  [Health Index/ARIMA](notebooks/04_fd001_health_index_arima.ipynb) and
  [cross-dataset and improvement study](notebooks/05_cross_dataset_and_improvement.ipynb).
  Each notebook recomputes metrics from stored predictions with package
  functions and `assert`s that they match the artifacts; none fits a model or
  writes to `outputs/`.

Final handoff:

- The [20-slide presentation deck](slides/cmapss_fd001_presentation.pptx)
  ([PDF](slides/cmapss_fd001_presentation.pdf)) is sized for a 30-minute talk in
  five parts: problem and data, Big Data pipeline, time-series analysis,
  modeling and evaluation, and extensions and conclusions. Slide values are read
  from `outputs/` by [`slides/extract_slide_data.py`](slides/extract_slide_data.py);
  the [speech script](handoff/SPEECH_SCRIPT_VI.pdf) gives timing per slide. The
  [handoff checklist](handoff/FINAL_HANDOFF.md) lists validation.

Installation, data validation, notebook, ETL, feature-generation, modeling and
time-series instructions are available below. The dashboard requires the
optional `dashboard` dependency group.

## Objectives

The report addresses five primary questions on FD001:

1. How trustworthy and suitable is FD001 for an academic predictive-maintenance study?
2. Which sensor signals exhibit measurable degradation across engine life cycles?
3. Do Pandas and PySpark produce matching ETL outputs on the same input?
4. How accurately can RUL be predicted from the most recent 30 operating cycles?
5. How can model results be communicated as transparent, reproducible early-warning evidence?

A supplemental question tests how a model fitted on FD001 transfers to FD002--FD004 under a common 10-cycle protocol.

The primary objective is **RUL prediction**. Supervised regression is the modeling formulation because the target is a numeric number of remaining cycles; it is not the title or purpose of the project.

## Dataset

The repository uses the four simulated run-to-failure NASA C-MAPSS subsets. FD001 is the primary subset for detailed analysis; FD002--FD004 support the supplemental evaluation.

| Subset | Train rows | Test rows | Train engines | Test engines and labels |
| --- | ---: | ---: | ---: | ---: |
| FD001 | 20,631 | 13,096 | 100 | 100 |
| FD002 | 53,759 | 33,991 | 260 | 259 |
| FD003 | 24,720 | 16,596 | 100 | 100 |
| FD004 | 61,249 | 41,214 | 249 | 248 |

Each subset has `train_FD00x.txt`, `test_FD00x.txt`, and `RUL_FD00x.txt` files. Train histories run to failure; test histories are truncated, with one separate RUL label per test engine at its final observed cycle. Counts come from the [dataset-aware ETL artifacts](outputs/cross_dataset/README.md).

Each observation contains:

- Engine identifier.
- Operating cycle.
- Three operational settings.
- Twenty-one sensor measurements.

FD001 represents one operating condition and one High-Pressure Compressor degradation mode. FD002--FD004 add operating conditions and fault modes. The signals contain simulated sensor noise. Supplemental FD002--FD004 metrics are recorded in `outputs/cross_dataset`.

### Provenance and Integrity

- [NASA Open Data record](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- [Archived dataset with DOI and checksum](https://doi.org/10.5281/zenodo.15346912)
- Recorded ZIP checksum: `MD5 79a22f36e80606c69d0e9e4da5bb2b7a`
- Local provenance notes: [`data/raw/DATA_SOURCE.md`](data/raw/DATA_SOURCE.md)

The data is simulated, not field telemetry. Results support reproducible academic
comparison with FD001 as the primary subset and FD002--FD004 as supplemental
evaluations; they do not establish real-world deployment performance.

## Data Architecture

```text
NASA raw text files
        |
        v
Bronze: immutable source files
        |
        v
Silver: typed and validated observations in Parquet
        |
        v
Gold: engine summary, rolling features, predictions, metrics, and SQLite tables
        |
        +-------------------+
        |                   |
        v                   v
Statistical analysis   RUL prediction models
        |                   |
        +---------+---------+
                  v
        Figures, Streamlit dashboard,
        and final LaTeX report
```


### Data relationships and tool boundaries

Each row is identified by `(split, engine_id, cycle)`. An engine has many cycles;
each cycle contains 3 operational settings, 21 sensors, and a train RUL label.
For an engine ending at cycle 200, the observation at cycle 170 has RUL = 30.
The rolling window at cycle 170 uses cycles 141--170 from the same engine to
create 126 features (six summaries for each of 21 sensors).

| Step | Tools | Main output |
| --- | --- | --- |
| Read and validate | Python, Pandas | Schema, keys, cycle continuity, train RUL |
| ETL | Pandas, PySpark, Parquet, SQLite | Silver/Gold and parity |
| Features | Pandas, NumPy | 30-cycle window, 126 features |
| Modeling | scikit-learn | GroupKFold and HistGradientBoostingRegressor |
| Evaluation | scikit-learn, SciPy | MAE, RMSE, NASA score, bootstrap CI |
| Visualization | Matplotlib, Seaborn, Streamlit | Figures, dashboard, results |

FD001 is not large enough to claim Big Data by volume alone. The report shows
scalability through Bronze/Silver/Gold ETL, Parquet, PySpark parity, and 1x/10x/100x benchmarks.

Pandas is used in the EDA notebook, ETL reference path, statistics and model
preparation. PySpark reproduces the ETL boundaries for parity; recorded runtime
benchmarks measure Pandas only. The original dataset is only a few megabytes, so the
repository will not claim that FD001 itself demonstrates large-scale data
processing.

### Data access and transformations

The [data-processing walkthrough](handoff/DATA_WALKTHROUGH.md) documents the
implemented behavior: reading the two observation files, unioning train/test
rows while retaining `split`, creating train RUL, building rolling windows,
selecting 126 feature columns, and joining on `(split, engine_id, cycle)`. It
also records the actual query boundary: SQLite contains only `engine_summary`
and `train_targets`; observations are read from Silver Parquet.

FD001--FD004 are processed in separate subset directories. The repository does
not union all four subsets into one table because the current key has no
`dataset_id`. The implemented relationships are one-to-many from an engine to
its observations and one-to-one joins for features/targets and final
predictions/test labels. No many-to-many relationship, Airflow DAG, or Power BI
artifact is implemented. See the [editable relationship diagram](assets/diagrams/data-relations.drawio)
and [SVG preview](assets/diagrams/data-relations.svg).

In the ETL, “normalization” means schema and datatype validation. Numeric
scaling is a separate step: Ridge uses `StandardScaler` inside each training
fold, while the Health Index uses `StandardScaler` before PCA. HGB consumes
features on their original scale; rolling feature generation does not fit one
global scaler.

## Methodology

### Data Quality

- Validate the 26-column schema and numeric types.
- Enforce uniqueness of `(split, engine_id, cycle)`.
- Verify sequential cycles within each engine.
- Record row counts before and after every transformation.
- Keep train and test engine identifiers scoped to their own splits.

### Statistical Analysis

- Compare early-life and late-life sensor behavior per engine.
- Report mean paired differences and bootstrap 95% confidence intervals.
- Use paired Wilcoxon signed-rank tests where appropriate.
- Control multiple testing across sensors with Benjamini-Hochberg FDR.
- Estimate metric uncertainty with engine-level bootstrap resampling.

### Feature Engineering

The implemented feature representation uses a trailing window of 30 cycles. It keeps all 21 sensors in the feature artifact so that sensor selection is not performed on the full train set. For each sensor, the generated features are:

- Latest value.
- Mean and sample standard deviation (`ddof=1`).
- Minimum and maximum.
- Linear trend slope over positions 1--30 within the window.

Windows never cross engine boundaries and never use future observations. Engines with fewer than 30 cycles are rejected rather than padded or shortened.

### RUL Prediction

The implemented first modeling pass compares:

1. A naive median-RUL baseline.
2. Ridge regression as a regularized linear baseline.
3. HistGradientBoostingRegressor as the nonlinear candidate model.

Model selection uses five-fold grouped cross-validation, with `engine_id` as the group.
The official NASA test labels remain isolated until final predictions have been
written. HistGradientBoostingRegressor is currently selected. Final test metrics
are mean engine-level MAE 15.3490, RMSE 20.5413, and NASA score 1513.2577, with
bootstrap confidence intervals recorded in `outputs/modeling/tables/final_metrics.json`.

### FD002--FD004 Evaluation

The raw archive now contains FD001--FD004. The additional matrix uses the same
pipeline with a 10-cycle window because the shortest FD004 test engine has 19
cycles; it does not replace the published FD001 result that uses a 30-cycle
window. Within-subset MAE is 21.9258 on FD001, 22.3556 on FD002, 28.3255 on
FD003, and 27.2218 on FD004. Keeping the model fit on FD001 frozen and
transferring it to other subsets gives MAE 60.3215 on FD002, 47.3949 on FD003,
and 58.5731 on FD004. The FD001 model should therefore not be applied directly
to another subset without retraining or domain adaptation. See the [transfer
evaluation artifact](outputs/cross_dataset/README.md).

### ARIMA Analysis

ARIMA is a secondary time-series analysis, not the primary RUL model. The
implemented pipeline fits a standardized PCA Health Index on train observations
and forecasts the final 30% of 10 representative validation engines. It reports:

- Forecast MAE and RMSE.
- 95% prediction-interval coverage.
- Augmented Dickey-Fuller stationarity diagnostics.
- Ljung-Box residual autocorrelation diagnostics.

The current run selects ARIMA order by an AIC grid on each 70% fit segment only.
All 10 engines have converged fits; forecast MAE is 0.1515--1.0372 and 95%
prediction-interval coverage is 21.43%--97.03%. ADF does not reject
non-stationarity on the fit segments. After adjusting Ljung--Box by `p+q`,
engines 69 and 70 have p-values below 0.05. ARIMA is not applied directly to
the deterministic training RUL countdown; Health Index forecast is not RUL.
On the same holdouts, mean MAE is 0.4110 for ARIMA, 0.7469 for drift and 0.9478
for last-value; this comparison is descriptive and has no claim of statistical
superiority.
The fixed 30-cycle OOF warning protocol produces 44 before-threshold, 7 at-threshold,
and 49 after-threshold first warnings. Lead time has median 30 cycles and range
19--55 cycles. Because OOF predictions also selected the model, this is internal
descriptive evidence rather than an independent post-selection evaluation.

## Big Data Scope

The Big Data component focuses on lifecycle and reproducibility rather than overstating dataset volume:

- Equivalent Pandas and PySpark ETL implementations.
- Parquet as the analytical storage format.
- SQLite as a portable local query and reporting layer.
- Row-count, key, and aggregate parity checks between processing engines.
- Scalability benchmarks at 1×, 10×, and 100× volume.

Replicated records used in the scalability benchmark will receive remapped engine identifiers and remain isolated from scientific analysis and model training.

## Evaluation Protocol

The estimand, unit of analysis, split/horizon, metric role, and uncertainty
method are recorded in
[`outputs/modeling/tables/evaluation_protocol.csv`](outputs/modeling/tables/evaluation_protocol.csv).
It keeps OOF train model selection, final test-engine evaluation, Health Index
trajectory forecasting, train-only low-RUL warning analysis, and the descriptive
Pandas benchmark as separate protocols. The warning artifact uses a fixed
`RUL <= 30` threshold on selected-model OOF predictions; it is descriptive and
does not use test labels.

- Use a trailing window of 30 cycles.
- Split validation folds by engine, never by overlapping rows or windows.
- Fit scalers, feature selection, PCA, and models using training folds only.
- Select models using mean engine-level MAE.
- Refit the selected model on all training engines.
- Produce exactly one final prediction for each of the 100 test engines.
- Join predictions with `RUL_FD001.txt` only during final evaluation.
- Report aggregate metrics, confidence intervals, and error slices by RUL range.

No target score is declared in advance. A negative result remains valid if the protocol is preserved and the failure modes are analyzed honestly.

## Visualizations

- Engine-life distribution.
- Sensor variability and raw-cycle trajectories.
- Train-only Health Index trajectories and holdout forecasts for the
  selected engines in [the PCA--ARIMA artifacts](outputs/timeseries/README.md).
- Final test-set true-versus-predicted RUL scatter and residuals by
  RUL range in [the modeling artifacts](outputs/modeling/README.md).
- Per-engine inspection through the Streamlit dashboard and persisted
  out-of-fold predictions for grouped validation analysis.
- Scope note: normalized sensor trajectories and a dedicated validation scatter
  are not separate persisted figures in the current handoff.

An illustrative warning may be shown when predicted RUL is at most 30 cycles. This is not presented as an optimized maintenance threshold because FD001 contains no maintenance costs, downtime, or intervention records.

## Repository Structure

Current repository structure:

```text
.
├── data/
│   ├── raw/                 # Original FD001--FD004 files and provenance notes
│   └── archive/             # Data retained for comparison, not active modeling
├── references/              # Bibliography, reading guide, papers, assignment context
├── notebooks/               # Executed: 00 data walkthrough, 01 EDA, 02 statistics, 03–05 result review
├── src/cmapss/              # data/, statistics/, etl/, features/, modeling/, timeseries/, eda.py, validate.py,
│                            # dashboard.py (artifact loaders), dashboard_app.py (Streamlit UI)
├── scripts/                 # Notebook builders, report figures and improvement-study verification
├── tests/                   # Data contract, CLI, EDA, statistics, ETL, features, modeling, timeseries, and reproducibility tests
├── outputs/validation/      # Data-validation reports
├── outputs/eda/             # EDA tables, figures, methods, and validation
├── outputs/statistics/      # Train-only sensor inference artifacts and validation
├── outputs/etl/             # Silver/Gold Parquet, SQLite, metadata, benchmark
├── outputs/features/        # Gold rolling feature Parquet and metadata
├── outputs/modeling/        # Grouped CV, predictions, model artifact, final metrics
├── outputs/timeseries/      # Train-only PCA--ARIMA Health Index artifacts
├── outputs/cross_dataset/   # FD001--FD004 ETL, features, and transfer results
├── assets/diagrams/         # Editable draw.io sources, SVG previews, URL handles
├── pyproject.toml           # Package and direct dependencies
├── requirements.txt         # Pip installation: local package and runtime libraries
├── requirements-dev.txt     # Notebook and test dependencies
├── requirements-etl.txt     # Optional PySpark ETL dependencies
├── requirements-dashboard.txt # Optional Streamlit dashboard dependencies
├── uv.lock                  # Locked dependency versions
├── report/                  # LaTeX source, PDF, figures, and evidence
├── slides/                  # 20-slide deck, PDF export, data extractor and builder
│   └── archive/             # Superseded decks and builders
├── handoff/                 # Demo script and final handoff checklist
│   └── archive/             # Legacy speech source
├── app.py                   # Streamlit entry point (calls cmapss.dashboard_app)
├── README.md                # English documentation
└── README.vi.md             # Vietnamese documentation
```

Local agent-support files are intentionally excluded from GitHub through `.gitignore`.

## Reproducibility

The repository provides one documented path from immutable raw data to processed artifacts, metrics, figures, dashboard data, and the final report. Reproducibility requirements include:

- Seed `42` for stochastic models and bootstrap; GroupKFold does not shuffle.
- Version-pinned Python dependencies.
- Logged row counts and validation outcomes.
- Persisted model configurations and evaluation results.
- Separation of scientific data from synthetic scalability data.
- Automated checks for schema, leakage, window boundaries, and prediction count.

## Getting Started

With Python 3.12 available, create the environment and install dependencies
once from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Then run data validation:

```powershell
.venv\Scripts\python -m cmapss.validate
```

This reads FD001, validates the data, creates and checks train RUL in memory,
and writes the [validation report](outputs/validation/fd001.json).
It does not read test labels.

[`requirements.txt`](requirements.txt) includes runtime dependencies and installs
the local package through `-e .`. For notebooks, development, or tests, install
[`requirements-dev.txt`](requirements-dev.txt) instead.

The data contract and train-only paired-data flow live in `src/cmapss/`:

```text
contracts.py     Column names, dtypes, and the validation exception
readers.py       Read source files into DataFrames
validation.py    Validate schemas, keys, and cycles
targets.py       Create and validate train RUL
statistics/
  config.py       Fixed inference configuration
  pairs.py        Build and validate engine/sensor pairs
  inference.py    Bootstrap, Wilcoxon, FDR, and sensor summaries
  validation.py   Validate sensor inference tables
  artifact_io.py  CSV/JSON, hash, and round-trip readers
  metadata.py     Run metadata and validation payloads
  artifacts.py    Write CSV/JSON artifacts and round-trip validation
  reproducibility.py
                  Repeated-run validation, runtime audit, and artifact comparison
statistics_artifacts.py
                  Compatibility import for the artifact writer
```

For example, a train engine ending at cycle 200 has `rul = 200 - cycle`.
The target function rejects test observations. See the [data contract](data/CONTRACTS.md)
for conditions and validation limits.

### Overall data pipeline

The public draw.io diagram below shows the detailed FD001 path: EDA, paired
statistics, and PCA--ARIMA read train data independently; Silver/Gold feeds
rolling features and RUL modeling; test labels are joined only after final
predictions. The [report pipeline figure](report/figures/pipeline.pdf) also
marks the supplemental FD001--FD004 evaluation scope.

![FD001 overall data pipeline](assets/diagrams/fd001-pipeline.png)

[Editable Draw.io source](assets/diagrams/fd001-pipeline.drawio) ·
[Report PDF figure](report/figures/pipeline.pdf)

### Run the EDA notebook

After installing `requirements-dev.txt`, open the notebook from the repository root:

```powershell
.venv\Scripts\python -m jupyterlab notebooks/01_fd001_eda.ipynb
```

Select the `.venv` Python kernel and **Restart Kernel and Run All Cells**.
This analyzes only train and recreates 12 CSV tables and 11 PNG figures.
Validated fresh-kernel runs produced identical EDA artifacts in the same
environment. The repository-wide test suite passed **188 tests** (run on
2026-09-28), including data, statistics, ETL, feature, modeling, PCA--ARIMA,
cross-dataset, dashboard, and end-to-end reproducibility checks.

![FD001 train EDA workflow](assets/diagrams/fd001-eda.svg)

[Editable diagram](assets/diagrams/fd001-eda.drawio) ·
[Methods, findings, and limits](outputs/eda/README.md)

### Run the statistics notebook

After installing `requirements-dev.txt`, open the train-only inference notebook:

```powershell
.venv\Scripts\python -m jupyterlab notebooks/02_fd001_inference.ipynb
```

Running all cells recreates `outputs/statistics/tables/engine_sensor_pairs.csv`,
`outputs/statistics/tables/sensor_inference.csv`, `run_info.json`, and
`validation.json`, together with
`outputs/statistics/figures/difference_distributions.png` and
`outputs/statistics/figures/effect_size_ci.png`. For example, the current
artifact has 2,100 engine-sensor pairs and 21 sensor-level inference rows. Six
globally constant sensors remain in the output as `all_zero_difference`; 15
testable sensors reject the null after Benjamini-Hochberg FDR under the fixed
protocol. These results are train-only and are not feature selection.

![FD001 train statistics workflow](assets/diagrams/fd001-statistics.svg)

[Editable draw.io source](assets/diagrams/fd001-statistics.drawio) ·
[Browser URL file](assets/diagrams/fd001-statistics.drawio-url.txt) ·
[Methods, tables, validation, and limits](outputs/statistics/README.md)

For example, the 100 train engines have a median lifetime of 199 cycles.
Six sensors are globally constant; `sensor_6` varies globally but is constant
within 38 engines. The inference figures show each sensor in its own panel so that
different sensor scales are not ranked directly. `sensor_6` illustrates that a
small mean difference can coexist with a small p-value, while `sensor_9`
illustrates why the per-engine distribution should be read together with the
mean and CI. These are train-only findings, not model feature selection. A
repeated-run check ran the notebook twice from fresh kernels; the two CSV files and two PNG files
matched, and the runtime audit blocked probes that tried to open `test_FD001.txt`
and `RUL_FD001.txt`.

Run the repeated-run statistics validation with:

```powershell
.venv\Scripts\python -m cmapss.statistics.reproducibility `
  --project-root . `
  --notebook notebooks/02_fd001_inference.ipynb `
  --output-dir outputs/statistics `
  --work-dir tmp/statistics-reproducibility `
  --timeout 1800
```

This is a local `PYTHONPATH` runtime audit, not a filesystem sandbox, CI
enforcement, or verified branch protection.

### Run the ETL

The Pandas reference pipeline reads and validates both FD001 observation files,
writes Silver/Gold artifacts, and records the 1x/10x/100x benchmark:

```powershell
.venv\Scripts\python -m cmapss.etl `
  --raw-dir data/raw `
  --output-dir outputs/etl `
  --benchmark-scales 1,10,100 `
  --benchmark-repetitions 2
```

Install the optional PySpark extra and provide Java/JAVA_HOME before requesting
parity:

```powershell
uv sync --frozen --extra etl
.venv\Scripts\python -m cmapss.etl `
  --with-spark `
  --raw-dir data/raw `
  --output-dir outputs/etl
```

See [ETL artifacts and validation](outputs/etl/README.md) for schemas, row
counts, benchmark measurements, and the Pandas/PySpark parity result.

### Generate rolling features

After ETL has written Silver observations, create Gold rolling features:

```powershell
.venv\Scripts\python -m cmapss.features `
  --silver-observations outputs/etl/silver/observations.parquet `
  --output-dir outputs/features
```

The current artifact has 27,927 rows: 17,731 train windows and 10,196 test
windows. See [rolling feature artifacts and validation](outputs/features/README.md)
for the column contract, row counts, leakage checks, and limitations.

### Run RUL modeling

After ETL and rolling features exist, run grouped model training and final
evaluation:

```powershell
.venv\Scripts\python -m cmapss.modeling `
  --rolling-features outputs/features/gold/rolling_features.parquet `
  --train-targets outputs/etl/silver/train_targets.parquet `
  --test-labels data/raw/RUL_FD001.txt `
  --output-dir outputs/modeling
```

The current run selects `hist_gradient_boosting` with final mean engine-level
MAE 15.3490, RMSE 20.5413, and NASA score 1513.2577 on 100 test engines.
See [modeling artifacts and validation](outputs/modeling/README.md) for fold
metrics, final predictions, final evaluation, figures, and limitations.

### Run FD001--FD004 evaluation

To reproduce the within-subset and transfer matrix, run:

```powershell
.venv\Scripts\python -m cmapss.cross_dataset `
  --raw-dir data/raw `
  --output-root outputs/cross_dataset `
  --window-cycles 10
```

The command creates four within-subset evaluations and three FD001 transfer
evaluations. See the [cross-dataset artifact](outputs/cross_dataset/README.md).
The matrix uses a 10-cycle window because FD004 has a 19-cycle test engine; it
does not replace the published FD001 30-cycle result.

### Run the PCA--ARIMA analysis

After ETL and grouped modeling have written their artifacts, run the secondary
Health Index forecast:

```powershell
.venv\Scripts\python -m cmapss.timeseries `
  --observations outputs/etl/silver/observations.parquet `
  --fold-assignments outputs/modeling/tables/fold_assignments.csv `
  --output-dir outputs/timeseries
```

The current artifact contains 20,631 train Health Index rows and 704 holdout
forecasts from 10 engines. See [PCA--ARIMA artifacts and diagnostics](outputs/timeseries/README.md)
for the selection rule, metrics, figures, and limitations.

### Run the result-review notebooks

Notebooks 03–05 only read existing artifacts. They recompute metrics with
package functions and stop on an `assert` if a number differs from the file in
`outputs/`:

| Notebook | Content | Checked against |
|---|---|---|
| `03_fd001_modeling_evaluation.ipynb` | Per-fold MAE, test metrics, bootstrap CI, error by RUL range, RUL ≤ 30 warning | `outputs/modeling/tables/*` |
| `04_fd001_health_index_arima.ipynb` | Health Index trend, ARIMA holdout metrics, naive baselines, ADF/Ljung–Box | `outputs/timeseries/tables/*` |
| `05_cross_dataset_and_improvement.ipynb` | FD001–FD004 matrix, nested-CV ablation and paired CI | `outputs/cross_dataset/matrix.json`, `outputs/experiments/rul_improvement/20260927_full/*` |

Build and execute them from a fresh kernel with:

```powershell
.venv\Scripts\python scripts/build_results_notebooks.py --execute
```

Add `--only 04` to rebuild a single notebook.

### Run the reproducibility dashboard

Install the optional dashboard dependencies and start Streamlit after the
modeling and PCA--ARIMA artifacts exist:

```powershell
uv sync --extra dashboard
uv run --extra dashboard streamlit run app.py
```

The dashboard has eight pages. A page can be opened directly with a query
string, for example `http://localhost:8501/?page=Cross-dataset`.

| Page | Artifacts read | Content |
|---|---|---|
| Tổng quan (overview) | `outputs/modeling` | Test metrics and CI, pipeline, CV model comparison, suggested demo order |
| Dữ liệu & EDA | `outputs/eda`, `outputs/features` | Lifetime distribution, sensor–RUL Spearman, per-engine sensor trajectories |
| Đánh giá RUL | `outputs/modeling/predictions` | One test engine at its last cycle, scatter and residuals; OOF trajectory of a train engine per model |
| Cảnh báo sớm | `outputs/modeling/tables` | 30-cycle threshold, timing classes, trajectory with the first-warning marker |
| Cross-dataset | `outputs/cross_dataset/matrix.json` | MAE and CI for 4 within-subset and 3 transfer evaluations |
| Thử nghiệm cải thiện | `outputs/experiments/rul_improvement/20260927_full` | Ablation, paired CI, baseline versus selected-procedure trajectories |
| Health Index / ARIMA | `outputs/timeseries` | Forecast with 95% prediction interval, naive-baseline comparison, diagnostics |
| Provenance | `run_info.json` | Schema and isolation contract of the run |

Artifact loaders live in `src/cmapss/dashboard.py` and the UI in
`src/cmapss/dashboard_app.py`. `app.py` is only the entry point and still
exports `load_dashboard_data`, which `cmapss.reproducibility` imports, so
importing `app` does not require Streamlit. The dashboard does not retrain
models or read `data/raw/RUL_FD001.txt`. The data page uses the
`sensor_*__last` and `sensor_*__mean` Gold rolling-feature columns, so
trajectories start at cycle 30.

### Validate the end-to-end pipeline

Regenerate the scientific path from immutable FD001 raw files into a fresh
temporary root and compare table content, metrics, figures, raw hashes and the
dashboard loader with the checked-in artifacts:

```powershell
uv run python -m cmapss.reproducibility `
  --output-root tmp/e2e-repro-final `
  --repo-root . `
  --evidence-path tmp/e2e-repro-final/validation.json
```

The command runs the Pandas ETL path with a 1x benchmark, rolling features,
grouped RUL modeling and train-only PCA--ARIMA. It uses a fresh output root,
keeps raw hashes unchanged, and compares persisted tables with numeric tolerance
`rtol=1e-9`, `atol=1e-9`. PySpark parity and notebook reproducibility remain
separate checks documented in the validation evidence.

<details>
<summary>Development and optional checks</summary>

- Security update: pytest 9.0.3, JupyterLab 4.5.10, and nbconvert 7.17.1.
  Reinstall `requirements-dev.txt` for an existing environment. See the
  [dependency patch and regression evidence](outputs/validation/dependencies.md).
- After installing `requirements-dev.txt`, run `.venv\Scripts\python -m pytest`.
- Add `--check-test-labels` to the main command for a separate label integrity
  check. It does not join labels or compute model metrics.
- Use `--raw-dir` or `--output` only to change the default paths.
- `requirements*.txt` is exported from `uv.lock`, retaining pinned direct and
  transitive dependencies. `pyproject.toml` still declares package metadata.
  The validated environment is Python 3.12.14 on Windows.
- Existing uv users can run `uv run cmapss-validate`. The pip workflow above
  does not require uv.

If pytest cannot access the Windows Temp directory, use a dedicated directory:

```powershell
New-Item -ItemType Directory -Force tmp | Out-Null
.venv\Scripts\python -m pytest --basetemp=tmp/pytest-data-contract
```

Pytest clears its `--basetemp` directory on subsequent runs; reserve it for tests.

</details>

The [report PDF](report/main.pdf) presents the verified FD001 analyses and the
supplemental FD001--FD004 matrix. See [report instructions](report/README.md)
for source organization and PDF build commands.

## Academic Context

This repository supports a final report for the **Big Data Analysis** course.
The report distinguishes observed results from interpretations. All
generated analysis, code, citations, and conclusions remain subject to human
review and reproducibility checks.

## References

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). [Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation](https://doi.org/10.1109/PHM.2008.4711414).
- Ramasso, E., & Saxena, A. (2014). [Review and Analysis of Algorithmic Approaches Developed for Prognostics on CMAPSS Dataset](https://doi.org/10.36001/phmconf.2014.v6i1.2512).
- Li, X., Ding, Q., & Sun, J.-Q. (2018). [Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks](https://doi.org/10.1016/j.ress.2017.11.021).
- See [`references/reading-guide.md`](references/reading-guide.md) for the reference guide.

## Limitations

- FD001 is simulated and represents one operating condition and one fault mode.
- Operating cycles are not equivalent to hours or calendar time.
- RUL prediction does not by itself determine an optimal maintenance schedule.
- Scalability benchmarks do not create additional independent scientific evidence.
- Model performance on FD001 does not establish real-world generalization.

## License and Data Use

No license has been selected for the project source code yet. The NASA Open Data listing currently does not specify a dataset license, so redistribution and reuse should preserve source attribution and be reviewed against the source record before publication. A project license and final data-distribution policy will be added before the repository is presented as reusable software.
