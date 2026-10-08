<a href="README.md"><img src="https://flagcdn.com/20x15/us.png" alt="English" width="20" height="15"> English</a> · <a href="README.vi.md"><img src="https://flagcdn.com/20x15/vn.png" alt="Tiếng Việt" width="20" height="15"> Tiếng Việt</a>

<hr>

# Topic 8: IoT Data Analysis and Predictive Maintenance on NASA C-MAPSS

**Student:** Phan Nguyễn Mai Phương · **Student ID:** 25002711.

**Last updated 2026-10-08:** the report, slides and dashboard now follow the course
rubric order: data pipeline → denoising and time windows → degradation trend and
anomalies → RUL prediction → early warning and maintenance schedule. Added EWMA
denoising, Health Index anomaly detection and a predicted-maintenance report for
the 100 test engines. Model-improvement studies are grouped as extended experiments.

This repository supports the final report for the Big Data Analytics course,
**Topic 8 — IoT Data Analysis and Predictive Maintenance**: analyze equipment sensor
data to detect early signs of abnormal behavior and predict when maintenance is
needed. The data are the simulated NASA C-MAPSS turbofan engines; the main analysis
uses FD001, and FD002--FD004 appear only in extended experiments.

## Project Status

Main FD001 results:

| Step | Result | Artifacts |
|---|---|---|
| Data pipeline | Schema, key and cycle checks; Silver/Gold Parquet and SQLite; Pandas and PySpark match exactly; at 10× scale local Spark is about 3.4× faster than Pandas | `outputs/etl/`, `outputs/benchmarks/` |
| Denoising | EWMA (span 10) removes about 86% of short-term fluctuation across 14 trending sensors | `outputs/degradation/` |
| Time windows | 30-cycle window, 21 sensors × 6 statistics = 126 features | `outputs/features/` |
| Trend and anomalies | 15/21 sensors change significantly between early and late life; the Health Index flags an anomaly in 100/100 train engines, median 121 cycles before failure | `outputs/statistics/`, `outputs/degradation/` |
| RUL prediction | HistGradientBoosting, 5-fold GroupKFold by engine; test MAE 15.35 cycles (95% CI 12.73--18.09), RMSE 20.54 | `outputs/modeling/` |
| Early warning | Predicted RUL ≤ 30: no missed engine, median lead time 30 cycles, 49% late; a 40-cycle threshold reduces late warnings to 6% | `outputs/modeling/tables/` |
| Maintenance schedule | 21 test engines are flagged for immediate maintenance and all 21 are correct; the other 4 engines that truly need maintenance are scheduled 1--8 cycles later | `outputs/maintenance/` |

Deliverables:

- [Report PDF](report/main.pdf) (45 pages, [LaTeX source](report/README.md)).
- [Slides](slides/cmapss_fd001_presentation.pptx) ([PDF](slides/cmapss_fd001_presentation.pdf)),
  20 slides on the IUH template, built by [`slides/build_presentation.py`](slides/build_presentation.py).
- A nine-page [Streamlit dashboard](app.py) that only reads artifacts.
- Seven executed notebooks in [`notebooks/`](notebooks/), each identifying the student.
- [Presenter handbook](handoff/SPEECH_SCRIPT_VI.pdf) (each slide with what to point at and what
  to say, plus the dashboard demo), [handoff package](handoff/FINAL_HANDOFF.md) and
  [course alignment](handoff/COURSE_ALIGNMENT.md).

Extended experiments (nested CV, 42 history features, similarity-based RUL,
LightGBM, TCN, FD002--FD004 transfer, ARIMA) remain in
[`experiments/`](experiments/README.md) and `outputs/experiments/`. None of them
changes the main model or the warning policy.

Installation and step-by-step commands are in Getting Started. The dashboard uses
the optional `dashboard` dependency group.

## Objectives

The report answers five questions in pipeline order:

1. Are the data valid with respect to schema, keys and cycle continuity, and do the Pandas and PySpark ETL produce the same result?
2. How much sensor noise does denoising remove, and how are the data aggregated over time windows?
3. Which degradation trends do the sensors show, and can abnormal behavior be detected before failure?
4. How accurate is RUL prediction on the independent test set?
5. How well can predicted RUL and the anomaly signal be turned into early warnings and a per-engine maintenance schedule?

The topic-specific deliverable is a **predicted-maintenance report with an
early-warning trend chart**. RUL (Remaining Useful Life) is modeled with supervised
regression as a tool, not as the end goal.

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
preparation. PySpark reproduces ETL and rolling features for parity. The added
1x/10x benchmark measures both engines, including I/O and process-tree RSS; Spark
exports Parquet through Arrow on the driver. The original dataset is only a few megabytes, so the
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

The modeling pass compares:

1. A naive median-RUL baseline.
2. Ridge regression as a regularized linear baseline.
3. HistGradientBoostingRegressor (HGB) on the 126 window features.

Model selection uses five-fold grouped cross-validation, with `engine_id` as the group.
The official NASA test labels remain isolated until final predictions have been
written. HGB has the lowest cross-validation MAE and is the main model. Final test metrics
are mean engine-level MAE 15.3490, RMSE 20.5413, and NASA score 1513.2577, with
bootstrap confidence intervals recorded in `outputs/modeling/tables/final_metrics.json`.

### Denoising, anomalies and maintenance schedule

- **Denoising:** an EWMA with span 10 smooths each sensor within each engine,
  $s_t = \alpha x_t + (1-\alpha) s_{t-1}$ with $\alpha = 2/(\text{span}+1) \approx 0.18$;
  it only uses the current and past cycles.
- **Health Index:** StandardScaler and a one-component PCA fit on the denoised train
  data; no RUL labels are used.
- **Anomaly (level-1 warning):** each engine is compared with its own first 30
  cycles; an anomaly is flagged when the Health Index drops more than 3 standard
  deviations and stays there for 5 consecutive cycles.
- **Level-2 warning:** predicted RUL ≤ 30 cycles.
- **Maintenance schedule:** maintenance cycle = current cycle + ⌊predicted RUL − 30⌋.
  Status priority: needs maintenance (level 2), enhanced monitoring (level 1),
  normal. Test labels are read only after the schedule has been written.

Example: test engine 90 is at cycle 146 with predicted RUL 31.2, so maintenance is
due at cycle 147; it is under enhanced monitoring because its Health Index has been
abnormal since cycle 67.

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
ARIMA is an extended experiment and is not used for warnings.

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

- Engine-life distribution and raw sensor trajectories.
- Signals before and after denoising: [`denoise_before_after.png`](outputs/degradation/figures/denoise_before_after.png).
- Health Index, 3σ threshold and anomaly onset: [`anomaly_health_index.png`](outputs/degradation/figures/anomaly_health_index.png).
- Test-set true-versus-predicted RUL and residuals in [the modeling artifacts](outputs/modeling/README.md).
- Per-engine RUL ≤ 30 warning trajectories on train engines.
- Test-fleet status and maintenance planning: [`fleet_status.png`](outputs/maintenance/figures/fleet_status.png),
  [`maintenance_planning.png`](outputs/maintenance/figures/maintenance_planning.png).

The 30-cycle threshold is not an optimized maintenance threshold because FD001
contains no maintenance costs, downtime or intervention records.

## Repository Structure

Current repository structure:

```text
.
├── data/
│   ├── raw/                 # Original FD001--FD004 files and provenance notes
│   └── archive/             # Data retained for comparison, not active modeling
├── references/              # Bibliography, reading guide, papers, assignment context
├── notebooks/               # Executed: 00 data walkthrough, 01 EDA, 02 statistics, 03–06 result review
├── src/cmapss/              # data/, statistics/, etl/, features/, modeling/, timeseries/,
│                            # degradation/ (EWMA, anomalies), maintenance/ (schedule), eda.py, validate.py,
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
├── outputs/degradation/     # EWMA signals, Health Index, anomaly onset cycles
├── outputs/maintenance/     # Maintenance schedule for 100 test engines, planning and label check
├── outputs/cross_dataset/   # FD001--FD004 ETL, features, and transfer results
├── assets/diagrams/         # Editable draw.io sources, SVG previews, URL handles
├── pyproject.toml           # Package and direct dependencies
├── requirements.txt         # Pip installation: local package and runtime libraries
├── requirements-dev.txt     # Notebook and test dependencies
├── requirements-etl.txt     # Optional PySpark ETL dependencies
├── requirements-dashboard.txt # Optional Streamlit dashboard dependencies
├── uv.lock                  # Locked dependency versions
├── report/                  # LaTeX source, PDF, figures, and evidence
├── slides/                  # 20-slide deck on the IUH template, PDF export, data extractor and builder
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

With uv, `uv sync --frozen --no-dev` installs the runtime environment without
notebook or test dependencies. It includes `psutil` for the benchmark's memory
measurements; Spark requires `--extra etl` and Java. For example,
`python -m cmapss.etl.scalability --help` also works in the runtime environment.

After changing dependencies, update `uv.lock` and rerun the export command
printed at the top of each of the four `requirements*.txt` files. Run
`python -m pytest tests/test_dependency_exports.py` to check the exports against
the lock, including platform-specific pins for Linux.

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
environment. The 2026-10-08 test run achieved **440 passed, 1 skipped**;
the TCN test is skipped because `torch` is unavailable. The 2026-09-29 run
achieved **198 tests** and remains the historical evidence for data, statistics,
ETL, feature, modeling, PCA--ARIMA, cross-dataset, dashboard, and end-to-end
reproducibility checks.

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

The pipeline records the resolved `raw_dir` and protects it after a working
directory change. The ETL CLI stages Silver/Gold, the benchmark and requested
parity before publishing; handled write/publication errors restore the old
bundle. The manifest is published last and links auxiliary receipts to input
checksums and the ETL timestamp. `auxiliary_checks.parity.status=not_requested`
means this run did not execute Spark, even if a historical `parity.json` remains.
Files outside the writer's scope, such as README, are preserved. Multiple-file
publication does not guarantee concurrent readers/writers or sudden power loss;
close applications holding SQLite files before rerunning on Windows. If rollback
also fails, the error reports the retained backup directory for recovery.

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

Rolling features inherit the Bronze boundary from ETL metadata. For legacy
Silver without `raw_dir`, supply the actual `--raw-dir`, for example
`--raw-dir D:/datasets/CMAPSS`. This adds protection rather than disabling a
recorded boundary. The rolling writer also uses staging/rollback and rejects
overwriting the Silver input or its source manifest.

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

### Run denoising and anomaly detection

Reads only `data/raw/train_FD001.txt` and uses no RUL labels:

```powershell
.venv\Scripts\python -m cmapss.degradation
```

The command writes `outputs/degradation/`: the noise-reduction table, anomaly onset
cycles for the 100 train engines, a sensitivity table for the anomaly rule
(`tables/anomaly_sensitivity.csv`: z threshold 2/3/4 × persistence 3/5/10 on the raw
and EWMA Health Index), per-cycle raw/EWMA signals and Health Index, and two figures.
All nine configurations flag 100/100 engines with at least 50 cycles of lead time;
they differ only in how early they fire and how many engines fire right after the
baseline window.

### Run the maintenance schedule

After `outputs/modeling/predictions/final_predictions.parquet` exists:

```powershell
.venv\Scripts\python -m cmapss.maintenance
```

The command fits the Health Index on train, only transforms test, writes the
schedule, and only then reads `RUL_FD001.txt` to build `status_vs_actual.csv`.

### Run the result notebooks

Notebooks 03–06 read the saved artifacts, plot them and discuss the results. They
do not fit models or write to `outputs/`; the consistency checks between code and
artifacts live in `tests/`.

| Notebook | Content | Reads |
|---|---|---|
| `03_fd001_modeling_evaluation.ipynb` | Per-fold MAE, test metrics, bootstrap CI, error by RUL range, RUL ≤ 30 warning | `outputs/modeling/tables/*` |
| `04_fd001_health_index_arima.ipynb` | Health Index trend, ARIMA holdout metrics, naive baselines, ADF/Ljung–Box | `outputs/timeseries/tables/*` |
| `05_cross_dataset_and_improvement.ipynb` | FD001–FD004 matrix, nested-CV ablation and paired CI | `outputs/cross_dataset/matrix.json`, `outputs/experiments/rul_improvement/20260928_rebuild/*` |
| `06_fd001_degradation_maintenance.ipynb` | EWMA denoising, Health Index, anomaly detection, maintenance schedule and label check | `outputs/degradation/*`, `outputs/maintenance/*` |

Open them in JupyterLab and run all cells, or execute them from the command line:

```powershell
.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace notebooks/06_fd001_degradation_maintenance.ipynb
```

### Run the reproducibility dashboard

Install the optional dashboard dependencies and start Streamlit after the
modeling, PCA--ARIMA, `cmapss.degradation` and `cmapss.maintenance` artifacts exist:

```powershell
uv sync --extra dashboard
uv run --extra dashboard streamlit run app.py
```

The dashboard has nine pages in pipeline order. A page can be opened directly with
a query string, for example `http://localhost:8501/?page=Lịch bảo trì`.

| Page | Artifacts read | Content |
|---|---|---|
| Tổng quan (overview) | `outputs/modeling`, `outputs/degradation`, `outputs/maintenance` | Four headline numbers, 7-step pipeline, suggested demo order |
| Dữ liệu & EDA | `outputs/eda`, `outputs/features` | Lifetime distribution, sensor–RUL Spearman, sensor trajectories |
| Khử nhiễu (denoising) | `outputs/degradation` | Raw versus EWMA signal per engine/sensor, noise reduction |
| Xu hướng & bất thường (trend & anomalies) | `outputs/degradation`, `outputs/statistics` | Health Index, 3σ threshold, anomaly onset, lead-time distribution, sensitivity grid |
| Dự đoán RUL (RUL prediction) | `outputs/modeling` | CV model comparison, one test engine, OOF trajectories |
| Cảnh báo sớm (early warning) | `outputs/modeling/tables` | 30-cycle threshold, timing classes, warning trajectory |
| Lịch bảo trì (maintenance schedule) | `outputs/maintenance` | 100-engine table, planning chart, per-engine detail, label check |
| Thử nghiệm mở rộng (extended experiments) | `outputs/cross_dataset`, `outputs/experiments`, `outputs/timeseries` | Tabs: transfer, improvement, ARIMA |
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

The 2026-10-07 code verification achieved **426 passed, 1 skipped** (TCN requires
the unavailable torch package). Coverage includes custom Bronze boundaries,
ETL/rolling fault injection, dependency exports and the shared XGBoost/LightGBM
lifecycle. See the [completed plan](handoff/archive/CODE_RELIABILITY_PLAN_20261007.md)
and [runtime evidence](handoff/archive/CODE_RELIABILITY_20261007.json) for commands,
Windows/Linux limits and real spawn worker checks.

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
- The maintenance schedule uses a fixed 30-cycle threshold that is not cost-calibrated; the enhanced-monitoring level is broad (41/59 engines have true RUL above 60).
- Anomaly detection cannot measure a false-alarm rate because there is no label for degradation onset.
- Scalability benchmarks do not create additional independent scientific evidence.
- Model performance on FD001 does not establish real-world generalization.

## License and Data Use

No license has been selected for the project source code yet. The NASA Open Data listing currently does not specify a dataset license, so redistribution and reuse should preserve source attribution and be reviewed against the source record before publication. A project license and final data-distribution policy will be added before the repository is presented as reusable software.
