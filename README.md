<a href="README.md"><img src="https://flagcdn.com/20x15/us.png" alt="English" width="20" height="15"> English</a> · <a href="README.vi.md"><img src="https://flagcdn.com/20x15/vn.png" alt="Tiếng Việt" width="20" height="15"> Tiếng Việt</a>

<hr>

# NASA C-MAPSS RUL Prediction Pipeline

A reproducible Big Data course project for analyzing multivariate turbofan sensor data and predicting Remaining Useful Life (RUL) on the NASA C-MAPSS FD001 dataset. The project emphasizes data provenance, statistically sound analysis, leakage-safe evaluation, scalable ETL, and explainable visual reporting.

## Project Status

This repository now provides **FD001 data validation, train RUL targets, a completed train-only EDA notebook (milestone 1), persisted M2.3a--M2.4 train-only sensor inference artifacts (milestone 2), Pandas/PySpark Silver/Gold ETL with benchmark and parity artifacts (M3.1), and deterministic 30-cycle rolling features (M3.2)**.

Available now:

- The original NASA C-MAPSS FD001 files and source documentation.
- Readers, validators, and tests for row counts, engines, schema, keys, cycles, and train RUL.
- A [data contract](data/CONTRACTS.md), JSON validation CLI, and dependency lockfile.
- An [executed EDA notebook](notebooks/01_fd001_eda.ipynb), 12 CSV tables,
  11 figures, and [analysis notes and validation evidence](outputs/eda/README.md).
- A curated bibliography and reading guide.
- A [LaTeX report](report/README.md) with a [PDF](report/main.pdf) summarizing
  the current validated results, a data dictionary, glossary of report
  terminology and abbreviations, decision rationale, and validation/EDA/M2.3a--M2.4
  interpretations.
  EDA milestone 1, milestone 2 train-only statistics, M3.1 ETL parity, and
  M3.2 rolling features have passed independent validation; models and the
  dashboard remain next.
- A cited theoretical background and a [milestone 2 statistical protocol](report/sections/statistical-protocol.tex).
  The paired comparison uses each sensor's mean over the first and last 30 cycles of every train engine.
  The M2.0 statistical configuration was fixed before inference.
- A train-only `build_engine_sensor_pairs` helper that creates and validates one
  first/last 30-cycle pair for every engine and all 21 sensors.
- M2.2 helpers for mean paired differences, percentile bootstrap confidence intervals,
  Wilcoxon permutation, and Benjamini-Hochberg FDR under the fixed M2.0 configuration.
- An [executed M2.3a inference notebook](notebooks/02_fd001_inference.ipynb),
  two validated CSV tables, run metadata, and [validation notes](outputs/statistics/README.md).
  The outputs keep all 21 sensors and do not read test observations or test labels.
- M2.3b figures generated from those artifacts:
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

Not implemented yet:

- RUL models and ARIMA analysis.
- RUL models, ARIMA analysis, and the Streamlit dashboard.

Installation, data validation, notebook, ETL, and feature-generation instructions are available below. Model training, evaluation, and dashboard entry points are not implemented yet.

## Objectives

The project is designed to answer four questions:

1. How trustworthy and suitable is FD001 for an academic predictive-maintenance study?
2. Which sensor signals exhibit measurable degradation across engine life cycles?
3. How accurately can RUL be predicted from the most recent 30 operating cycles?
4. How can model results be communicated as transparent, reproducible early-warning evidence?

The primary objective is **RUL prediction**. Supervised regression is the modeling formulation because the target is a numeric number of remaining cycles; it is not the title or purpose of the project.

## Dataset

The project uses **NASA C-MAPSS FD001**, a simulated run-to-failure turbofan dataset published by the NASA Ames Prognostics Center of Excellence.

| File | Purpose | Local size |
| --- | --- | ---: |
| `train_FD001.txt` | Complete run-to-failure histories for 100 engines | 20,631 rows |
| `test_FD001.txt` | Truncated histories for 100 different engines | 13,096 rows |
| `RUL_FD001.txt` | Ground-truth RUL at the final observed cycle of each test engine | 100 labels |

Each observation contains:

- Engine identifier.
- Operating cycle.
- Three operational settings.
- Twenty-one sensor measurements.

FD001 represents one operating condition and one High-Pressure Compressor degradation mode. The signals contain simulated sensor noise.

### Provenance and Integrity

- [NASA Open Data record](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- [Archived dataset with DOI and checksum](https://doi.org/10.5281/zenodo.15346912)
- Recorded ZIP checksum: `MD5 79a22f36e80606c69d0e9e4da5bb2b7a`
- Local provenance notes: [`data/raw/DATA_SOURCE.md`](data/raw/DATA_SOURCE.md)

The data is simulated, not field telemetry. Results therefore support reproducible academic comparison on FD001, but do not by themselves establish real-world deployment performance.

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

Pandas is used in the EDA notebook, ETL reference path, statistics and model
preparation. PySpark reproduces the ETL boundaries and supports a transparent
scalability benchmark. The original dataset is only a few megabytes, so the
repository will not claim that FD001 itself demonstrates large-scale data
processing.

## Planned Methodology

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

The implemented M3.2 representation uses a trailing window of 30 cycles. It keeps all 21 sensors in the feature artifact so that sensor selection is not performed on the full train set. For each sensor, the generated features are:

- Latest value.
- Mean and sample standard deviation (`ddof=1`).
- Minimum and maximum.
- Linear trend slope over positions 1--30 within the window.

Windows never cross engine boundaries and never use future observations. Engines with fewer than 30 cycles are rejected rather than padded or shortened.

### RUL Prediction

The planned comparison includes:

1. A naive median-RUL baseline.
2. Ridge regression as a regularized linear baseline.
3. HistGradientBoostingRegressor as the nonlinear candidate model.

Model selection uses five-fold grouped cross-validation, with `engine_id` as the group. The official NASA test labels remain isolated until the final evaluation. Primary and secondary metrics include engine-level MAE, RMSE, and the NASA asymmetric scoring function.

### ARIMA Analysis

ARIMA is a secondary time-series analysis, not the primary RUL model. It will forecast a train-fitted Health Index on representative validation engines and report:

- Forecast MAE and RMSE.
- 95% prediction-interval coverage.
- Augmented Dickey-Fuller stationarity diagnostics.
- Ljung-Box residual autocorrelation diagnostics.

ARIMA will not be applied directly to the deterministic training RUL countdown.

## Big Data Scope

The Big Data component focuses on lifecycle and reproducibility rather than overstating dataset volume:

- Equivalent Pandas and PySpark ETL implementations.
- Parquet as the analytical storage format.
- SQLite as a portable local query and reporting layer.
- Row-count, key, and aggregate parity checks between processing engines.
- Scalability benchmarks at 1×, 10×, and 100× volume.

Replicated records used in the scalability benchmark will receive remapped engine identifiers and remain isolated from scientific analysis and model training.

## Evaluation Protocol

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

- Engine-life distribution (completed in milestone 1).
- Sensor variability and raw-cycle trajectories (completed in milestone 1).
- Planned: sensor and Health Index trajectories over normalized life.
- Planned: true versus predicted RUL on validation engines.
- Planned: test-set prediction scatter and residual distributions.
- Planned: per-engine and RUL-range error analysis.
- Planned: a Streamlit dashboard for inspecting individual engine trajectories and model outputs.

An illustrative warning may be shown when predicted RUL is at most 30 cycles. This is not presented as an optimized maintenance threshold because FD001 contains no maintenance costs, downtime, or intervention records.

## Repository Structure

Current and planned public project structure:

```text
.
├── data/
│   ├── raw/                 # Original FD001 files and provenance notes
│   └── archive/             # Data retained for comparison, not active modeling
├── references/              # Bibliography, reading guide, and available papers
├── notebooks/               # Executed FD001 train EDA and statistics notebooks
├── src/cmapss/              # data/, statistics/, etl/, features/, eda.py, validate.py
├── tests/                   # Data contract, CLI, EDA, statistics, ETL, features, and reproducibility tests
├── outputs/validation/      # Data-validation reports
├── outputs/eda/             # EDA tables, figures, methods, and validation
├── outputs/statistics/      # M2.3a--M2.4 inference artifacts and validation
├── outputs/etl/             # Silver/Gold Parquet, SQLite, metadata, benchmark
├── outputs/features/        # Gold rolling feature Parquet and metadata
├── assets/diagrams/         # Editable draw.io sources, SVG previews, URL handles
├── pyproject.toml           # Package and direct dependencies
├── requirements.txt         # Pip installation: local package and runtime libraries
├── requirements-dev.txt     # Notebook and test dependencies
├── requirements-etl.txt     # Optional PySpark ETL dependencies
├── uv.lock                  # Locked dependency versions
├── report/                  # Living LaTeX/PDF report, progress and evidence
├── app.py                   # Planned Streamlit dashboard entry point
├── README.md                # English documentation
└── README.vi.md             # Vietnamese documentation
```

Local agent-support files are intentionally excluded from GitHub through `.gitignore`.

## Reproducibility

The completed repository will provide one documented path from immutable raw data to processed artifacts, metrics, figures, dashboard data, and the final report. Reproducibility requirements include:

- Fixed random seed `42` for data splitting and stochastic models.
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
  config.py       Fixed M2.0 inference configuration
  pairs.py        Build and validate engine/sensor pairs
  inference.py    Bootstrap, Wilcoxon, FDR, and sensor summaries
  validation.py   Validate sensor inference tables
  artifact_io.py  CSV/JSON, hash, and round-trip readers
  metadata.py     Run metadata and validation payloads
  artifacts.py    Write CSV/JSON artifacts and round-trip validation
  reproducibility.py
                  M2.4 validation, runtime audit, and artifact comparison
statistics_artifacts.py
                  Compatibility import for the artifact writer
```

For example, a train engine ending at cycle 200 has `rul = 200 - cycle`.
The target function rejects test observations. See the [data contract](data/CONTRACTS.md)
for conditions and validation limits.

### Run the EDA notebook

After installing `requirements-dev.txt`, open the notebook from the repository root:

```powershell
.venv\Scripts\python -m jupyterlab notebooks/01_fd001_eda.ipynb
```

Select the `.venv` Python kernel and **Restart Kernel and Run All Cells**.
This analyzes only train and recreates 12 CSV tables and 11 PNG figures.
The validated runs used fresh kernels and produced identical artifacts in the
same environment; the EDA checkpoint suite passed **69 tests**. The current
repository-wide suite, including M2.3a artifact export tests, the M2.4 runtime
audit, M3.1 ETL checks, and M3.2 feature checks, passed **105 tests**.

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
within 38 engines. The M2.3b figures show each sensor in its own panel so that
different sensor scales are not ranked directly. `sensor_6` illustrates that a
small mean difference can coexist with a small p-value, while `sensor_9`
illustrates why the per-engine distribution should be read together with the
mean and CI. These are train-only findings, not model feature selection. M2.4
ran the notebook twice from fresh kernels; the two CSV files and two PNG files
matched, and the runtime audit blocked probes that tried to open `test_FD001.txt`
and `RUL_FD001.txt`.

Run the complete M2.4 validation with:

```powershell
.venv\Scripts\python -m cmapss.statistics.reproducibility `
  --project-root . `
  --notebook notebooks/02_fd001_inference.ipynb `
  --output-dir outputs/statistics `
  --work-dir tmp/statistics-reproducibility-m24 `
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

After ETL has written Silver observations, create the M3.2 Gold rolling features:

```powershell
.venv\Scripts\python -m cmapss.features `
  --silver-observations outputs/etl/silver/observations.parquet `
  --output-dir outputs/features
```

The current artifact has 27,927 rows: 17,731 train windows and 10,196 test
windows. See [rolling feature artifacts and validation](outputs/features/README.md)
for the column contract, row counts, leakage checks, and limitations.

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

Pandas/PySpark Silver/Gold parity, benchmark artifacts, and M3.2 rolling
features are available; models and dashboard remain. M2.4 repeated-run validation has passed.
The [report draft](report/main.pdf) is updated alongside
verified work: its progress log records dates, rationale, artifacts, checks,
and remaining work. See [report instructions](report/README.md) for source
organization and PDF build commands. The current draft includes verified data
validation, selected initial EDA observations, M3.1 ETL measurements, and M3.2 rolling-feature evidence, not
model evaluation results.

## Academic Context and AI Disclosure

This repository supports a final report for the **Big Data Analysis** course. The report will distinguish observed results from proposed explanations and will disclose the use of AI/LLM assistance in accordance with course requirements. All generated analysis, code, citations, and conclusions remain subject to human review and reproducibility checks.

## References

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). [Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation](https://doi.org/10.1109/PHM.2008.4711414).
- Ramasso, E., & Saxena, A. (2014). [Review and Analysis of Algorithmic Approaches Developed for Prognostics on CMAPSS Dataset](https://doi.org/10.36001/phmconf.2014.v6i1.2512).
- Li, X., Ding, Q., & Sun, J.-Q. (2018). [Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks](https://doi.org/10.1016/j.ress.2017.11.021).
- See [`references/reading-guide.md`](references/reading-guide.md) for the complete reading plan.

## Limitations

- FD001 is simulated and represents one operating condition and one fault mode.
- Operating cycles are not equivalent to hours or calendar time.
- RUL prediction does not by itself determine an optimal maintenance schedule.
- Scalability benchmarks do not create additional independent scientific evidence.
- Model performance on FD001 does not establish real-world generalization.

## License and Data Use

No license has been selected for the project source code yet. The NASA Open Data listing currently does not specify a dataset license, so redistribution and reuse should preserve source attribution and be reviewed against the source record before publication. A project license and final data-distribution policy will be added before the repository is presented as reusable software.
