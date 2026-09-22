<a href="README.md"><img src="https://flagcdn.com/20x15/us.png" alt="English" width="20" height="15"> English</a> · <a href="README.vi.md"><img src="https://flagcdn.com/20x15/vn.png" alt="Tiếng Việt" width="20" height="15"> Tiếng Việt</a>

<hr>

# NASA C-MAPSS RUL Prediction Pipeline

A reproducible Big Data course project for analyzing multivariate turbofan sensor data and predicting Remaining Useful Life (RUL) on the NASA C-MAPSS FD001 dataset. The project emphasizes data provenance, statistically sound analysis, leakage-safe evaluation, scalable ETL, and explainable visual reporting.

## Project Status

This repository now provides **FD001 data validation, train RUL targets, a completed train-only EDA notebook (milestone 1), and the paired-data contract for milestone 2**.

Available now:

- The original NASA C-MAPSS FD001 files and source documentation.
- Readers, validators, and tests for row counts, engines, schema, keys, cycles, and train RUL.
- A [data contract](data/CONTRACTS.md), JSON validation CLI, and dependency lockfile.
- An [executed EDA notebook](notebooks/01_fd001_eda.ipynb), 12 CSV tables,
  11 figures, and [analysis notes and validation evidence](outputs/eda/README.md).
- A curated bibliography and reading guide.
- A [living LaTeX report](report/README.md) with a [PDF draft](report/main.pdf),
  work progress, decision rationale, and traceable validation/initial EDA observations.
  EDA milestone 1 passed independent validation; inferential analysis belongs to milestone 2.
- A cited theoretical background and a [milestone 2 statistical protocol](report/sections/statistical-protocol.tex).
  The paired comparison uses the mean of the first and last 30 cycles per engine.
  Detailed statistical settings remain proposals; inferential tests have not run.
- A train-only `build_engine_sensor_pairs` helper that creates and validates one
  first/last 30-cycle pair for every engine and all 21 sensors. It has no
  persisted statistical output and does not perform inference.

Not implemented yet:

- Silver/Gold ETL pipelines in Pandas and PySpark.
- Bootstrap, Wilcoxon/FDR inference, the milestone 2 notebook, and feature engineering.
- RUL models and ARIMA analysis.
- Streamlit dashboard and completed final report (the LaTeX draft has started).

Installation, data validation, and notebook instructions are available below. Model training, evaluation, and dashboard entry points are not implemented yet.

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

## Planned Data Architecture

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
Gold: rolling features, predictions, metrics, and SQLite tables
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

Pandas is used in the EDA notebook and will also support statistics and model preparation. PySpark will reproduce the ETL path and support a transparent scalability benchmark. The original dataset is only a few megabytes, so the repository will not claim that FD001 itself demonstrates large-scale data processing.

## Planned Methodology

### Data Quality

- Validate the 26-column schema and numeric types.
- Enforce uniqueness of `(split, engine_id, cycle)`.
- Verify sequential cycles within each engine.
- Record row counts before and after every transformation.
- Keep train and test engine identifiers scoped to their own splits.

### Statistical Analysis

- Compare early-life and late-life sensor behavior per engine.
- Report effect sizes and bootstrap 95% confidence intervals.
- Use paired Wilcoxon signed-rank tests where appropriate.
- Control multiple testing across sensors with Benjamini-Hochberg FDR.
- Estimate metric uncertainty with engine-level bootstrap resampling.

### Feature Engineering

The primary representation uses a trailing window of 30 cycles. For each non-constant sensor, the planned features are:

- Latest value.
- Mean and standard deviation.
- Minimum and maximum.
- Linear trend slope.

Windows never cross engine boundaries and never use future observations.

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
├── notebooks/               # Executed FD001 train EDA
├── src/cmapss/              # data/: contracts; eda.py; statistics.py; validate.py
├── tests/                   # Data contract, CLI, EDA, and paired-data tests
├── outputs/validation/      # Data-validation reports
├── outputs/eda/             # EDA tables, figures, methods, and validation
├── assets/diagrams/         # Editable draw.io sources and SVG exports
├── pyproject.toml           # Package and direct dependencies
├── requirements.txt         # Pip installation: local package and runtime libraries
├── requirements-dev.txt     # Notebook and test dependencies
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
statistics.py     Build and validate 30-cycle engine/sensor pairs
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
repository-wide suite, including the M2.1a pair contract, passed **86 tests**.

![FD001 train EDA workflow](assets/diagrams/fd001-eda.svg)

[Editable diagram](assets/diagrams/fd001-eda.drawio) ·
[Methods, findings, and limits](outputs/eda/README.md)

For example, the 100 train engines have a median lifetime of 199 cycles.
Six sensors are globally constant; `sensor_6` varies globally but is constant
within 38 engines. These are descriptive findings, not model feature selection
or inferential significance claims. Milestone 2 will address paired analysis,
effect sizes, bootstrap, and multiple testing.

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

Silver/Gold ETL, PySpark, rolling features, inference, models, and dashboard
are not implemented yet. The [report draft](report/main.pdf) is updated alongside
verified work: its progress log records dates, rationale, artifacts, checks,
and remaining work. See [report instructions](report/README.md) for source
organization and PDF build commands. The current draft includes verified data
validation and selected initial EDA observations, not model evaluation results.

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
