<a href="README.md"><img src="https://flagcdn.com/20x15/us.png" alt="English" width="20" height="15"> English</a> · <a href="README.vi.md"><img src="https://flagcdn.com/20x15/vn.png" alt="Tiếng Việt" width="20" height="15"> Tiếng Việt</a>

<hr>

# NASA C-MAPSS RUL Prediction Pipeline

A reproducible Big Data course project for analyzing multivariate turbofan sensor data and predicting Remaining Useful Life (RUL) on the NASA C-MAPSS FD001 dataset. The project emphasizes data provenance, statistically sound analysis, leakage-safe evaluation, scalable ETL, and explainable visual reporting.

## Project Status

This repository is currently in the **planning and data-preparation stage**.

Available now:

- The original NASA C-MAPSS FD001 files and source documentation.
- Verified row, engine, schema, key, and cycle-continuity checks.
- A curated bibliography and reading guide.

Not implemented yet:

- Pandas and PySpark processing pipelines.
- Statistical analysis and feature engineering.
- RUL models and ARIMA analysis.
- Streamlit dashboard and final LaTeX report.

Commands for installation, training, evaluation, and dashboard execution will be added only after their corresponding code and dependency manifests exist.

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

Pandas will be used for exploratory analysis, statistics, and model preparation. PySpark will reproduce the ETL path and support a transparent scalability benchmark. The original dataset is only a few megabytes, so the repository will not claim that FD001 itself demonstrates large-scale data processing.

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

## Planned Visualizations

- Engine-life distribution.
- Sensor variability and trend summaries.
- Sensor and Health Index trajectories over normalized life.
- True versus predicted RUL on validation engines.
- Test-set prediction scatter and residual distributions.
- Per-engine and RUL-range error analysis.
- A Streamlit dashboard for inspecting individual engine trajectories and model outputs.

An illustrative warning may be shown when predicted RUL is at most 30 cycles. This is not presented as an optimized maintenance threshold because FD001 contains no maintenance costs, downtime, or intervention records.

## Repository Structure

Current and planned public project structure:

```text
.
├── data/
│   ├── raw/                 # Original FD001 files and provenance notes
│   └── archive/             # Data retained for comparison, not active modeling
├── references/              # Bibliography, reading guide, and available papers
├── notebooks/               # Planned EDA and experiment notebooks
├── src/                     # Planned reusable pipeline and modeling modules
├── tests/                   # Planned data-contract and pipeline tests
├── outputs/                 # Planned metrics, figures, and reproducible artifacts
├── report/                  # Planned LaTeX report
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

The executable pipeline has not been implemented yet. Until a dependency manifest and runnable entry points are added, the repository should be treated as a research project under active development rather than a ready-to-run package.

Once implementation begins, this section will document:

1. Supported Python and Java versions.
2. Environment creation and dependency installation.
3. Data validation and pipeline commands.
4. Model training and final evaluation commands.
5. Streamlit dashboard startup.
6. XeLaTeX report compilation.

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
