# Requirements

## Functional requirements

- Ingest national team fixture and statistics data from API-Football and Elo ratings on a scheduled basis
- Build Bronze/Silver/Gold data layers with schema validation
- Train and compare 9 candidate models across 5 model families (see below) using a three-environment pipeline
- Measure permutation-based feature importance for all models in the experimental environment
- Tune models by CV Poisson NLL; rank for promotion by CV NLL; promote top 4 to QA environment
- Backtest QA models against the 2022 World Cup; report RPS and RMSE
- Promote the single best QA model to deploy environment; refit on the full Gold dataset available at run time
- Simulate tournament outcomes via Monte Carlo from predicted score distributions
- Apply home advantage correction at inference time for USA, Canada, and Mexico in the 2026 WC
- Retrain automatically when fresh data arrives from both sources

## Non-functional requirements

- Reproducibility: all data, artifacts, and model runs must be reproducible (DVC + MLflow)
- Leakage safety: no future information may appear in any feature at training or inference time
- Observability: all training runs, parameters, metrics, and artifacts must be logged to MLflow
- Auditability: raw source data is immutable (Bronze); transformations are code-versioned (Git)
- Simplicity: no overengineering; every tool in the stack must earn its place
- Local-first executability: the full pipeline must run without any cloud dependency
- Testability: core transformation and promotion logic must have unit tests
- Reproducible deployment packaging: model artifacts are packaged via MLflow pyfunc; pipeline jobs are containerized with Docker

## Candidate models (Experimental environment — 9 models)

| # | Model                    | Family       | Library       |
|---|--------------------------|--------------|---------------|
| 1 | Poisson GLM              | Statistical  | scipy.optimize (custom MLE) |
| 2 | Negative Binomial GLM    | Statistical  | statsmodels                 |
| 3 | Bayesian Poisson (PyMC)  | Bayesian     | pymc          |
| 4 | SARIMAX (ARIMAX)         | Time-Series  | statsmodels   |
| 5 | Ridge Regression         | ML baseline  | scikit-learn  |
| 6 | Random Forest            | ML ensemble  | scikit-learn  |
| 7 | XGBoost                  | ML boosting  | xgboost       |
| 8 | LSTM                     | Deep Learning| keras         |
| 9 | 1D CNN                   | Deep Learning| keras         |

All models target home_goals and away_goals as count/distribution targets.
Match outcomes (W/D/L) are derived downstream from score distributions.

## Modeling hypotheses

- H1: Poisson GLM will be a strong baseline; football goal counts are well-approximated by Poisson processes
- H2: XGBoost will outperform linear models by capturing non-linear interactions (Elo differential, competition tier, form)
- H3: SARIMAX will capture temporal momentum effects not visible to cross-sectional models
- H4: LSTM and 1D CNN will underperform relative to their complexity due to limited national team data volume
- H5: Bayesian Poisson will produce better-calibrated uncertainty estimates than frequentist alternatives

## Feature importance

- Permutation importance computed for all 9 models in the experimental environment
- Logged to MLflow alongside training metrics for each run
- Compared across model families in the report

## Environment pipeline

1. Experimental: all 9 models tuned via Optuna with walk-forward CV on pre-WC 2022 data using Poisson NLL; ranked by CV NLL; top 4 advance
2. QA: promoted models retrained on full pre-WC 2022 data and backtested against 2022 World Cup holdout; KPIs: RPS, RMSE
3. Deploy: best QA model refitted on the full Gold dataset available at run time (including WC 2022 and all subsequent matches); evaluation run logged separately for audit; refitted model registered and promoted in MLflow

## Threats to validity

See docs/threats_to_validity.md for full list.
