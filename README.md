# wc-mlops

Minimal but serious end-to-end MLOps project for football World Cup prediction.

## Goal
Build a level-1-style MLOps pipeline that:
- ingests football and Elo data
- builds Bronze / Silver / Gold layers
- engineers time-aware features
- trains 10 candidate models (mean-rate Poisson baseline + 9 feature-based candidates)
- compares them in MLflow
- promotes the best validated model
- supports code-based deployment and tournament simulation
- includes explicit code and data versioning

## Modeling objective
The main prediction target is goals / score distributions, not direct match outcomes.
Match outcomes such as home win / draw / away win are derived later from predicted score distributions and used in Monte Carlo tournament simulation.

## Principles
- keep it simple
- local-first
- test core logic
- avoid leakage
- do not overengineer
- version code with Git
- version data/artifacts with DVC

## Prerequisites

- **macOS**: Xcode (not just Command Line Tools) is required for PyMC/PyTensor C compilation. Install from the App Store and run `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`.

## Stack
- Python
- pandas / numpy / scikit-learn
- statsmodels
- xgboost
- MLflow
- pytest
- Docker
- Git
- DVC
- DagsHub (MLflow tracking + DVC remote)
- GCP Cloud Run Jobs / Cloud Scheduler (deployment in progress)

## Current status
- Bronze: historical API-Football + Elo TSVs (2018–2026)
- Silver: schema-validated, season-partitioned Parquet; team mapping, Elo join, competition tier
- Gold v2: match-level features (~6,784 rows); champion XGBoost (holdout RPS 0.2109 on WC 2022)
- Models: 10 candidates trained, tuned (Optuna), QA backtest, MLflow registry (`wc_staging` / `wc_production`)
- Inference + Monte Carlo tournament simulation (2026 bracket, 48 teams)
- Monitoring: per-match live RPS/NLL scoring for all 9 models during WC 2026
- DagsHub: integrated as MLflow remote and DVC remote
- **In progress:** thesis Phase 1 model fixes, Phase 2 feature changes, GCP deployment (target June 10)

## Versioning state
- DVC initialized; raw data tracked via `data/raw.dvc`
- DagsHub connected for MLflow experiment tracking and DVC artifact remote
- Project remains runnable locally without cloud

## Structure
- `src/ingestion`: raw source ingestion (API-Football, Elo)
- `src/silver`: cleaning and standardization
- `src/gold`: match-level feature engineering
- `src/models`: training, evaluation, promotion
- `src/inference`: prediction and tournament simulation
- `src/monitoring`: live WC 2026 performance scoring and alerting
- `src/pipeline`: daily trigger (ingest → gold → train/refit → inference → monitor)
- `tests/`: unit and integration tests
- `docs/`: architecture, specs, plans, literature, thesis notes
- `data_samples/`: small sample inputs for local development and tests
