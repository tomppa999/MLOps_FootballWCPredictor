# wc-mlops

Minimal but serious end-to-end MLOps project for football World Cup prediction.

## Goal
Build a level-1-style MLOps pipeline that:
- ingests football and Elo data
- builds Bronze / Silver / Gold layers
- engineers time-aware features
- trains several models (Poisson GLM, Negative Binomial GLM, XGBoost, Bayesian Poisson, SARIMAX, Ridge, Random Forest, LSTM, CNN)
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

## Planned stack
- Python
- pandas / numpy / scikit-learn
- statsmodels
- xgboost
- MLflow
- pytest
- Docker
- Git
- DVC
- DagsHub later as remote/collaboration layer

## Current status
- Historical Bronze data ingested (API-Football + Elo TSVs, 2018–2026)
- Silver pipeline implemented: schema-validated, season-partitioned Parquet
- Team mapping, Elo join, competition tier, knockout detection complete
- Tests cover core Silver transformations
- Gold feature engineering and modeling layers are next (Phase 3)

## Versioning state
- DVC initialized locally; data/artifacts tracked via DVC, not Git
- DagsHub repo not yet created; project runs fully without it

## Structure
- `src/ingestion`: raw source ingestion (API-Football, Elo)
- `src/silver`: cleaning and standardization
- `src/gold`: match-level feature engineering (planned)
- `src/models`: training, evaluation, promotion (planned)
- `src/inference`: prediction and simulation (planned)
- `src/monitoring`: drift/performance reports (planned)
- `tests/`: unit and integration tests
- `docs/`: architecture decisions, specs, acceptance criteria
- `data_samples/`: small sample inputs for local development and tests