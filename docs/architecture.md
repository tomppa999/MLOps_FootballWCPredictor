# Architecture

## Architecture style
Minimal level-1-style MLOps pipeline with local-first implementation and cloud-ready extension points.

## Versioning and reproducibility
- Git is used for code versioning.
- DVC is used for data, model artifacts, and reproducible pipeline support.
- DagsHub is planned as a later remote/collaboration layer, but does not yet exist for this project.
- The project must work locally before any DagsHub integration.

## Data layers

### Bronze
Immutable raw inputs:
- raw API-Football JSON responses
- raw Elo TSV snapshots
- timestamped source captures

These raw artifacts should be versionable with DVC.

### Silver
Cleaned and standardized match-level base data:
- one row per match
- parsed fixtures and match statistics from API-Football
- canonical team mapping applied
- schema-validated match metadata
- resolved pre-match Elo values joined from Elo TSV history
- optional resolved post-match Elo values for analysis/debugging, not for model features

Silver is the main cleaned base table for downstream feature generation.
It should not require separate persisted team-history tables as part of the formal v1 contract.
Any reshaping needed to compute rolling pre-match features can be created during the Gold build as intermediate logic rather than as a separate persisted Silver artifact.

These intermediate artifacts may also be versioned with DVC where useful.

### Gold
Modeling-ready match-level features:
- one row per match
- home-side feature columns
- away-side feature columns
- selected difference features
- time-aware rolling form variables
- tactical profile features
- tactical cluster features
- pre-match Elo features
- contextual match features

Gold datasets used for training should be reproducible and versionable.

## Model layer
Train and compare 9 candidates across 5 families:

| # | Model                   | Family        | Library                     |
|---|-------------------------|---------------|-----------------------------|
| 1 | Poisson GLM             | Statistical   | scipy.optimize (custom MLE) |
| 2 | Negative Binomial GLM   | Statistical   | statsmodels                 |
| 3 | Bayesian Poisson (PyMC) | Bayesian      | pymc          |
| 4 | SARIMAX (ARIMAX)        | Time-Series   | statsmodels   |
| 5 | Ridge Regression        | ML baseline   | scikit-learn  |
| 6 | Random Forest           | ML ensemble   | scikit-learn  |
| 7 | XGBoost                 | ML boosting   | xgboost       |
| 8 | LSTM                    | Deep Learning | keras         |
| 9 | 1D CNN                  | Deep Learning | keras         |

All models should share a common training/evaluation interface where practical.

## Modeling target
The main modeling objective is to predict goals / score distribution parameters rather than direct outcomes.
Outcome probabilities and tournament simulations are derived later from the score distributions.

## Promotion logic
A model can only be promoted if:
- data validation passes
- schema checks pass
- training succeeds
- evaluation beats the current production baseline on the chosen primary metric

After selection, the winning model type and hyperparameters are refitted on
the full Gold dataset available at run time. The refitted model is the
artifact that gets registered and promoted via the `champion` alias in the MLflow
model registry. The evaluation run that justified the selection is linked via an
`evaluation_run_id` parameter on the production-refit run.

## Deployment
Use code-based deployment:
- preprocessing code
- feature schema
- model artifact
- inference code
- simulation logic

## Monitoring
Later phase:
- data drift
- prediction distribution checks
- delayed performance tracking after real results arrive