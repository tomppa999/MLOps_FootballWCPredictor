# Requirements

## Functional requirements

- Ingest national team fixture and statistics data from API-Football and Elo ratings on a scheduled basis
- Build Bronze/Silver/Gold data layers with schema validation
- Train and compare Poisson, Negative Binomial, and XGBoost goal-prediction models
- Promote the best model based on explicit, testable criteria
- Simulate tournament outcomes via Monte Carlo from predicted score distributions
- Retrain automatically when fresh data arrives from both sources

## Non-functional requirements

- Reproducibility: all data, artifacts, and model runs must be reproducible (DVC + MLflow)
- Leakage safety: no future information may appear in any feature at training or inference time
- Observability: all training runs, parameters, metrics, and artifacts must be logged to MLflow
- Auditability: raw source data is immutable (Bronze); transformations are code-versioned (Git)
- Simplicity: no overengineering; every tool in the stack must earn its place
- Local-first executability: the full pipeline must run without any cloud dependency; Cloud Run and Cloud Scheduler are the production deployment targets but never a prerequisite
- Testability: core transformation and promotion logic must have unit tests
- Reproducible deployment packaging: model artifacts are packaged via MLflow pyfunc (inference code + environment spec); pipeline jobs are containerized with Docker; both are reproducible from versioned inputs