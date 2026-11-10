# AGENTS.md

## Project intent
This repository implements a minimal but serious MLOps pipeline for football World Cup prediction.

## Non-negotiable constraints
- Keep the solution simple.
- Do not overengineer.
- Prefer the lowest-friction implementation that still demonstrates real MLOps.
- Use code-based deployment only.
- Use test-driven development for core pipeline logic.
- Preserve strict time-aware anti-leakage behavior.
- Do not depend on xG.
- Do not reimplement Elo formulas.
- On Elo fetch failure, use the last validated snapshot and log it.
- Include explicit code and data versioning support through Git and DVC.

## Approved stack
- Python
- pandas / numpy / scikit-learn
- statsmodels
- xgboost
- keras (for LSTM and CNN candidates)
- MLflow
- pytest
- Docker
- Git
- DVC
- local-first file storage
- DagsHub later as remote/collaboration layer
- GCS / Cloud Run Jobs / Cloud Scheduler later

## Current project state
- DagsHub is integrated as MLflow tracking remote and DVC artifact remote.
- The live WC 2026 experiment is finished; the pipeline is retired. No new deployment work.
- Work happens on the `thesis` branch.
- Focus: statistical analysis (RQ1–RQ3) and reproducibility. See HANDOFF.md for open tasks.
- Do not change reconstructed datasets, model states, or upstream pipeline stages without explicit approval.
- All analysis outputs must be produced by DVC stages (`dvc repro <stage>`), never by ad-hoc scripts.
- Never run `dvc repro -f` without `-s` (a bare forced repro runs upstream strands for ~4h40).
- Run `pytest tests/analysis -q` after any analysis change.

## Disallowed additions without explicit approval
- Databricks
- Spark
- Airflow
- Kubernetes
- dbt
- Feast
- extra orchestration or platform tools

## Coding style
- Favor small, explicit modules and functions.
- Use src/ layout.
- Write tests for all critical transformations and interfaces.
- Keep comments concise and useful.
- Prefer deterministic behavior.
- Add type hints where practical.

## Architecture rules
- Bronze = raw immutable data
- Silver = cleaned and standardized data
- Gold = one row per match, modeling-ready features
- No future information in features
- Keep training/inference schema stable
- Candidate models: Mean-rate Poisson (baseline), Poisson GLM, Negative Binomial GLM, XGBoost, Bayesian Poisson, SARIMAX, Ridge, Random Forest, LSTM, CNN
- Models predict goals / goal-distribution parameters first, not direct outcomes
- Outcomes are derived later from score distributions and Monte Carlo simulation
- Log all candidate runs to MLflow
- Promotion logic must be explicit and testable
- Use Git for code versioning
- Use DVC for data, artifacts, and reproducibility support

## Working style
- Read docs before changing structure.
- Implement local-first.
- Stop after meaningful milestones and summarize clearly.