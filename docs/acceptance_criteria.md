# Acceptance Criteria

## Phase 1 acceptance criteria
- repository bootstraps cleanly
- docs reflect the agreed architecture and constraints
- small representative sample raw data is present for local development and tests
- sample raw data can be read and parsed
- initial ingestion logic exists for both API-Football and Elo TSV inputs
- Silver cleaning works on sample inputs
- Gold match-level feature builder skeleton exists
- tests run successfully
- project structure is coherent and maintainable
- repository is prepared for local DVC usage

## Phase 2 acceptance criteria
- historical backfill ingestion is implemented for the selected training window
- Bronze contains reproducible raw snapshots for the real historical data pull
- DVC is initialized
- key data/artifact paths are identified for DVC tracking
- versioning/reproducibility documentation exists (see [versioning_and_reproducibility.md](versioning_and_reproducibility.md))
- the project remains fully usable without DagsHub
- optional instructions for later DagsHub connection are documented

## Phase 3 acceptance criteria — Gold and modeling

### Gold
- Gold feature table is one row per match
- all features are strictly time-aware and leakage-safe
- feature set includes: Elo ratings, rolling form indicators, competition tier, is_knockout, is_neutral, home advantage flag
- home advantage inference correctly flags USA, Canada, and Mexico playing in their host country for 2026 WC group stage
- DVC pipeline (`dvc.yaml`) defines all three stages: raw → silver → gold

### Experimental environment
- all 9 candidate models can train through a common interface
- all runs are logged to MLflow (params, metrics, artifacts)
- promotion gate threshold is pre-defined in config before any training runs
- permutation importance is computed and logged for each model
- models that exceed the threshold (max 4) are tagged for QA promotion in MLflow

### QA environment
- promoted models are backtested against 2022 World Cup data
- the following KPIs are computed and logged: Ranked Probability Score (RPS), Brier score, outcome accuracy
- the single best QA model is identified and tagged for deploy promotion

### Deploy environment
- best model is retrained on full Gold dataset
- artifact is serialized via MLflow pyfunc
- Monte Carlo tournament simulation runs from deployed model predictions
- simulation correctly handles the 2026 FIFA bracket (12 groups, best-8-of-12 third-placers, FIFA lookup table for R32 matchups)

## Phase 4 acceptance criteria
- containerized jobs run reproducibly
- scheduled refresh/retrain flow is defined
- deployment artifacts are versioned
- monitoring outputs are generated
- simulation outputs can be derived from predicted score distributions
