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
- versioning/reproducibility documentation exists
- the project remains fully usable without DagsHub
- optional instructions for later DagsHub connection are documented

## Phase 3 acceptance criteria
- all three candidate models can train through a common flow
- the modeling target is goals / score distributions, not direct outcome classification as the main target
- MLflow logging is implemented
- evaluation outputs are reproducible
- promotion logic exists and is testable

## Phase 4 acceptance criteria
- containerized jobs run reproducibly
- scheduled refresh/retrain flow is defined
- deployment artifacts are versioned
- monitoring outputs are generated
- simulation outputs can be derived from predicted score distributions