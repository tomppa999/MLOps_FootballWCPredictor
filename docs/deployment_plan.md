# Deployment Plan

## Deployment style
Code-based deployment only.

## What gets deployed
- preprocessing code
- feature-building logic
- model artifact selected by the pipeline
- inference code
- tournament simulation code

## Why
This allows continuous retraining and keeps production behavior tied to versioned code rather than a standalone model blob.

## Reproducibility and versioning
- code versioning through Git
- data and artifact versioning through DVC
- model artifacts should be reproducible from versioned inputs and code
- DagsHub may later be used as the remote layer for DVC data/artifact storage and collaboration

## Current state
- no DagsHub repo exists yet
- deployment and reproducibility design must not depend on DagsHub being available now

## Modeling output
The deployed prediction layer should produce goal-related outputs or score distribution parameters.
Outcome probabilities and tournament outcomes are derived downstream from those outputs.

## Phasing
### Phase 1
Local-first pipeline and tests

### Phase 2
DVC initialization and tracked data/artifact workflow

### Phase 3
Containerized execution

### Phase 4
Optional DagsHub remote integration

### Phase 5
Cloud Run Jobs + Cloud Scheduler + cloud storage

## Output types
- batch score distribution predictions
- tournament simulation outputs
- monitoring artifacts