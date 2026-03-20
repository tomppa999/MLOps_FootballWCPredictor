# Versioning and Reproducibility

## Goal
This project should support reproducible ML work through explicit versioning of:
- code
- raw data snapshots
- derived datasets where useful
- model artifacts where useful
- pipeline definitions

## Tools
- Git for code versioning
- DVC for data, artifacts, and pipeline reproducibility
- DagsHub later as optional remote/collaboration layer

## Current state
- no DagsHub repository has been created yet
- the project must remain fully operational locally
- DagsHub should be treated as a later integration, not a prerequisite

## DVC usage intent
Use DVC for:
- raw Bronze snapshots that are too large or too important to leave unmanaged
- selected Silver/Gold artifacts where reproducibility matters
- model artifacts if tracked outside MLflow artifact storage
- pipeline stage reproducibility later through `dvc.yaml`

## Practical approach
Start simple:
1. initialize DVC locally
2. identify data/artifact folders worth tracking
3. keep the repository working without any remote configured
4. later, connect a DVC remote
5. optionally use DagsHub as that remote

## Future DagsHub integration
When ready:
- create a DagsHub repository
- connect the Git repository or push code there
- configure the DVC remote using DagsHub’s provided setup commands
- verify push/pull of DVC-tracked artifacts

## Important principle
Do not let versioning tooling bloat the project. The purpose is reproducibility and clean provenance, not tool theater.