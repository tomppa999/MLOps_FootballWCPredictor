# Repo Plan

## Implementation order

### Phase 1 — Bootstrap (complete)
1. bootstrap repo structure
2. add small sample raw data and reference mapping files
3. initialize local DVC setup
4. implement Bronze ingestion/parsing for sample inputs

### Phase 2 — Historical data (complete)
5. implement real historical backfill ingestion for the chosen training window (2018–2026)
6. implement Silver cleaning and standardization
7. add tests and validation checks

### Phase 3 — Gold, modeling, and environments
8. implement Gold match-level feature engineering
   - rolling form, Elo diff, competition tier, is_knockout, is_neutral, home advantage flag
   - strictly time-aware; no leakage
9. define dvc.yaml pipeline (raw → silver → gold) in one pass once Gold is validated
10. implement common model training interface (one entry point per model family)
11. add MLflow logging for all experimental runs
12. compute and log permutation feature importance per model
13. implement experimental gate (pre-defined threshold, max 4 promoted)
14. implement QA backtesting against 2022 World Cup (RPS, Brier, outcome accuracy)
15. implement deploy retraining on full Gold + MLflow pyfunc serialization

### Phase 4 — Simulation, deployment, monitoring
16. implement Monte Carlo tournament simulation
    - handle 2026 FIFA bracket (12 groups, best-8-of-12 third-placers, R32 lookup table)
    - apply home advantage correction for USA/CAN/MEX group stage
17. containerize pipeline jobs
18. define scheduled refresh/retrain flow
19. add monitoring outputs (drift, performance)
20. optionally connect DVC remote to DagsHub
21. prepare cloud deployment (Cloud Run Jobs + Cloud Scheduler)

## Design rules
- local-first
- small modules
- no premature cloud complexity
- no unnecessary tools
- version code with Git
- version data and key artifacts with DVC
- do not assume DagsHub already exists
- do not assume the real historical dataset already exists before the ingestion pipeline is built
- promotion logic must be explicit, config-driven, and testable
- Gold schema must remain stable and consistent across all 9 model families
