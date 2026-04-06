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

### Phase 3 — Gold, modeling, and environments (Gold + trigger complete)
8. ~~implement Gold match-level feature engineering~~ (complete)
9. ~~define dvc.yaml pipeline (raw → silver → gold)~~ (complete)
10. ~~implement daily pipeline trigger with dual-source freshness checks~~ (complete)
11. ~~change trigger freshness gate from AND to OR; add --mode CLI arg~~ (complete)
12. ~~implement common model training interface (one entry point per model family)~~ (complete)
13. ~~add MLflow logging for all experimental runs~~ (complete)
14. ~~compute and log permutation feature importance per model~~ (complete)
15. ~~implement experimental gate (top-4 by CV RPS promoted)~~ (complete)
16. ~~implement QA backtesting against 2022 World Cup (RPS primary, RMSE secondary)~~ (complete)
17. ~~implement deploy retraining on full Gold + MLflow pyfunc serialization~~ (complete)
18. ~~implement continuous training dispatch in trigger (retrain on 10+ new gold rows, inference-only otherwise)~~ (complete)
19. freeze model at tournament start; inference-only mode during WC

### Phase 3.5 — GCP Cloud Run and MLflow remote
20. set up DagsHub MLflow tracking as remote
21. set up GCP project, Artifact Registry, Secret Manager
22. fix Dockerfile entrypoint, build and push Docker image
23. create Cloud Run Job + two Cloud Scheduler entries (daily 4:30 UTC + tournament every 30 min)
24. test end-to-end trigger via manual Cloud Run Job execution

### Phase 4 — Simulation, deployment, monitoring
24. implement Monte Carlo tournament simulation
    - handle 2026 FIFA bracket (12 groups, best-8-of-12 third-placers, R32 lookup table)
    - apply home advantage correction for USA/CAN/MEX group stage
25. add monitoring outputs (drift, performance)

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
