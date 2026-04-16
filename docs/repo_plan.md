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

### Phase 3 — Gold, modeling, and environments (complete)
8. ~~implement Gold match-level feature engineering~~ (complete)
9. ~~define dvc.yaml pipeline (raw → silver → gold)~~ (complete)
10. ~~implement daily pipeline trigger with dual-source freshness checks~~ (complete)
11. ~~change trigger freshness gate from AND to OR; add --mode CLI arg~~ (complete)
12. ~~implement common model training interface (one entry point per model family)~~ (complete)
13. ~~add MLflow logging for all experimental runs~~ (complete)
14. ~~compute and log permutation feature importance per model~~ (complete)
15. ~~implement experimental gate (tuning by Poisson NLL; all 9 promoted to QA)~~ (complete)
16. ~~implement QA backtesting against 2022 World Cup (RPS primary, RMSE secondary)~~ (complete)
17. ~~implement deploy retraining on full Gold + MLflow pyfunc serialization~~ (complete)
18. ~~implement continuous training dispatch in trigger (retrain on 10+ new gold rows, inference-only otherwise)~~ (complete)
19. ~~freeze model at tournament start; inference-only mode during WC~~ (complete)
 
### Phase 3.5 — GCP Cloud Run and MLflow remote
20. set up DagsHub MLflow tracking as remote
21. set up GCP project, Artifact Registry, Secret Manager
22. fix Dockerfile entrypoint, build and push Docker image
23. create Cloud Run Job + two Cloud Scheduler entries (daily 4:30 UTC + tournament every 30 min)
24. test end-to-end trigger via manual Cloud Run Job execution

Note: `trigger.py` auto-propagates `DAGSHUB_USERNAME`/`DAGSHUB_TOKEN` env vars
to `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` at startup if the
MLflow vars are not already set. This avoids duplicating credentials in the
Cloud Run Job environment configuration.

### Phase 4 — Inference and simulation pipeline (complete)
25. ~~generate tournament config (`data/tournament/wc2026.json`): 12 groups, R32 bracket, 495 best-third mappings, KO bracket with venues, venue→country mapping~~ (complete)
26. ~~implement inference feature builder (`src/inference/features.py`): generate all C(48,2)=1128 WC pairings and deterministic group fixtures from `wc2026.json`, parse played WC results from Bronze (filtered to current WC season), compute rolling features reusing Gold machinery~~ (complete)
27. ~~implement prediction module (`src/inference/predict.py`): load frozen champion from MLflow, predict λ_h/λ_a, compute analytical outcome probs~~ (complete)
28. ~~implement Monte Carlo simulation (`src/inference/simulation.py`): per-match scoreline sampling, full tournament bracket sim (group stage + best-third ranking + R32 → Final), ET scaling 30/90, penalty coin flip~~ (complete)
29. ~~implement inference logging (`src/inference/logging.py`): log predictions, scoreline distributions, tournament advancement probs, group-position probs, and KO pairing frequencies as MLflow artifacts~~ (complete)
30. ~~implement inference orchestrator (`src/inference/run.py`): generate all-pairs + group fixtures → predict all 1128 pairings → filter unplayed group matches for scoreline sampling → simulate with full rate table and locked results → log~~ (complete)
31. ~~wire `run_inference_and_simulation()` into trigger: runs after every path (full pipeline, refit, inference-only, below-threshold auto)~~ (complete)
32. ~~add tests for all inference modules (24 tests) + update trigger tests (3 new dispatch tests)~~ (complete)
33. ~~update docs: fix tactical column count, venue_country mapping, simulation design, ET/penalty/tiebreaker threats to validity, correct holdout match count~~ (complete)

### Phase 5 — Monitoring and deployment
34. add monitoring outputs (drift, performance)
35. add DagsHub integration

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
