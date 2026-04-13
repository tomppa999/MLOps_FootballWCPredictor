---
name: Pipeline fixes and reruns
overview: Fix the deploy-phase champion gate (tie handling + graceful failure), wire MLflow to DagsHub, promote all 9 models to QA, add pipeline_run_id tagging, update docs/tests, then run the pipeline twice (current data + lookback backfill) and analyze results.
todos:
  - id: fix-champion-gate
    content: Change >= to > in deploy phase and catch ChallengeFailed in run_full_pipeline
    status: completed
  - id: update-gate-tests
    content: Update test docstring, add tie-promotes test, run pytest
    status: completed
  - id: wire-dagshub
    content: Make setup_mlflow() read MLFLOW_TRACKING_URI from env, ensure .env is loaded
    status: completed
  - id: fix-mlflow3-register
    content: Fix MLflow 3.x register_model by using model_uri from log_model instead of runs:/ URI
    status: completed
  - id: first-pipeline-run
    content: Run pipeline with current data, verify DagsHub tracking
    status: completed
  - id: promote-all-9
    content: Remove TOP_K_FOR_QA filter, send all 9 models to QA
    status: completed
  - id: add-pipeline-run-id
    content: Add pipeline_run_id tag to all MLflow runs (pipeline.py + tuning.py)
    status: completed
  - id: retro-tag-runs
    content: Retroactively tag existing DagsHub runs with pipeline_run_id
    status: completed
  - id: update-tests-topk
    content: Update TestTopKSelection tests for all-models-advance
    status: completed
  - id: update-docs
    content: Update report, modeling_plan, requirements, acceptance_criteria, repo_plan
    status: completed
  - id: run-tests-step4
    content: Run pytest to verify all step 4 changes
    status: completed
  - id: second-pipeline-run
    content: Run pipeline with all 9 models in QA (user runs manually)
    status: pending
  - id: lookback-ingestion
    content: Run --lookback-days 28 ingestion, dvc repro to rebuild gold
    status: pending
  - id: third-pipeline-run
    content: Run pipeline on enlarged dataset (user runs manually)
    status: pending
  - id: analyze-results
    content: Compare results across runs, update report with findings
    status: pending
isProject: false
---

# Pipeline Fixes, DagsHub Integration, and Full Reruns

## Step 1: Fix deploy-phase champion gate — DONE

Two changes in `src/models/pipeline.py`:

**1a.** Changed `>=` to `>` on line 342 so ties promote.

**1b.** Wrapped `run_deploy_phase(winner)` in `try/except ChallengeFailed` in `run_full_pipeline` so a worse challenger logs a message and returns `winner.qa_run_id` instead of crashing.

**1c.** Updated test docstring from `>=` to `>`, added `test_challenger_ties_promotes`.

**1d.** All 19 tests passed.

## Step 2: Wire MLflow to DagsHub remote — DONE

- `src/models/mlflow_utils.py`: Added `import os`, changed `TRACKING_URI` to `os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")`.
- `src/pipeline/trigger.py`: Added `from dotenv import load_dotenv; load_dotenv()` at top of `main()`.

## Step 2.5: Fix MLflow 3.x model registration — DONE

**Root cause:** MLflow 3.10.0 stores model artifacts under `models:/{model_id}` namespace, not under `{run_id}/artifacts/model/`. DagsHub accepts the model log but `register_model("runs:/{run_id}/model")` fails because run-level artifact listing returns empty and the `logged_model` fallback isn't supported by DagsHub.

**Fix:**
- `src/models/pipeline.py`: `_log_model_artifact` now returns `model_info.model_uri` (the `models:/` URI).
- Both `run_qa_phase` and `run_deploy_phase` capture this URI and pass it to `register_model(model_uri=..., model_name=...)`.
- `src/models/mlflow_utils.py`: `register_model` signature changed from `(run_id, artifact_path, *, model_name)` to `(model_uri, *, model_name)`.

Verified against DagsHub: registration succeeded with `models:/m-...` URI.

## Step 3: Run the pipeline (first successful run) — DONE

Run completed April 12, 2026. All experimental, QA, and deploy runs tracked to DagsHub.

**Note on existing DagsHub runs:** There are three groups of runs in experiment `0`:
- **Failed run (April 11):** All experimental phases completed, QA crashed on first `register_model`. Tagged retroactively as `pipeline_run_id = "run_001_failed"` in step 4f.
- **Debug runs (April 11):** 3 runs with names starting with `debug_` from the MLflow 3.x fix verification. Tagged as `pipeline_run_id = "debug"` in step 4f.
- **Successful run (April 12):** Full pipeline completed. Tagged as `pipeline_run_id = "run_002"` in step 4f.

## Step 4: Promote all 9 models to QA + pipeline_run_id + update docs/tests

### 4a. Remove TOP_K filter (all 9 models advance to QA)

In `src/models/pipeline.py`:
- Remove `TOP_K_FOR_QA: int = 4` (line 69)
- Change `top_k = results[:TOP_K_FOR_QA]` to `top_k = results` (all models advance)
- Update the log message on lines 219-223 to reflect all models advancing

### 4b. Update tests

In `tests/models/test_pipeline.py`:
- Remove `TOP_K_FOR_QA` from the import (line 16)
- `TestTopKSelection` class (lines 148-183):
  - `test_sorted_by_cv_nll_ascending`: Change `results[:TOP_K_FOR_QA]` → `results`, assert `len(top_k) == 5` (all models), keep sort-order assertions
  - `test_worst_model_excluded`: Remove or convert to pure sort-order test (with all models advancing, none are excluded)
  - `test_fewer_than_k_models_returns_all`: Remove (no longer meaningful without a K cutoff)

### 4c. Update docs (5 files)

**`docs/report/wc_mlops_report.tex`:**
- Line 54: "top-4 candidates" → "all nine candidates"
- Lines 232-233: "The top four candidates by CV NLL are selected" → "All nine candidates advance"
- Line 799: "top four candidates" → "all nine candidates"

**`docs/modeling_plan.md`:**
- Line 155: "All TOP_K QA finalists" → "All nine QA finalists" (or "All candidates")
- Line 172: "top 4 by CV NLL advance to QA" → "all nine advance to QA"

**`docs/requirements.md`:**
- Line 9: "promote top 4 to QA environment" → "promote all 9 to QA environment"
- Line 60: "top 4 advance" → "all 9 advance"

**`docs/acceptance_criteria.md`:**
- Line 38: "top 4 models by CV NLL are selected for QA promotion" → "all 9 models advance to QA"

**`docs/repo_plan.md`:**
- Line 24: "top-4 by CV NLL promoted" → "all 9 promoted"

### 4d. Run tests

```bash
pytest tests/models/test_pipeline.py -v
```

### 4e. Add `pipeline_run_id` tag to all MLflow runs

**`src/models/pipeline.py`:**
- Add `from datetime import datetime, timezone` at top
- In `run_full_pipeline`: generate `pipeline_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")` once
- Add `pipeline_run_id: str | None = None` kwarg to `run_experimental_phase`, `run_qa_phase`, `run_deploy_phase`
- Each phase includes `"pipeline_run_id": pipeline_run_id` in its `tags={}` dict (the `best_*` runs at line 191, QA runs at line 279, deploy run at line 367)

**`src/models/tuning.py`:**
- Add `pipeline_run_id: str | None = None` kwarg to `run_tuning` (line 115)
- Add `"pipeline_run_id": pipeline_run_id` to the parent tuning run's tags (line 181)
- Pass `pipeline_run_id` into `_make_objective` and add it to nested trial tags (line 97)

**`src/models/pipeline.py` (caller side):**
- Pass `pipeline_run_id` from `run_experimental_phase` through to `run_tuning` (line 171)

### 4f. Retroactively tag existing DagsHub runs

One-off script (run manually, not committed). Logic:

```python
import mlflow, os
os.environ['MLFLOW_TRACKING_URI'] = '...'  # from .env
os.environ['MLFLOW_TRACKING_USERNAME'] = '...'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '...'

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
client = mlflow.tracking.MlflowClient()

runs = client.search_runs('0', max_results=5000)
for run in runs:
    name = run.info.run_name or ''
    # April 12 runs = successful pipeline run
    start_date = run.info.start_time  # epoch ms
    april_12_midnight_utc = 1775952000000  # 2026-04-12T00:00:00Z

    if name.startswith('debug_'):
        tag = 'debug'
    elif start_date >= april_12_midnight_utc:
        tag = 'run_002'
    else:
        tag = 'run_001_failed'

    client.set_tag(run.info.run_id, 'pipeline_run_id', tag)
    print(f'{name}: {tag}')
```

Adjust the `april_12_midnight_utc` timestamp if needed based on actual run start times.

### 4g. Run tests again

```bash
pytest tests/models/test_pipeline.py -v
```

## Step 5: Run the pipeline again (all 9 in QA) — user runs manually

Same command as step 3. Expected: ~75 min Phase 1 + ~2 min Phase 2 (now 9 models) + deploy. The `pipeline_run_id` tag will be set automatically.

## Step 6: Lookback ingestion (3-4 weeks)

```bash
python -m src.ingestion.download_api_football_national_matches --lookback-days 28
dvc repro
```

## Step 7: Run the pipeline again (new data) — user runs manually

```bash
source .env && export MLFLOW_TRACKING_URI DAGSHUB_TOKEN DAGSHUB_USERNAME
export MLFLOW_TRACKING_USERNAME=$DAGSHUB_USERNAME
export MLFLOW_TRACKING_PASSWORD=$DAGSHUB_TOKEN
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%dT%H:%M:%SZ')
from src.models.pipeline import run_full_pipeline
run_full_pipeline()
"
```

## Step 8: Analyze results for the report

After step 7, compare across all runs (filter by `pipeline_run_id`):
- Model rankings (CV NLL and holdout RPS/NLL) across runs
- Feature importance stability
- Whether the ranking reversal persists with more data
- Champion progression in MLflow registry

Update `docs/report/wc_mlops_report.tex` with findings.
