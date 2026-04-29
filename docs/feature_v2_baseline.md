# Gold Feature v2 — Baseline Anchor

_Captured at 2026-04-29T14:34:43Z_

## v1 champion (pre-feature-expansion)

| Field | Value |
|---|---|
| run_id | `b57274ad247c41d7ac217c50668103a2` |
| model_name | `xgboost` |
| pipeline_run_id | `None` |
| stage | `production-refit` |
| registry name | `wc_production` |
| registry version | `2` |
| gold_row_count at train time | `6663` |

### Holdout metrics (QA phase)

| Metric | Value |
|---|---|
| `qa_holdout_nll` | `2.9199` |
| `qa_holdout_rmse_away` | `1.0980` |
| `qa_holdout_rmse_home` | `1.3880` |
| `qa_holdout_rps` | `0.2118` |

### Best params

| Param | Value |
|---|---|
| `colsample_bytree` | `0.968499838325458` |
| `learning_rate` | `0.01113450981305486` |
| `max_depth` | `3` |
| `n_estimators` | `366` |
| `reg_lambda` | `0.0010119131740279753` |
| `subsample` | `0.6379451427257596` |

## v2 — to be filled after retraining

- pipeline_run_id: TBD
- new champion model_name: TBD
- new champion run_id: TBD
- new champion registry version: TBD
- holdout metrics: TBD

## v1 → v2 delta (filled in Phase 4)

| Model | v1 holdout RPS | v2 holdout RPS | Δ |
|---|---|---|---|
| _to fill_ | | | |

