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

## v2 champion (post-feature-expansion)

| Field | Value |
|---|---|
| run_id | `ba8638c38a1a4e8b8d1de228bc7ba439` |
| model_name | `xgboost` |
| pipeline_run_id | `20260429T161823Z` |
| stage | `production-refit` |
| registry name | `wc_production` |
| registry version | `4` |
| gold_row_count at train time | `6784` |
| evaluation_run_id (QA) | `f25b4dae7d4045a9813ffc832aad5cab` |

### Holdout metrics (QA phase)

| Metric | Value |
|---|---|
| `qa_holdout_nll` | `2.9155` |
| `qa_holdout_rmse_away` | `1.0947` |
| `qa_holdout_rmse_home` | `1.3882` |
| `qa_holdout_rps` | `0.2109` |

### Best params

| Param | Value |
|---|---|
| `colsample_bytree` | `0.8502132109222367` |
| `learning_rate` | `0.011242244538805183` |
| `max_depth` | `3` |
| `n_estimators` | `368` |
| `reg_lambda` | `0.2467461498190776` |
| `subsample` | `0.6872195981680427` |

## v1 → v2 delta (QA holdout RPS, all 9 candidates)

Champion family unchanged (XGBoost both v1 and v2). The auto-promotion gate
fired because v2 XGBoost strictly improved on v1 XGBoost's holdout RPS.

| Model | v1 holdout RPS | v2 holdout RPS | Δ |
|---|---|---|---|
| **xgboost** (champion) | 0.2118 | **0.2109** | **−0.0009** |
| random_forest | 0.2148 | 0.2121 | −0.0027 |
| cnn | 0.2296 | 0.2248 | −0.0048 |
| ridge | 0.2136 | 0.2141 | +0.0004 |
| sarimax | 0.2178 | 0.2183 | +0.0005 |
| poisson_glm | 0.2148 | 0.2159 | +0.0011 |
| negbin_glm | 0.2155 | 0.2170 | +0.0015 |
| bayesian_poisson | 0.2149 | 0.2166 | +0.0017 |
| lstm | 0.2133 | 0.2207 | +0.0074 |

### New-feature top-5 importance hits

- `elo_sum`: top-5 in random_forest (#2), poisson_glm (#5), negbin_glm (#4),
  sarimax (#3) — most useful new feature.
- `rest_diff`: top-5 in bayesian_poisson (#2).
- `away_days_since_last_match`: top-5 in bayesian_poisson (#4).
- `home_days_since_last_match`: not in any top-5 (the derived `rest_diff` does
  surface).
- `is_cross_confederation`: not in any top-5 — confounded with
  `competition_tier` (cross-confederation games are mostly friendlies).
  **Removed from Gold schema** after this analysis (CORE 13→12, FULL 29→28).

