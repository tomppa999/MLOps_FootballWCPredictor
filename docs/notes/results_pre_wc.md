# Pre-WC Results

Numbers captured during Phase 1 and Phase 2 implementation.
Maps to thesis Chapter 5, Section 5.1 (model selection and baseline comparison).
Add rows/sections as each implementation step completes.

---

## Baseline: Gold v2 champion (2026-04-29)

Model: XGBoost, run `ba8638c3`, pipeline run `20260429T161823Z`, Gold rows: 6784.
Holdout: WC 2022 only (64 matches).

| Model            | v2 holdout RPS | Notes                        |
|------------------|---------------|------------------------------|
| xgboost          | **0.2109**    | champion                     |
| random_forest    | 0.2121        |                              |
| ridge            | 0.2141        |                              |
| poisson_glm      | 0.2159        |                              |
| bayesian_poisson | 0.2166        | evaluation collapse (1b)     |
| negbin_glm       | 0.2170        | evaluation collapse (1b)     |
| sarimax          | 0.2183        |                              |
| lstm             | 0.2207        |                              |
| cnn              | 0.2248        |                              |

---

## After Phase 1 fixes (1a / 1b / 1c)

_Fill after QA rerun._

Mean-rate Poisson baseline RPS: [fill]

| Model            | holdout RPS | Δ vs v2 | Notes |
|------------------|-------------|---------|-------|
| mean_rate_poisson | [fill]     | —       | baseline floor |
| xgboost          | [fill]      | [fill]  |       |
| ...              |             |         |       |

NegBin fitted alpha (dispersion): [fill]
Bayesian Poisson: posterior samples retained? Y/N

---

## After A.1 (drop tactical features)

_Fill after refit._

Gold rows after drop: [fill]

| Model | RPS before | RPS after | Δ |
|-------|-----------|-----------|---|
| xgboost | 0.2109 | [fill] | [fill] |
| ... | | | |

Notable: did any model improve meaningfully after dropping tactical features?

---

## After A.2 (add rolling Elo-change)

_Fill after refit._

Elo-change window selected (3 / 5 / 10 matches): [fill]
Justification: [fill]

| Model | RPS before | RPS after | Δ |
|-------|-----------|-----------|---|
| ... | | | |

Was `rolling_elo_change` in any model's top-5 importance? [fill]

---

## After A.3 (expanded holdout)

_Fill after data split change._

New holdout match count: [fill] (target ~347)
Match counts by tournament:

| Tournament     | Matches in holdout |
|----------------|--------------------|
| WC 2022        | 64                 |
| AFCON 2024     | [fill]             |
| Asian Cup 2024 | [fill]             |
| Gold Cup 2023  | [fill]             |
| Copa 2024      | [fill]             |
| EURO 2024      | [fill]             |
| Gold Cup 2025  | [fill]             |
| AFCON 2025     | [fill] (group stage only) |
| **Total**      | [fill]             |

| Model | WC-only RPS | Expanded holdout RPS | Δ |
|-------|------------|---------------------|---|
| xgboost | 0.2109 | [fill] | [fill] |
| ... | | | |

---

## After A.4 (time-decay + importance weights)

_Fill after refit with sample weights._

Optimised `half_period_years` per model (Optuna [1.0, 5.0]):

| Model            | half_period_years | Notes                  |
|------------------|-------------------|------------------------|
| mean_rate_poisson | 3.0 (fixed)      | Ley et al. known opt.  |
| poisson_glm      | [fill]            |                        |
| negbin_glm       | [fill]            |                        |
| bayesian_poisson | [fill]            |                        |
| ridge            | [fill]            |                        |
| random_forest    | [fill]            |                        |
| xgboost          | [fill]            |                        |
| lstm             | [fill]            |                        |
| cnn              | [fill]            |                        |

RPS delta vs pre-weights baseline (expanded holdout):

| Model | without weights | with weights | Δ |
|-------|----------------|-------------|---|
| xgboost | [fill] | [fill] | [fill] |
| ... | | | |

---

## Thesis champion selection

_Fill after Phase 2 complete (model selection step)._

Selected top 3 + baseline = 4 models for live WC experiment:

| Role | Model | Holdout RPS |
|------|-------|------------|
| Baseline | mean_rate_poisson | [fill] |
| Champion 1 | [fill] | [fill] |
| Champion 2 | [fill] | [fill] |
| Champion 3 | [fill] | [fill] |

Selection rationale (diversity of families): [fill]

Frozen run_id (both modes start here): [fill]
MLflow aliases assigned: `champion_frozen` = [fill], `champion_per_round` = [fill]
