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

_Fill after refit with sample weights (A.5 interim values in baselines.py)._

RPS delta vs pre-weights baseline (expanded holdout):

| Model | without weights | with weights | Δ |
|-------|----------------|-------------|---|
| xgboost | [fill] | [fill] | [fill] |
| ... | | | |

---

## After A.6 (marginal half-period tuning)

Run: `python -m src.models.half_period_tuning [--n-trials N]`

Strategy: freeze A.5 best hyperparameters per model; 1-D Optuna over
`half_period_years ∈ [1.0, 5.0]` (25 trials, minimise walk-forward CV NLL).

### First run (2026-06-03 → 06-04) — INTERIM, pre-Poisson-GLM-fix

`CV NLL @ 3yr` = A.5 value (same frozen hyperparameters at half-life 3.0).
`CV NLL @ tuned` = best of the 25 sampled half-periods. Because TPE never
samples exactly 3.0, a positive Δ does **not** mean "worse than 3yr" — at the
nearest-to-3yr trial each stable model reproduces its A.5 value within noise,
so `min over [1,5]` would be ≤ the 3yr value if 3.0 were sampled.

| Model             | half_period_years | CV NLL @ tuned | CV NLL @ 3yr | Δ NLL   | Notes |
|-------------------|-------------------|----------------|--------------|---------|-------|
| mean_rate_poisson | 3.0 (fixed)       | —              | —            | —       | unweighted (no-info floor) |
| sarimax           | 3.0 (fixed)       | —              | —            | —       | unweighted (no statsmodels hook) |
| poisson_glm       | 1.232             | 2.7623         | 2.7567       | +0.0056 | **UNRELIABLE** — fit blew up (NLL 21.78, 3.67); landscape non-smooth. Must fix before trusting. |
| negbin_glm        | 2.901             | 2.9708         | 2.9705       | +0.0003 | flat → 3yr optimal (noise) |
| bayesian_poisson  | 1.933             | 2.7519         | 2.7518       | +0.0001 | flat → 3yr optimal; persistent MCMC divergences (up to 94/trial) |
| ridge             | 4.803             | 2.8304         | 2.8349       | −0.0045 | **real gain** — deterministic, genuinely prefers ~4.8yr |
| random_forest     | 3.832             | 2.7678         | 2.7677       | +0.0001 | flat → 3yr optimal (noise) |
| xgboost           | 4.918             | 2.7713         | 2.7703       | +0.0010 | ~flat; mild preference for long half-life (within RNG) |
| lstm              | 3.745             | 2.8532         | 2.8150       | +0.0382 | flat in hp (2.853–2.869 across range); offset vs A.5 = Keras training nondeterminism, not hp effect |
| cnn               | 3.465             | 3.0252         | 3.0269       | −0.0017 | flat → reproduces A.5 within noise |

Interim per-model best half-periods (from A.6 run, for reference only — NOT yet
written to `TUNED_HALF_PERIODS`, pending Poisson-GLM fix + validated rerun):
`poisson_glm 1.23, negbin_glm 2.90, bayesian_poisson 1.93, ridge 4.80,
random_forest 3.83, xgboost 4.92, lstm 3.75, cnn 3.46`.

Takeaways:
- Half-period objective is **flat** for negbin / random_forest / bayesian / lstm /
  cnn → 3yr (Ley et al.) is at/near optimal; sub-0.002 deltas are noise.
- **ridge** is the one model with a real, clean preference for a longer ~4.8yr
  half-life. xgboost weakly agrees (~4.9yr) but within RNG.
- **poisson_glm is numerically unstable** under reweighting and its result is not
  trustworthy — fix first, then rerun A.6.
- Deviation from 3yr is a reportable methodology finding (ridge is the headline).

### Validated rerun (after Poisson-GLM fix) — PENDING

**Code ready (2026-06-04):** `BivariatePoisson.fit` normalizes weights to mean 1;
`tune_half_period` enqueues `half_period_years=3.0` before TPE. Run:

`conda activate modelops && python -m src.models.half_period_tuning`

Then paste the printed `TUNED_HALF_PERIODS` block into `src/models/config.py`.
For each model, keep 3.0 if the pinned 3.0 trial beats or ties the TPE best.

---

## After A.5 (first full thesis refit: A.1–A.4 combined, fixed 3yr half-period)

A.1–A.4 were implemented without intermediate individual refits; A.5 is therefore
the first measured checkpoint reflecting all four changes simultaneously:
in-game statistics dropped (A.1), rolling Elo-change added (A.2), holdout expanded
to WC 2022 + continental tournaments (A.3), time-decay + match-importance sample
weights added with fixed 3yr half-period (A.4). Half-period tuning (A.6) is not yet
applied — these are interim values. MLflow pipeline run: `pipeline_a5_*` (2026-06-02).

Holdout: WC 2022 + AFCON 2024 + EURO 2024 + Gold Cup 2023 + Gold Cup 2025 +
Copa América 2024 + Asian Cup 2024 (~347 matches).

| Model             | cv_nll  | holdout_nll | rmse_home | rmse_away | **holdout_rps** | qa_wall_sec |
|-------------------|---------|-------------|-----------|-----------|-----------------|-------------|
| xgboost           | 2.7703  | 2.7788      | 1.0211    | 1.1818    | **0.18242**     | 1.3s        |
| poisson_glm       | 2.7567  | 2.7976      | 1.0323    | 1.2002    | 0.18316         | 10.3s       |
| bayesian_poisson  | 2.7518  | 2.7893      | 1.0197    | 1.2022    | 0.18320         | 264.9s      |
| random_forest     | 2.7677  | 2.7932      | 1.0196    | 1.1864    | 0.18331         | 1.0s        |
| negbin_glm        | 2.9705  | 2.7969      | 1.0278    | 1.2071    | 0.18377         | 1.0s        |
| sarimax           | 2.9973  | 2.8981      | 1.0147    | 1.2077    | 0.18595         | 4.2s        |
| ridge             | 2.8346  | 2.8175      | 1.0021    | 1.2240    | 0.18819         | 4.1s        |
| lstm              | 2.8150  | 2.8302      | 1.0661    | 1.2072    | 0.18898         | 6.2s        |
| cnn               | 3.0269  | 2.8672      | 1.0080    | 1.3012    | 0.20181         | 4.1s        |
| mean_rate_poisson | 3.2358  | 2.9787      | 1.0701    | 1.3601    | 0.22872         | 4.1s        |

Sorted by holdout RPS ascending (lower = better). All metrics from `qa_*` MLflow runs.

Notable observations:
- Top 4 (xgboost, poisson_glm, bayesian_poisson, random_forest) are within 0.001 RPS.
- bayesian_poisson has the best cv_nll (2.7518) and holdout_nll (2.7893) but ranks 3rd on RPS.
- cv_nll ranking diverges noticeably from holdout_nll for negbin_glm and sarimax (better CV, worse holdout).
- holdout_rmse_away is consistently higher than holdout_rmse_home across all models.
- ridge achieves the lowest rmse_home (1.0021) despite ranking 7th on RPS.
- bayesian_poisson qa_wall_sec (265s) reflects full QA pass (CV + holdout); shadow refit is a single `.fit()` — not a production concern.

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
