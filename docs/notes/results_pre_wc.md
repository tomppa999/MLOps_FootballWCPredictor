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

### Final decisions (2026-06-04) — no rerun

First-run evidence sufficient. `TUNED_HALF_PERIODS` populated in `src/models/config.py`:

| Model | half_period_years | Rationale |
|---|---|---|
| poisson_glm | 3.0 | Unreliable first-run (optimizer blow-ups); flat landscape → 3yr |
| negbin_glm | 3.0 | Δ NLL = +0.0003 (noise) |
| ridge | **4.803** | Δ NLL = −0.0045, deterministic — the only real finding |
| random_forest | 3.0 | Δ NLL = +0.0001 (noise) |
| xgboost | 3.0 | "Best" at 4.92 was +0.001 vs 3yr → 3yr preferred |
| bayesian_poisson | 3.0 | Δ NLL = +0.0001 (flat) |
| lstm | 3.0 | Flat landscape; A.6 offset = Keras nondeterminism |
| cnn | 3.0 | Δ NLL = −0.0017 (within noise / TPE artefact) |

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

## Post-A.6 full pipeline rerun (2026-06-05)

First full pipeline run with `TUNED_HALF_PERIODS` applied (ridge at 4.803yr,
all others at 3.0yr). A.1–A.6 all active. Holdout: expanded set (WC 2022 +
continental tournaments, ~347 matches). All metrics from `qa_*` MLflow runs.

Sorted by holdout RPS ascending (lower = better):

| Model             | cv_nll  | holdout_nll | rmse_home | rmse_away | **holdout_rps** | qa_wall_sec |
|-------------------|---------|-------------|-----------|-----------|-----------------|-------------|
| xgboost           | 2.7691  | 2.7823      | 1.0201    | 1.1850    | **0.18289**     | 1.7s        |
| bayesian_poisson  | 2.7517  | 2.7894      | 1.0201    | 1.2024    | 0.18316         | 239.2s      |
| negbin_glm        | 2.9704  | 2.7961      | 1.0273    | 1.2064    | 0.18373         | 4.2s        |
| poisson_glm       | 2.7565  | 2.7939      | 1.0277    | 1.1991    | 0.18389         | 1.0s        |
| random_forest     | 2.7665  | 2.8002      | 1.0289    | 1.1891    | 0.18392         | 1.4s        |
| sarimax           | 2.9989  | 2.8900      | 1.0148    | 1.2077    | 0.18583         | 4.2s        |
| ridge             | 2.8330  | 2.8158      | 1.0019    | 1.2239    | 0.18813         | 3.3s        |
| lstm              | 2.8179  | 2.8958      | 1.0535    | 1.2293    | 0.19299         | 5.2s        |
| cnn               | 3.0417  | 2.9287      | 1.0255    | 1.3387    | 0.20916         | 2.8s        |
| mean_rate_poisson | 3.2358  | 2.9787      | 1.0701    | 1.3601    | 0.22872         | 4.1s        |

### Comparison vs A.5 (pre-half-period-tuning) 

Only ridge changed half-period (3.0 → 4.803yr). Other models re-tuned
hyperparameters via Optuna, so small differences reflect stochastic variation.

| Model             | A.5 RPS   | Post-A.6 RPS | Δ RPS    | Notes |
|-------------------|-----------|--------------|----------|-------|
| xgboost           | 0.18242   | 0.18289      | +0.00047 | within Optuna noise |
| bayesian_poisson  | 0.18320   | 0.18316      | −0.00004 | stable |
| negbin_glm        | 0.18377   | 0.18373      | −0.00004 | stable |
| poisson_glm       | 0.18316   | 0.18389      | +0.00073 | within noise; ranking swapped with bayesian |
| random_forest     | 0.18331   | 0.18392      | +0.00061 | within Optuna noise |
| sarimax           | 0.18595   | 0.18583      | −0.00012 | stable |
| ridge             | 0.18819   | 0.18813      | −0.00006 | small gain from tuned half-period |
| lstm              | 0.18898   | 0.19299      | +0.00401 | Keras nondeterminism |
| cnn               | 0.20181   | 0.20916      | +0.00735 | Keras nondeterminism |
| mean_rate_poisson | 0.22872   | 0.22872      |  0.00000 | deterministic (no params) |

### Interpretation

**Overall picture is healthy.** All tuned models beat the mean-rate-poisson
baseline by a wide margin (~0.04 RPS). The ranking is stable across runs.

- **Top 5 are tightly clustered** (RPS 0.1829–0.1839). XGBoost wins but the
  margin over bayesian_poisson / negbin_glm / poisson_glm / random_forest is
  within noise. This suggests a performance ceiling for the current feature set
  and holdout composition.
- **No overfitting signal.** Most models have holdout NLL better than CV NLL,
  likely because tournament matches are more structured than the training mix
  of friendlies and qualifiers. LSTM is the main exception (cv_nll 2.818 vs
  holdout_nll 2.896), consistent with mild overfitting on limited data.
- **Deep learning underperforms.** CNN (rank 9) and LSTM (rank 8) sit clearly
  below the statistical and tree-based models. The dataset (~6700 rows) is too
  small for neural nets to justify their parameter count, and Keras training
  nondeterminism adds noise across runs.
- **Ridge half-period tuning effect is marginal.** A.6 moved ridge from
  3.0 → 4.803yr, yielding a Δ RPS of −0.00006 on holdout. The gain is real
  (deterministic, reproducible in CV) but negligible in practice.
- **holdout_rmse_away > holdout_rmse_home** across all models. Away goals are
  harder to predict, which is consistent with the higher variance of away
  scoring in international football.
- **A.5 → post-A.6 deltas are negligible** for all non-neural models (|Δ| < 0.001).
  Confirms A.6 did not disrupt existing results and the pipeline is stable.

---

## Thesis champion selection

Roster locked 2026-06-06 (selection rationale and freeze process: see
`docs/notes/decisions.md`). RPS values from the A.7 / post-A.6 refit
(expanded holdout, ~347 matches). Frozen weights are deferred to the
pre-tournament re-fit (≈ June 10–11), so run_id / aliases stay `[fill]`
until the freeze.

Selected 3 champions + baseline = 4 models for live WC experiment:

| Role | Model | Holdout RPS | Family |
|------|-------|------------|--------|
| Baseline | mean_rate_poisson | 0.22872 | no-information floor |
| Champion 1 | xgboost | 0.18289 | tree ensemble (boosting) |
| Champion 2 | poisson_glm | 0.18389 | frequentist Poisson GLM |
| Champion 3 | bayesian_poisson | 0.18316 | Bayesian Poisson GLM |

Selection rationale (diversity of families): top 5 span only two families
(tree ensembles + Poisson-likelihood GLMs, the latter triplicated) and cluster
within 0.001 RPS. Chose one representative per distinct paradigm — boosted
trees / frequentist GLM / Bayesian — to test cadence effects across model
classes rather than near-duplicates. negbin_glm dropped (same family as
poisson_glm, overdispersion barely moves RPS); random_forest dropped (second
tree, no uncertainty channel). bayesian_poisson kept for its native posterior
uncertainty (RQ2 / RQ3) despite mild MCMC divergences.

Frozen run_id (both modes start here): [fill after pre-tournament re-fit]
MLflow aliases assigned: `champion_frozen` = [fill], `champion_per_round` = [fill]
