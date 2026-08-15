# Thesis Plan — Consolidated

Branch strategy: **`main`** (model fixes) → **`thesis`** (thesis-specific work)
MLflow experiment: `wc_mlops_thesis`
WC 2026 starts: **June 11, 2026**

> **Active sprint execution → `docs/plans/golive_runbook.md`** (day-by-day
> June 7–11 deployment worklist). This file remains the canonical plan; the
> runbook points at the section IDs below rather than duplicating them.

---

## Framing

**Candidate titles:**

1. The effect of model update frequency on calibration and uncertainty
   resolution in multi-round probabilistic prediction
2. Adaptive retraining strategies for probabilistic forecasting in staged
   sequential events
3. Retraining cadence as an experimental factor in sequential tournament
   prediction

**Use case:** 2026 FIFA World Cup (48 teams, expanded format, 7 natural
retraining boundaries).

**Goal:** Quantify how retraining cadence affects the predictive performance
and uncertainty dynamics of probabilistic goal models in a multi-round
tournament setting.

**Research questions:**

- **RQ1 (Accuracy):** Does per-round retraining improve predictive accuracy
  (RPS, NLL) compared to a frozen pre-tournament model, and does the
  improvement vary across tournament phases (group stage vs knockout)?
- **RQ2 (Uncertainty):** How does retraining cadence affect the rate and
  pattern of uncertainty resolution in tournament advancement probabilities,
  measured via Shannon entropy trajectories?
- **RQ3 (Concept drift):** To what extent does concept drift manifest within
  a single 5-week tournament window, and does per-round retraining mitigate
  its effects on calibration?

**Experimental factor:** Two modes — frozen (pre-tournament snapshot, never
refitted) vs per-round (refitted at each matchday boundary, 7 retraining
events).

**WC 2022 role:** Frozen-model retrospective only. Descriptive context, not
a manipulated condition (avoids confounding format effects with inter-
tournament changes).

---

## Sequencing

### Phase 1 — Model fixes (`main` branch, weeks 1–2)

Correctness fixes needed by both paper and thesis. No paper.tex changes now;
those will be derived from the thesis later.

- [x] **1a** Monitoring threshold → naive-baseline floor
- [x] **1b** Fix NegBin and Bayesian Poisson evaluation collapse
- [x] **1c** Add symmetric mean-rate Poisson baseline (as threshold, not
  peer candidate — intercept-only case of Maher 1982)
- [x] **1d** Drift monitoring — Threats to Validity paragraph (once thesis
  document exists)

### Phase 2 — Thesis feature/data changes (`thesis` branch, weeks 2–3)

- [x] **A.1** Drop in-game statistics features
- [x] **A.2** Add rolling Elo-change (window tied to rolling goals window)
- [x] **A.3** Expand holdout to continental tournaments
- [x] **A.4** Add time-decay + match-importance sample weights (fixed 3yr;
  Optuna half-period tuning deferred to refit)
- [x] **A.5** Refit all 10 candidates on thesis feature set with fixed 3yr
  half-period (verifies Phase 1 fixes — eyeball that NegBin/Bayes metrics
  no longer collapse to Poisson)
  - [x] Update all 10 values in `HOLDOUT_RPS_BASELINES` in
    `src/monitoring/baselines.py` from the A.5 `qa_holdout_rps` metrics
    (interim values — only affects monitoring log context, not alert logic)
- [x] **A.6** Tune `half_period_years` (Optuna float `[1.0, 5.0]`) for
  each weighted model; report per-model optimised values in methodology.
  Deviation from 3yr is a reportable finding.
  - [x] Implemented marginal 1-D tuner (`src/models/half_period_tuning.py`),
    persistence wiring, tests (decision: marginal, not joint — see
    `docs/notes/decisions.md`).
  - [x] First run done 2026-06-03/04. Finding: objective flat for most models
    (3yr ≈ optimal); **ridge** genuinely prefers ~4.8yr (Δ NLL = −0.0045).
  - [x] **`TUNED_HALF_PERIODS` populated (2026-06-04):** ridge = 4.803; all
    other weighted models = 3.0. No rerun performed — first-run evidence
    sufficient; only ridge shows a real, deterministic gain. See
    `docs/notes/decisions.md` for per-model reasoning.
- [x] **A.7** Refit all 10 candidates with tuned half-periods from A.6
  - [x] Updated all 10 values in `HOLDOUT_RPS_BASELINES` in
    `src/monitoring/baselines.py` from the A.7 `holdout_rps` metrics
    (final frozen values for WC 2026 monitoring context; 2026-06-05)
- [x] **A.8** Select top 3 that beat the baseline → thesis live experiment uses
  these 3 + baseline (4 models total). **Roster locked 2026-06-06:** xgboost,
  poisson_glm, bayesian_poisson + mean_rate_poisson baseline. Rationale and
  full process in `docs/notes/decisions.md`; recorded in
  `docs/notes/results_pre_wc.md`.

#### A.9 — Narrow candidates to the selected champions

Code currently fits/QAs all 10 candidates. Before deployment, restrict the
live pipeline to the 4-model roster while keeping the full set reproducible
for the thesis appendix.

- [x] **`EXPERIMENT_MODELS` added to `src/models/config.py` (2026-06-07):**
  `["xgboost", "poisson_glm", "bayesian_poisson", "mean_rate_poisson"]`.
  All 10 candidate modules kept intact for offline reproducibility.
- [x] **Simulation narrowed to roster (Option A, 2026-06-07):** `run.py`
  loops `simulate_tournament()` over `EXPERIMENT_MODELS`. Champion simulates
  from its dedicated prediction path (reliable fallback); the other 3 roster
  models simulate from filtered rows of `all_models_predictions_df`
  (best-effort — skipped gracefully if predictions are missing).
  `predictions_all_models.csv` still covers all 10 as RPS shadows;
  simulation is roster-only.
- [x] **Multi-model tournament artifacts (2026-06-07):** `tournament_probabilities.csv`,
  `group_positions.csv`, and `ko_pairings.csv` are now long-format with a leading
  `model_name` column stacking all simulated roster models. `scoreline_distributions.csv`
  stays champion-only (not a byproduct of the per-model sims). `champion_model_name`
  and `simulated_models` are logged as MLflow params on every inference run.
- [x] **Dashboard champion filter (2026-06-07):** `load_artifacts.py` filters the
  three multi-model artifacts to the champion's `model_name` before returning to
  Streamlit. Backward-compatible: pre-Option-A runs without a `model_name` column
  pass through unchanged.
- [x] **Confirmed `HOLDOUT_RPS_BASELINES` unchanged** — keeps all 10 for
  monitoring context logging; no code change needed.
- [x] **Shadow-load URI bug fixed (2026-06-06).** `load_shadow_model` used the
  obsolete `runs:/<run_id>/model` URI; under MLflow 3.x the model lives at the
  version `source` (`models:/m-<id>`), so the download hung ~4 min then failed and
  every shadow was unloadable (`predictions_all_models.csv` was champion-only the
  whole time). Now loads via `models:/<name>/<version>`, mirroring `load_champion`.
  Verified: full cycle loads all 10 candidates (11280 rows) in 89.3s. See
  `docs/notes/decisions.md`.
- [x] **Defense-in-depth (per-model isolation, 2026-06-07):** broadened
  `except ValueError` → `except (ValueError, MlflowException)` in
  `run_prediction_all_models` (`src/inference/predict.py`) so a registry error on
  one shadow skips that model without aborting the loop.
  
#### A.10 — Freeze champions for both modes (pre-tournament snapshot)

Deferred to the **last responsible moment before kickoff (≈ June 10–11)** so
the snapshot includes the final 2026 friendlies as training rows. This is a
**re-fit, not a re-tune** (level 1 only; hyperparameters + half-periods stay
locked from A.6/A.7). See `decisions.md` "Champion freeze process".

**Prerequisite:** B.2 must exist first — A.10 assigns B.2's `champion_frozen`
and `champion_per_round` aliases. Run B.1–B.4 before this step.

- [x] `bayesian_poisson` refit hardening: `target_accept=0.9`, `tune_steps=1000`,
  `max_eta=10.0`, `prior_sigma` capped at 2.0 — done with B.3.
- [ ] Build the latest full Gold table (last friendlies included).
- [ ] `run_champion_refit` for each of the 3 champions on full Gold
  (no Optuna, no half-period search, no re-selection).
- [ ] Register the single snapshot under **both** aliases `champion_frozen`
  and `champion_per_round` (B.2) — both modes start identical.
- [ ] Record frozen `run_id` + assigned aliases in
  `docs/notes/results_pre_wc.md` (currently `[fill]`).
- [ ] Can be folded into a C.9 pre-WC test run rather than a separate step.

### Phase 3 — Cadence infrastructure (`thesis` branch, weeks 2–3)

**Ordering (decided 2026-06-07):** B.1–B.4 run **before A.10** — A.10 consumes
B.2's dual aliases, so the alias scheme must exist first. Built in parallel
with the GCP skeleton (C.1–C.7). See `golive_runbook.md`.

- [x] **B.1** Snapshot metadata tagging
- [x] **B.2** Dual MLflow aliases + per-mode dispatch
- [x] **B.3** Per-round refit trigger
- [x] **B.4** Per-mode monitoring

### Phase 4 — GCP deployment (`thesis` branch, by June 10)

**Ordering (decided 2026-06-07):** C.1–C.7 start **first** (Jun 7), deploying
the current champion as-is to prove the plumbing on fresh friendly data. Only
C.9's dual-mode assertions depend on B.2 + A.10, so C.9 splits into an early
plumbing pass (Jun 9) and a final experiment pass (Jun 10). See
`golive_runbook.md`.

- [x] **C.1** Containerise the trigger
- [x] **C.2** Service account + permissions
- [x] **C.3** Secret Manager
- [x] **C.4** Artifact Registry + image push
- [x] **C.5** Cloud Run Job (created; resource reduction to 4 GiB / 2 vCPU still pending)
- [x] **C.6** Cloud Scheduler `daily-pipeline-trigger` created + enabled (`0 4 * * *`);
  WC hourly flip deferred to Jun 11 kickoff
- [~] **C.7** DVC remote on GCS — **deferred to post-WC** (DagsHub remote verified)
- [ ] **C.8** Observability (log-based metric, alerting policy)
- [x] **C.9** (first pass, Jun 9): dual-mode dispatch, all 4 models simulated, DVC push
  succeeded, gold 6921 rows. Final pass (alias resolution) pending A.10.

### During WC (June 11 – July 19)

- [ ] Monitor pipeline runs daily; fix failures fast
- [ ] Verify both modes produce tagged snapshots after each match
- [ ] Verify per-round refit fires at matchday boundaries

### Post-WC (July onward)

- [ ] **D.1** Trajectory extraction and entropy analysis
- [ ] **D.2** Structural-driver regression
- [ ] **D.3** WC 2022 frozen-model retrospective
- [ ] **D.4** Thesis writing
- [ ] **D.5** Derive paper revision from thesis

---

## Phase 1 — Model fixes (on `main`)

**Two baselines (different purposes):**

- **Mean-rate Poisson** (1c): a proper predictive model (lambda ≈ 1.32).
  Acts as a *model-level threshold* — any candidate that cannot beat its
  holdout RPS has no demonstrated feature value.
- **Naive-random RPS = 0.235** (1a): a *monitoring alert threshold*. The
  expected RPS of a uniform W/D/L predictor. Not a model; it triggers an
  alert when a deployed model degrades to useless levels.

### 1a. Monitoring threshold → naive-baseline floor

**Problem:** The current `1.3 × per-model baseline` threshold permits an
RPS of ~0.29 — worse than a uniform-random predictor (~0.235).

- [x] In `src/monitoring/baselines.py`:
  - Add `NAIVE_BASELINE_RPS: Final[float] = 0.235` with docstring
    (derivation: uniform-random predictor, neutral venue, p_draw ≈ 0.25)
  - Keep `HOLDOUT_RPS_BASELINES` for context logging
  - Remove or deprecate `ALERT_FACTOR`
- [x] In `src/monitoring/monitor.py:evaluate_alert_threshold`:
  - Change condition to `rolling_rps > NAIVE_BASELINE_RPS`
  - Update log message
- [x] Update tests referencing `ALERT_FACTOR`

### 1b. Fix NegBin and Bayesian Poisson evaluation collapse

**Problem:** Both models collapse to standard Poisson at evaluation time.
NegBin's fitted dispersion is discarded; Bayesian Poisson's posterior is
reduced to its mean before scoring. H4 was likely incorrectly rejected.

#### Model interface

- [x] In `src/models/base.py`:
  - Add `distribution_family` property (default `"poisson"`)
  - Keep `predict()` backward-compatible

#### NegBin GLM (`src/models/candidates/negbin_glm.py`)

- [x] Store fitted `alpha` from statsmodels GLM result
- [x] Extend `predict()` to optionally return `(lambda_h, lambda_a, alpha_h,
  alpha_a)`
- [x] Set `distribution_family = "negbin"`

#### Bayesian Poisson (`src/models/candidates/bayesian_poisson.py`)

- [x] Retain posterior samples (replace `.mean(dim=["chain", "draw"])`
  collapse with stored sample arrays)
- [x] Add method to draw lambda samples for a given X (vectorised matmul)
- [x] Set `distribution_family = "bayesian_poisson"`

#### Evaluation (`src/models/evaluation.py`)

- [x] Add `compute_outcome_probs_nb(lambda, alpha, max_goals=10)` using
  `scipy.stats.nbinom.pmf`
- [x] Add `compute_outcome_probs_bayes(lambda_h_samples, lambda_a_samples,
  max_goals=10)` — averages Poisson PMFs across N posterior samples
- [x] Add `compute_mean_nll_nb(...)` and `compute_mean_nll_bayes(...)`
- [x] Add dispatcher `compute_outcome_probs_dispatch(model, X)` that selects
  scoring function based on `model.distribution_family`

#### Training / tuning

- [x] In `src/models/tuning.py`: NegBin Optuna trials minimise NB NLL
- [x] In `src/models/tuning.py`: Bayesian Poisson trials minimise MC
  posterior NLL
- [x] Verify NegBin/Bayes no longer collapse to Poisson at evaluation time
  during the post-A.4 refit (Phase 2; no separate pre-Phase-2 QA run)

### 1c. Add symmetric mean-rate Poisson baseline

**Goal:** A no-features reference threshold. Any model that cannot beat it
has no demonstrated feature value. The intercept-only case of the
independent Poisson framework (Maher 1982).

**Maher citation context:** Dixon & Coles (1997) is the most influential
Maher citation. It adds a low-score correction factor ρ for scorelines
{0-0, 1-0, 0-1, 1-1} and exponential time-decay on parameters. However,
the correction only improves exact-scoreline prediction — it is
mathematically irrelevant for RPS (which depends only on goal difference,
and the common term cancels; Maher 1982 Section 4 shows this). For
national teams and RPS evaluation, Dixon & Coles adds no benefit over
independent Poisson. Ley et al. (2019) is the national-team adaptation of
Maher/D&C and is the direct state-of-the-art reference for this project.
Groll et al. (2019) establish the broader Poisson goal-modeling framework
and RPS as the standard evaluation metric.

- [x] New file `src/models/candidates/mean_rate_poisson.py`:
  - `MeanRatePoisson(BaseModel)`, no hyperparameters
  - `fit(X, y)`: `self._lambda = (y[:, 0].mean() + y[:, 1].mean()) / 2`
  - `predict(X)`: `(np.full(n, self._lambda), np.full(n, self._lambda))`
  - `distribution_family = "poisson"`, `name = "mean_rate_poisson"`
- [x] Register in `src/models/config.py` candidate list
- [x] Skip Optuna — fits in milliseconds
- [x] Walk-forward CV NLL and QA holdout RPS via existing harness

### 1d. Drift monitoring acknowledgement

- [x] Thesis text (once document structure exists, see D.4): paragraph in
  Threats to Validity acknowledging the absence of statistical data-drift
  monitoring and explaining the rationale (goals are statistically stable,
  Elo is self-correcting). Not blocking any code work.
  Captured in `docs/threats_to_validity.md`.

---

## Phase 2 — Data and feature changes (on `thesis`)

### A.1. Drop in-game statistics features

Permutation importance shows tactical features contribute essentially
nothing; ~49% coverage gap biases training toward UEFA/CONMEBOL.

- [x] In `src/gold/schema.py`:
  - Remove `ROLLING_SHOT_COLUMNS` and `ROLLING_TACTICAL_COLUMNS` from
    `FEATURE_COLUMNS` (keep in `GOLD_COLUMNS` for transparency)
- [x] Update tests asserting feature count or specific feature names
- [x] Refit all candidates on the slimmer feature set

### A.2. Add rolling Elo-change

Captures opponent-quality-adjusted form. Elo values reflect long-term
strength but change slowly; rolling Elo-change captures recent relative
performance in a single number. Full coverage, no new data source needed.

**Preliminary analysis:**
- Qualitative case: Norway rose ~200 Elo points (1724 → 1922) over 18
  months via strong Nations League results, but latest friendlies show a
  dip — rolling Elo-change captures this while absolute Elo stays high.
- Quantitative: Partial correlation with goal margin after controlling for
  elo_pre and rolling_goals_for is weak (r = -0.028, ΔR² = 0.0008 in a
  linear model). Signal may be nonlinear; easy to test via feature ablation
  after refitting. Costs nothing to include (full coverage, single feature).

**Window size:** Preliminary analysis used 5 matches (matching existing
rolling goals window). Candidates to test: 3 (short-term / tournament form),
5 (default), 10 (smoother, ~12–18 months). Treat as a hyperparameter and
tune during the model refit/QA cycle — compute the feature at multiple
windows, evaluate holdout RPS, select the best.

- [x] In `src/gold/rolling_features.py:_build_team_history`:
  - Include `elo_pre` and `elo_post` in per-team history rows
- [x] In `_rolling_for_team`:
  - Compute `rolling_elo_change = (last_n["elo_post"] -
    last_n["elo_pre"]).sum()`
  - Make window size configurable (default 5, test 3/5/10)
- [x] In `src/gold/schema.py`:
  - Add `home_team_rolling_elo_change` and `away_team_rolling_elo_change`
  - Add to `FEATURE_COLUMNS` and `GOLD_DTYPES` (`Float64`)
- [x] Tests:
  - Correctness on synthetic team history
  - Strict time-awareness (no future matches in window)
  - Empty history → NaN

### A.3. Expand holdout

Training cutoff stays at `WC_2022_START`. Holdout = union of WC 2022 +
continental tournament finals through WC 2026. Expands holdout from 64 to
~347 matches (5.4×).

**Preliminary analysis:**
- KS tests: no tournament's goal distribution differs significantly from
  WC 2022 (all p > 0.6). Pooled tier-1 vs tier-2 confederations: KS=0.030,
  p=0.999.
- Poisson lambda ranges from 1.11 (Copa) to 1.34 (WC 2022); Gold Cup 2023
  is an outlier at 1.69.
- Elo level differs (~1860 WC/EURO/Copa vs ~1545 AFCON/AsianCup/GoldCup)
  but models condition on elo_pre, so this is accounted for.
- All tournaments have 84–97% neutral-venue matches.

Tournament match counts:

| Tournament       | Matches | Status           |
|------------------|---------|------------------|
| WC 2022          |      64 | complete         |
| AFCON 2024       |      52 | complete         |
| Asian Cup 2024   |      51 | complete         |
| Gold Cup 2023    |      31 | complete         |
| Copa 2024        |      31 | complete         |
| EURO 2024        |      51 | complete         |
| Gold Cup 2025    |      31 | complete         |
| AFCON 2025       |      36 | group stage only |
| **Total**        | **~347**|                  |

- [x] In `src/models/data_split.py`:
  - Define tournament date/tier constants
  - Replace single-window holdout mask with union of date-range + tier
    filters
  - Training mask stays `date_utc < WC_2022_START`
- [x] Tests:
  - Each tournament contributes expected match count
  - No date overlap between training and holdout
  - Tier filter excludes friendlies/qualifiers inside tournament windows

### A.4. Add exponential time-decay and match-importance sample weights

Both Ley et al. (2019) and Groll et al. (2019) weight training samples by
recency and match importance. This is standard in the field and likely
explains part of our RPS gap vs literature benchmarks (~0.21–0.23 vs
~0.16–0.19).

**Time decay:** `w_time = 0.5^(days_ago / half_period)` with Half Period =
3 years as the default (Ley et al.'s optimum for national teams across all
Poisson variants). A match from 3 years ago contributes 50% as much as
today's match.

**Match importance:** Weights taken from Ley et al. (2019), who adopted
them from the pre-2018 FIFA ranking methodology. Cite as Ley et al.'s
weights, not as "current FIFA weights" (FIFA changed their ranking system
in August 2018). Reuse `competition_tier` from Gold:
- Tier 1 (World Cup): weight 4
- Tier 2 (continental final): weight 3
- Tier 3 (qualifier, Nations League): weight 2.5
- Tier 4 (friendly): weight 1

**Dual role of `competition_tier`:** Using it as both a sample weight and
as a predictor feature is correct and not redundant. The weight improves
training signal quality (fitting parameters on a WC-match-emphasised
distribution). The feature lets the model adjust its predicted lambda at
inference time for the specific competition type being predicted. These
serve different purposes and are complementary.

Final sample weight = `w_time × w_importance`.

- [x] In training harness (`src/models/tuning.py` or `data_split.py`):
  - Compute `days_ago` from `date_utc` relative to training cutoff
  - Compute `w_time = 0.5 ** (days_ago / (3 * 365.25))`
  - Map `competition_tier` → importance weight
  - Pass `sample_weight = w_time * w_importance` to model fit
- [x] XGBoost: `sample_weight` parameter in DMatrix
- [x] GLMs (Poisson, NegBin): frequency/exposure weights in statsmodels
- [x] Bayesian Poisson: weighted likelihood (scale log-likelihood per obs)
- [x] Use a fixed 3-year half-life for all weighted models (Ley et al.
  optimum). `mean_rate_poisson` is **not** weighted (no-information floor;
  weighting only nudges its single constant — see
  `docs/notes/decisions.md`). SARIMAX is left unweighted (no per-observation
  weight concept in its state-space MLE). Optuna tuning of
  `half_period_years` is **deferred** to the model-selection/refit step below.
- [x] Tests:
  - Weight of a match exactly half_period days ago = 0.5 × importance
  - Recent WC match has highest weight
  - Very old friendly has near-zero weight

### Model selection for live experiment

Happens here (after A.1–A.4), not in Phase 1. One QA cycle serves both
purposes: verify Phase 1 evaluation fixes and select thesis champions.
The model selection must use the thesis feature set and expanded holdout.

- [x] **A.5** Refit all 10 candidates on the thesis feature set with fixed
  3yr half-period (slimmer features from A.1, plus rolling Elo-change from
  A.2; verify NegBin/Bayes metrics differ from standard Poisson).
  Smoke-test run — confirms A.1–A.4 are wired up correctly.
  - [x] Update all 10 values in `HOLDOUT_RPS_BASELINES` with A.5 holdout RPS
- [x] **A.6** Tune `half_period_years` (Optuna float `[1.0, 5.0]`) for
  each weighted model; recompute `w_time` per trial (one numpy op). Thread
  per-row `days_ago` + `competition_tier` through the tuning objective.
  Report per-model optimised values in methodology; deviation from 3 years
  is a reportable finding.
- [x] **A.7** Refit all 10 candidates with per-model tuned half-periods
  from A.6. This is the selection run.
- [x] Evaluate on expanded holdout (A.3)
- [x] Rank by holdout RPS; select top 3 that beat the mean-rate baseline
- [x] Prefer diversity of model families (e.g. one GLM, one tree, one
  Bayesian) if performance is close
- [x] These 3 + mean-rate baseline = 4 models in the live WC experiment
  (A.8 roster: xgboost, poisson_glm, bayesian_poisson + mean_rate_poisson)
- [x] Narrow live pipeline to the 4-model roster (A.9) — `EXPERIMENT_MODELS` and
  `LIVE_SHADOW_MODELS` wired; done 2026-06-07.
- [ ] Freeze champions for both modes (A.10) — Jun 11 pre-kickoff. See the A.10
  steps in the Sequencing section above for the detailed checklist.

---

## Phase 3 — Cadence experiment infrastructure (on `thesis`)

**Scope note (Option A, decided 2026-06-07):** the live experiment runs the
**4-model roster × 2 cadence modes** end to end — each of xgboost, poisson_glm,
bayesian_poisson, mean_rate_poisson is predicted **and simulated** in both the
frozen and per-round modes (needed for cross-family RQ2 entropy; mean_rate is
the flat floor). Only the `champion` (xgboost) is shown in Streamlit, via the
`champion` alias + a `model_name` artifact filter — **no new registered model
is required** for the display distinction. All B-section work below therefore
loops over `{roster model} × {cadence_mode}`, not a single champion. The 6
non-selected candidates stay as match-level RPS shadows (never simulated).

### B.1. Snapshot metadata tagging

Every inference cycle logs structured metadata for post-WC trajectory
reconstruction.

- [x] In `src/inference/run.py`:
  - Accept `cadence_mode`, `matchday_label`,
    `matches_completed_in_matchday`, `total_matches_completed`
  - Derive matchday/sequence from `parse_wc_results()` output
- [x] In `src/inference/logging.py:log_inference_artifacts`:
  - Add four fields to MLflow `params` dict
- [x] In `src/monitoring/monitor.py`:
  - Tag monitoring rows with `cadence_mode`

### B.2. Dual MLflow aliases and per-mode dispatch

Each roster model runs in two modes side by side during WC: frozen (never
refitted) and per-round (refitted at matchday boundaries).

- [x] Define MLflow registry aliases for the display champion: `champion_frozen`,
  `champion_per_round` (`champion` stays for backward compat, points at frozen).
  The other 3 roster models resolve by `(model_name, cadence_mode)` tag search
  (same tag-based path shadows already use), so no per-model alias proliferation.
- [x] **Degrade gracefully:** until A.10 assigns the dual aliases, the per-mode
  dispatch must fall back to the existing single `champion` path so the pre-WC
  pipeline (run locally or on the freshly deployed GCP job) keeps working.
- [x] In `src/models/mlflow_utils.py`:
  - Add `get_production_run_id(alias: str)` to fetch the display champion by alias
  - Add a `(model_name, cadence_mode)` resolver for the non-champion roster
- [x] In `src/pipeline/trigger.py:dispatch_training_or_inference`:
  - Loop over both cadence modes; each call runs the full roster simulation
    (Option A) and logs artifacts tagged with `cadence_mode` (B.1)
  - Pre-MD1: always run both modes independently (no reuse optimisation)
- [x] Streamlit reads only the `champion` (xgboost) artifact set — filter the
  logged artifacts by `model_name` (update `src/dashboard/load_artifacts.py`);
  dashboard pinned to `cadence_mode=frozen` lineage
- [x] Per-round refit logic only fires for the per-round mode (see B.3)

### B.3. Per-round refit trigger

Matchday-boundary detection fires refit for the per-round mode only.

- [x] In `src/pipeline/trigger.py`:
  - `_last_per_round_refit_matchday()` reads the `per_round_refit_matchday`
    tag from the `champion_per_round` run; returns `None` (pre-A.10) or the
    last matchday label.
  - On boundary change → `run_per_round_refit(df, matchday=next_matchday)`
    refit **all 4 EXPERIMENT_MODELS roster entries** → each registered with
    `cadence_mode=per_round` + `model_name` tags; display champion (xgboost)
    also gets the `champion_per_round` alias on `wc_production`; the other 3
    go to `wc_shadow`.
  - `promote_to_production` generalized to accept `alias=` kwarg.
  - Pre-A.10 fallback: when `champion_per_round` alias absent, legacy
    delta-based refit path is used unchanged.
- [x] Store `per_round_refit_matchday` as MLflow run tag on each refit run
- [x] Frozen mode never refits during WC
- [x] 7 expected refit events: MD1, MD2, MD3, R32, R16, QF, SF
- [x] **Concurrency guard:** whole-run `fcntl` lockfile in `trigger.main()`
  prevents within-host overlap (skip tick if lock held). On Cloud Run Jobs there
  is no `--max-instances` flag — cross-execution overlap is prevented by
  keeping task timeout (50 min) under the scheduler interval (120 min / every-2h),
  plus `--tasks 1 --parallelism 1`. Each execution has a fresh container
  filesystem so the lockfile doesn't persist cross-execution. WC refits are
  spaced hours apart; the guards cover the edge case of a slow bayesian MCMC
  refit approaching the interval limit.

### B.4. Per-mode monitoring

- [x] In `src/monitoring/monitor.py:score_completed_wc_matches`:
  - Look up pre-kickoff inference run for each mode separately
  - Emit rows tagged with `cadence_mode`
- [x] `evaluate_alert_threshold` groups by `(cadence_mode, model_name)`
- [x] `log_monitoring_run` creates `monitor_<mode>_<model_name>` runs

---

## Phase 4 — GCP deployment (on `thesis`)

### C.1. Containerise the trigger

- [x] Wire Dockerfile CMD to `python -m src.pipeline.trigger --mode=auto`
- [x] Verify `pyproject.toml` declares all runtime dependencies
- [x] Add `.dockerignore` (exclude `data/`, `mlruns/`, `.git/`, notebooks)
- [x] Local build + run with mounted credentials

### C.2. Service account and permissions

- [x] Create `wc-mlops-trigger@<project>.iam`
- [x] Grant: `roles/run.invoker`, `roles/artifactregistry.reader`,
  `roles/secretmanager.secretAccessor`, `roles/logging.logWriter`,
  `roles/storage.objectAdmin`

### C.3. Secret Manager

- [x] Create secrets: API-Football key, DagsHub username, DagsHub token,
  MLflow tracking URI
- [x] Wire as env vars in Cloud Run Job definition (C.5). Four secrets map to
  **six** env vars — `dagshub-username` and `dagshub-token` each serve two
  names (`entrypoint.sh` needs `DAGSHUB_*`; MLflow needs `MLFLOW_TRACKING_*`):
  ```
  MLFLOW_TRACKING_URI=mlflow-tracking-uri:latest
  MLFLOW_TRACKING_USERNAME=dagshub-username:latest
  MLFLOW_TRACKING_PASSWORD=dagshub-token:latest
  DAGSHUB_USERNAME=dagshub-username:latest
  DAGSHUB_TOKEN=dagshub-token:latest
  API_FOOTBALL_KEY=api-football-key:latest
  ```

### C.4. Artifact Registry + image push

- [x] Create Docker repo in Artifact Registry
- [x] Build for **`linux/amd64`** (Cloud Run; arm64 Mac builds fail otherwise):
  `docker buildx build --platform linux/amd64 ...`
- [x] Tag with a **unique timestamp** (not `:latest` alone). Deployed tags so
  far: `20260608b` (initial), `20260609a` (entrypoint fix + guard + B.1–B.4).
- [x] **Rebuild only when code changes.** Jun 10 rebuild pending (timeout change).
- [x] Commit `dvc.lock` / `data/raw.dvc` before build — done Jun 9 (7708 files).

### C.5. Cloud Run Job

- [x] Create `wc-mlops-trigger` job with `--image` set to the **dated tag**
  from C.4 (not `:latest`)
- [x] Attach C.3 secrets via `--set-secrets` (six env vars; see C.3)
- [x] **`--tasks 1 --parallelism 1`** — ensures one container per execution.
- [x] Timeout: **50 min** (`--task-timeout 3000`). Python-level shadow-refit
  budget is 30 min (`SHADOW_REFIT_TOTAL_TIMEOUT_S`; raised from 20 min Jun 9,
  per-model cap removed — each model gets the full remaining budget). First
  successful full run was ~35 min, well within both limits.
- [x] Mode: **`auto` via `PIPELINE_MODE` env var** (default in `entrypoint.sh`).
- [x] On code changes: rebuild (C.4) → `gcloud run jobs update --image <dated-tag>`
- [ ] Lower resources: 8 GiB / 4 vCPU → 4 GiB / 2 vCPU
  (`gcloud run jobs update --memory 4Gi --cpu 2`). First successful run (~35 min)
  confirms routine cycles fit. **Unblocked — do before WC.**

### C.6. Cloud Scheduler

Two schedules hit the same `wc-mlops-trigger` job; **both use `mode=auto`**
(default `PIPELINE_MODE`). Only the cron frequency changes — do not switch to
`inference_only` (that skips per-round refit at matchday boundaries).

- [x] **Pre-WC** (now – June 10): daily 04:00 UTC. Cron: `0 4 * * *`.
  Job: `daily-pipeline-trigger`. Created and enabled.
- [ ] **WC** (June 11 – July 19): every 2 hours. Cron: `0 */2 * * *`.
  At kickoff (19:00 UTC Jun 11): pause `daily-pipeline-trigger`, enable every-2h.
  Keeps task timeout (50 min) safely under the 120-min interval.
- [ ] Post-WC: pause both

### C.7. DVC remote on GCS

**Deferred to post-WC.** DagsHub remote is working (blobs + git pointers
verified Jun 9, 7779 files on remote). Migrating days before kickoff is
avoidable risk for marginal gain (only upside is dropping the DagsHub token
from Secret Manager in favour of IAM). Revisit after the tournament.

- [~] Create bucket `gs://wc-mlops-dvc-<project>/`
- [~] `dvc remote add -d gcs gs://wc-mlops-dvc-<project>/`
- [~] Verify `dvc push` works from inside the container

### C.8. Observability

- [ ] Cloud Logging captures stdout/stderr
- [ ] Log-based metric on `ALERT` warnings → alerting policy → email
- [ ] Cloud Monitoring alert on job failure (non-zero exit) → email
- [ ] **Partial-roster guard:** a champion-only cycle exits 0 (the all-model path
  is wrapped in a `try`), so job-failure alerts miss it. Log an `ALERT`/warning when
  `predictions_all_models.csv` has fewer than the expected roster count of
  `model_name`s, so a silently degraded cycle is caught. (Root cause of the
  2026-06-06 case is fixed; this guards future regressions.)

### C.9. Pre-WC test runs

**First pass — done Jun 9 (execution `wc-mlops-trigger-qgwsf`):**
- [x] Trigger Cloud Run Job manually
- [x] Both modes' MLflow runs appear (`cadence_mode=frozen` + `cadence_mode=per_round`)
- [x] Both artifact sets written with correct metadata tags
- [x] All 4 roster models simulated in both modes (`bayesian_poisson, mean_rate_poisson,
  poisson_glm, xgboost`)
- [x] DVC push succeeded (gold 6921 rows, 9 files pushed)

**Final pass — pending A.10 (Jun 11 pre-kickoff):**
- [x] `champion_frozen` and `champion_per_round` aliases resolve
- [x] Monitoring runs empty pre-WC
- [x] Both alias-tagged artifact sets written correctly

---

## Phase 5 — Post-WC analysis (July onward)

### D.1. Trajectory and entropy analysis

- [ ] Extract per-model `tournament_probabilities` from all tagged MLflow
  snapshots, keyed by `(model_name, cadence_mode)` → build trajectory DataFrames
  (4 roster models × 2 modes, per Option A)
- [ ] Compute Shannon entropy per snapshot: normalise 48-team advancement
  vector to `p_i / 32`, then `H = -Σ p_i log(p_i)`. Decompose per-group.
- [ ] Plot entropy resolution curves: frozen vs per-round **for each roster
  model** (cross-family robustness of the cadence effect), with mean_rate_poisson
  as the flat floor

### D.2. Structural-driver regression

- [ ] Dependent variable = |Δp_advance| for each team at each match
  completion
- [ ] Regressors: third-place boundary proximity, groups completed, Elo
  gap, matchday number, kickoff slot

### D.3. WC 2022 frozen-model retrospective

- [ ] Replay 2022 results in order, snapshot trajectories, compute entropy
- [ ] Compare resolution curves (descriptive, not causal)

### D.4. Thesis writing

- [ ] Check university thesis guidelines for page/word limits
- [ ] Structure: Introduction → Background/Literature → Methodology → Data
  & Features → Experimental Setup → Results (RQ1, RQ2, RQ3) → Discussion →
  Conclusion
- [ ] Keep in Overleaf

### D.5. Derive paper revision

- [ ] After thesis is complete: extract paper from thesis
- [ ] Update tables, discussion, hypotheses based on thesis findings

### Traps to remember

- Advancement probabilities sum to 32, not 1 — normalise before entropy
- Keep "who advances" vs "who wins" uncertainty separate
- Reuse existing `round_advancement_probabilities` artifacts
- Mean-rate Poisson baseline provides the flat entropy floor (no information
  enters the model, so entropy never resolves)
- **Goal correlation in Monte Carlo simulation:** Independent Poisson is
  fine for RPS (goal *difference* distribution is identical regardless of
  covariance — Ley et al. 2019 show this mathematically). But for exact
  scoreline simulation, correlation exists (~0.2 per Maher 1982; Dixon &
  Coles 1997 add a low-score adjustment). If group tiebreakers depend on
  exact scorelines (goals scored, head-to-head), this could matter slightly.
  Decision: keep independent Poisson for simplicity; acknowledge in
  limitations.

### Literature benchmarks (for context in thesis)

- Poisson ranking models on national teams: RPS ~0.165 (Ley et al. 2019)
- Hybrid RF on World Cup matches: RPS ~0.187–0.190 (Groll et al. 2019)
- Bookmakers on World Cup matches: RPS ~0.188–0.194 (Groll et al. 2019)
- Our models are not directly comparable (different training regime,
  feature-based not per-team-parameter, evaluated on different holdouts)
  but these provide ceiling context
