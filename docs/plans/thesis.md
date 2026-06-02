# Thesis Plan — Consolidated

Branch strategy: **`main`** (model fixes) → **`thesis`** (thesis-specific work)
MLflow experiment: `wc_mlops_thesis`
WC 2026 starts: **June 11, 2026**

---

## Framing

**Candidate titles** (to discuss Monday):

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

- [ ] **1a** Monitoring threshold → naive-baseline floor
- [ ] **1b** Fix NegBin and Bayesian Poisson evaluation collapse
- [ ] **1c** Add symmetric mean-rate Poisson baseline (as threshold, not
  peer candidate — intercept-only case of Maher 1982)
- [ ] **1d** Drift monitoring — Threats to Validity paragraph (once thesis
  document exists)
- [ ] Rerun full QA with all 10 candidates (verify fixes)

### Phase 2 — Thesis feature/data changes (`thesis` branch, weeks 2–3)

- [ ] **A.1** Drop in-game statistics features
- [ ] **A.2** Add rolling Elo-change (pending prof approval Monday)
- [ ] **A.3** Expand holdout to continental tournaments (pending prof
  approval Monday)
- [ ] **A.4** Add time-decay + match-importance sample weights
- [ ] Refit all 10 candidates on thesis feature set
- [ ] Select top 3 that beat the baseline → thesis live experiment uses
  these 3 + baseline (4 models total)
- [ ] Freeze champions for both modes

### Phase 3 — Cadence infrastructure (`thesis` branch, weeks 2–3)

- [ ] **B.1** Snapshot metadata tagging
- [ ] **B.2** Dual MLflow aliases + per-mode dispatch
- [ ] **B.3** Per-round refit trigger
- [ ] **B.4** Per-mode monitoring

### Phase 4 — GCP deployment (`thesis` branch, by June 10)

- [ ] **C.1–C.9** Containerise, deploy, test

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

- [ ] In `src/monitoring/baselines.py`:
  - Add `NAIVE_BASELINE_RPS: Final[float] = 0.235` with docstring
    (derivation: uniform-random predictor, neutral venue, p_draw ≈ 0.25)
  - Keep `WC2022_RPS_BASELINES` for context logging
  - Remove or deprecate `ALERT_FACTOR`
- [ ] In `src/monitoring/monitor.py:evaluate_alert_threshold`:
  - Change condition to `rolling_rps > NAIVE_BASELINE_RPS`
  - Update log message
- [ ] Update tests referencing `ALERT_FACTOR`

### 1b. Fix NegBin and Bayesian Poisson evaluation collapse

**Problem:** Both models collapse to standard Poisson at evaluation time.
NegBin's fitted dispersion is discarded; Bayesian Poisson's posterior is
reduced to its mean before scoring. H4 was likely incorrectly rejected.

#### Model interface

- [ ] In `src/models/base.py`:
  - Add `distribution_family` property (default `"poisson"`)
  - Keep `predict()` backward-compatible

#### NegBin GLM (`src/models/candidates/negbin_glm.py`)

- [ ] Store fitted `alpha` from statsmodels GLM result
- [ ] Extend `predict()` to optionally return `(lambda_h, lambda_a, alpha_h,
  alpha_a)`
- [ ] Set `distribution_family = "negbin"`

#### Bayesian Poisson (`src/models/candidates/bayesian_poisson.py`)

- [ ] Retain posterior samples (replace `.mean(dim=["chain", "draw"])`
  collapse with stored sample arrays)
- [ ] Add method to draw lambda samples for a given X (vectorised matmul)
- [ ] Set `distribution_family = "bayesian_poisson"`

#### Evaluation (`src/models/evaluation.py`)

- [ ] Add `compute_outcome_probs_nb(lambda, alpha, max_goals=10)` using
  `scipy.stats.nbinom.pmf`
- [ ] Add `compute_outcome_probs_bayes(lambda_h_samples, lambda_a_samples,
  max_goals=10)` — averages Poisson PMFs across N posterior samples
- [ ] Add `compute_mean_nll_nb(...)` and `compute_mean_nll_bayes(...)`
- [ ] Add dispatcher `compute_outcome_probs_dispatch(model, X)` that selects
  scoring function based on `model.distribution_family`

#### Training / tuning

- [ ] In `src/models/tuning.py`: NegBin Optuna trials minimise NB NLL
- [ ] In `src/models/tuning.py`: Bayesian Poisson trials minimise MC
  posterior NLL
- [ ] Re-run Experimental → QA on pre-WC 2022 data for both candidates
  (other seven candidates unchanged)

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

- [ ] New file `src/models/candidates/mean_rate_poisson.py`:
  - `MeanRatePoisson(BaseModel)`, no hyperparameters
  - `fit(X, y)`: `self._lambda = (y[:, 0].mean() + y[:, 1].mean()) / 2`
  - `predict(X)`: `(np.full(n, self._lambda), np.full(n, self._lambda))`
  - `distribution_family = "poisson"`, `name = "mean_rate_poisson"`
- [ ] Register in `src/models/config.py` candidate list
- [ ] Skip Optuna — fits in milliseconds
- [ ] Walk-forward CV NLL and QA holdout RPS via existing harness

### 1d. Drift monitoring acknowledgement

- [ ] Thesis text (once document structure exists, see D.4): paragraph in
  Threats to Validity acknowledging the absence of statistical data-drift
  monitoring and explaining the rationale (goals are statistically stable,
  Elo is self-correcting). Not blocking any code work.

---

## Phase 2 — Data and feature changes (on `thesis`)

### A.1. Drop in-game statistics features

Permutation importance shows tactical features contribute essentially
nothing; ~49% coverage gap biases training toward UEFA/CONMEBOL.

- [ ] In `src/gold/schema.py`:
  - Remove `ROLLING_SHOT_COLUMNS` and `ROLLING_TACTICAL_COLUMNS` from
    `FEATURE_COLUMNS` (keep in `GOLD_COLUMNS` for transparency)
- [ ] Update tests asserting feature count or specific feature names
- [ ] Refit all candidates on the slimmer feature set

### A.2. Add rolling Elo-change (pending Monday approval)

Captures opponent-quality-adjusted form. Elo values reflect long-term
strength but change slowly; rolling Elo-change captures recent relative
performance in a single number. Full coverage, no new data source needed.

**Preliminary analysis (to present Monday):**
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

- [ ] In `src/gold/rolling_features.py:_build_team_history`:
  - Include `elo_pre` and `elo_post` in per-team history rows
- [ ] In `_rolling_for_team`:
  - Compute `rolling_elo_change = (last_n["elo_post"] -
    last_n["elo_pre"]).sum()`
  - Make window size configurable (default 5, test 3/5/10)
- [ ] In `src/gold/schema.py`:
  - Add `home_team_rolling_elo_change` and `away_team_rolling_elo_change`
  - Add to `FEATURE_COLUMNS` and `GOLD_DTYPES` (`Float64`)
- [ ] Tests:
  - Correctness on synthetic team history
  - Strict time-awareness (no future matches in window)
  - Empty history → NaN

### A.3. Expand holdout (pending Monday approval)

Training cutoff stays at `WC_2022_START`. Holdout = union of WC 2022 +
continental tournament finals through WC 2026. Expands holdout from 64 to
~310+ matches (4.9×).

**Preliminary analysis (to present Monday):**
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

- [ ] In `src/models/data_split.py`:
  - Define tournament date/tier constants
  - Replace single-window holdout mask with union of date-range + tier
    filters
  - Training mask stays `date_utc < WC_2022_START`
- [ ] Tests:
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

- [ ] In training harness (`src/models/tuning.py` or `data_split.py`):
  - Compute `days_ago` from `date_utc` relative to training cutoff
  - Compute `w_time = 0.5 ** (days_ago / (3 * 365.25))`
  - Map `competition_tier` → importance weight
  - Pass `sample_weight = w_time * w_importance` to model fit
- [ ] XGBoost: `sample_weight` parameter in DMatrix
- [ ] GLMs (Poisson, NegBin): frequency/exposure weights in statsmodels
- [ ] Bayesian Poisson: weighted likelihood (scale log-likelihood per obs)
- [ ] Tune Half Period via Optuna: add `half_period_years` as a float
  hyperparameter (range `[1.0, 5.0]`) for each model whose training uses
  sample weights. Recompute `w_time` per trial — one cheap numpy operation.
  For `mean_rate_poisson` baseline, fix at 3 years (Ley et al. optimum).
  Report the per-model optimised values in the methodology section; any
  deviation from 3 years is an interesting reportable finding.
- [ ] Tests:
  - Weight of a match exactly half_period days ago = 0.5 × importance
  - Recent WC match has highest weight
  - Very old friendly has near-zero weight

### Model selection for live experiment

Happens here (after feature changes), not in Phase 1. The model selection
must use the thesis feature set and expanded holdout.

- [ ] Refit all 10 candidates on the thesis feature set (slimmer features
  from A.1, plus rolling Elo-change from A.2 if approved)
- [ ] Evaluate on expanded holdout (A.3) if approved, else WC 2022 only
- [ ] Rank by holdout RPS; select top 3 that beat the mean-rate baseline
- [ ] Prefer diversity of model families (e.g. one GLM, one tree, one
  Bayesian) if performance is close
- [ ] These 3 + mean-rate baseline = 4 models in the live WC experiment
- [ ] Freeze champions for both modes (frozen + per-round start identical)

---

## Phase 3 — Cadence experiment infrastructure (on `thesis`)

### B.1. Snapshot metadata tagging

Every inference cycle logs structured metadata for post-WC trajectory
reconstruction.

- [ ] In `src/inference/run.py`:
  - Accept `cadence_mode`, `matchday_label`,
    `matches_completed_in_matchday`, `total_matches_completed`
  - Derive matchday/sequence from `parse_wc_results()` output
- [ ] In `src/inference/logging.py:log_inference_artifacts`:
  - Add four fields to MLflow `params` dict
- [ ] In `src/monitoring/monitor.py`:
  - Tag monitoring rows with `cadence_mode`

### B.2. Dual MLflow aliases and per-mode dispatch

Two model artifacts run side by side during WC: frozen (never refitted) and
per-round (refitted at matchday boundaries).

- [ ] Define MLflow registry aliases: `champion_frozen`, `champion_per_round`
  (`champion` stays for backward compat, points at frozen)
- [ ] In `src/models/mlflow_utils.py`:
  - Add `get_production_run_id(alias: str)` to fetch by alias
- [ ] In `src/pipeline/trigger.py:dispatch_training_or_inference`:
  - Loop over both modes: load each mode's champion, predict, simulate, log
    artifacts with the appropriate `cadence_mode`
- [ ] Per-round refit logic only fires for the `champion_per_round` alias

### B.3. Per-round refit trigger

Matchday-boundary detection fires refit for the per-round mode only.

- [ ] In `src/pipeline/trigger.py`:
  - Compare `next_matchday` against last per-round refit's recorded matchday
  - On boundary change → `run_champion_refit` on current Gold → register
    under `champion_per_round`
- [ ] Store last per-round refit matchday as MLflow tag on the refit run
- [ ] Frozen mode never refits during WC
- [ ] 7 expected refit events: MD1, MD2, MD3, R32, R16, QF, SF

### B.4. Per-mode monitoring

- [ ] In `src/monitoring/monitor.py:score_completed_wc_matches`:
  - Look up pre-kickoff inference run for each mode separately
  - Emit rows tagged with `cadence_mode`
- [ ] `evaluate_alert_threshold` groups by `(cadence_mode, model_name)`
- [ ] `log_monitoring_run` creates `monitor_<mode>_<model_name>` runs

---

## Phase 4 — GCP deployment (on `thesis`)

### C.1. Containerise the trigger

- [ ] Wire Dockerfile CMD to `python -m src.pipeline.trigger --mode=auto`
- [ ] Verify `pyproject.toml` declares all runtime dependencies
- [ ] Add `.dockerignore` (exclude `data/`, `mlruns/`, `.git/`, notebooks)
- [ ] Local build + run with mounted credentials

### C.2. Service account and permissions

- [ ] Create `wc-mlops-trigger@<project>.iam`
- [ ] Grant: `roles/run.invoker`, `roles/artifactregistry.reader`,
  `roles/secretmanager.secretAccessor`, `roles/logging.logWriter`,
  `roles/storage.objectAdmin`

### C.3. Secret Manager

- [ ] Create secrets: API-Football key, DagsHub username, DagsHub token,
  MLflow tracking URI
- [ ] Wire as env vars in Cloud Run Job definition

### C.4. Artifact Registry + image push

- [ ] Create Docker repo in Artifact Registry
- [ ] Tag and push image

### C.5. Cloud Run Job

- [ ] Create `wc-mlops-trigger` job
- [ ] Resources: 4 GiB / 2 vCPU routine, 8 GiB / 4 vCPU initial
- [ ] Timeout: 60 min initial, 15 min routine
- [ ] Args: `--mode=auto`

### C.6. Cloud Scheduler

- [ ] **Pre-WC** (now – June 10): daily 04:30 UTC. Cron: `0 6 * * *`
- [ ] **WC** (June 11 – July 19): every 30 min. Cron: `*/30 * * * *`.
  Pause pre-WC schedule.
- [ ] Post-WC: pause both

### C.7. DVC remote on GCS

- [ ] Create bucket `gs://wc-mlops-dvc-<project>/`
- [ ] `dvc remote add -d gcs gs://wc-mlops-dvc-<project>/`
- [ ] Verify `dvc push` works from inside the container

### C.8. Observability

- [ ] Cloud Logging captures stdout/stderr
- [ ] Log-based metric on `ALERT` warnings → alerting policy → email
- [ ] Cloud Monitoring alert on job failure (non-zero exit) → email

### C.9. Pre-WC test runs

- [ ] Trigger Cloud Run Job manually 2–3 times before June 10
- [ ] Verify:
  - Both modes' MLflow runs appear
  - Both artifact sets written with correct metadata tags
  - `champion_frozen` and `champion_per_round` aliases resolve
  - Monitoring runs empty pre-WC
  - DVC push succeeds

---

## Phase 5 — Post-WC analysis (July onward)

### D.1. Trajectory and entropy analysis

- [ ] Extract `tournament_probabilities.csv` from all tagged MLflow
  snapshots per mode → build trajectory DataFrames
- [ ] Compute Shannon entropy per snapshot: normalise 48-team advancement
  vector to `p_i / 32`, then `H = -Σ p_i log(p_i)`. Decompose per-group.
- [ ] Plot entropy resolution curves (frozen vs per-round vs mean-rate
  baseline floor)

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
