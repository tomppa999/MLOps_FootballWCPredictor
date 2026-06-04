# Key Design Decisions

Captures rationale for non-obvious choices. Maps to thesis Chapter 3 (Methodology)
and Chapter 6 (Discussion / Limitations). Add a bullet whenever you make a decision
that won't be obvious from the code alone.

---

## Baseline model: Maher / Dixon & Coles

- Mean-rate Poisson = Maher's Model 0. Defensible, well-cited.
- Dixon & Coles (1997) is the key Maher citation but irrelevant here:
  its low-score correction (ρ for 0-0, 1-0, 0-1, 1-1) only improves
  exact scoreline prediction, not RPS (goal difference distribution is
  identical — bivariate correction cancels in X−Y).
- Ley et al. (2019) IS the national-team successor to Maher/D&C.
  Frame it as a lineage: Maher → Dixon & Coles → Ley et al.
- See also: `docs/literature/prediction_framing.md`

## Half-period: fixed at 3yr in A.4/A.5; marginal Optuna tuning in A.6

- Ley et al. found 3 years optimal for all Poisson variants on national teams.
- **A.4/A.5 (fixed phase): 3-year half-life for all weighted models.**
- **A.6 (tuning phase): marginal 1-D Optuna search** — freeze A.5 hyperparameters per
  model and tune only `half_period_years ∈ [1.0, 5.0]` (25 trials, minimise walk-forward
  CV NLL). Entry point: `python -m src.models.half_period_tuning`.
- **Why marginal (not joint):** Isolates the half-period effect cleanly. Joint tuning from
  scratch would discard A.5's purpose, confound half-period with re-tuned hyperparameters,
  and re-run expensive searches. The interaction between half-period and other hyperparameters
  is weak for these models.
- **Persistence:** tuned values are pasted manually into `TUNED_HALF_PERIODS` in
  `config.py`. All downstream paths (deploy, champion refit, shadow refit) read
  `ChampionMeta.half_period_years` from MLflow params (default 3.0 for pre-A.6 runs).
- `half_period_years` is in `_DEPLOY_INTERNAL_PARAMS` so it is never passed to model
  constructors. It is used only in `make_splits(..., half_period_years=...)`.
- Report per-model values in thesis methodology. Deviation from 3yr is a finding.
- `days_ago` measured relative to each split's most recent match date (so the newest
  match has weight ≈ importance; avoids negative `days_ago` / weights > 1 on full-data refit).
- **Mean-rate Poisson baseline: not weighted at all** (plain unweighted grand mean). Its
  prediction is a single constant per match, so time-weighting would only nudge the scalar
  and muddy its role as the no-information floor / flat entropy floor (Phase 5). It accepts
  `sample_weight` and ignores it.

## A.6 first-run findings (2026-06-03/04) — interim, pre-Poisson-GLM-fix

- **The half-period objective is nearly flat for most models.** negbin, random_forest,
  bayesian_poisson, lstm, cnn move CV NLL only in the 3rd–4th decimal across [1,5] → the
  Ley et al. 3yr default is at/near optimal. Sub-0.002 deltas vs 3yr are noise.
- **ridge is the one clean, real finding:** deterministic, genuinely prefers a longer
  ~4.8yr half-life (2.8349 → 2.8304). xgboost weakly agrees (~4.9yr) but within RNG.
- **TPE-never-samples-3.0 artifact:** because the 1-D search is continuous, hp=3.0 is
  never evaluated, so a model's reported "best of 25" can sit marginally above its 3yr
  value even though `min over [1,5]` must be ≤ it. This is a sampling artifact, not
  degradation. Fix for the rerun: pin 3.0 into the search (`study.enqueue_trial` or an
  explicit grid incl. 3.0), making "deviation from 3yr" exact. If a model's tuned best
  does not beat 3.0, keep 3.0.
- **poisson_glm instability (first run):** custom weighted MLE at alpha≈9.9 blew up in
  some A.6 trials (NLL 21.78, 3.67). Root cause: raw weight totals vary with
  `half_period_years` (short hp → low sum, long hp → high sum) while the L2 penalty
  `0.5 * alpha * Σβ²` is fixed, so the objective scale shifts and the optimizer can fail.
- **Fix (2026-06-04):** in `BivariatePoisson.fit`, normalize weights to mean 1
  (`w *= len(w) / w.sum()`) before the MLE; fall back to zero init if L-BFGS-B does not
  report success. Relative time-decay × importance structure is unchanged; global scale
  no longer confounds half-period comparison. **3.0 pinned** via `study.enqueue_trial`
  in `tune_half_period`. Full A.6 rerun still required to populate `TUNED_HALF_PERIODS`.
- **bayesian_poisson** NLL is flat but it emits persistent MCMC divergences (up to 94 in a
  trial). NLL/means are stable, so usable, but flag as a reliability caveat for RQ2
  (uncertainty); consider raising `target_accept`/`tune_steps` for the champion refit.
- **lstm/cnn run-to-run variance:** lstm's A.6 best (2.8532) sits ~0.04 above its A.5 NLL
  (2.8150); the hp landscape is flat, so the offset is Keras training nondeterminism
  across process runs, not a half-period effect.
- **Decision:** stay marginal (joint won't fix Poisson and the interaction is weak).
  Code fix + 3.0 pin landed 2026-06-04; user runs full A.6 rerun, then pastes validated
  values into `TUNED_HALF_PERIODS`. Until then `TUNED_HALF_PERIODS` stays empty → 3yr fallback.

## Competition tier: dual role (sample weight + predictor feature)

- As sample weight: tells the optimizer WC matches are more informative signal.
- As predictor feature: lets the model adjust lambda for the match being predicted.
- Complementary, not redundant — they operate at different stages (training vs inference).
- Weights from Ley et al. (1 / 2.5 / 3 / 4) = pre-2018 FIFA methodology.
  Cite as "Ley et al.'s weights", NOT "current FIFA weights" (FIFA changed 2018).

## Bayesian Poisson: weighted (tempered) likelihood (A.4)

- PyMC has no `sample_weight` kwarg. `pm.Poisson(observed=y)` adds an unweighted
  `Σ log P(y_i | λ_i)` to the model log-density.
- To apply time-decay × importance weights, replace `observed=` with a manual term:
  `pm.Potential((w * pm.logp(pm.Poisson.dist(mu=λ), y)).sum())`.
- This yields a *weighted / tempered pseudo-posterior*: `prior × Π P(y_i|λ_i)^{w_i}`,
  not an exact Bayesian posterior.
- Consequence: posterior credible intervals no longer reflect the true sample size —
  weighting shrinks the *effective* N (down-weighted old matches contribute less
  information). This is the intended recency emphasis, but the uncertainty is
  conditioned on the weighting scheme. Flag as a one-line methodology caveat
  (relevant to RQ2 / uncertainty interpretation).
- Weighting is training-only; per-match predictive spread still comes from the
  posterior over β, so `predict_samples()` (distribution-aware scoring) is unaffected.
- Same weighted-likelihood idea as the GLMs (Poisson GLM scales its per-obs log-PMF;
  NegBin uses statsmodels `var_weights`) — just expressed via `pm.Potential`.

## Bayesian Poisson: MCMC divergences during A.5 tuning

- During A.5 (Optuna tuning, 40 trials, 500 draws + 500 tune steps per trial),
  two trials produced PyMC divergence warnings: 25 divergences (trial ~1) and
  20 divergences (trial ~2). Full log: `logs/pipeline_a5_20260602T130157.log`.
- At ~5% of draws (25/500), this is in the "mild but acceptable" range. Posterior
  means used in `predict()` are still reliable; the tails may be slightly underexplored.
- Root cause: the weighted likelihood via `pm.Potential` (A.4) changes the effective
  curvature of the posterior, making NUTS harder to navigate for some hyperparameter
  combinations proposed by Optuna.
- Mitigation if Bayesian Poisson is a champion candidate: raise `tune_steps` to
  1000–2000 and add `target_accept=0.9` to `pm.sample()` in the final champion refit.
  Not needed for A.5 smoke test.

## Tactical features: dropped (A.1)

- Permutation importance negligible across all 9 models.
- ~49% coverage gap biases training toward UEFA/CONMEBOL.
- Kept in GOLD_COLUMNS for transparency; removed from FEATURE_COLUMNS.

## Rolling Elo-change: included despite weak linear signal (A.2)

- Partial correlation r = −0.028, ΔR² = 0.0008 in a linear model. Weak linear signal.
- Included because: full coverage, no cost, potential nonlinear capture by tree models.
- Justified by ablation (test holdout RPS with/without after refit).

## Holdout expansion: continental tournaments 2022–2025 (A.3)

**Goal distributions** (per-team goals, Poisson lambda):
- KS tests: no tournament's goal distribution differs significantly from WC 2022 (all p > 0.6).
- Pooled tier-1 (WC/EURO/Copa) vs tier-2 (AFCON/Asian Cup/Gold Cup): KS = 0.030, p = 0.999.
- Gold Cup 2023 is a lambda outlier (1.69 vs WC 2022 = 1.34). Acknowledged, kept.
- Expands holdout from 64 → ~347 matches (4.9×). Needed for stable RPS estimation.

**Elo-difference distributions** (|home_elo_pre − away_elo_pre|):

| Tournament     |  N | Mean | Median | Std | p vs WC 2022 |
|----------------|----|------|--------|-----|--------------|
| WC 2022        | 64 |  188 |    168 | 125 | —            |
| AFCON 2024     | 52 |  161 |    141 | 113 | 0.164        |
| EURO 2024      | 51 |  157 |    128 | 107 | 0.140        |
| Gold Cup 2023  | 31 |  206 |    185 | 150 | 0.967        |
| Gold Cup 2025  | 31 |  221 |    202 | 123 | 0.205        |
| Copa 2024      | 31 |  209 |    231 | 153 | 0.432        |
| Asian Cup 2024 | 51 |  253 |    210 | 164 | 0.063        |

- No tournament is statistically different from WC 2022 in Elo-diff distribution (all p > 0.05).
- Asian Cup 2024 is the borderline case (p = 0.063, KS = 0.240). It has more extreme
  mismatches: ~10% of matches have a 500+ Elo gap vs ~3% for WC 2022. This reflects the
  wider range of team quality in AFC.
- AFCON 2024 vs Asian Cup 2024 is the only pairwise comparison that crosses p < 0.05
  (p = 0.049, KS = 0.261) — the two tier-2 tournaments differ from each other more
  than either differs from WC 2022.

**WC 2026 group stage benchmark** (current Elo values, computed pre-tournament):
- Mean |Elo diff| across all 72 group matches: **226** (std = 144).
- This is higher than WC 2022 (188) due to the 48-team expansion bringing in weaker teams.
- It is closer to Asian Cup 2024 (253) than to WC 2022.
- Argument for keeping Asian Cup: its wider spread is arguably more representative of
  WC 2026's matchup landscape than WC 2022 was.
- Most balanced group: D (USA, Paraguay, Australia, Turkey) — mean |Δ| = 100.
- Most lopsided group: H (Spain, Cape Verde, Saudi Arabia, Uruguay) — mean |Δ| = 358;
  Spain vs Cape Verde = 608 Elo points, the single largest gap in WC 2026 group stage.

## Goal-based prediction vs direct W/D/L

- See `docs/literature/prediction_framing.md` for the full argument.
- Summary: Ley et al. ALSO predict goals (Poisson) and get their best RPS that way.
  Approach is correct. RPS gap explained by missing time-decay, holdout difficulty,
  and feature-based vs parameter-based estimation. Not a modeling failure.

## WC 2022 role: retrospective only, not a manipulated condition

- Using WC 2022 as a second experimental condition (alongside WC 2026) would confound
  format effects with inter-tournament changes. Avoided by design.
- WC 2022 appears in results only as a descriptive frozen-model replay (D.3).

## Per-round mode: re-fit only, not re-tune or re-select

- The experimental factor (RQ1/RQ3) is *retraining cadence*. To attribute any RPS/NLL
  difference to cadence, everything except the data the parameters are fit on must be
  held constant. The per-round mode therefore re-fits **parameters only** (GLM
  coefficients, tree splits, posterior) with **fixed, pre-tournament-tuned
  hyperparameters and a fixed champion architecture**.
- Three distinct levels of "updating": (1) re-fit = re-estimate parameters; (2) re-tune
  = re-run Optuna hyperparameter search; (3) re-select = full exp→qa→deploy with possible
  architecture change. Only level (1) is used per round. Operationally this is
  `run_champion_refit` fired at each matchday boundary (B.3); the half-period and all
  hyperparameters come from the one-time pre-tournament A.6/A.7 tuning.
- Why not re-tune per round: each boundary adds only ~dozens of matches (shrinking through
  the knockouts). Optuna on that increment has essentially no signal and would fit
  hyperparameters to noise; re-fitting *parameters* on it is already aggressive. Re-tuning
  would also confound two axes (fresher parameters vs different hyperparameters), so a
  result could no longer be attributed to cadence. It would additionally inject Optuna
  search stochasticity, hurting round-over-round reproducibility.
- Aligns with standard MLOps usage: "retraining" in production normally means scheduled
  re-fit; hyperparameter tuning is a rarer offline activity.
- Honest tradeoff (state in methodology/limitations): per-round runs with hyperparameters
  optimized on the larger, partly different-regime pre-tournament data, not re-optimized
  for the small tournament set. This is the correct call because the alternative
  (re-tuning on tiny data) is worse, and it matches realistic production practice.
- Both modes always *infer* on fresh feature values (updated Elo, rolling windows). The
  frozen mode adapts to the tournament through features only; the per-round mode also
  adapts through parameters. The experiment isolates the parameter-update effect.
- Terminology hygiene for the thesis: define "retraining" once as parameter re-estimation
  (re-fit), explicitly excluding hyperparameter re-tuning and model re-selection. The RQ
  labels use "retraining" in this scoped sense.

## Independent Poisson for simulation

- Bivariate extension (Maher 1982, Dixon & Coles 1997) matters for exact scorelines
  but not for RPS. For group tiebreakers that depend on goal counts, independent
  Poisson introduces a small approximation. Acknowledged in limitations.
- Decision: keep independent Poisson for simplicity.

## CV objective vs model-selection metric: NLL within families, RPS across families

- Optuna minimises each model's native log-likelihood (NLL) via walk-forward CV.
  For Poisson GLM this is Poisson NLL; for NegBin it is NB NLL; for Bayesian Poisson
  it is MC posterior NLL. This is maximum likelihood within the distribution family —
  consistent with Ley et al. (2019) and Groll et al. (2019).
- NLL values are **not comparable across distribution families**: NegBin spreads
  probability mass more broadly (heavier tails), so its NLL is arithmetically higher
  than Poisson NLL even when the model is better calibrated. Comparing CV NLL between
  Poisson GLM and NegBin is meaningless.
- Holdout **RPS** is the cross-model selection metric. RPS operates on predicted
  outcome probabilities (H/D/A), not on the PMF shape, making it distribution-agnostic
  and directly comparable across all candidates. This split (NLL for tuning, RPS for
  selection) is standard in the literature.
- Thesis methodology note: "Hyperparameter search minimises each model's native
  log-likelihood via walk-forward CV; cross-model comparison uses holdout RPS as
  the common scoring metric, consistent with Ley et al. (2019)."

## Home advantage for host nations in 2026

- USA, Canada, Mexico play in their home country for group stage.
- `is_neutral` overridden to False for these matches at inference time — not a Gold feature.
- For KO rounds: venue assigned from `data/tournament/wc2026.json` bracket mapping.
