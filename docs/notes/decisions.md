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

## Half-period: fixed at 3 years now; Optuna tuning deferred

- Ley et al. found 3 years optimal for all Poisson variants on national teams.
- **A.4 (this phase): fixed 3-year half-life for all weighted models.** Implements the
  full weighting mechanics (the valuable, reusable part) without the search-space cost.
- **Optuna tuning of `half_period_years` ∈ [1.0, 5.0] is deferred to the refit /
  model-selection cycle** (the post-A.4 step). Adding it is low overhead (one numpy
  recompute per trial) and the weight code is structured so the change stays localized.
- Report per-model values in thesis methodology once tuned. Deviation from 3y is a finding.
- `days_ago` measured relative to each split's most recent match date (so the newest
  match has weight ≈ importance; avoids negative `days_ago` / weights > 1 on full-data refit).
- **Mean-rate Poisson baseline: not weighted at all** (plain unweighted grand mean). Its
  prediction is a single constant per match, so time-weighting would only nudge the scalar
  and muddy its role as the no-information floor / flat entropy floor (Phase 5). It accepts
  `sample_weight` and ignores it.

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

## Independent Poisson for simulation

- Bivariate extension (Maher 1982, Dixon & Coles 1997) matters for exact scorelines
  but not for RPS. For group tiebreakers that depend on goal counts, independent
  Poisson introduces a small approximation. Acknowledged in limitations.
- Decision: keep independent Poisson for simplicity.

## Home advantage for host nations in 2026

- USA, Canada, Mexico play in their home country for group stage.
- `is_neutral` overridden to False for these matches at inference time — not a Gold feature.
- For KO rounds: venue assigned from `data/tournament/wc2026.json` bracket mapping.
