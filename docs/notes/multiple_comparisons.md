# Multiple-comparison strategy for RQ1–RQ3

## Decision and status

Approved on 31 October 2026. The four families are implemented in the RQ
stages and appear in the regenerated tables. The
[findings memo](findings_memo.md) records the agreed interpretation.

**The family revision occurred after inspection of the results**, including an
earlier Holm analysis that counted all reported rows. That implementation and
its outputs were selectively rolled back. These families are not prespecified
or preregistered. The primary, secondary, and exploratory labels below describe
the approved reporting hierarchy, not a hierarchy established before observing
the results. Revised correction does not remove the limitations of this
post-result choice.

Use Holm correction at familywise alpha 0.05 within each approved core family.
There are four core families, containing 54 tests in total: 9 + 9 + 18 + 18.
The error-control scope is each family separately, not all 54 tests jointly or
the entire study. Benjamini–Hochberg is not part of this strategy.

Holm allows arbitrary dependence between tests, provided the individual null
p-values are valid. It does not repair temporal dependence or exchangeability
problems within the underlying resampling tests. The usual fixed-family
guarantee must not be presented as eliminating the effects of revising families
after seeing results. See the [R statistical documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
and Holm (1979), *A Simple Sequentially Rejective Multiple Test Procedure*.

## Shared implementation and reporting rules

- Retain the existing statistical tests, resampling settings, seeds, raw
  p-values, effect estimates, sample sizes, and bootstrap confidence intervals.
- Apply `statsmodels.stats.multitest.multipletests(method="holm")` only to
  canonical rows in the four core families below, using their original
  full-precision `p_perm` values before CSV serialization.
- Append `p_holm` without replacing any original columns. Only canonical core
  rows receive adjusted values. Non-core rows retain their raw statistics and
  have a missing adjusted value; missing means outside the correction scope,
  not nonsignificant or p = 1.
- Validate family membership and cardinality explicitly. Each listed core
  comparison must occur exactly once and have a finite p-value in [0, 1]. A
  missing, duplicated, or invalid core comparison is an error, not a reason to
  silently shrink the family.
- Repeated displays of a hypothesis do not add tests. Identify their canonical
  source; do not select the smallest raw p-value across repeated Monte Carlo
  evaluations. Preserve the original raw statistics on duplicate display rows.
- Confidence intervals remain pointwise 95% bootstrap intervals. They are not
  simultaneous or Holm-adjusted intervals. Report effects and intervals beside
  raw and adjusted p-values; do not imply that a pointwise interval excluding
  zero establishes significance after correction.
- Make formal correction-based claims only for the approved core comparisons.
  Exploratory and descriptive results may retain nominal p-values, clearly
  labelled as unadjusted, without formal significance claims. Their promotion
  to inferential families requires a separate decision.

The three treatment-contrast models are `xgboost`, `poisson_glm`, and
`bayesian_poisson`. Shared matches and reconstruction inputs do not turn their
comparisons into exact duplicate hypotheses, but they limit independent
replication claims.

## RQ1: predictive accuracy

Canonical source: `docs/tables/rq1_paired_phase.csv`, produced by stage `rq1`.
Select `phase_scheme = group_ko`, `phase` in `{all, group, KO}`, and the three
treatment-contrast models.

- **Core family `rq1_rps`: 9 tests.** Three cells × three models, `metric = rps`.
  RPS is the primary outcome-probability accuracy readout.
- **Core family `rq1_nll`: 9 tests.** The same cells and models, `metric = nll`.
  NLL is the secondary core readout for scoreline-distribution accuracy.
- **Exploratory RMSE set: 9 comparisons**, using the same selectors and
  `metric = rmse`. No Holm correction or formal significance claims by default.

`all` covers 80 treated matches, MD2–Final; `group` covers MD2–MD3 (48 matches)
and `KO` covers the remaining 32 matches. Negative differences
(`per_round − frozen`) favour per-round refitting for all three metrics.
Separate RPS and NLL families support metric-specific conclusions, not a
study-wide claim selected because any accuracy metric passes a threshold.

The `group` and `all` rows in `group_r32_r16plus` repeat the canonical
`group_ko` hypotheses. The R32 phase comparison also repeats the R32 round
comparison. Neither repetition increases a family count. R32/R16+ subdivisions,
individual rounds, and cumulative curves remain exploratory/descriptive.

MD1 is a construction-enforced placebo tie and is excluded from treatment
families. Cumulative views overlap the same observations and include MD1 zeros;
they are not independent confirmations. Their changed denominators and
bootstrap distributions must not be represented as identical estimates.

The nine tests per metric compare cadences within cells. They do not directly
test whether the group and knockout effects differ from each other. Different
significance labels across phases do not establish phase heterogeneity.

## RQ2: uncertainty resolution

Canonical source: `docs/tables/rq2_paired_stage.csv`, produced by stage `rq2`.

**Core family `rq2_entropy`: 18 tests.** Select `window = treated`,
`grain = state`, the three treatment-contrast models, and all six targets:
`entropy_r32`, `entropy_r16`, `entropy_qf`, `entropy_sf`, `entropy_final`, and
`entropy_winner`. Adjust all 18 together, not separately by target or model.

The core claim concerns average entropy differences over the observed treated
states for each advancement target. Winner entropy may be the headline, but
remains in this family. The common treated window supports comparison of
targets over the same observed states; the existing data have 88 treated states
per fitted model. Positive differences (`H_frozen − H_per_round`) indicate
lower per-round entropy, not necessarily better accuracy or calibration.

Keep the following outside the core family:

- `treated_unresolved`: exploratory sensitivity to excluding already locked
  targets; its observation window varies by target.
- Cycle grain: sensitivity to snapshot weighting and repeated quiet periods.
- Matchday trajectories: descriptive calendar views, including small late
  samples; a single state does not supply a usable resampling test.
- `all` window: descriptive mixture of treatment and pre-treatment periods.
- Placebo labels 1–2: diagnostics, including the premature-refit residue, not
  clean treatment-effect evidence.
- `mean_rate_poisson`: negative-control/Monte Carlo diagnostic, not one of the
  three fitted-model treatment comparisons. Its observed gaps need not be
  exactly zero.
- Already locked target comparisons: structural diagnostics, not additional
  treatment hypotheses. Zero gaps retained within a core treated-window mean
  are part of that estimand, not extra tests.

Alternative windows and cycle/state grains generally change the observations
or their weighting. They are not exact duplicates, but neither are they
independent replications of the broad uncertainty claim. State deduplication
does not establish temporal independence. The existing tests compare entropy
levels; they do not directly test a change in the rate of entropy resolution.

## RQ3: calibration and concept-drift probes

Canonical core source: `docs/tables/rq3_calibration_phase.csv`, produced by
stage `rq3`. Select `phase_scheme = group_ko`, `phase` in `{all, group, KO}`,
and the three treatment-contrast models. These cells use the same treated
match counts as RQ1.

**Core family `rq3_calibration`: 18 tests.** ECE and reliability
(`metric` in `{ece, rel}`) × three models × three cells. Adjust both metrics
together. ECE is primary and reliability is supporting; combining them avoids
choosing whichever calibration readout gives the stronger result. Negative
differences (`per_round − frozen`) favour per-round calibration.

**Exploratory score/decomposition set: 18 comparisons.** Brier score and
resolution (`metric` in `{bs, res}`) × the same models and cells. These remain
uncorrected, without formal significance claims by default. Brier score is
lower-is-better; resolution is higher-is-better. Neither alone identifies
calibration drift. Outcome-only uncertainty (`unc`) is excluded because its
cadence difference is structurally zero.

**Exploratory/descriptive trend set: 24 tests.** The 12 rows in
`docs/tables/rq3_trend.csv` contain two hypotheses each:

- `p_frozen`: frozen ECE/REL slope > 0, by one-sided period-label permutation.
- `p_diff`: frozen slope − per-round slope > 0, by one-sided per-match arm swap.

The count is two metrics × three models × two variants (`all5`, `treated4`) ×
two hypotheses. No Holm correction or formal significance claims are approved
for this set. Both variants remain visible as sensitivity analyses. They use
different periods, so are not exact duplicate tests; neither is selected because
its result is more favourable. Four or five periods provide limited trend
evidence. Level differences alone do not establish drift or its mitigation.

MD2, MD3, and R32 period rows repeat the corresponding individual-round
hypotheses. MD1 also repeats in period, round, and first cumulative displays.
These repetitions add no core tests. Keep other period/round and cumulative
views descriptive. For nonlinear calibration statistics, the cumulative
104-match endpoint is not numerically equivalent to the treated 80-match
analysis. MD1 stays outside treatment families but remains in the explicitly
labelled `all5` descriptive trend variant.

## Reproduction status

The local regeneration verified the original estimates, raw p-values, and
bootstrap intervals against the previous tables. The separate fresh-clone
check is pending for the final versioned revision. Use the
[targeted DVC workflow](../../README.md#reproduce-the-thesis-analysis) and
the pinned reconstruction inputs.
