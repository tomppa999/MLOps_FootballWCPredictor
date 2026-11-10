# Findings memo

Interpretation of the regenerated World Cup 2026 results for thesis Chapters 5
and 6. Use the DVC-produced CSVs linked below for numerical citations.

## Reading the results

- The comparison is frozen versus per-round refitting for the three selected
  models. It covers one tournament and two policies. Both model selection under
  frozen conditions and shared match outcomes limit the scope of inference.
- Holm is applied separately to four core families: 9 RPS, 9 NLL, 18 entropy,
  and 18 calibration comparisons. The families were revised after inspection
  of results. See the [decision and rationale](multiple_comparisons.md).
  A blank `p_holm` means the row is outside those families. Round, cumulative,
  alternative-window, and trend results are exploratory or descriptive.
- Raw p-values and pointwise 95% bootstrap intervals depend on the resampling
  assumptions. The RQ1 and RQ2 bootstrap samples individual paired observations
  with replacement. The sign-flip test requires the joint null distribution
  to be invariant to the flips. Shared teams, model fits, and tournament
  history make independence and exchangeability uncertain. Holm adjusts
  across tests with valid raw p-values. It leaves the intervals pointwise and
  cannot correct dependence within the resampling units.
- RQ1 and RQ3 use `d = per_round − frozen`. Negative differences favour
  refitting for RPS, NLL, RMSE, ECE, reliability, and Brier score. RQ2 uses
  `d = H_frozen − H_per_round`, so positive differences indicate lower
  per-round entropy. RQ3 Brier resolution is higher-is-better and differs
  from the uncertainty-resolution concept in RQ2.

## RQ1 — Predictive accuracy

[Phase comparisons](../tables/rq1_paired_phase.csv) ·
[Round and cumulative comparisons](../tables/rq1_paired_rounds.csv) ·
[Pairing coverage](../tables/rq1_pairing_coverage.csv)

The canonical phase cells cover **80 treated matches**, comprising 48 group
matches (MD2–MD3) and 32 knockout matches. XGBoost, Poisson GLM, and Bayesian
Poisson have a cadence contrast. The other five tracked models have tied
outputs across cadences. MD1 enters the round and cumulative views as a
construction-enforced tie. The analysis assigns the frozen forecast to both
MD1 arms to represent the intended policy after a premature live refit. The
raw live artifacts and provenance retain the original incident.

### Overall treated matches

- **XGBoost RPS.** 0.14147 frozen to 0.13712 per-round, an observed decrease
  of 0.00436 (3.1%). The pointwise CI is [−0.00830, −0.00033], raw p = 0.0340,
  and Holm p = 0.2720.
- **Poisson GLM RPS.** d = −0.00291, CI [−0.00654, +0.00070], raw p = 0.1207,
  Holm p = 0.5714.
- **Bayesian Poisson RPS.** d = −0.00260, CI [−0.00550, +0.00033], raw
  p = 0.0841, Holm p = 0.5045.
- **NLL.** d = −0.00632 for XGBoost, +0.01166 for Poisson GLM, and −0.00632
  for Bayesian Poisson. All overall intervals include zero. The signs differ
  across models and all nine adjusted NLL p-values equal 1.

**Interpretation.** Mean RPS was modestly lower under refitting on the observed
treated matches for all three models. Three RPS rows crossed the nominal
0.05 threshold and none crossed it after Holm (minimum adjusted p = 0.1548).
No NLL row crossed either threshold. The nominal XGBoost RPS CI and its
adjusted p-value use different error criteria. Report both and avoid an
equivalence claim from a failure to reject.

### Phase, round, and other score patterns

Group-stage mean RPS differences were −0.00412, −0.00523, and −0.00452 for
XGBoost, Poisson GLM, and Bayesian Poisson. Knockout means in the same order
were −0.00470, +0.00057, and +0.00027. The GLMs show a descriptive difference
in pattern by phase. The approved tests compare cadences **within** each
phase. A direct group-minus-knockout contrast was outside the agreed scope,
so the phase-variation part of RQ1 is only answered descriptively.

MD2 favoured refitting for all three models by observed mean RPS. R16
(n = 8) showed the opposite pattern. The small later round samples and
overlapping cumulative views warrant descriptive reporting. The 104-match
cumulative endpoint includes 24 MD1 ties. The exploratory RMSE estimates
were mixed and the overall intervals included zero.

## RQ2 — Tournament uncertainty and forecast concentration

[Target comparisons](../tables/rq2_paired_stage.csv) ·
[Matchday comparisons](../tables/rq2_paired_matchday.csv) ·
[Pairing coverage](../tables/rq2_pairing_coverage.csv) ·
[Target trajectories](../figures/entropy-stages.png) ·
[Winner trajectory](../figures/entropy-trajectory.png)

Shannon entropy measures how spread out the forecast mass is across teams.
For winner probabilities it directly measures the concentration of the
winner distribution. For a multi-team advancement target, the probabilities
are normalised by their sum before computing entropy. That measures
concentration across teams. The joint uncertainty over sets of qualifiers
would require a distribution over those sets. A locked field of k teams has
entropy ln(k) and a zero cadence gap, with ln(1) = 0 for the settled winner.

The headline compares all six targets across the **88 treated states per
model**. Pairing uses the preceding frozen cycle within 30 minutes. Timestamp
proximity alone cannot verify that every pair had exactly the same settled
information. Excluding early placebo labels avoids mixing in a premature
refit. The core sample excludes the synthetic outage snapshots. Deduplication
prevents runs of unchanged snapshots from receiving extra weight. Successive
changed states still share earlier results and fitted models.

### Treated winner entropy

- **XGBoost.** 2.30172 frozen to 2.22585 per-round, d = +0.07587 nats,
  CI [+0.06438, +0.08766], raw p ≈ 0.00010, Holm p ≈ 0.00180.
- **Bayesian Poisson.** 2.19676 to 2.15748, d = +0.03928 nats,
  CI [+0.03518, +0.04337], raw p ≈ 0.00010, Holm p ≈ 0.00180.
- **Poisson GLM.** d = +0.00016 nats, CI [−0.01011, +0.01014],
  raw p = 0.9777, Holm p = 1.

**Interpretation.** Twelve of 18 comparisons cross 0.05 after Holm under
the existing test assumptions. Eleven show lower per-round entropy
(XGBoost from R16 through winner and Bayesian Poisson on all six targets).
The remaining comparison shows higher per-round entropy for Poisson GLM
at R32 (d = −0.00320 nats). Concentration changes depend on the model and
target. Forecast concentration can change independently of correctness.

### Targets versus calendar trajectories

The six target comparisons ask who is expected to reach each stage at
comparable tournament states. The calendar view shows how winner entropy
and its cadence gap evolve as results arrive. In the descriptive cycle view,
the average winner-entropy gap peaks at R32 for all three fitted models.
Later state cells are small and the gap varies in sign and size. These
patterns describe the observed trajectories. The approved tests compare
levels, so a faster resolution rate remains untested. A time-to-threshold
analysis would need a defined time axis and remaining-uncertainty threshold
and is outside the current scope.

The all-cycle and unresolved-only windows change the sample or its weights.
The earlier D.1 cycle-weighted results are descriptive context. The mean-rate
Poisson trajectory is a non-refitting simulation diagnostic and changes as
results settle. It is outside the correction family.

## RQ3 — Calibration and evidence about drift

[Phase comparisons](../tables/rq3_calibration_phase.csv) ·
[Round and cumulative comparisons](../tables/rq3_calibration_rounds.csv) ·
[Trend comparisons](../tables/rq3_trend.csv) ·
[Calibration figure for the three contrast models](../figures/calibration-contrast-group-vs-ko.png)

ECE and the Brier reliability component are pooled in the 18-test core
family. Brier score and resolution remain exploratory. Brier uncertainty
depends on the observed outcomes and has a structural zero difference.
Calibration set statistics use a paired match bootstrap and a per-match
forecast arm swap. The latter requires forecast-label exchangeability under
the null, a stronger condition than equality of aggregate ECE.

### Levels and the knockout ECE result

- Overall ECE differences were −0.01437 for XGBoost, +0.01129 for Poisson
  GLM, and −0.00357 for Bayesian Poisson. All six overall ECE and reliability
  intervals include zero and all six Holm p-values are 1. Observed overall
  Brier scores were slightly lower in the refitted arm and their pointwise
  intervals include zero.
- **XGBoost knockout ECE** is the sole corrected threshold crossing.
  ECE was 0.16599 frozen and 0.13913 per-round (d = −0.02686, n = 32).
  Raw p ≈ 0.00010 and Holm p ≈ 0.00180, while the pointwise CI is
  **[−0.04415, +0.02063]**. The disagreement is present before Holm.
  The percentile bootstrap estimates variation under resampling of matches.
  The arm-swap test compares the result with a null distribution under
  exchangeable forecast labels. ECE also depends on binning.
  Report both results and treat the isolated contrast as uncertain.

### Trends and the limits of identifying drift

The exploratory trend fits use four or five ordered periods from the 80
treated or 104 total matches. The frozen-trend test permutes match period
labels while preserving counts. That requires exchangeability across periods,
which changes in teams, outcomes, and forecast difficulty may undermine.
The slope-difference test swaps arms within matches. Its exchangeability
assumption likewise needs justification. The simulated ECE noise floor is a
reference under outcomes drawn from the forecasts, with no role as a CI or
proof that a late increase is an artefact.

Frozen ECE falls and then rises for XGBoost. The fitted treated-only frozen
ECE slopes are positive in all three models, with unadjusted p-values above
0.05. Poisson GLM has some nominal slope-difference results, while the XGBoost
treated-only ECE slope difference is negative. The views remain descriptive.
With a short tournament, four or five periods, and small late-stage samples,
calibration fluctuations are difficult to distinguish from changing match
composition and sampling noise. **Concept drift and its mitigation remain
unresolved.**

## Provenance and follow-up scope

The live XGBoost match contrast and the reconstructed frozen GLM forecasts
have different provenance. RQ2 also uses audited reconstruction for frozen
trajectories. Keep the audit evidence and the MD1 handling in Methods. The
Strand 5 identity check reproduces frozen champion goal rates on all 423
cycles to a maximum difference of 4.441e−16. See the
[reconstruction record](wc_live.md#done--strand-5-cadence-correct-frozen-entropy-supersedes-strand-4)
and the [validity discussion](../threats_to_validity.md#evaluation-threats).

The thesis results use the existing analyses. The proposed pre-match Elo-gap
follow-up for RQ1 was declined.
Draw versus decisive outcomes after regulation time is an optional
descriptive breakdown. The cost comparison was dropped because the needed
attribution data are unavailable. A paper decision remains open, with
auditable retraining experiments and the relationship between forecast
concentration and predictive performance as candidate angles.

The local checks of the regenerated Holm tables and figures precede a
separate fresh-clone check of the final Git and DVC revision. See the
[reproduction instructions](../../README.md#reproduce-the-thesis-analysis).
