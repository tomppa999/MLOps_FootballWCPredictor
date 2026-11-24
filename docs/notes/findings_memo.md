# Findings memo

Interpretation of the regenerated World Cup 2026 results for thesis Chapters 5
and 6. Use the DVC-produced CSVs linked below for numerical citations.

RQ1 and RQ3 now use **90-minute outcomes and goal counts**, including stoppage
time and excluding extra time and shootouts. Five knockout matches change from
decisive outcomes to draws: 29 draws across 104 matches, rather than 24.
The joint Bivariate Poisson replay uses the historical fitted parameters for
match probabilities and tournament simulation. The other models retain their
archived match and logged tournament forecasts. Four reconstructed outage
points use publicly reported results in both arms of each model. The paired
RQ2 tests exclude these points. [Settlement-only comparison](regulation_time_comparison.md).

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

[Phase comparisons](../tables/regulation_time/rq1_paired_phase.csv) ·
[Round and cumulative comparisons](../tables/regulation_time/rq1_paired_rounds.csv) ·
[Pairing coverage](../tables/regulation_time/rq1_pairing_coverage.csv)

The canonical phase cells cover **80 treated matches**, comprising 48 group
matches (MD2–MD3) and 32 knockout matches. XGBoost, Poisson GLM, and Bayesian
Poisson have a cadence contrast. The other five tracked models have tied
outputs across cadences. MD1 enters the round and cumulative views as a
construction-enforced tie. The analysis assigns the frozen forecast to both
MD1 arms to represent the intended policy after a premature live refit. The
raw live artifacts and provenance retain the original incident.

### Overall treated matches

- **XGBoost RPS.** 0.14629 frozen to 0.14399 per-round, an observed decrease
  of 0.00230 (1.6%). The pointwise CI is [−0.00624, +0.00163], raw p = 0.2550,
  and Holm p = 1.
- **Poisson GLM RPS.** d = −0.00261, CI [−0.00613, +0.00101], raw p = 0.1489,
  Holm p = 0.8933.
- **Bayesian Poisson RPS.** d = −0.00162, CI [−0.00451, +0.00134], raw
  p = 0.2791, Holm p = 1.
- **NLL.** d = −0.00171 for XGBoost, +0.01670 for Poisson GLM, and −0.00157
  for Bayesian Poisson. All overall intervals include zero. All nine adjusted
  NLL p-values equal 1.

**Interpretation.** Mean overall RPS was modestly lower under refitting for all
three models, but all overall intervals include zero. Two group-stage RPS
comparisons cross the raw 0.05 threshold: Poisson GLM (p = 0.0113, Holm
p = 0.1017) and Bayesian Poisson (p = 0.0172, Holm p = 0.1376). None of the
nine RPS or nine NLL comparisons survives Holm. The earlier nominal overall
XGBoost result disappears under 90-minute scoring. Failure to reject does not
establish equivalence.

### Phase, round, and other score patterns

Group-stage mean RPS differences were −0.00412, −0.00639, and −0.00452 for
XGBoost, Poisson GLM, and Bayesian Poisson. Knockout means in the same order
were +0.00044, +0.00305, and +0.00273. All three models therefore show a
descriptive difference in pattern by phase. The approved tests compare
cadences **within** each phase. A direct group-minus-knockout contrast was outside the agreed scope,
so the phase-variation part of RQ1 is only answered descriptively.

MD2 favoured refitting for all three models by observed mean RPS. R16
(n = 8) showed the opposite pattern. The small later round samples and
overlapping cumulative views warrant descriptive reporting. The 104-match
cumulative endpoint includes 24 MD1 ties. The exploratory mean absolute
goal-error summaries (labelled RMSE in the monitoring schema) are higher under refitting overall. Poisson GLM has
d = +0.01525, CI [+0.000014, +0.03097], but raw p = 0.0625; this is outside
Holm and illustrates that the bootstrap interval and sign-flip p-value need
not agree at the nominal boundary. The other two overall intervals include zero.

## RQ2 — Tournament uncertainty and forecast concentration

[Target comparisons](../tables/rq2_paired_stage.csv) ·
[Matchday comparisons](../tables/rq2_paired_matchday.csv) ·
[Pairing coverage](../tables/rq2_pairing_coverage.csv) ·
[Target overview](../figures/entropy-stages.png) ·
[R32 standalone trajectory](../figures/entropy-targets/entropy-r32.png) ·
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
- **Bivariate Poisson GLM.** 2.19768 to 2.16719, d = +0.03049 nats,
  CI [+0.02569, +0.03524], raw p ≈ 0.00010, Holm p ≈ 0.00180.

**Interpretation.** Seventeen of the 18 core comparisons show lower per-round
entropy after Holm adjustment. These cover all six targets for both fitted Poisson
models and R16 through winner for XGBoost. For Bivariate Poisson, the
frozen-minus-per-round difference increases from +0.00058 nats for R32 to
+0.03049 for winner. All six adjusted p-values round to 0.00180. The same
88 treated pairs were retained by the corrected state-selection rule.
The archived independent projection gave a winner gap of +0.00016 and
a negative R32 gap. These effects measure forecast concentration.

### Targets versus calendar trajectories

The six target comparisons ask who is expected to reach each stage at
comparable tournament states. The calendar view shows how winner entropy
and its cadence gap evolve as results arrive. Both entropy figures use the
frozen series for both arms before the first valid MD1-to-MD2 refit. The
change applies to the plotted trajectories only. In the descriptive cycle view,
the average winner-entropy gap peaks at R32 for all three fitted models.
Later state cells are small and the gap varies in sign and size. These
patterns describe the observed trajectories. The figures stage also produces a
standalone plot for each target in `docs/figures/entropy-targets/`. Individual
plots use the visible target range, clearly labelled when the y-axis is zoomed.
All roster curves, the baseline and reconstructed markers remain visible; the
common-scale overview is retained. The approved tests compare
levels, so a faster resolution rate remains untested. A time-to-threshold
analysis would need a defined time axis and remaining-uncertainty threshold
and is outside the current scope.

The all-cycle and unresolved-only windows change the sample or its weights.
The earlier D.1 cycle-weighted results are descriptive context. The mean-rate
Poisson trajectory is a non-refitting simulation diagnostic and changes as
results settle. It is outside the correction family.

### Entropy at the first valid refit

On 18 June the corrected plotted changes were R32 +0.02%, R16 −0.13%,
QF −0.26%, SF −0.09%, Final +0.35%, and Winner −0.09%. Winner entropy
moved from 2.5595 to 2.5572 nats. Using the preceding live per-round snapshot
as the reference gives +0.23% for Winner. Both calculations use the same
post-refit forecast. [Boundary estimates for all six targets](../tables/rq2_refit_boundary.csv).

The earlier independent-marginal simulation showed a rise on all six targets
at this boundary. The fitted common goal component increased after MD1.
Independent draws at the marginal means inflated the goal-difference
variance by twice this component. The joint replay uses the two team-specific
rates for the difference and one shared draw per simulated scoreline.
The 24 new matches had about 1.5% of total training weight and 38.5% within
World Cup matches. These weights and the refit checks describe the historical
fit. A causal attribution to weighting or regularisation requires a separate
comparison. [Historical refit diagnostic](poisson_refit_diagnostic.md).

## RQ3 — Calibration and evidence about drift

[Phase comparisons](../tables/regulation_time/rq3_calibration_phase.csv) ·
[Round and cumulative comparisons](../tables/regulation_time/rq3_calibration_rounds.csv) ·
[Trend comparisons](../tables/regulation_time/rq3_trend.csv) ·
[Calibration figure for the three contrast models](../figures/regulation-time/calibration-contrast-group-vs-ko.png)

ECE and the Brier reliability component are pooled in the 18-test core
family. Brier score and resolution remain exploratory. Brier uncertainty
depends on the observed outcomes and has a structural zero difference.
Calibration set statistics use a paired match bootstrap and a per-match
forecast arm swap. The latter requires forecast-label exchangeability under
the null, a stronger condition than equality of aggregate ECE.

### Calibration levels

- Overall ECE differences are −0.00047 for XGBoost, +0.00878 for Poisson GLM,
  and −0.00461 for Bayesian Poisson. All nine ECE and nine reliability
  comparisons have raw p > 0.05, Holm p = 1, and CIs including zero.
- **XGBoost knockout ECE** is 0.13836 frozen and 0.13105 per-round
  (d = −0.00731, n = 32). Raw p = 0.6010, Holm p = 1, and the pointwise CI
  is [−0.04091, +0.03935]. The earlier significant result and its p-value–CI
  disagreement belong to the superseded extra-time scoring version.

No core comparison provides statistically significant evidence of improved
calibration. This does not establish equal calibration or absence of drift.

### Trends and the limits of identifying drift

The exploratory trend fits use four or five ordered periods from the 80
treated or 104 total matches. The frozen-trend test permutes match period
labels while preserving counts. That requires exchangeability across periods,
which changes in teams, outcomes, and forecast difficulty may undermine.
The slope-difference test swaps arms within matches. Its exchangeability
assumption likewise needs justification. The simulated ECE noise floor is a
reference under outcomes drawn from the forecasts, with no role as a CI or
proof that a late increase is an artefact.

The treated-only frozen ECE slopes are positive for all three models, with
raw p = 0.491–0.584. The ECE rise and mitigation tests all have p > 0.05.
For Bivariate Poisson the ECE slope-difference p-values are 0.0953 for four
periods and 0.1475 for five. The corresponding supporting reliability tests
give p = 0.0186 and 0.0519. These trend tests are exploratory.
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

The thesis results use 90-minute RQ1/RQ3 scoring and the corrected joint
Poisson RQ2 analysis. The current calibration figures are under
`docs/figures/regulation-time/`. The proposed pre-match Elo-gap
follow-up for RQ1 was declined.
Draw versus decisive outcomes after regulation time is an optional
descriptive breakdown. The cost comparison was dropped because the needed
attribution data are unavailable. A paper decision remains open, with
auditable retraining experiments and the relationship between forecast
concentration and predictive performance as candidate angles.

An independent checkout restored the six input roots into an empty DVC
cache and reproduced the same SHA-256 hashes for all 30 verified outputs,
including the ten figures. Repeating the target skipped every stage.
[Reproduction instructions](../../README.md#reproduce-the-thesis-analysis).
