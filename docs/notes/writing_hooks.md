# Writing Hooks

Arguments, framings, and specific points to use when writing the thesis.
These are things that don't belong in the code or plan but need to survive
to writing time (August 2026+). Add anything you'd struggle to reconstruct
from the codebase alone.

---

## Why this problem is hard: WC 2026 format complexity

The 2026 WC uses 12 groups with the best 8 of 12 third-placers advancing —
C(12,8) = 495 possible bracket configurations. A single goal difference in any
group can restructure the entire knockout tree. This makes uncertainty
quantification non-trivial and is a concrete motivation for entropy-based
uncertainty tracking (RQ2). EURO evidence since introducing the best-third
format shows a meaningful increase in draws on matchday 3, consistent with
teams gaming the format once qualification mathematics are known.

## Goal dependence affects match and tournament forecasts

The fitted Bivariate Poisson shares one Poisson draw between the two scores.
That draw cancels in their difference. The correct Skellam rates are the
two team-specific rates. Sampling independently at the larger marginal means
inflates goal-difference variance by twice the shared rate. This changes
win/draw/loss probabilities, RPS and the distribution of tournament outcomes.

## Entropy normalisation trap (critical for D.1 / RQ2)

Normalise each advancement probability vector by its own sum, which equals
the number of slots in the target field. The R32 denominator is 32 and
winner probabilities sum to 1. Entropy measures concentration across teams.
A locked field of k teams has entropy ln(k). A settled winner has entropy
zero. Mean-rate Poisson is a non-refitting diagnostic. Its tournament
trajectory still changes as match results settle.

## The thesis experiment does not require beating Ley et al.

The experiment compares paired forecasts from frozen and per-round policies
within each selected model. Absolute scores help explain model selection
(Chapter 5.1). The cadence comparison uses score differences on the same
matches. The [findings memo](findings_memo.md#rq1--predictive-accuracy)
records the scope and interpretation of the individual comparisons.

## The correct peer RPS benchmark is Groll et al., not Ley et al.

Ley et al. evaluate on ALL non-friendly national team matches (2008–2017),
including qualification matches where favorites consistently dominate.
Groll et al. evaluate on WC matches only (WC 2018, 64 matches) and get
RPS 0.190–0.194 — a far more appropriate ceiling for WC-only evaluation.
Use Ley et al. as the goal-model literature reference; use Groll et al.
as the WC-performance ceiling reference.

## Home advantage override is inference-time, not a Gold feature

USA, Canada, Mexico play in their home country for group stage.
`is_neutral` is overridden to False for these matches at inference time only.
This is documented in `feature_spec.md`. No Gold schema change needed.
For KO rounds: venue assigned from `data/tournament/wc2026.json`.

## Frozen vs per-round: both modes start from the identical pre-WC champion

Both `champion_frozen` and `champion_per_round` aliases point to the same
pre-tournament model at the start of WC. The intended design changes the
model at the gated per-round refit, holding the starting model fixed.
The [findings memo](findings_memo.md#rq1--predictive-accuracy) explains the
MD1 convention following the premature refit and the analysis scope.

## RQ3 (concept drift) is the hardest to answer definitively

Calibration levels and trends are exploratory evidence about drift.
Their interpretation depends on fixture composition, sample size, and
resampling assumptions. In these data concept drift and its mitigation
remain unresolved. See the [findings memo](findings_memo.md#rq3--calibration-and-evidence-about-drift)
and [validity discussion](../threats_to_validity.md#evaluation-threats).

## WC 2022 retrospective (D.3) is descriptive, not causal

Only frozen mode is available for 2022 (no live per-round refitting occurred).
The replay shows entropy resolution curves under frozen conditions for a prior
WC, providing descriptive context. Not a controlled comparison — different teams,
format, match quality, external conditions. Acknowledge explicitly.
