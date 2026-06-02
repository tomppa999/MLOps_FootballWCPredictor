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

## Independent Poisson is fine for RPS — but not for tiebreakers

Bivariate extension (Maher 1982, Dixon & Coles 1997) matters for exact scorelines
but not for outcome probabilities. The common W term cancels in X−Y, so P(home win),
P(draw), P(away win) are identical for independent and bivariate Poisson.
Ley et al. confirm empirically: RPS 0.1651 (bivariate) vs 0.1653 (independent).

For group tiebreakers that depend on goals scored (not just goal difference),
this approximation could matter slightly. Decision: keep independent Poisson,
acknowledge in limitations.

## Entropy normalisation trap (critical for D.1 / RQ2)

Advancement probabilities sum to 32 (one per R16 slot), not 1.
Normalise before computing Shannon entropy: p_i = P_advance_i / 32.
Otherwise H is inflated by log(32) ≈ 3.47 nats across all snapshots.

Mean-rate Poisson baseline provides the flat entropy floor: no information
enters the model, so entropy never resolves. Use this as the lower bound
in entropy resolution plots.

## The thesis experiment does not require beating Ley et al.

The experiment tests whether per-round retraining moves the needle versus a
frozen model. The null hypothesis is: frozen RPS ≈ per-round RPS across all
phases. A model that is 0.02 worse than Ley et al. in absolute terms can still
produce a clean cadence effect if the effect size is consistent. Absolute RPS
only matters for the model selection step (Chapter 5.1); the cadence comparison
is entirely within the same model family.

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
pre-tournament model at the start of WC. They diverge only if per-round
refit produces a new `champion_per_round` artifact after MD1. This design
ensures any observed difference is attributable solely to cadence, not to
different initial conditions.

## RQ3 (concept drift) is the hardest to answer definitively

Drift is measured via calibration degradation over time, not via a formal
drift test (no automated statistical drift detection in the pipeline — see
`threats_to_validity.md`). The argument is: if frozen calibration degrades
as the tournament progresses and per-round calibration does not, this is
consistent with concept drift mitigation. Cannot claim causal identification
without a third arm.

## WC 2022 retrospective (D.3) is descriptive, not causal

Only frozen mode is available for 2022 (no live per-round refitting occurred).
The replay shows entropy resolution curves under frozen conditions for a prior
WC, providing descriptive context. Not a controlled comparison — different teams,
format, match quality, external conditions. Acknowledge explicitly.
