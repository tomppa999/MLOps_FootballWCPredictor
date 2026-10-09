# Findings memo

Bullet claims with numbers and caveats, one section per RQ. Every number here
is read from a `dvc.yaml` stage output; the paper and thesis cite those CSVs,
not this file. Regenerate with `dvc repro figures`.

Conventions: `d = metric_per_round − metric_frozen` per (match, model). All
metrics are lower-is-better, so **negative d = per_round better**. CI = 95 %
percentile bootstrap on the mean of d (B = 10 000, seed 0). p = paired
sign-flip permutation, two-sided. `frac_improved` = share of matches with d < 0.

---

## RQ1 — Accuracy: per_round vs frozen

Sources: `docs/tables/rq1_paired_phase.csv`, `docs/tables/rq1_paired_rounds.csv`,
`docs/tables/rq1_pairing_coverage.csv` (stage `rq1`, `src/analysis/rq1_paired.py`).

### Design facts that constrain every claim

- Only **3 of 8 models** have a real cadence contrast: `xgboost` (live both
  arms), `poisson_glm` and `bayesian_poisson` (frozen arm reconstructed by
  D.1 Strand 1). The other five are byte-identical across arms on all 104
  matches. They never appear in a paired row.
- **MD1 is a placebo round by design and by construction.** The first
  correctly gated per_round refit ran after MD1, so both arms should have
  served the same model. The live per_round artifact did not: the premature
  Jun 12 refit (v16, `wc_live.md` MD1, `chronology.md`) left a residue on
  22/24 MD1 matches for all three contrast models (|mean d_RPS| ≤ 0.0005,
  max |d| = 0.011). Following the documented convention ("use
  `champion_frozen` predictions for both cadence modes"), `rq_datasets` now
  sets **per_round MD1 := frozen twin** at source for all 8 models
  (provenance `md1_frozen_copied`, 192 rows, 66 of which carried different
  values). `rq1_paired` asserts the MD1 tie and fails otherwise.
- The treatment is therefore active on **80 matches** (MD2–Final). MD1 is
  excluded from every phase cell (group = MD2+MD3, n = 48) and enters the
  round table only as an exact tie, so cumulative rows keep the 104-match
  denominator (23 % of pairs tied at the Final).
- `n_diff = 80/104` for each contrast model in the coverage table; 0 for the
  other five.
- ~30 tests are reported (3 models × 3 metrics × several cells). No
  correction applied; read the table for consistency of direction, not for
  any single p.

### Headline (RPS, treated matches, `phase = all`, n = 80)

- xgboost −0.0044 [−0.0083, −0.0003], p = 0.034, improved 56 %
  (frozen 0.1415 → per_round 0.1371, −3.1 % relative).
- poisson_glm −0.0029 [−0.0065, +0.0007], p = 0.12, improved 56 %.
- bayesian_poisson −0.0026 [−0.0055, +0.0003], p = 0.084, improved 61 %.
- Claim: per-round retraining lowers RPS by ~0.003–0.004 for every model
  that was actually refit; direction is unanimous, the CI excludes zero only
  for the champion. Relative size is 2–3 % of RPS.
- Cumulative over all 104 (MD1 ties included, `rounds / Final / cumulative`):
  xgboost −0.0034 [−0.0064, −0.0003] p = 0.039; poisson_glm −0.0022
  [−0.0051, +0.0006]; bayesian_poisson −0.0020 [−0.0042, +0.0002]. Same
  picture, diluted by 24 zeros.

### Phase decomposition (RPS)

- **Group = MD2–3 (n = 48):** xgboost −0.0041 [−0.0091, +0.0008] p = 0.11;
  poisson_glm −0.0052 [−0.0103, −0.0003] p = 0.049;
  bayesian_poisson −0.0045 [−0.0079, −0.0009] p = 0.017 (improved 69 %).
- **KO (n = 32):** xgboost −0.0047 [−0.0112, +0.0018] p = 0.17 (improved
  59 %); poisson_glm +0.0006 [−0.0044, +0.0055]; bayesian_poisson
  +0.0003 [−0.0046, +0.0050]. The GLMs are a wash in the knockout stage.
- **R32 (n = 16) vs R16+ (n = 16):** R32 negative for all three
  (xgboost −0.0058, GLMs −0.0036 / −0.0035; CIs all straddle 0). R16+
  flips positive for the GLMs (+0.0047, +0.0040; CIs straddle 0) and stays
  negative for xgboost (−0.0036 [−0.0132, +0.0063]).
- Claim: **the cadence effect is a group-stage effect.** It is largest and
  most consistent on MD2–MD3, where the refit ingests same-tournament group
  matches, and fades (xgboost) or reverses (GLMs) once the slate is small
  and the matchups are tight. Answer to the RQ1 sub-question "does it vary by
  phase": yes, in the direction of group > knockout.

### Round-level (RPS, `grain = round`)

- **MD2 is the single decisive round** and the only round where all three
  models have CIs excluding zero: xgboost −0.0069 [−0.0133, −0.0007]
  p = 0.049; poisson_glm −0.0087 [−0.0162, −0.0016] p = 0.030;
  bayesian_poisson −0.0069 [−0.0105, −0.0031] p = 0.002 (improved 79 %).
- MD3 negative for all three but small (−0.0013 to −0.0022), CIs wide.
- **R16 (n = 8) is the one round where per_round is worse for every model:**
  xgboost +0.0063 [−0.0028, +0.0153]; poisson_glm +0.0123 [+0.0035, +0.0206]
  p = 0.054; bayesian_poisson +0.0117 [+0.0031, +0.0199] p = 0.058. With
  n = 8 this is a flag, not a finding; `wc_live.md` R16 attributes the round
  to compressed favourite strength, not upsets.
- QF (n = 4): xgboost −0.0189, 4/4 improved, but p = 0.13 (the permutation
  test cannot go below 2/16 with n = 4). SF and Final have n = 2; the
  CIs are meaningless and p is 1.0 by construction. Report n, do not
  interpret.
- Cumulative RPS (MD1 ties included) for xgboost is negative with CI
  excluding 0 from R32 onward (R32 −0.0033 [−0.0064, −0.0002]) and stays
  there through the Final; it dips back to include 0 only at R16
  (−0.0025 [−0.0056, +0.0005]). For the GLMs the cumulative CI excludes 0
  through R32 and includes 0 from R16 on. The cumulative curve is the
  paper's "did it pay off across the tournament" figure.

### NLL and RMSE

- **NLL: no effect.** Treated n = 80: xgboost −0.0063 [−0.0303, +0.0177];
  poisson_glm +0.0117 [−0.0143, +0.0381]; bayesian_poisson −0.0063
  [−0.0225, +0.0093]. No cell in any scheme has a CI excluding 0. Signs
  differ between models.
- **RMSE: no effect, if anything slightly worse for the GLMs.**
  poisson_glm n = 80 +0.0121 [−0.0033, +0.0280], group +0.0162; xgboost
  −0.0020 [−0.0179, +0.0148].
- Claim: retraining improved the **outcome** probabilities (RPS on W/D/L)
  without improving the **goal-rate** fit (NLL, RMSE on λ). Consistent with
  the live observation (`wc_live.md` MD2–R32) that RPS is where the refit
  reliably wins while NLL/RMSE trade blows within noise. For the paper this
  is the sharper statement than "per_round is better": the refit moved the
  calibration of the three-way outcome, not the expected goals.

### Caveats to carry into the writing

- Two of the three contrasts are half-reconstructed (frozen arm offline,
  per_round arm live). The MD1 residue that the source fix removed
  (premature-refit artifact vs true frozen) was ~0.0005 RPS in the mean,
  an order of magnitude below the MD2 effect but the same order as the MD3
  effect; treat MD3-sized differences with that in mind.
- The MD1 fix is a documented analysis convention, not a re-serving: the raw
  live per_round artifact (`data/analysis/live/monitoring_per_round.csv`)
  still holds the v16 residue and `source_run_id` points back to it.
- The frozen arms of the two GLMs are the same reconstruction, so their
  near-identical numbers are partly shared inputs, not independent
  replication. Treat "3 models agree" as 2 independent lines of evidence
  (xgboost; the GLM pair).
- n = 80 for a ~0.003 effect gives power well below 0.8 at α = 0.05; the
  CIs that straddle 0 for the GLMs are the expected consequence, not
  evidence of no effect.
- No third arm (e.g. refit-every-2-rounds), so the dose-response is
  unobserved and cadence *frequency* claims stay qualitative.

---

## RQ2 — Uncertainty resolution: frozen vs per_round

Sources: `docs/tables/rq2_paired_stage.csv`, `docs/tables/rq2_paired_matchday.csv`,
`docs/tables/rq2_pairing_coverage.csv` (stage `rq2`, `src/analysis/rq2_paired.py`);
figure `docs/figures/entropy-stages.png`. Sign convention for this section only:
`d = H_frozen − H_per_round` (positive = per_round has resolved more uncertainty).
Internally `summarise()` still sees `−d`, so `frac_improved` = share with d > 0.
CI / p as in the header. Memo cites **state-grain** CI/p; cycle-grain means are
kept for comparability with the `thesis.md` D.1 table.

### Design facts that constrain every claim

- Twin = the frozen cycle immediately preceding each per_round cycle within
  30 min (`merge_asof` backward). 426 candidate pairs, of which 4 are synthetic
  Strand-4/5 snapshots (dropped) → **422 live pairs**. One frozen cycle (SF,
  Jul 12 10:06) has no twin. Offsets 7.5–15.0 min (median 8.7).
- `matchday_label` here is the cycle's pipeline tag (roughly next/in-progress
  round), **not** the matchday of a scored match. Labels `1` and `2` (80 pairs:
  1 pre-opener + 79 cycles Jun 12–18 while MD1 was still being played) are a
  **placebo window**. The premature Jun 12 v16 refit left per_round serving an
  accidental early model until the first *correctly gated* refit on Jun 18
  (`wc_live.md` MD1, `chronology.md`). Including those cycles would mix
  "accidental early refit vs frozen" into the cadence contrast. This is
  **not** the RQ1 MD1 fix: RQ1 overwrote per_round *match* rows for round 1
  (`md1_frozen_copied`); RQ2 entropy cycles were never patched that way.
  Headline cells use the **treated** window (labels `3`…`Complete` = after
  Jun 18, n = 342 cycles / 88 states for the three fitted models).
- **Cycle grain** = every paired snapshot. **State grain** = drop runs where
  the 12-value frozen+per_round entropy vector did not change (quiet stretch:
  no new result, no refit, same seed). Of 422 cycles only **122 distinct
  states** remain for xgboost / bayesian_poisson / poisson_glm and **88** for
  mean_rate_poisson. Cycle-grain CIs are overconfident; cite state grain for
  inference, cycle means only for comparability with `thesis.md` D.1.
- Two cuts of the same data must not be conflated: **stage columns** =
  six different questions at the same cycles ("who reaches R32?" … "who
  wins?"); **matchday rows** = when on the calendar the *winner*-entropy gap
  is large. "Grows toward the winner" is the first cut; "peaks at R32" is
  the second.
- Stage columns lock once that field is decided (`treated_unresolved`
  shrinks accordingly): `entropy_r32`/`entropy_r16` from label R16, `qf`
  from QF, `sf` from SF, `final` from Final, `winner` from Complete. After
  lock the level is ln(n_teams_in_field) (e.g. R32 → ln 32 ≈ 3.47), not 0;
  only winner goes to 0. Code asserts |d| ≤ tol after lock.
- Roster is the four models with a Monte Carlo bracket: xgboost,
  bayesian_poisson, poisson_glm, mean_rate_poisson. Mean-rate is the
  intended **entropy floor** (never refit; both arms identical by design).
- The frozen arm is a Strand 5 reconstruction, not the live log
  (shadow-resolution contamination; faithfulness rests on
  `s5.champion_identity`, max |Δλ| = 4.441e-16).

### Headline (`entropy_winner`, treated, state grain)

- xgboost +0.0759 [+0.0644, +0.0877], p = 0.0001, improved 94 %
  (cycle mean +0.0645 on n = 342; frozen 1.90 → per_round 1.84).
- bayesian_poisson +0.0393 [+0.0352, +0.0434], p = 0.0001, improved 98 %
  (cycle mean +0.0300).
- poisson_glm +0.0002 [−0.0101, +0.0101], p = 0.98, improved 60 %
  (cycle mean +0.0099 looks positive only because quiet stretches overweight
  the R32/R16 gap; at state grain the GLM is a wash).
- mean_rate_poisson +0.0004 [−0.0001, +0.0010], p = 0.14 (cycle mean
  +0.0005) — floor as designed.
- Cycle-grain `all` (n = 422) reproduces `thesis.md` D.1 exactly: xgboost
  +0.0552, bayesian +0.0244, poisson_glm +0.0080, mean_rate +0.0004.
- Claim: **per_round resolves more residual winner-uncertainty than frozen
  for xgboost and bayesian_poisson; the GLM does not, once serial dependence
  is removed.** Magnitude is model-specific (xgboost ≈ 2× bayesian); any
  pooled cadence effect is unsupported. Relative to the absolute entropy
  level (~2.2 nats at the treated state mean) the xgboost gap is ~3 %.

### Stage decomposition (treated_unresolved, state grain)

- These are **six belief targets at the same cycles**, not "winner entropy
  as the Final approaches." `entropy_r32` = who reaches R32; …
  `entropy_winner` = who wins.
- For xgboost the gap **grows toward the winner column**: r32 ≈ 0, r16
  +0.0126, qf +0.0254, sf +0.0412, final +0.0543, winner +0.0767 (all from
  r16 on have CIs excluding 0; improved ≥ 95 %). Cadence barely moves who
  reaches R32; it rewrites deep-bracket / winner beliefs.
- bayesian_poisson same shape, smaller: r32 +0.0011, r16 +0.0102, …,
  winner +0.0397; every stage CI excludes 0.
- poisson_glm **flips sign on early stages**: r32 −0.0041, r16 −0.0025,
  sf −0.0078 (CIs exclude 0 or touch it), winner +0.0002. The GLM's
  cadence effect is not a monotone "per_round resolves more".
- mean_rate is flat at every stage (|mean d| ≤ 0.0005).
- Claim: for the two models that move, the cadence effect on uncertainty is
  a **deep-bracket belief** effect, not an R32-advancement effect.
  Complementary to RQ1 (accuracy was a group-stage / match-slate effect).
  Do not read this as "the gap gets bigger late in July" — that is the
  matchday cut below.

### Matchday trajectory (`entropy_winner`)

- Calendar cut of **winner entropy only** (what `entropy-trajectory.png`
  shows). **R32 is the peak** for every fitted model (cycle means): xgboost
  +0.136, bayesian +0.048, poisson_glm +0.045. State grain for xgboost R32:
  +0.1265 [+0.1134, +0.1395], p = 0.0001, improved 100 % (n = 36 states).
- Already visible in late group: MD3 positive for xgboost (+0.033 state)
  and bayesian (+0.038). Not "invisible until knockout." poisson_glm is
  **negative on MD3** (−0.055 [−0.063, −0.046], improved only 6 %) before
  flipping positive at R32.
- From R16 the gap **shrinks**. At **SF / Final xgboost changes sign**
  (−0.007 / −0.006): frozen winner-entropy slightly *lower* than per_round
  (dashed under solid on the figure). n = 3 / 1 states — report n, do not
  interpret. Bayesian stays non-negative through Final. Complete is exactly
  tied (winner decided, H = 0 for both arms).
- Placebo (labels 1–2, state): xgboost +0.016 [+0.010, +0.023] — the v16
  residue is real but an order of magnitude below the R32 peak; bayesian
  and the GLM are consistent with zero there.
- Claim: resolution advantage accumulates through the group stage and
  **peaks once the KO bracket opens (R32)**; it is not monotone through
  the knockout and is not "largest near the Final."

### Caveats to carry into the writing

- Frozen entropy is reconstructed (Strand 5); live frozen log was
  unusable. Cite `s5.champion_identity` when claiming faithfulness.
- Cycle grain (n = 422) matches the thesis table but inflates certainty;
  always prefer the state grain (n ≈ 88 treated) for CI/p claims.
- poisson_glm's cycle-vs-state disagreement is the dependence warning in
  action — do not cite its cycle-grain p.
- Placebo window is analysis policy on entropy *cycles* (exclude from
  headline), not a re-serving and not the RQ1 match-level MD1 overwrite;
  raw per_round entropy artifacts still hold the v16 residue for labels
  `1`/`2`.
- Many cells (QF n = 5, SF n = 3, Final n = 1 at state grain); report n,
  do not lean on those p-values — including the late XGB sign flip.
- No multiple-testing correction across stages × models × windows.

---

## RQ3 — Calibration drift: frozen vs per_round

Sources: `docs/tables/rq3_calibration_rounds.csv`,
`docs/tables/rq3_calibration_phase.csv`, `docs/tables/rq3_trend.csv`
(stage `rq3`, `src/analysis/rq3_calibration_drift.py`); figures
`docs/figures/calibration-group-vs-ko.png` (8 models, MD1 in group) and
`docs/figures/calibration-contrast-group-vs-ko.png` (3 contrast models,
group = MD2–3). Sign convention as in the header: `d = per_round − frozen`
(negative = per_round better for ECE, REL, BS). RES is higher-is-better;
UNC is outcome-only so d ≡ 0. ECE / REL / RES are set statistics (paired
match bootstrap CI + arm-swap p); BS reuses `summarise()` on per-match
scores. B = 10 000, seed 0.

### Design facts that constrain every claim

- Same contrast roster and MD1 placebo as RQ1: only xgboost / poisson_glm /
  bayesian_poisson; MD1 per_round := frozen at source; MD1 excluded from
  `group_ko` cells and enters the round table as an exact tie.
- **No third arm.** Cadence is binary (frozen vs per_round). The planned
  claim ceiling is therefore "consistent with concept-drift mitigation",
  never a causal identification (`writing_hooks.md`,
  `threats_to_validity.md` — no automated drift monitor in the pipeline).
- ECE at n = 16–24 sits near its own consistency noise floor
  (`ece_noise_floor_*` ≈ 0.15–0.19). Period-level ECE levels are noisy;
  interpret differences and slopes, not absolute ECE.
- Trend periods are the five blocks with n ≥ 16: MD1, MD2, MD3, R32, R16+
  (24/24/24/16/16). R16 / QF / SF / Final alone have n ≤ 8
  (`interpretable = false`); report n, do not interpret.
- Two independent lines of evidence (xgboost; the GLM pair shares the
  Strand-1 frozen reconstruction). Pre-set verdict rule: **supports** =
  frozen slope > 0 with p < 0.05 and slope_diff p < 0.05 in both lines;
  **consistent with** = frozen slope > per_round slope in both variants
  (`all5`, `treated4`) for both lines, but p ≥ 0.05; else **does not
  support**.

### Headline (treated matches, `phase = all`, n = 80)

- ECE: xgboost −0.0144 [−0.0305, +0.0175] p = 0.12; poisson_glm +0.0113
  [−0.0134, +0.0344]; bayesian_poisson −0.0036 [−0.0340, +0.0184]. All
  CIs include 0; signs disagree.
- REL: all three CIs include 0 (xgboost +0.0023; poisson_glm +0.0113;
  bayesian −0.0092).
- BS (per-match): xgboost −0.0090 [−0.0185, +0.0008] p = 0.074;
  poisson_glm −0.0072 [−0.0161, +0.0012]; bayesian −0.0058
  [−0.0122, +0.0009]. Direction unanimous (per_round slightly sharper)
  but no CI excludes 0. Relative size ~1–2 % of BS.
- Claim: on the full treated slate there is **no clear cadence effect on
  calibration**. The mild BS edge for per_round is the same order as the
  RQ1 RPS gain and is consistent with "outcome probabilities moved a
  little", not with a calibration rescue.

### Phase decomposition

- **Group = MD2–3 (n = 48):** ECE diffs straddle 0 for all three
  (xgboost +0.0019; poisson_glm +0.0145; bayesian −0.0149). BS favours
  per_round for bayesian (−0.0093 [−0.0170, −0.0009] p = 0.033) and
  sits on the edge for poisson_glm (−0.0120 [−0.0248, +0.0001]
  p = 0.062).
- **KO (n = 32):** ECE point estimates favour per_round (xgboost
  −0.0269, bayesian −0.0111, poisson_glm −0.0049). xgboost's arm-swap
  p = 0.0001 is extreme, but the bootstrap CI still includes 0
  [−0.0442, +0.0206] — set-stat CI and p disagree; prefer the CI.
  BS in KO is a wash for the GLMs (|d| ≤ 0.0004) and only weakly
  negative for xgboost (−0.0103, CI includes 0).
- Claim: unlike RQ1 (cadence effect = group-stage RPS), any calibration
  gap that appears is a **knockout ECE point-estimate** story with
  unstable inference, not a group-stage finding.

### Period trajectory (ECE / REL)

- Frozen ECE is a **U-shape**, not a monotone climb: MD1 high
  (xgboost 0.199; GLMs ~0.23) → falls through MD2–R32 (xgboost 0.153
  at R32) → rises at R16+ (xgboost 0.216; GLMs ~0.23). Per_round tracks
  the same shape. The late rise coincides with the noise floor rising
  to ~0.19 and with n = 16.
- Per-period cadence diffs: MD1 ≡ 0 (placebo). MD2–MD3 CIs all include
  0. R32 xgboost ECE −0.0164 with p = 0.004 but CI
  [−0.0416, +0.0360] includes 0 (same CI/p disagreement). R16+ negative
  for all three (xgboost −0.011; poisson_glm −0.022; bayesian −0.030)
  with CIs that still include 0.
- Cumulative ECE through Final (n = 104): xgboost −0.0052, poisson_glm
  +0.0081, bayesian −0.0083; every CI includes 0. No cumulative
  separation accumulates the way RQ1's cumulative RPS does for xgboost.

### Trend test ("does frozen calibration degrade?")

- Statistic: OLS slope of period-level ECE (and REL) on period index.
  `p_frozen` = one-sided period-label permutation (preserves period
  sizes, so small-n ECE bias stays under the null). `p_diff` = one-sided
  arm-swap for `slope_frozen − slope_per_round > 0`. Power: 4–5 points.
- **all5 (MD1–R16+, n = 104):** frozen ECE slopes ≈ 0 and n.s.
  (xgboost +0.0010 p = 0.75; poisson_glm +0.0028 p = 0.70; bayesian
  −0.0003 p = 0.78). slope_diff is positive for all three but only
  poisson_glm's p_diff = 0.014 clears 0.05 (REL p_diff = 0.005); the
  other two are n.s. bayesian's frozen ECE slope is negative.
- **treated4 (MD2–R16+, n = 80):** every frozen ECE slope is positive
  (xgboost +0.012 p = 0.41; poisson_glm +0.026 p = 0.10; bayesian
  +0.017 p = 0.29) — the U-shape's right arm — but **none** have
  p_frozen < 0.05. xgboost's slope_diff is **negative** (−0.0025):
  per_round's ECE rises at least as fast. poisson_glm keeps
  p_diff = 0.030 on ECE / 0.011 on REL.
- Claim: the data do **not** show that frozen calibration degrades over
  the tournament. The late ECE uptick is shared by both arms, is within
  the noise floor, and is the expected small-n artefact the period
  permutation was designed to absorb. poisson_glm's significant
  slope_diff cells are a single reconstructed line of evidence and do
  not meet the pre-set bar on their own.

### Caveats to carry into the writing

- Ceiling claim is "consistent with", and even that bar is not cleared
  (see Verdict). Do not write "per_round mitigated concept drift".
- ECE / REL at n ≤ 24 are dominated by binning noise; always cite
  `ece_noise_floor_*` next to a period-level ECE.
- Set-statistic CI and arm-swap p can disagree (KO / R32 xgboost ECE);
  when they do, do not hang a claim on the p alone.
- Two of three contrasts share the Strand-1 frozen reconstruction; treat
  "3 models" as 2 lines of evidence, as in RQ1.
- No multiple-testing correction across periods × models × metrics ×
  variants.

### Verdict

**Does not support.** Frozen ECE does not rise monotonically; frozen
slopes are indistinguishable from zero under the size-preserving null;
xgboost's treated4 slope_diff has the wrong sign; the only cells with
p_diff < 0.05 sit on the shared GLM reconstruction. The RQ3 hypothesis
("frozen calibration degrades; per_round mitigates it") is not
consistent with these tables. What the data do support is weaker and
already covered by RQ1: a small, mostly group-stage sharpening of
outcome probabilities (BS / RPS), not a calibration-drift story.
