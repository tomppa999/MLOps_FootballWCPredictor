# Errors overview — WC 2026 pipeline

Synthesis of the defects and outages that produced the post-tournament TODOs, and how each was
addressed. Chronology and per-round detail live in `wc_live.md`; this file is the
cross-cutting view that feeds Chapter 3 (threats to validity) and Chapter 6 (limitations).

Ten live-pipeline defects fall into four buckets: one silently invalidated a *result*, four cost
*artifacts* that can be rebuilt, four cost *operations* that self-healed, and one was a silent
train/serve skew (§8). Only the first bucket threatens a research question. **All four artifact
holes were rebuilt in D.1 (Aug 15) and regenerated Sep 4 once the repair's own defects (§2.3,
§2.4) were fixed; the result-invalidating bug was corrected offline and in code, for RQ1 by
Strand 1 and for RQ2 only by Strand 5; the train/serve skew is documented rather than repaired.**
The whole reconstruction now re-derives offline from DVC-tracked inputs via
`dvc pull && dvc repro figures` — see the README.

Four further defects belong to the *repair* rather than the pipeline. The first version of the
D.1 backfill re-predicted rows it should have copied, briefly introducing a spurious effect of
its own (§2.1); the first version of the frozen-shadow replay leaked each match's own result
into its features, briefly reversing the RQ1 finding (§2.2); the constant that fixed that leak
was itself a guess, and admitted results for matches still being played (§2.3); and the first
reconstruction of the four missing entropy snapshots filled them with end-of-tournament models
(§2.4). All four are recorded here because a reconstruction that manufactures — or inverts —
signal is a sharper threat to validity than any of the outages. §2.5 records the mirror case: the
one place where the replay is *right* and the log is wrong, so reproducing the log would be the
error.

---

## 1. Result-invalidating: the shadow-resolution bug

**What.** `wc_shadow` has no cadence alias, and frozen shadow refits write no `cadence_mode`
tag, so `load_shadow_model` resolved both cadences to the newest (per_round) version. The
logged "frozen" rows for the three refit-eligible roster shadows (`poisson_glm`,
`bayesian_poisson`, `mean_rate_poisson`) are the per_round artifact, for the entire tournament.

**Why it is the highest-cost defect.** It produced no error, no alert and no missing data. It
was found only by noticing that two columns which should differ were byte-identical. It leaves
`xgboost` as the study's only clean frozen↔per_round contrast — one model instead of four.

**Not affected.** The champion in both cadences (alias-separated), the four never-refit shadows
(`negbin_glm`, `ridge`, `sarimax`, `random_forest`, where identical columns are *correct*), and
the entire per_round column for every model.

**Addressed by.** **DONE (Aug 15) — D.1 Strand 1.** Offline frozen-shadow rebuild for
`poisson_glm` (v88) and `bayesian_poisson` (v90) completed; tournament frozen means RPS
0.15845 / 0.15788, NLL 2.92722 / 2.92649, RMSE 0.89821 / 0.89823 (n=104, after the §2.2
correction rerun) — see
`data/reconstruction/strand1_frozen_shadow/`. Cadence-aware resolution fix and regression
tests under `tests/models/` also landed. The `mean_rate_poisson` rebuild was dropped by
decision: its λ spread across the tournament is ~2e-3 and symmetric, so refit-vs-frozen is
noise there. As-logged, `xgboost` was the only clean contrast; after D.1, `poisson_glm` and
`bayesian_poisson` also have separable frozen↔per_round columns.

**But Strand 1 only cleaned RQ1.** It rebuilds per-match metrics, and RQ1 is a match-level
question. RQ2 asks how tournament-level uncertainty resolves, so it reads entropy off the
bracket simulations — and Strand 2 replays the *logged* λ values, which for every frozen cycle
are the per_round artifact this section describes. The frozen entropy trajectory therefore
stayed contaminated for the three weeks between the two repairs (Aug 15 → Sep 4): RQ1 said
"frozen and per_round differ for three models", while RQ2 was still comparing per_round against
itself for all but `xgboost`. **Corrected by Strand 5 (§2.4)**, which re-predicts all ~1,128
pairings on each of the 423 frozen cycles from the pre-tournament pin instead of trusting the
log. The
effect it recovers is real but small — `poisson_glm` frozen-minus-per_round `entropy_winner`
mean +0.0080 nats — and it did not exist in the data at all before Strand 5.

---

## 2. Artifact-level: four holes, one replay

Three of the four share the same mechanics — replay from the `wc2026-end-of-tournament` tag,
rebuild each match's pre-kickoff feature row from the DVC-versioned Gold snapshot (strict
`inference_timestamp < kickoff`), then re-predict or re-simulate. The fourth (the dropped rows)
is a pure copy and must not be re-predicted at all, for the reason given in §2.1. They were
folded into a single D.1 pass rather than four separate jobs.

**STATUS: DONE — all four holes replayed (Aug 15), then regenerated Sep 4 under the corrected
settlement rule of §2.3 and the regime-pinned Strand 5 of §2.4.** Audit: **67 pass / 10 warn /
0 fail** (`data/reconstruction/audit_report.json`, 77 checks: 20 Strand 1, 33 Strand 2, 15
Strand 3, 7 Strand 5, 2 RQ2 provenance). These totals are not comparable line-for-line with the
Aug 15 figure of 83 / 7 / 0: Strand 4 was retired as an input, so its checks left the roster and
Strand 5's took their place.

The 10 warns are expected and none of them is a numerical disagreement. Seven are visibility
limits in the offline audit rather than findings — the monitoring export it reads covers 54 of
104 matches, so it cannot confirm the 832/832 coverage claim or the byte identity of the four
Strand 3 cadence copies from that source alone. Two are the cadence-asymmetric cycle counts left
by the silent skips of §3 (frozen 423 vs per_round 422). The last is entropy monotonicity:
577 of 3,376 rows violate `entropy_r32 ≥ … ≥ entropy_winner`, worst upward jump 0.0048 nats,
which is Monte Carlo noise at 10k simulations and not a resolution failure. **"Same numbers"
means 0 fails, not 0 warns.** RQ-ready merges in `data/analysis/`.

| Hole | Cause | Missing | Replay action | Status |
|---|---|---|---|---|
| Frozen shadow column | shadow-resolution bug (§1) | True frozen rows, `poisson_glm` + `bayesian_poisson`, all rounds | Predict with the untagged `stage=shadow-refit` versions; recompute RPS / NLL / RMSE | **DONE** Strand 1 |
| Bracket artifacts | host-rate scramble, KO-locking bug, pen-winner bug (Jul 7 → `20260712a`) | Valid `tournament_probabilities.csv`, `ko_pairings.csv`, `ko_fixtures.csv` from Jul 7 on, and the RQ2 entropy curves derived from them | Re-run `simulate_tournament` on corrected code, same seeds, locked KO results. Analysis window starts at the last pre-tournament frozen cycle (`ANALYSIS_START`, 2026-06-11 16:27:37 UTC); 34 dry-run cycles are excluded. | **DONE** Strand 2 (3,376 cycle entropy rows); its frozen half replays the logged λ and so inherits §1, re-predicted by Strand 5 (§2.4) |
| Four dropped rows | `_safe_shadow_predict` 120 s load timeout on a last pre-kickoff cycle | 2 × `ridge` per_round (MD3), 1 × `random_forest` frozen (R32), 1 × `mean_rate_poisson` frozen (MD1) — 830/832 as-logged | Copy each row byte-for-byte from its twin in the other cadence; never-refit ⇒ the twin *is* what the missing cycle would have logged | **DONE** Strand 3 → **832/832** |
| Four entropy snapshots | R32 scheduler pause (3), R16 IPv6 stall (1) | Intermediate locked states in the RQ2 trajectory | Re-simulate each state with the round's `_seed_from_string` seed and the model versions that served that round | **DONE** Strand 5 (32 synthetic rows). Strand 4 did this first with terminal models and is superseded — §2.4 |

All three models in the dropped-row hole (`ridge`, `random_forest`, `mean_rate_poisson`) are
never refitted, so the backfill reduced in practice to four cadence copies and no prediction at
all: 2 × `ridge` per_round from frozen, 1 × `random_forest` frozen from per_round, 1 ×
`mean_rate_poisson` frozen from per_round. The audit verifies all four against their twins
(`s3.copy.*`) and fails if any backfill row is not a copy (`s3.all_copied`).
**Achieved: 832/832 per cadence** (`data/analysis/rq1_matches.csv`, 1,664 rows).

The backfill list is **final at four rows**. Three further silent shadow-skips fired later
(`negbin_glm` Jul 13 08:15, `ridge` Jul 13 08:17, `random_forest` Jul 14 20:21) but each was
followed by a successful cycle before the affected kickoff, so they left no holes.

### 2.1 The repair that manufactured an effect

**What.** The first version of Strand 3 re-predicted the two `ridge` per_round rows offline —
pinned `wc_shadow` artifact, DVC Gold snapshot before kickoff — instead of copying them from the
frozen twin. `ridge` is never refitted, so the two cadences must be identical by construction.
The offline replay landed on a marginally different feature snapshot than the live cycle had
used, leaving `ridge` with a frozen↔per_round delta of **~0.0003 mean RPS** on 2 of 104 matches.

**Why it matters more than its size.** The real RQ1 cadence effect is ≤0.0033 RPS. A repair to a
*data-completeness* hole therefore produced a spurious cadence signal of the same order of
magnitude as the finding under study, on a model that by definition cannot have one. Read
naively, the table showed four models with a cadence effect instead of three.

**How it was caught.** Not by the audit, which passed: the re-predicted rows were internally
consistent, correctly scored and correctly keyed. It surfaced only from the invariant check —
*never-refit models must be bit-identical across cadences* — applied per model over all 104
matches. `ridge` differed on exactly 2.

**Addressed by.** **DONE (Aug 15).** Strand 3 is now a uniform copy step over
`BACKFILL_FIXTURES`, each spec declaring `copy_from_cadence`; the pinned-model and Gold
time-travel imports are gone from that driver. `audit_reconstruction` gained `s3.all_copied`,
which fails if any backfill row is not a cadence copy, and its twin-identity check now resolves
the source cadence per spec (so the `--frozen-monitoring` export is consulted for `ridge`).
Strand 3 audit: 14 pass / 0 warn / 0 fail against the full exports; 8 pass / 7 warn / 0 fail in
the committed report, where the audit sees only the 54-match monitoring snapshot and reports the
unverifiable half as warns. Downstream `rq1` / `rq3` / `round_leaderboards` were
rebuilt; `ridge` now shows 0/104 cadence differences and its tournament RPS is 0.1589 in both
modes.

**Lesson for Chapter 3.** Reconstruction is not free of the failure modes it repairs. Where a
quantity cannot have changed, copy it rather than recompute it, and encode the invariant as a
test — otherwise the reconstruction's own noise floor competes with the effect being measured.

### 2.2 The repair that leaked the outcome it was predicting

**What.** The first version of Strand 1 called `parse_wc_results_before_kickoff` with each
match's own kickoff as the cutoff and no `settle_delta`. That function counts a match as
settled when `kickoff <= cutoff`, so **every replayed match had its own final score appended to
Gold before its features were built**, with `reference_date` set to kickoff + 1 day so the
rolling window actually read it. The function's docstring warns about exactly this, and Strand 2
guards against it by passing `SETTLE_DELTA` (2 h); Strand 1 did not. Confirmed on the tournament
opener, where augmented Gold held 6946 rows before a single match had been played.

A second, independent defect rode along: `build_inference_features` derives
`days_since_last_match` from Gold concatenated with the *upcoming* rows, and live inference
passes all 1131 pairings at once while the replay passed one fixture at a time. The two paths
disagreed even given identical Gold and an identical model.

**Why it matters more than its size.** The leak was worth **+0.0018 (bayesian_poisson) /
+0.0019 (poisson_glm) mean RPS** to the frozen arm — the same order as the RQ1 cadence effect
under study, and in the direction that *reversed* it. Pre-correction the table showed frozen
beating per-round for both reconstructed models while `xgboost` (untouched by Strand 1) showed
the opposite; the disagreement was an artifact. After correction all three refit-eligible models
favour per-round, and the group/knockout split becomes coherent.

**How it was caught.** Not by the audit, which passed 19/19: every reconstructed row was
internally consistent and its RPS recomputed from λ to 1e-15. It surfaced from a physical
invariant — *no refit had occurred before MD1, so frozen and per-round must be bit-identical
there*. `xgboost` satisfied it for exactly the first two matches (the premature refit fired
after match 2); the two reconstructed models violated it from match 1.

**Addressed by.** **DONE (Aug 15).** Strand 1 passes `SETTLE_DELTA` and predicts through the
full pairing batch via new `replay_common` helpers (`snapshot_key`, `batch_lambdas`,
`predict_fixture_from_batch`). All 208 rows regenerated; `rq1` / `rq2` / `rq3` /
`round_leaderboards` rebuilt. Validation: the opener now reproduces the live λ exactly
(2.083377 / 0.656733). `audit_reconstruction` gained `s1.leakage`, which fails if any match sees
its own or an unsettled result, and `tests/analysis/test_strand1_frozen_shadow.py` pins the
settle-delta pass-through, the self-leakage invariant and the batch shape.

**Lesson for Chapter 3.** The as-of cutoff is the single most dangerous parameter in a
point-in-time replay, and a default of "no delay" is a leak waiting to happen. Two further
points generalise: an audit that only checks internal consistency will certify a leaking
dataset, so validity checks must encode *physical* invariants (nothing may depend on its own
outcome; quantities that cannot differ must be identical); and a replay must reproduce the
*shape* of the original computation, because batch-dependent features make single-row inference
a different function.

### 2.3 The fix from §2.2 was a guess, and the guess was two hours

**What.** `SETTLE_DELTA = 2 h` — introduced in §2.2 to stop a match leaking its own score —
treats a match as known to the pipeline once `kickoff + 2 h <= cutoff`. Live never worked that
way: it took a result only once an API-Football ingestion had returned it in a *finished*
status, and the frozen cadence fires at kickoff + ~2 h 06 min. For a match still being played
at that moment the proxy handed the replay a result live did not have. Seven frozen cycles and
their seven per_round twins were affected: Jun 23 00:06, Jul 1 22:06, Jul 2 02:06, Jul 3 20:06,
Jul 4 00:06, Jul 7 18:06, Jul 7 22:06 UTC. Six of the seven matches were knockout ties decided
in extra time or penalties (Belgium–Senegal AET, Argentina–Cape Verde AET, Australia–Egypt PEN,
Switzerland–Colombia PEN, plus USA–Bosnia); one was an interrupted group match (France–Iraq,
status `INT` before `FT`); and Argentina–Egypt finished in normal time but was still `2H`
elapsed 90 at the 18:05 ingestion. The retained Bronze window snapshots record all of these
statuses directly, so the diagnosis rests on the versioned data, not inference.

**Why it matters.** It is future information in the features — the one class of defect the
project treats as non-negotiable — and it is not small where it lands: 93 of 1,128 pairings per
model on an affected cycle, with |Δλ| from 1.3e-2 to 1.3e-1 against a `METRIC_TOL` of 1e-9. The
bracket simulation then locked a team as advancing that live still had in the balance, so the
RQ2 entropy at those points read lower than live's.

**A wrong hypothesis first.** The initial reading blamed `resolve_gold_commit` for picking a
stale Gold because the live `dvc.lock` commit landed late. That is disproved: every one of the
423 frozen cycles has its Gold commit land 16–33 s *before* it (median 32.6 s) and no commit
ever lands within 34 min *after* a cycle. The direction was also backwards — live's lambdas on
the seven cycles are bit-identical to the previous cycle and match the replay's *previous-cycle*
values to 2–4e-16, so the replay was ahead of live, not behind. Gold selection was never wrong.

**Addressed by.** `snapshot_inputs --only settlement` walks the Bronze committed alongside each
Gold version and writes `data/reconstruction/inputs/fixture_settlement.csv`: per fixture, the
first commit whose snapshot showed it finished, and the status at that moment. 104 fixtures,
none unresolved; 97 resolved on the first probe and exactly the seven problem matches on the
second. `parse_wc_results_before_kickoff` is snapshot-first and falls back to `SETTLE_DELTA`
per fixture only when the snapshot is absent. No allowlist and no loosened tolerance.
Measured blast radius before re-running anything: 14 of 845 cycles for Strands 2 and 5, and 32
of 104 Strand 1 match states. Every change removes a result; none adds one.

**RQ1 and RQ3 are provably untouched, for two independent reasons.** In all 32 changed Strand 1
states the dropped result involves *neither* of the two teams being predicted — a team that
played two hours ago is not kicking off again the same night — so no rolling form feature moves.
The one global channel, `reference_date` (`max(known WC date) + 1 day`), does shift in all 32,
but it can only reach `days_since_last_match` / `rest_diff`, and those were already pinned to 0
by the batch skew in §8. Re-running Strand 1 confirmed it: all 208 rows are bit-identical to the
pre-fix output, max |Δλ| exactly 0, mean RPS 0.158166 unchanged. The correction is therefore
confined to RQ2.

`audit_strand5` encodes the invariant that would have caught this on day one:
`s5.champion_identity` requires the frozen champion to reproduce the logged lambdas on **all**
423 frozen cycles within `METRIC_TOL`. The champion is the right probe because
`champion_frozen` v15 was never touched by the shadow-resolution bug, so any deviation is a
replay-input error rather than a model difference. Against the pre-fix outputs it failed on
exactly 7/423.

**Acceptance evidence (Sep 4 re-run).** Both identity checks flipped, which is what makes them
regression tests rather than tautologies:

| check | before the settlement fix | after |
|---|---|---|
| `s5.champion_identity` | FAIL, 7 of 423 cycles, worst \|Δλ\| 1.301e-01 | **PASS**, all 423, max 4.441e-16 |
| `s5.host_identity` | FAIL, worst \|Δλ\| 9.951e-02 | **PASS**, 33,276 pairings, max 4.441e-16 |

Measured scope after the re-run matches the prediction made before it: Strand 2 moved 56 of
3,376 rows — exactly the 14 cycles × 4 models — with `entropy_winner` shifting up to 0.0896 nats,
and Strand 1 moved zero rows. `s5.post_refit_divergence` still passes, with the frozen GLMs
diverging from live's contaminated rows by 4.76e-01 (`poisson_glm`), 3.72e-01
(`bayesian_poisson`) and 2.02e-03 (`mean_rate_poisson`); had the settlement fix flattened that
divergence it would have meant Strand 5 was reproducing the bug rather than correcting it.

**Lesson for Chapter 3.** §2.2 concluded that the as-of cutoff is the most dangerous parameter
in a point-in-time replay; the sequel is that *replacing a leaky default with a plausible
constant is still a guess*. Two hours was defensible for a 90-minute match and wrong for extra
time, penalties and long stoppage. Where the production system recorded what it knew — here the
versioned Bronze, committed atomically with the Gold it produced — the replay should read that
record instead of modelling it. The reconstruction is only as trustworthy as its worst
assumption about the past, and an assumption that is right 831 times out of 845 is still an
assumption.

### 2.4 The replay that filled a gap with models from the future

**What.** Strand 4 reconstructed the four missing entropy snapshots (§2, row 4) by calling
`run_inference_and_simulation` at reconstruction time with an overridden timestamp and a
truncated result set. Overriding *when* the cycle claims to run does not change *which* models
it loads: that call resolves the champion and shadow registries through their live aliases, so
in August it picked up the terminal, end-of-tournament versions. The R32 and R16 gaps were
therefore filled by models fitted on rounds that had not yet been played at the state being
reconstructed.

**Why it matters.** It is the same class of defect as §2.2 — future information in a
point-in-time replay — but arriving through the model rather than the features, which makes it
easier to miss. It also lands precisely where it does the most damage: the four snapshots exist
only to give RQ2's entropy trajectory resolution across the knockout rounds, so an
over-informed model at those four points flatters exactly the "uncertainty resolves as the
tournament progresses" claim the trajectory is evidence for.

**Addressed by.** **DONE (Sep 4) — superseded by Strand 5.** Strand 5 pins each snapshot to the
versions that actually served that regime, read from
`data/reconstruction/inputs/models/manifest.csv`: R32 gets `wc_production` v19 plus `wc_shadow`
v104–106, R16 gets v20 plus v107–109, and the frozen arm always gets the pre-tournament pin
(v15, v88–90). Strand 4's data directory was deleted; `src/analysis/strand4_entropy.py` remains
importable because `build_entropy_snapshot_specs` is still the source of truth for *which* four
states are missing, and its `run_strand4_entropy` docstring records the supersession.
`audit_strand4` stays dispatchable via `--strand 4` and now reports `s4.exists` as a fail, which
is the correct answer: those outputs should not come back.

**Lesson for Chapter 3.** A replay has two clocks, not one — the as-of date of the data and the
as-of version of the model — and overriding the first does nothing to the second. Pin model
versions explicitly from a recorded manifest; an alias is a mutable pointer and will always
resolve to the present.

### 2.5 Where the replay is right and live was wrong: host orientation

**What.** Commit `01942e41` (Jun 27, "orient host KO rates without swapping; emit host-vs-host
dual predictions") fixed how `src/inference/features.py` handles the three World Cup hosts. Every
frozen cycle before Jun 27 16:08 UTC was served by the pre-fix code, so a faithful replay on
corrected code *must* differ from the log on the 2–4 pairings per cycle that involve a
`WC_2026_HOSTS` team, and emits 1,131 pairings where live logged 1,128 (three extra host-vs-host
reverse rows).

**Why it is recorded here.** Not as a defect to repair — Strand 5 is the correct side of this
disagreement — but because it is the one place where "the reconstruction reproduces the log" is
the *wrong* acceptance criterion. Stated carelessly, an identity check would either be relaxed
until it stopped catching real errors, or would push the replay back onto buggy code to match a
log that was already known to be wrong.

**Addressed by.** `s5.host_identity` excludes host pairings before the Jun 27 fix and expects
exactly three unmatched rows per affected cycle, then requires exact agreement everywhere else:
33,276 pairings, max |Δλ| = 4.441e-16. The host exclusion is scoped by timestamp, so it stops
applying the moment the fixed code went live. `METRIC_TOL` stays at 1e-9 and there is no skip
list — if identity ever fails for a model outside this exclusion, the pinned version is wrong
and the fix belongs in `manifest.csv`, not in the tolerance.

---

## 3. Operational: four incidents, all self-healed

None cost a prediction or a Gold row. All 104 matches were scored pre-kickoff in both cadences.

- **Jun 29–30 scheduler pause** — `wc-pipeline-trigger` left paused after a deploy; **22.01 h**
  (Jun 29 09:23 → Jun 30 07:32), 3 R32 cycles missed. Cost: entropy-trajectory resolution only.
- **Jul 5–6 IPv6 ELO stall** — every run killed by the 90 min task timeout at the ELO freshness
  check; **at least 25.8 h** (last inference Jul 5 06:14, next after Jul 6 08:00). Cost:
  1 entropy snapshot.
- **Jul 12 02:05 `dvc push` timeout** — `CalledProcessError` at `trigger.py:280`, before the
  commit gate. This run logged **no inference at all**; the 02:16 / 02:25 pair is the retry, not
  a second cycle. No snapshot, no commit. Superseded by `8df3d581` at 02:15.
- **Jul 12 10:51 `git push` rejected** (non-fast-forward) — `CalledProcessError` at
  `trigger.py:287`. Unlike the 02:05 crash this one died **mid-cycle**, after the frozen pass and
  before per_round, leaving an unpaired frozen run (`5af66ed18a1c`, 10:06:46) and a **Jul 12 count
  of 8 frozen vs 7 per_round**. Data was pushed but the local commit was discarded by the next
  cycle's hard reset, leaving orphaned blobs on the DVC remote. Superseded by `7254a5f` at 11:10.

The two crashes therefore failed at different points in the cycle, not both in the push gate. The
8-vs-7 asymmetry is a logging artifact of the mid-cycle kill and has **no downstream completeness
impact** — it is not one of the four holes in §2 and does not extend the backfill list, which
stays final at four rows. Beyond that the crashes cost only the point-in-time view at those
timestamps: because Gold is cumulative, each retry snapshot is a superset of the failed one, so no
match rows are unrecoverable. Verified — see §5.

---

## 4. Explained, not open: `mean_rate_poisson` at the naive floor

The model breached the naive RPS floor from match 40 onward and plateaued at rolling-24 RPS
0.2562 against a 0.2350 floor, generating 198 ALERT lines. The alert behaved exactly as
designed; the model is the problem, not the monitoring.

Root cause is `lambda_h == lambda_a` in 100% of rows — no home/host advantage at all. This
belongs to the host-advantage work item, not the refit item, and it is a finding rather than a
defect to repair.

---

## 5. Prerequisite: Gold / DVC history integrity — VERIFIED (Aug 10)

Everything in §2 depends on per-cycle Gold history being intact enough to rebuild each match's
pre-kickoff feature row. Two independent checks, both against the tag:

1. **Remote completeness.** `dvc status --cloud --all-commits data/gold` reported `0 files` to
   transfer on every Gold snapshot it walked (500+), i.e. no object missing from the DagsHub
   remote. The run was stopped near the end once the pattern was uniform; no missing object was
   ever reported.
2. **Pre-kickoff coverage.** For all **104 of 104** matches there is a Gold commit strictly
   before kickoff. Median lead time 1.89 h, consistent with the 1–2 h cadence.

Gold history is rich: **437 distinct `data/gold` hashes across the 454 commits** touching
`dvc.lock`, so nearly every cycle produced a genuinely new state.

The two Jul 12 gaps do not appear among the worst lead times — the retry cycles restored normal
cadence well before the next kickoff. **Conclusion: the rebuild tolerates the two missing
point-in-time snapshots; no recovery attempt is needed.** Do not run `dvc gc` while the Jul 12
10:51 orphans may still be wanted, and never `dvc repro` during D.1.

---

## 6. Decisions not to fix

Three classes of defect are documented rather than repaired. For the first two the pipeline
retires at the end of the tournament and both mitigations are purely preventive — they pay off
only on future inference cycles, of which there are none. The third (§8) is skipped for a
different reason: re-serving the tournament under a corrected feature definition would not be a
reconstruction of what happened.

- **Silent shadow-skips.** The dual-mode inference layer trades completeness for resilience: one
  bad shadow cannot kill a cycle, at the cost of silent per-cycle holes. Recommended hardening:
  in-cycle retry on the shadow load/predict child, ERROR-level logging, and a per-cycle
  model-count completeness check.
- **Push failures.** An unattended CT pipeline that commits its own data state needs the push
  step to be idempotent and retry-safe, because the remote is the one dependency it cannot
  control. The crash-and-retry pattern worked here only because the cadence was frequent and the
  data model cumulative; a daily cadence or a mutable Gold would have turned the same two
  failures into real snapshot loss.

The one code change that *will* ship is the cadence-aware shadow resolution fix (§1), because it
invalidates a result rather than an operation. The Strand 3 copy-not-re-predict change (§2.1)
and the Strand 1 settle-delta / batch-feature change (§2.2) ship for the same reason: they are
analysis-side code that still runs.

---

## 7. Observability limits

Two limits compound, and together they are a Chapter 3 point rather than a footnote.

**Killed runs are invisible to MLflow.** Every run in both outage windows reads `FINISHED` — the
stalled jobs died before reaching `start_run`, so the tracking server holds no record that they
were ever attempted. Outages are detectable only as an absence (a gap between consecutive run
start times), never as a failed run. Cloud Logging is the sole positive evidence.

**Cloud Logging expires.** Application logs before Jul 11 06:00 UTC are permanently gone
(`_Default`, 30-day retention), including the Jul 10 QF matchday. Audit logs survive 400 days in
`_Required`. The exported window Jul 11 06:08 – Jul 20 10:31 is preserved in
`logs/logs_qf_final.txt`.

The consequence: the R32 and R16 outages sit *before* the retained window, so they are
reconstructible only because the export was taken when it was. An unattended pipeline whose
failure mode is invisible to its experiment tracker, with a 30-day expiry on the only evidence
that remains, has a real observability hole — the fix is to open the run before the work, so a
crash leaves a `FAILED` run rather than nothing.

This limits narration, not reconstruction: Gold, DVC and MLflow carry everything D.1 needs.

---

## 8. Train/serve skew: three rest features were constant at inference

**What.** `build_inference_features` computes `days_since_last_match` by concatenating Gold
history with the *upcoming* fixture rows and taking each team's previous match by date. Live
inference passes all 1131 WC pairings in one call and they all carry the same `reference_date`,
so for nearly every pairing the "previous match" resolves to another hypothetical pairing on
that same date rather than the team's real last fixture.

**Effect.** `home_days_since_last_match`, `away_days_since_last_match` and `rest_diff` were
**0 for essentially every live prediction, for the whole tournament**. The models were trained
on real Gold values, so three inputs were pinned to a constant at serving time. Measured on the
pre-WC Gold snapshot for the opener, the correct values were 6 / 5 / 1 against 0 / 0 / 0 served.

**Why it does not threaten the RQs.** The skew is uniform: it hits every model and both cadences
identically, so the frozen↔per_round contrast is unaffected, and RQ1 / RQ2 / RQ3 conclusions
stand. It is a *capability* loss (three features carried no signal) rather than a *comparison*
bias. It was found while diagnosing §2.2, since the correct single-row behaviour and the buggy
batch behaviour disagreed.

**Decision: documented, not repaired.** Fixing it would require re-serving the entire tournament
with a different feature definition, which is not a reconstruction of what happened. It belongs
in Chapter 6 limitations. Before claiming the features were low-value anyway, check the
`rest_diff` coefficient in the Poisson GLM and its gain in `xgboost` — that claim is currently
untested.

**Lesson for Chapter 3.** A feature whose value depends on the *batch it is served in* is a
latent train/serve skew: it is correct in training, correct in unit tests on single rows, and
wrong in production, with no error raised anywhere. Batch-invariance of feature construction
deserves an explicit test.
