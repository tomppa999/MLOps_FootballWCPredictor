# Errors overview — WC 2026 pipeline

Synthesis of the defects and outages that produced the post-tournament TODOs, and how each was
addressed. Chronology and per-round detail live in `wc_live.md`; this file is the
cross-cutting view that feeds Chapter 3 (threats to validity) and Chapter 6 (limitations).

Ten live-pipeline defects fall into four buckets: one silently invalidated a *result*, four cost
*artifacts* that can be rebuilt, four cost *operations* that self-healed, and one was a silent
train/serve skew (§8). Only the first bucket threatens a research question. **All four artifact
holes were rebuilt in D.1 (Aug 15); the result-invalidating bug was corrected offline and in
code; the train/serve skew is documented rather than repaired.**

Two further defects belong to the *repair* rather than the pipeline: the first version of the
D.1 backfill re-predicted rows it should have copied, briefly introducing a spurious effect of
its own (§2.1), and the first version of the frozen-shadow replay leaked each match's own result
into its features, briefly reversing the RQ1 finding (§2.2). Both are recorded here because a
reconstruction that manufactures — or inverts — signal is a sharper threat to validity than any
of the outages.

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

---

## 2. Artifact-level: four holes, one replay

Three of the four share the same mechanics — replay from the `wc2026-end-of-tournament` tag,
rebuild each match's pre-kickoff feature row from the DVC-versioned Gold snapshot (strict
`inference_timestamp < kickoff`), then re-predict or re-simulate. The fourth (the dropped rows)
is a pure copy and must not be re-predicted at all, for the reason given in §2.1. They were
folded into a single D.1 pass rather than four separate jobs.

**STATUS: DONE (Aug 15) — all four holes replayed.** Audit: 83 pass / 7 warn / 0 fail
(`data/reconstruction/audit_report.json`). RQ-ready merges in `data/analysis/`.

| Hole | Cause | Missing | Replay action | Status |
|---|---|---|---|---|
| Frozen shadow column | shadow-resolution bug (§1) | True frozen rows, `poisson_glm` + `bayesian_poisson`, all rounds | Predict with the untagged `stage=shadow-refit` versions; recompute RPS / NLL / RMSE | **DONE** Strand 1 |
| Bracket artifacts | host-rate scramble, KO-locking bug, pen-winner bug (Jul 7 → `20260712a`) | Valid `tournament_probabilities.csv`, `ko_pairings.csv`, `ko_fixtures.csv` from Jul 7 on, and the RQ2 entropy curves derived from them | Re-run `simulate_tournament` on corrected code, same seeds, locked KO results. Analysis window starts at the last pre-tournament frozen cycle (`ANALYSIS_START`, 2026-06-11 16:27:37 UTC); 34 dry-run cycles are excluded. | **DONE** Strand 2 (3,376 cycle entropy rows) |
| Four dropped rows | `_safe_shadow_predict` 120 s load timeout on a last pre-kickoff cycle | 2 × `ridge` per_round (MD3), 1 × `random_forest` frozen (R32), 1 × `mean_rate_poisson` frozen (MD1) — 830/832 as-logged | Copy each row byte-for-byte from its twin in the other cadence; never-refit ⇒ the twin *is* what the missing cycle would have logged | **DONE** Strand 3 → **832/832** |
| Four entropy snapshots | R32 scheduler pause (3), R16 IPv6 stall (1) | Intermediate locked states in the RQ2 trajectory | Re-run `run_inference_and_simulation` per state with the round's `_seed_from_string` seed | **DONE** Strand 4 (32 synthetic rows) |

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
Strand 3 audit: 14 pass / 0 warn / 0 fail. Downstream `rq1` / `rq3` / `round_leaderboards` were
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
