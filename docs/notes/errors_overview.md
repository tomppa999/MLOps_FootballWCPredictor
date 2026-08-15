# Errors overview — WC 2026 pipeline

Synthesis of the defects and outages that produced the post-tournament TODOs, and how each was
addressed. Chronology and per-round detail live in `wc_live.md`; this file is the
cross-cutting view that feeds Chapter 3 (threats to validity) and Chapter 6 (limitations).

Nine incidents fall into three buckets: one silently invalidated a *result*, four cost
*artifacts* that can be rebuilt, and four cost *operations* that self-healed. Only the first
bucket threatens a research question. **All four artifact holes were rebuilt in D.1 (Aug 15);
the result-invalidating bug was corrected offline and in code.**

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
0.15654 / 0.15607, NLL 2.87747 / 2.87265, RMSE 0.87936 / 0.87780 (n=104) — see
`data/reconstruction/strand1_frozen_shadow/`. Cadence-aware resolution fix and regression
tests under `tests/models/` also landed. The `mean_rate_poisson` rebuild was dropped by
decision: its λ spread across the tournament is ~2e-3 and symmetric, so refit-vs-frozen is
noise there. As-logged, `xgboost` was the only clean contrast; after D.1, `poisson_glm` and
`bayesian_poisson` also have separable frozen↔per_round columns.

---

## 2. Artifact-level: four holes, one replay

All four share the same mechanics — replay from the `wc2026-end-of-tournament` tag, rebuild each
match's pre-kickoff feature row from the DVC-versioned Gold snapshot (strict
`inference_timestamp < kickoff`), re-predict or re-simulate. They were folded into a single D.1
pass rather than four separate jobs.

**STATUS: DONE (Aug 15) — all four holes replayed.** Audit: 76 pass / 12 warn / 0 fail
(`data/reconstruction/audit_report.json`). RQ-ready merges in `data/analysis/`.

| Hole | Cause | Missing | Replay action | Status |
|---|---|---|---|---|
| Frozen shadow column | shadow-resolution bug (§1) | True frozen rows, `poisson_glm` + `bayesian_poisson`, all rounds | Predict with the untagged `stage=shadow-refit` versions; recompute RPS / NLL / RMSE | **DONE** Strand 1 |
| Bracket artifacts | host-rate scramble, KO-locking bug, pen-winner bug (Jul 7 → `20260712a`) | Valid `tournament_probabilities.csv`, `ko_pairings.csv`, `ko_fixtures.csv` from Jul 7 on, and the RQ2 entropy curves derived from them | Re-run `simulate_tournament` on corrected code, same seeds, locked KO results | **DONE** Strand 2 (3,500 cycle entropy rows) |
| Four dropped rows | `_safe_shadow_predict` 120 s load timeout on a last pre-kickoff cycle | 2 × `ridge` per_round (MD3), 1 × `random_forest` frozen (R32), 1 × `mean_rate_poisson` frozen (MD1) — 830/832 as-logged | Same predict path; never-refit ⇒ version resolution is deterministic | **DONE** Strand 3 → **832/832** |
| Four entropy snapshots | R32 scheduler pause (3), R16 IPv6 stall (1) | Intermediate locked states in the RQ2 trajectory | Re-run `run_inference_and_simulation` per state with the round's `_seed_from_string` seed | **DONE** Strand 4 (32 synthetic rows) |

Two of these overlapped: the frozen rebuild covered `mean_rate_poisson` via the live
per_round copy path, and the `random_forest` frozen row was copied from its byte-identical
per_round twin. The backfill therefore reduced in practice to the two `ridge` re-predictions.
**Achieved: 832/832 per cadence** (`data/analysis/rq1_matches.csv`, 1,664 rows).

The backfill list is **final at four rows**. Three further silent shadow-skips fired later
(`negbin_glm` Jul 13 08:15, `ridge` Jul 13 08:17, `random_forest` Jul 14 20:21) but each was
followed by a successful cycle before the affected kickoff, so they left no holes.

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

Two classes of defect are documented rather than repaired, because the pipeline retires at the
end of the tournament and both mitigations are purely preventive — they pay off only on future
inference cycles, of which there are none.

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
invalidates a result rather than an operation.

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
