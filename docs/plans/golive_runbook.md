# Go-Live Runbook — June 7–11, 2026

Active execution slice for the pre-WC deployment sprint. The canonical plan
(with full checklist bodies and rationale) stays in `thesis.md`; this file is
the day-by-day worklist and points at those section IDs rather than copying
them. Ephemeral — folds into `wc_live.md` once the WC starts.

- **Branch:** `thesis`
- **WC kickoff:** June 11, 2026
- **Goal:** deploy the pipeline to GCP and verify it runs on fresh friendly
  data *before* kickoff, with the cadence experiment (frozen vs per-round)
  wired in.

---

## Guiding order (why)

- **B.1–B.4 precede A.10.** A.10 consumes B.2's dual aliases (it assigns
  `champion_frozen` / `champion_per_round` to the frozen run), so the alias
  scheme must exist first. This is the dependency direction, not a workaround.
- **GCP (C.1–C.7) starts first, in parallel with B.*.** Friendlies (Jun 7–10)
  are the only fresh-data test window before kickoff, and GCP is unfamiliar, so
  it gets the most buffer. C.1–C.7 deploy whatever champion exists today; only
  C.9's dual-mode assertions need B.2 + A.10.
- **A.10 stays at the last responsible moment (Jun 10)** so the frozen snapshot
  includes the final friendlies.

---

## Day-by-day timeline 

### Jun 7–8 — GCP skeleton on the current champion ‖ B.1–B.4 in parallel

GCP track (deploy code as-is, single `champion` alias, pre-B.2):
- [x] **C.1** Containerise the trigger
- [x] **C.2** Service account + permissions
- [x] **C.3** Secret Manager
- [x] **C.4** Artifact Registry + image push
- [x] **C.5** Cloud Run Job
- [x] **C.6 (created)** Cloud Scheduler `daily-pipeline-trigger` ENABLED at
  `0 4 * * *`. The flip to WC hourly is moved to **Jun 11** (kickoff section).
- [~] **C.7** DVC remote on GCS — **deferred to post-WC.** The DagsHub remote
  works (blobs + git pointers verified Jun 9). Migrating ~32 MB to GCS days
  before kickoff is avoidable risk for marginal gain (only upside is dropping
  the DagsHub token from Secret Manager). Revisit after the tournament.

Thesis track (on `thesis`):
- [x] **B.1** Snapshot metadata tagging
- [x] **B.2** Dual aliases + per-mode dispatch — **must degrade gracefully**
  (fall back to `champion`) while the dual aliases don't exist yet
- [x] **B.3** Per-round refit trigger (bayesian hardening already done)
- [x] **B.4** Per-mode monitoring

> **HARD GATE:** C.1–C.5 working (build → deploy → one successful manual job
> exiting 0 on real friendly data) by **end of Jun 8**, so Jun 9–10 are
> verification, not debugging. If GCP slips, the local pipeline is the fallback
> for the freeze.

### Jun 9 — redeploy with B.1–B.4 merged + first C.9 (dress rehearsal)
- [x] Redeploy image with B.1–B.4 (deployed `20260609a`; also carries the
  entrypoint pointer-checkout fix + the collapsed-raw guard)
- [x] **C.9** (first pass): dual-mode dispatch ran on GCP (frozen + per_round),
  metadata tags landed, `predictions_all_models.csv` covered the full roster
  (all 4 models simulated), DVC push succeeded. Clean run = gold 6921 rows.
  - Also resolved: the Cloud Run truncation incident (entrypoint `git reset
    --mixed` left `raw.dvc`/`dvc.lock` unmaterialized → `dvc pull` fetched ~13
    files → truncated pointer clobbered `thesis`). Fixed in `entrypoint.sh`.

### Jun 10 — test the hourly cadence on live friendly data
- [ ] Build + push new image (carries the timeout change + any other pending
  code changes); `docker build --platform linux/amd64 -t ...:20260610a .` →
  `docker push` → `gcloud run jobs update --image ...:20260610a`
- [ ] Temporarily switch `daily-pipeline-trigger` to hourly (`0 * * * *`) for a
  short window (watch 1–2 ticks), then revert to daily. Validates the cadence
  mechanics + the 50-min-under-60-min timing budget on real friendly data.
  - Caveats: pre-A.10 each tick may fire the legacy delta refit on every new
    friendly (extra MLflow runs / compute — cosmetic); each cycle must stay
    < 60 min (today's full run was ~35 min); do **not** kick off a manual job
    while a scheduled tick is running (Cloud Run Jobs have no cross-execution
    lock); remember to revert to daily afterwards.

### Jun 11 (pre-kickoff, early) — A.10 freeze + final C.9
- The last 2026 friendly kicks off **01:00 UTC Jun 11**; its result is ingested
  by the 04:00 UTC daily tick. Run A.10 after that and before the **19:00 UTC**
  opening-match kickoff — plenty of buffer.
- [ ] **A.10** build latest full Gold (final friendlies) → `run_champion_refit`
  for the 3 champions → assign `champion_frozen` + `champion_per_round`.
  **Pause `daily-pipeline-trigger` during the manual freeze** to avoid an
  execution overlap (no cross-execution lock on Cloud Run Jobs).
- [ ] Record frozen `run_id` + aliases in `results_pre_wc.md` (fills the
  existing `[fill]` slots)
- [ ] **C.9** (final pass): both aliases resolve, both artifact sets written,
  monitoring runs empty pre-WC

### Jun 11 — kickoff (opening match 19:00 UTC)
- [ ] **C.6** flip scheduler: pause pre-WC daily, enable hourly (`0 * * * *`)
- [ ] First live cycle verified; start logging in `wc_live.md`

---

## Notes / decisions made during the sprint

- **Bayesian hardening (B.3 prerequisite) is done.** `target_accept=0.9`,
  `tune_steps=1000`, `max_eta=10.0` clip, `prior_sigma` capped at 2.0 are
  all in `BayesianPoissonModel.__init__` defaults.
- **B.3 concurrency guard:** a whole-run `fcntl` lockfile (`data/.trigger.lock`)
  in `trigger.main()` prevents a second local invocation from running while the
  first is still active (exit 0 / skip tick). On **Cloud Run Jobs** there is no
  `--max-instances` flag (that is a Services setting) — the cross-execution
  guard is keeping task timeout (50 min) under the scheduler interval (60 min
  hourly), plus `--tasks 1 --parallelism 1` (one container per execution). The
  lockfile covers local/manual overlap only. WC refits are spaced hours apart
  so real overlap is unlikely; the guards exist for the edge case of a slow
  bayesian MCMC refit approaching the interval limit.
- **Cloud Run Job resources:** start at 8 GiB / 4 vCPU (initial full pipeline);
  lower to 4 GiB / 2 vCPU (`gcloud run jobs update --memory 4Gi --cpu 2`)
  after the first successful run confirms routine cycles fit.
- **Image deploy (C.4/C.5):** build `linux/amd64` on arm64 Mac; tag with a
  unique timestamp (e.g. `20260608-1530`), not `:latest` alone; point the job
  at that tag. Rebuild **only on code changes** — data is ingested at runtime.
  Sprint redeploys: initial C.5 + Jun 9 (B.1–B.4). Details in thesis C.3–C.5.
- **Scheduler mode (C.6):** both pre-WC daily and WC hourly schedules use
  `mode=auto` (default `PIPELINE_MODE` in `entrypoint.sh`). Do **not** use
  `inference_only` — it skips B.3 per-round refits. No `--args` on the job;
  mode is env-driven only.
- **Shadow-refit timeout raised + per-model cap removed (Jun 9):**
  `SHADOW_REFIT_TOTAL_TIMEOUT_S = 1800` (30 min); `SHADOW_FIT_TIMEOUT_S`
  removed entirely. In the pre-WC full 8-model refit, bayesian_poisson (fit
  last) was starved by the 20-min overall cap (~8 min → timed out → prior
  version kept). The per-model cap was redundant since bayesian is the only
  slow model and it runs last — fast models are never blocked by it. Each model
  now gets the full remaining overall budget. Keep worst-case cycle < 50-min
  Cloud Run task timeout.
- (add live blockers / decisions here; promote durable ones to `decisions.md`)
