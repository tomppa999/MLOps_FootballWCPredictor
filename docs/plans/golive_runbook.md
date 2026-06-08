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
- [ ] **C.6** Cloud Scheduler (pre-WC daily + WC hourly paused)
- [ ] **C.7** DVC remote on GCS

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
- [ ] Redeploy image with B.1–B.4 (code change → new dated tag + `jobs update`;
  see thesis C.4)
- [ ] **C.9** (first pass): dual-mode dispatch runs on GCP, metadata tags land,
  `predictions_all_models.csv` covers the full roster, DVC push succeeds

### Jun 10 — A.10 freeze + final C.9
- [ ] **A.10** build latest full Gold (final friendlies) → `run_champion_refit`
  for the 3 champions → assign `champion_frozen` + `champion_per_round`
- [ ] Record frozen `run_id` + aliases in `results_pre_wc.md` (fills the
  existing `[fill]` slots)
- [ ] **C.9** (final pass): both aliases resolve, both artifact sets written,
  monitoring runs empty pre-WC

### Jun 11 — kickoff
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
- (add live blockers / decisions here; promote durable ones to `decisions.md`)
