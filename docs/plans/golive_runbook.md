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
- [ ] **C.1** Containerise the trigger
- [ ] **C.2** Service account + permissions
- [ ] **C.3** Secret Manager
- [ ] **C.4** Artifact Registry + image push
- [ ] **C.5** Cloud Run Job
- [ ] **C.6** Cloud Scheduler (pre-WC daily)
- [ ] **C.7** DVC remote on GCS

Thesis track (on `thesis`):
- [ ] **B.1** Snapshot metadata tagging
- [ ] **B.2** Dual aliases + per-mode dispatch — **must degrade gracefully**
  (fall back to `champion`) while the dual aliases don't exist yet
- [ ] **B.3** Per-round refit trigger (+ bayesian hardening, see note below)
- [ ] **B.4** Per-mode monitoring

> **HARD GATE:** C.1–C.5 working (build → deploy → one successful manual job
> exiting 0 on real friendly data) by **end of Jun 8**, so Jun 9–10 are
> verification, not debugging. If GCP slips, the local pipeline is the fallback
> for the freeze.

### Jun 9 — redeploy with B.1–B.4 merged + first C.9 (dress rehearsal)
- [ ] Redeploy image with B.1–B.4
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
- [ ] **C.6** flip scheduler: pause pre-WC daily, enable every-30-min
- [ ] First live cycle verified; start logging in `wc_live.md`

---

## Notes / decisions made during the sprint

- **Bayesian hardening moved to B.3.** `target_accept=0.9` + raised `tune_steps`
  is a prerequisite for B.3's 7 unattended per-round MCMC refits, not just
  A.10's one freeze fit. Do it with B.3 so the refit-cycle wall-time measured
  for the C.5 timeout / B.3 concurrency guard reflects hardened defaults.
- (add live blockers / decisions here; promote durable ones to `decisions.md`)
