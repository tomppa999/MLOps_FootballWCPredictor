# Chronology

Single dated event table for the thesis journal, §5.5 incident table, and Chapter 4 development timeline.

**Columns:** Date | ISO week | Phase | Event | Decision / outcome | Source | Feeds

**Feeds codes:** `journal` · `incident (§5.5)` · `timeline (Ch 4)` · `method (Ch 3)`

**Granularity:** Pre-May = coarse (≈1 row per milestone). May 15–Jun 10 = one row per event/decision. WC (Jun 11–Jul 19) = daily / per-incident. Post-WC = per D.1 strand and decision.

**Provenance note:** Git history in this repo starts **2026-03-20** (not May 15). Pre-May rows combine `git log` with the Gantt in `wc_mlops_report.tex` and are marked `reconstructed (coarse)` only where the Gantt is the sole evidence (Mar 9–19 Bronze window before the initial commit).

---

## Pre-May 2026 (sparse / coarse)

| Date | ISO week | Phase | Event | Decision / outcome | Source | Feeds |
|------|----------|-------|-------|--------------------|--------|-------|
| 2026-03-09 – 03-15 | 2026-W11 | Course pipeline | Bronze ingestion | First bronze layer from API-Football + Elo | `wc_mlops_report.tex` Gantt; reconstructed (coarse) | journal; timeline (Ch 4) |
| 2026-03-16 – 03-20 | 2026-W12 | Course pipeline | Silver pipeline | Cleaned/standardized matches | Gantt + git `abab0e4b` (2026-03-20 initial commit: ingestion + silver) | journal; timeline (Ch 4) |
| 2026-03-20 | 2026-W12 | Course pipeline | Repo + DVC bootstrap | Initial commit; DVC tracking for raw; `requirements.md` | git 2026-03-20 | journal; timeline (Ch 4) |
| 2026-03-23 – 03-31 | 2026-W13 | Course pipeline | Gold v1 | One-row-per-match Gold; DVC silver/gold stages | Gantt + git 2026-03-31 (`55501385`) | journal; timeline (Ch 4) |
| 2026-03-30 – 04-06 | 2026-W14 | Course modeling | Experimental / QA / deploy envs + pipeline trigger | 9-model candidate layer, MLflow tracking, training pipeline, tests | Gantt + git 2026-04-06 (`b8809306`) | journal; timeline (Ch 4); method (Ch 3) |
| 2026-04-06 | 2026-W15 | Course modeling | Drop team clusters from features | Clustering experiment closed; Elo ingestion hardened | git 2026-04-06 | journal; method (Ch 3) |
| 2026-04-09 | 2026-W15 | Course modeling | Tuning objective → Poisson NLL | Optuna switches from RPS to NLL within families | git 2026-04-09; `decisions.md` CV vs selection metric | journal; method (Ch 3) |
| 2026-04-11 | 2026-W15 | Course modeling | Alias-based MLflow promotion | Stage-based registry → aliases; holdout NLL; feature-importance artifacts | git 2026-04-11 | journal; timeline (Ch 4) |
| 2026-04-13 – 04-19 | 2026-W16 | Course modeling | Monte Carlo tournament simulation | 10k paths; ET/penalties; best-third R32; Streamlit dashboard; all-pairing predict | Gantt + git 2026-04-14…04-16 | journal; timeline (Ch 4); method (Ch 3) |
| 2026-04-13 | 2026-W16 | Course modeling | Drop top-K promotion gate | All 9 models advance to QA | git 2026-04-13 | journal; method (Ch 3) |
| 2026-04-27 – 05-02 | 2026-W18 | Course ops | Monitoring + shadow refit | Phase 5 monitoring; shadow refit path | Gantt + git 2026-04-29 | journal; timeline (Ch 4) |
| 2026-04-27 – 04-30 | 2026-W18 | Course modeling | Gold v2 features | `days_since_last_match`, `rest_diff`, `elo_sum`, `is_cross_confederation`; 6,784 rows | git 2026-04-29; `archive/feature_v2_baseline.md` | journal; method (Ch 3); timeline (Ch 4) |
| 2026-04-29 | 2026-W18 | Course modeling | Gold v2 champion snapshot | XGBoost champion holdout RPS 0.2109 (WC 2022, 64 matches); run `ba8638c3` | `results_pre_wc.md`; `feature_v2_baseline.md` | journal; results §5.1 |
| 2026-05-05 | 2026-W19 | Course write-up | Document Gold v2 iteration | Feature spec + report §7.2 + temporal tests | git 2026-05-05 | journal |
| 2026-05-11 – 05-17 | 2026-W20 | Course ops | Cloud deployment (course) | Docker / Cloud Run Jobs / Scheduler (course window in Gantt) | `wc_mlops_report.tex` Gantt | journal; timeline (Ch 4) |
| 2026-05-15 – 05-16 | 2026-W20 | Course paper | Course paper / Overleaf submission | `paper.tex` edits; permutation-importance table; Overleaf merge | git 2026-05-15…05-16 | journal |

---

## May 31 – Jun 10 (thesis prep + go-live)

| Date | ISO week | Phase | Event | Decision / outcome | Source | Feeds |
|------|----------|-------|-------|--------------------|--------|-------|
| 2026-05-31 | 2026-W22 | Thesis framing | Thesis plan + literature + `.env` MLflow fix | Pivot from course paper to thesis cadence experiment; thesis template | git 2026-05-31; `thesis.md` | journal; method (Ch 3) |
| 2026-06-01 | 2026-W23 | Data | Fix missing AFCON 2026 matches in ingestion | Holdout / Gold completeness for continental tournaments | git 2026-06-01 | journal; method (Ch 3) |
| 2026-06-02 | 2026-W23 | Phase 1 | Model-correctness fixes | Monitoring threshold → naive RPS 0.235; NegBin/Bayes eval collapse fixed; mean-rate Poisson baseline added | git 2026-06-02; `thesis.md` 1a–1c | journal; method (Ch 3) |
| 2026-06-02 | 2026-W23 | Phase 2 | Thesis feature/data changes (A.1–A.4) | Drop tactical features; rolling Elo-change; expanded continental holdout (~347); time-decay × importance weights; MLflow experiment rename | git 2026-06-02; `decisions.md` | journal; method (Ch 3) |
| 2026-06-02 | 2026-W23 | Docs | Consolidate project docs for thesis | Architecture / feature / modeling notes aligned | git 2026-06-02 | journal |
| 2026-06-03 – 06-04 | 2026-W23 | Phase 2 | A.5 refit + A.6 half-period tuning | Marginal Optuna on `half_period_years`; ridge → 4.803 yr; others stay 3.0; no rerun | `decisions.md` A.6; `results_pre_wc.md`; git 2026-06-04 | journal; method (Ch 3); results §5.1 |
| 2026-06-05 | 2026-W23 | Phase 2 | A.7 full refit with tuned half-periods | Selection run; `HOLDOUT_RPS_BASELINES` frozen for WC monitoring | `thesis.md` A.7; `results_pre_wc.md` | journal; results §5.1 |
| 2026-06-06 | 2026-W23 | Phase 2 | A.8 roster lock + shadow URI fix | Live roster: xgboost, poisson_glm, bayesian_poisson + mean_rate; MLflow 3.x `models:/` load fix | git 2026-06-06; `decisions.md` champion selection | journal; method (Ch 3) |
| 2026-06-07 | 2026-W23 | Phase 2/3 | A.9 Option A multi-model simulation | `EXPERIMENT_MODELS`; long-format tournament artifacts; dashboard champion filter | git 2026-06-07; `thesis.md` A.9 | journal; method (Ch 3); timeline (Ch 4) |
| 2026-06-07 – 06-08 | 2026-W24 | Go-live | GCP skeleton C.1–C.6 ‖ B.1–B.4 | Container, SA, secrets, Artifact Registry, Cloud Run Job, daily scheduler; dual-mode tags/aliases/refit/monitor | `golive_runbook.md`; `thesis.md` B.*, C.* | journal; timeline (Ch 4) |
| 2026-06-08 | 2026-W24 | Go-live | Harden live pipeline | FIFA h2h tiebreakers; Bayesian NUTS hardening; shadow timeouts; drop lstm/cnn from live shadows; collapsed-raw guard; `wc2026.json` in image | git 2026-06-08 | journal; timeline (Ch 4); incident (§5.5) |
| 2026-06-09 | 2026-W24 | Go-live | C.9 dress rehearsal + entrypoint fix | Dual-mode on GCP; 4 models simulated; Gold 6921; DVC push OK; `git reset --mixed` pointer materialization fix | `wc_live.md` pre-tournament; `golive_runbook.md`; git 2026-06-09 | journal; timeline (Ch 4); incident (§5.5) |
| 2026-06-09 | 2026-W24 | Go-live | Public repo publish | README rewrite; `.env.example`; untrack copyrighted binaries; GitHub default → `thesis` | `golive_runbook.md`; git 2026-06-09 | journal |
| 2026-06-09 | 2026-W24 | Go-live | C.7 DVC→GCS deferred | Keep DagsHub remote for tournament; revisit post-WC | `thesis.md` C.7; `golive_runbook.md` | journal; method (Ch 3) |
| 2026-06-10 | 2026-W24 | Go-live | Every-2h cadence test on friendlies | ~50 min run vs 120 min interval OK; revert to daily; shared MC seed across cadence modes | `wc_live.md`; git 2026-06-10 | journal; timeline (Ch 4) |
| 2026-06-10 | 2026-W24 | Phase 2 | Feature importance (3 champions) | Pre-WC importance snapshot for §5.1 | `results_pre_wc.md` 2026-06-10 | journal; results §5.1 |
| 2026-06-11 (pre-kickoff) | 2026-W24 | Go-live | A.10 freeze + final C.9 | Both aliases → frozen snapshot (xgboost v15); Gold 6945; monitoring empty pre-WC; WC scheduler enabled | `wc_live.md` pre-tournament; `results_pre_wc.md` aliases | journal; method (Ch 3); timeline (Ch 4) |

---

## WC 2026 live (Jun 11 – Jul 19)

| Date | ISO week | Phase | Event | Decision / outcome | Source | Feeds |
|------|----------|-------|-------|--------------------|--------|-------|
| 2026-06-11 | 2026-W24 | MD1 | Kickoff + DagsHub outage on go-live day | Symptom: Git/artifact HTTP 500 ~09:46–~19:5x UTC; first C.9 crash at `git push`. Root: DagsHub maintenance. Fix: re-run C.9 after recovery. TTR: same day. Predictions affected: **N** (freeze intact, Gold 6945). | `wc_live.md` MD1 | journal; incident (§5.5) |
| 2026-06-11 – 06-12 | 2026-W24 | MD1 | MD1 group stage (24 matches) | Both modes scored; pre-kickoff C.9; champion_frozen `5f5a313a…` (v15) | `wc_live.md` MD1 | journal; results (Ch 5) |
| 2026-06-12 | 2026-W24 | MD1 | Duplicate fixture_id crash after Match 1 | Symptom: concat duplicate Gold stub. Root: `wc_results_to_gold_rows` + nightly Gold. Fix: dedup in `run.py` + `temporal_features.py` (image `20260612a`). TTR: hours. Predictions affected: **first monitoring cycle delayed to after Match 2**. | `wc_live.md` MD1; git `b8f9c5a6` | journal; incident (§5.5) |
| 2026-06-12 | 2026-W24 | MD1 | Premature per_round refit (v16) | Symptom: refit after Match 1 on 6946 rows. Root: missing `per_round_refit_matchday` tag + `next_matchday` advances early. Outcome: 8th surplus refit; treat MD1 as frozen≡per_round for analysis. Predictions: **artefact, not designed cadence**. | `wc_live.md` MD1; Final refit table | journal; incident (§5.5); method (Ch 3) |
| 2026-06-14 | 2026-W24 | MD1 | Refit gate fix + monitoring speedup | Gate → `per_round_last_completed_matchday` (full round); monitoring skip-if-unchanged (image `20260614a/b`) | `wc_live.md` MD1; git `44dbd481`, `b814beaf` | journal; timeline (Ch 4) |
| 2026-06-17 | 2026-W25 | MD1/MD2 | Silent skip: frozen `mean_rate_poisson` row (POR–COD) | Symptom: 830/832 hole. Root: `_safe_shadow_predict` 120 s timeout. Fix: **none mid-flight**; Strand 3 copy post-WC. Predictions affected: **N** for RQ1 team-aware; completeness only. | `wc_live.md` MD2; `errors_overview.md` §2 | journal; incident (§5.5) |
| 2026-06-18 | 2026-W25 | MD2 | First correctly-gated MD1→MD2 refit | `champion_per_round` `c522bd72…` (v17); Gold 6969; intended cadence divergence begins | `wc_live.md` MD2 | journal; results (Ch 5); method (Ch 3) |
| 2026-06-18 – 06-24 | 2026-W25 | MD2 | MD2 group stage (24 matches) | Chalk slate; xgboost per_round leads MD2 RPS (~5% vs frozen) | `wc_live.md` MD2 | journal; results (Ch 5) |
| ~2026-06-18 | 2026-W25 | MD2 | Shadow-resolution bug discovered (analysis) | Symptom: frozen≡per_round for roster shadows. Root: `wc_shadow` no cadence alias / tag. Decision: **do not patch mid-tournament**; fix offline (D.1). As-logged clean contrast = xgboost only. | `wc_live.md` MD2; `errors_overview.md` §1 | journal; incident (§5.5); method (Ch 3) |
| 2026-06-22 | 2026-W26 | MD2 | `mean_rate_poisson` breaches naive RPS floor | Rolling-24 RPS 0.2398 > 0.235; structural (λ_h≡λ_a); alert designed correctly | `wc_live.md` Final; `errors_overview.md` §4 | journal; results (Ch 5) |
| 2026-06-24 – 06-28 | 2026-W26 | MD3 | MD3 group stage (24 matches) | Refit edge ~1%; champion not round leader; no clear third-place draw-gaming | `wc_live.md` MD3 | journal; results (Ch 5) |
| 2026-06-27 | 2026-W26 | MD3 | Silent skips: per_round `ridge` ×2 (COL–POR, COD–UZB) | Same timeout class; Strand 3 copies from frozen twins later | `wc_live.md` MD3; `errors_overview.md` §2 | journal; incident (§5.5) |
| 2026-06-27 | 2026-W26 | MD3 | Host-advantage scramble in KO simulation | Symptom: host KO rates swapped in `simulate_tournament`. Fix: orient without swap; host-vs-host dual predictions (commit `01942e41`). Affects RQ2 artifacts from Jul 7 path; RQ1 clean. Predictions (match λ): **N**. | `wc_live.md` MD3; git 2026-06-27 | journal; incident (§5.5); method (Ch 3) |
| 2026-06-28 | 2026-W26 | MD3→R32 | MD3-boundary refit | `wc_production` v19 `4dbbaec7…` feeds R32 | `wc_live.md` MD3 | journal; results (Ch 5) |
| 2026-06-28 – 07-04 | 2026-W27 | R32 | Round of 32 (16 matches) | Very chalky; per_round RPS edge ~5% (xgboost); RF leads round | `wc_live.md` R32 | journal; results (Ch 5) |
| 2026-06-29 | 2026-W27 | R32 | KO results never locked + refits mislabeled | Symptom: empty `ko_results`; every KO refit tagged "R32". Root: API round labels lack numeric suffix. Fix: team-set keys (image `20260629a`). Predictions/Gold: **N**; RQ2 locking: **Y until fix**. | `wc_live.md` R32; git 2026-06-29 | journal; incident (§5.5) |
| 2026-06-29 – 06-30 | 2026-W27 | R32 | Scheduler left paused after deploy | Symptom: `wc-pipeline-trigger` paused Jun 29 09:23 → Jun 30 07:32 (**22.01 h**); 3 R32 cycles missed. Root: ops (wrong job name on resume attempt). Fix: resume + catch-up. Predictions: **N** (RQ1 fallback OK). Cost: 3 RQ2 entropy snapshots. | `wc_live.md` R32; `errors_overview.md` §3 | journal; incident (§5.5) |
| 2026-06-30 | 2026-W27 | R32 | Silent skip: frozen `random_forest` (CIV–NOR) | Fourth completeness hole; Strand 3 copy later | `wc_live.md` R32 | journal; incident (§5.5) |
| 2026-07-04 | 2026-W27 | R32→R16 | R32-boundary refit | `wc_production` v20 `5e404f0c…` feeds R16 | `wc_live.md` R32 | journal; results (Ch 5) |
| 2026-07-04 – 07-07 | 2026-W27 | R16 | Round of 16 (8 matches) | Per_round **worse** on all metrics this round (~−3.6% RPS); all three hosts out; still leads cumulative | `wc_live.md` R16 | journal; results (Ch 5) |
| 2026-07-05 – 07-06 | 2026-W28 | R16 | IPv6 ELO freshness stall | Symptom: every run killed at 90 min timeout; ≥**25.8 h** (last inference Jul 5 06:14 → next after Jul 6 08:00). Root: `eloratings.net` AAAA + Cloud Run no IPv6; 30 s connect stall × ~244 slugs. Fix: force `AF_INET` (image `20260706a`). Predictions: **N**. Cost: 1 RQ2 snapshot (BRA–NOR ↔ MEX–ENG). | `wc_live.md` R16; `errors_overview.md` §3; git 2026-07-06 | journal; incident (§5.5) |
| 2026-07-08 | 2026-W28 | R16→QF | R16-boundary refit | `wc_production` v21 `5032d46f…` feeds QF | `wc_live.md` R16 | journal; results (Ch 5) |
| 2026-07-09 – 07-12 | 2026-W28 | QF | Quarter-finals (4 matches) | Large per_round RPS edge on small n; QF app logs aged out of Cloud Logging before export | `wc_live.md` QF; `errors_overview.md` §7 | journal; results (Ch 5); method (Ch 3) |
| 2026-07-12 02:05 | 2026-W28 | QF | `dvc push` timeout crash | Symptom: `CalledProcessError` at push; **no inference logged** this run. Root: transient DagsHub. Fix: next retry (`8df3d581` ~02:15). TTR: ~10 min. Predictions: **N**. | `wc_live.md` QF; `errors_overview.md` §3 | journal; incident (§5.5) |
| 2026-07-12 10:51 | 2026-W28 | QF | `git push` non-fast-forward crash | Symptom: died **mid-cycle** after frozen, before per_round → 8 vs 7 Jul 12 asymmetry; orphaned DVC blobs. Root: remote moved. Fix: next cycle hard reset (`7254a5f` ~11:10). TTR: ~19 min. Predictions: **N** (completeness). | `wc_live.md` QF; `errors_overview.md` §3 | journal; incident (§5.5) |
| 2026-07-12 | 2026-W28 | QF | Pen-winner locking bug | Symptom: ET/PEN winners via away-bias tie-break. Fix: `teams.winner` / `score.penalty` (image `20260712a`). Luck: prior picks happened to be correct. | `wc_live.md` QF; git 2026-07-12 | journal; incident (§5.5) |
| 2026-07-12 04:06 | 2026-W28 | QF→SF | QF-boundary refit | `wc_production` v22; shadows v113–v115 | `wc_live.md` QF | journal; results (Ch 5) |
| 2026-07-13 – 07-14 | 2026-W29 | SF | Silent skips that self-healed | `negbin_glm`, `ridge`, `random_forest` timeouts; successful cycle before kickoff → **no artifact holes**; backfill list stays at 4 | `wc_live.md` SF; `errors_overview.md` §2 | journal; incident (§5.5) |
| 2026-07-14 – 07-15 | 2026-W29 | SF | Semi-finals (2 matches) | Per_round wins all three metrics narrowly; RF leads round again | `wc_live.md` SF | journal; results (Ch 5) |
| 2026-07-15 22:06 | 2026-W29 | SF→Final | SF-boundary refit (terminal) | `wc_production` v23; shadows v116–v118; last refit of tournament | `wc_live.md` SF | journal; results (Ch 5) |
| 2026-07-18 – 07-19 | 2026-W29 | Final | Third place + Final | Both mispredicted at lowest confidence; xgboost per_round tournament RPS **0.1534** vs frozen **0.1567** (~2.1%) | `wc_live.md` Final | journal; results (Ch 5) |
| 2026-07-20 | 2026-W30 | End | Pipeline retired | Tag `wc2026-end-of-tournament`; scheduler paused; 104/104 scored both modes; 830/832 as-logged | `wc_live.md` summary | journal; timeline (Ch 4) |

### Incident lift table (for §5.5)

Compact view of operational incidents (self-healed or deferred). Reconstruction defects are in the post-WC section.

| Date | Symptom | Root cause | Fix | TTR | Predictions affected? |
|------|---------|------------|-----|-----|----------------------|
| Jun 11 | DagsHub 500 / push fail | Provider maintenance | Re-run C.9 | same day | N |
| Jun 12 | Duplicate fixture crash | Gold stub + nightly row | Dedup guards `20260612a` | hours | Delayed monitoring only |
| Jun 12 | Premature MD1 refit | Tag + next_matchday logic | Gate rewrite `20260614a` | 2 days | Artefact (treat as no-op) |
| Jun 17–30 | 4 silent shadow holes | 120 s child timeout | Document; Strand 3 copy | post-WC | Completeness only |
| Jun 27 | Host KO rate scramble | Swap in sim | Orient fix `01942e41` | same day | Match λ N; RQ2 Y |
| Jun 29 | KO unlock / mislabel | API round labels | Team-set keys `20260629a` | same day | RQ2 locking until fix |
| Jun 29–30 | 3 missed R32 cycles | Scheduler left paused | Resume + catch-up | **22.01 h** | N (RQ1); RQ2 snapshots Y |
| Jul 5–6 | All runs timeout | IPv6 AAAA stall | Force IPv4 `20260706a` | **≥25.8 h** | N; 1 RQ2 snapshot Y |
| Jul 12 02:05 | No inference logged | `dvc push` timeout | Retry | ~10 min | N |
| Jul 12 10:51 | Mid-cycle kill; 8 vs 7 | `git push` non-FF | Retry | ~19 min | N |
| Jul 12 | PEN winner bias | Away-bias tie-break | Winner/penalty fields `20260712a` | same day | Prior luck OK |
| Ongoing | mean_rate > 0.235 floor | λ_h≡λ_a always | Document (§4) | — | Alert only |

---

## Post-WC analysis (Jul 20 – Sep 13)

| Date | ISO week | Phase | Event | Decision / outcome | Source | Feeds |
|------|----------|-------|-------|--------------------|--------|-------|
| 2026-08-10 | 2026-W33 | D.1 prep | Gold/DVC history integrity verified | 104/104 matches have pre-kickoff Gold; remote complete; safe to replay | `errors_overview.md` §5 | journal; method (Ch 3) |
| 2026-08-10 | 2026-W33 | D.1 | Cadence-aware shadow resolution + reconstruction drivers | Code fix for §1; D.1 strand scaffolding | git 2026-08-10 | journal; method (Ch 3); timeline (Ch 4) |
| 2026-08-11 | 2026-W33 | D.1 | Replay orientation + settle guards | Home-oriented λ; settled-only locks; shadow timeout 300 s for replay | git 2026-08-11 | journal; method (Ch 3) |
| 2026-08-15 | 2026-W33 | D.1 | Strand 1–3 first pass + RQ datasets | Frozen shadows rebuilt (poisson_glm, bayesian_poisson); bracket replay; 4-row backfill → 832/832; merge to `thesis` | git 2026-08-15; `errors_overview.md` §1–§2 | journal; results (Ch 5); method (Ch 3) |
| 2026-08-15 | 2026-W33 | D.1 repair | §2.1 Strand 3 manufactured effect | Re-predict of never-refit `ridge` → spurious ~0.0003 RPS cadence delta. Fix: **copy-not-repredict** + `s3.all_copied` audit. | `errors_overview.md` §2.1 | journal; method (Ch 3); incident (§5.5) |
| 2026-08-15 | 2026-W33 | D.1 repair | §2.2 Strand 1 own-result leakage | Missing `SETTLE_DELTA` reversed RQ1. Fix: settle-delta + batch features; opener λ matches live. | `errors_overview.md` §2.2 | journal; method (Ch 3); incident (§5.5) |
| 2026-08-17 | 2026-W34 | Docs | Archive pre-WC presentation | Move under `docs/archive/presentation/` | git 2026-08-17 | journal |
| 2026-09-03 | 2026-W36 | D.1 | Offline Strand 2 + mark Strand 4 superseded | Freeze MLflow inputs to DVC; entropy/calibration figures added | git 2026-09-03 | journal; results (Ch 5) |
| 2026-09-04 | 2026-W36 | D.1 repair | §2.3 settlement guess (2 h) | `SETTLE_DELTA` admitted unfinished matches; Bronze snapshot-first settlement. RQ1 unchanged; RQ2 corrected. `s5.champion_identity` 7 fails → 0. | `errors_overview.md` §2.3; git 2026-09-04 | journal; method (Ch 3); incident (§5.5) |
| 2026-09-04 | 2026-W36 | D.1 | Strand 5 cadence-correct frozen entropy | Re-predict all frozen cycles from pre-tournament pin; regime-pinned gap snapshots (supersedes Strand 4 future-model fill §2.4) | `thesis.md` D.1; `errors_overview.md` §2.4; git 2026-09-04 | journal; results RQ2; method (Ch 3) |
| 2026-09-04 | 2026-W36 | D.1 | RQ2 merge + audit | 3,412 entropy rows; provenance column; audit 67 pass / 10 warn / 0 fail | `thesis.md` D.1; `errors_overview.md` §2 | journal; results RQ2 |
| 2026-09-04 | 2026-W36 | Finding | Train/serve skew on rest features | `days_since_last_match` / `rest_diff` ≈ 0 at live inference (batch shape). Decision: **document, do not re-serve**. Uniform → RQs unaffected. | `errors_overview.md` §8 | journal; method (Ch 3); discussion (Ch 6) |
| 2026-09-04 | 2026-W36 | Decision | Do not fix silent-skip / push hardening | Pipeline retires; document in Ch 3/6 (`thesis_prose.md`) | `errors_overview.md` §6; `wc_live.md` decisions | journal; method (Ch 3) |
| 2026-09-05 | 2026-W36 | D.1 | Expose reconstruction as DVC stages | `dvc pull && dvc repro figures` path; track analysis inputs | git 2026-09-05; `dvc.yaml` | journal; timeline (Ch 4); method (Ch 3) |
| 2026-09-13 | 2026-W37 | D.1 | Fresh-clone reproducibility verified | Packaging + docs; Strand 5 recorded; Strand 4 retired | git 2026-09-13; `thesis.md` D.1 STATUS | journal; method (Ch 3) |

### RQ2 entropy headline (from D.1, for findings memo)

Frozen minus per_round `entropy_winner` (nats), live cycles, n=422 paired: xgboost +0.0552 · bayesian_poisson +0.0244 · poisson_glm +0.0080 · mean_rate +0.0004. Sign consistent in the mean; magnitude model-specific; every model flips sign somewhere. Source: `thesis.md` D.1.

---

## Source gaps

Periods with little or no contemporaneous evidence — be explicit in the journal that these are thin or reconstructed:

| Period | ISO weeks | Evidence | Journal note |
|--------|-----------|----------|--------------|
| Mar 9–19 | W11 | Gantt only (pre-git) | Coarse reconstruction |
| Apr 17–26 | W16–W17 | Few commits between MC sim and Gold v2 | Thin |
| May 6–14 | W19–W20 | Between Gold v2 docs and paper edits | Thin |
| May 17–30 | W21–W22 | Between course paper and thesis pivot | Thin |
| Jul 21 – Aug 9 | W30–W32 | 6 auto-update commits Jul 20 only; then silence until Aug 10 | Post-tournament pause |
| Aug 18 – Sep 2 | W34–W35 | One archive commit Aug 17; then silence until Sep 3 | Analysis pause |
| Sep 6–12 | W36–W37 | Between DVC stages (Sep 5) and fresh-clone docs (Sep 13) | Thin |

Auto-update commits (`data: auto-update pipeline …`) during WC are omitted from event rows; they are evidence of cadence health, not decisions.

---

## Cross-check checklist

Against `errors_overview.md` §2–§3 operational + result defects:

- [x] Shadow-resolution bug (§1)
- [x] Four artifact holes + Strand 1/2/3/5 (§2)
- [x] Strand 3 manufactured effect (§2.1)
- [x] Strand 1 leakage (§2.2)
- [x] Settlement 2 h guess (§2.3)
- [x] Strand 4 future-model fill (§2.4)
- [x] Host orientation live-vs-replay (§2.5)
- [x] Scheduler pause Jun 29–30 (§3)
- [x] IPv6 stall Jul 5–6 (§3)
- [x] `dvc push` Jul 12 02:05 (§3)
- [x] `git push` Jul 12 10:51 (§3)
- [x] mean_rate floor (§4)
- [x] Gold integrity Aug 10 (§5)
- [x] Do-not-fix decisions (§6)
- [x] Observability limits (§7)
- [x] Train/serve skew (§8)

Against `thesis.md` dated checkboxes:

- [x] Phase 1a–1c (2026-06-02)
- [x] A.1–A.4 (2026-06-02)
- [x] A.5/A.6 (2026-06-03/04)
- [x] A.7 (2026-06-05)
- [x] A.8 (2026-06-06)
- [x] A.9 (2026-06-07)
- [x] B.1–B.4 + C.1–C.6 / C.9 (Jun 7–11)
- [x] D.1 done Sep 4 / repro Sep 13
- [ ] D.2, D.3, D.4, D.5 still open (correctly absent as completed events)
