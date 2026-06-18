# WC 2026 Live Log

Daily notes during the tournament (June 11 – July 19, 2026).
Maps to thesis Chapter 5 (RQ1–RQ3 results) and Chapter 3 (threats to validity).

Format: one entry per meaningful event (pipeline run, refit, failure, anomaly).
Granularity: don't log every 30-min inference cycle — log deviations, failures,
and end-of-matchday summaries.

---

## Pre-tournament deployment (Jun 1–10)

- Jun 9: GCP dress rehearsal (C.9 first pass) — dual-mode dispatch ran (frozen + per_round), metadata tags landed, `predictions_all_models.csv` covered all 4 models, DVC push succeeded. Clean run = gold 6921 rows.
- Jun 10: every-2h cadence test on live friendly data — validated scheduler mechanics (~50-min run vs 120-min interval), reverted to daily afterwards.
- Jun 11: final C.9 pass (`inference_only`) — both cadence modes ran, both artifact sets written.
- Jun 11: Both MLflow aliases resolve (`champion_frozen`, `champion_per_round`)? **Y** — both point at the frozen snapshot (display champion xgboost).
- Jun 11: DVC push from container succeeded? **Y** (verified Jun 9; Jun 11 local push also succeeded post-DagsHub-recovery).
- Jun 11: Monitoring runs appear empty pre-WC? **Y** — `No settled WC 2026 matches yet — monitoring no-op`.

---

## Matchday 1 — Group Stage Round 1 (Jun 11–12)

Matches played: 24

Pipeline:
- Both modes logged inference artifacts? **Y** (pre-kickoff C.9 pass; Gold 6945 rows).
- `champion_frozen` run_id used: `5f5a313ad5c14199aef0a791d2e4041a` (v15, xgboost, gold_row_count=6945; `4f6ce4f0808b43fcb68d4acf10c5f1a8` was the inference cycle run_id, not the champion model run_id) (per_round inference: `5272ce72261043228b1455b0a06ed5cb`).
- `champion_per_round` refit fired at MD1 boundary? **Y** — fired Jun 18 after all 24 MD1 fixtures reached FT. Image `20260618b` with the three `parse_wc_results` fixes enabled the gate to see `last_completed_matchday = "1"` (previously stuck at "0").
  - New `champion_per_round` run_id after refit: `c522bd72cd274889829d06f10165b76e`
  - Gold rows at refit time: 6969
- Any failures or anomalies: **DagsHub maintenance outage on go-live day** (~09:46–~19:5x UTC). Git + artifact service down: crashed the first C.9 `inference_only` run at `git push`, then blocked all artifact reads/writes (HTTP 500 on `artifacts/list` and download, for every run). Re-ran C.9 successfully after git recovered; dashboard/artifact access restored once the `mlflow-artifacts` proxy came back. Freeze stayed intact throughout (Gold unchanged at 6945 rows, run metadata healthy, aliases correct).
- **Jun 12: post-Match-1 inference cycle crashed** — duplicate `fixture_id` when `wc_results_to_gold_rows` returned Match 1's result as a stub while the nightly pipeline had already landed it in Gold. `pd.concat` produced two rows with the same `fixture_id`; `add_days_since_last_match` then set a non-unique index, corrupting feature output. Fixed in commit `b8f9c5a6` (deduplication guard in `run.py` + defensive `drop_duplicates` in `temporal_features.py`), deployed as image `20260612a`. First successful monitoring cycle ran after Match 2. Post-fix: pipeline auto-commits resumed normally (multiple `data: auto-update pipeline 2026-06-12` commits confirm stable hourly cycles).
- **Jun 12: `champion_per_round` refitted prematurely (v16 / local v8)** — two compounding causes: (1) the frozen champion run (v15/v7) had no `per_round_refit_matchday` tag, so `_last_per_round_refit_matchday()` defaulted to `"0"`; (2) `next_matchday` advances to `"2"` as soon as the *first* group-stage match settles (`max_group_matchday + 1` logic in `features.py`), not after all 24 MD1 games are done (12 groups × 2). Combined: condition `"2" != "0"` fired on the first `auto`-mode pipeline run after Match 1 settled, retraining on only 6946 Gold rows (+1 vs frozen). **For MD1 analysis: use `champion_frozen` predictions for both cadence modes** — v16 is a premature artefact and carries no meaningful new information.
- **Jun 14: fix deployed (image `20260614a`)** — gate switched to a new `per_round_last_completed_matchday` tag (keyed on a fully-completed round, not `next_matchday`). The new signal is derived from `parse_wc_results` counting scheduled vs finished fixtures per round; the refit fires only when `finished >= scheduled > 0` for a full round, propagating in order MD1 → MD2 → MD3 → R32 → R16 → QF → SF → Final. v16's stale `per_round_refit_matchday = "2"` tag is irrelevant to the new gate (it reads a different tag, defaulting to `"0"` for v16), so the next correct refit will fire as soon as all 24 MD1 fixtures are settled. No double-fire: a completed round is tagged on the new model version and skipped on subsequent ticks. See commit `44dbd481`.
- **Jun 14: monitoring efficiency improvement (image `20260614b`)** — monitoring now skips MLflow re-logging when the settled match count is unchanged since the last cycle (gate queries `n_scored_matches` from the most recent monitoring run; fails safe to "log anyway" if DagsHub is unreachable), and per-match metric writes are batched into a single `log_batch` call instead of one HTTP request per metric. Reduces the monitoring phase from ~12 min to <1 s on no-change ticks. See commit `b814beaf`.

Observations:

MD1 leaderboard — all 24 matches, `per_round` cadence (frozen identical):

| Model             | Mean RPS ↑ | Mean RMSE | Mean NLL |
|-------------------|------------|-----------|----------|
| ridge             | **0.2074** | 0.9218    | 3.027    |
| xgboost           | 0.2077     | 0.9812    | 3.111    |
| random_forest     | 0.2089     | 0.9789    | 3.089    |
| poisson_glm       | 0.2131     | 0.9576    | 3.061    |
| bayesian_poisson  | 0.2142     | 0.9625    | 3.063    |
| negbin_glm        | 0.2170     | 0.9699    | 3.074    |
| **mean_rate (floor)** | 0.2181 | 0.9901 | 3.253    |
| sarimax           | 0.2208     | 0.9530    | 3.034    |

- No alert window breach — all 7 team-aware models finished below the naive floor (0.218). First 24-match alert window now active.
- RPS spread is narrow (ridge 0.207 to sarimax 0.221); ridge leads on RPS but has the best RMSE too. Sarimax trails on RPS despite a low NLL — the draw penalty lands harder for it.
- Worst match: CIV 1–0 ECU — models had Ivory Coast as heavy underdogs (p_home ~0.13); sarimax RPS 0.663, negbin 0.596. GHA 1–0 PAN similar (RF/negbin ~0.54–0.55). Both genuine upsets for the WC format.
- Best match: GER 7–1 CUR — negbin RPS 0.008, poisson_glm 0.012; correctly priced dominant favorite at p_home ~0.85–0.88.
- SARIMAX anomaly: λ_away = 1e-06 for ESP vs CPV — degenerate near-zero rate (numerically clipped). No crash but flagged as reliability concern for high-asymmetry fixtures.
- Frozen and per_round predictions identical throughout all of MD1. Divergence begins MD2 (per_round refit fired Jun 18, new `champion_per_round` run_id `c522bd72...`).

---

## Matchday 2 — Group Stage Round 2 (~Jun 17–18)

Matches played: [fill]

Pipeline:
- Both modes logged? **Y** — frozen @ 14:01 UTC, per_round @ 14:09 UTC; Gold 6969 rows, 24 MD1 matches settled.
- per_round refit fired? N/A — MD2 boundary refit fires only after all 24 MD2 fixtures settle. Current: 0/24.
- Any failures: None. First clean dual-mode run since `20260618b` deployment.

Observations:
-

---

## Matchday 3 — Group Stage Round 3 (~Jun 23–25)

Matches played: [fill]

Pipeline:
- Both modes logged? Y/N
- per_round refit fired? Y/N. New run_id:
- AFCON/format effect: unusual draw frequency on matchday 3 (gaming third-place
  qualification math)? Note if observed.
- Any failures:

Observations:
-

---

## Round of 32 (~Jun 27 – Jul 2)

Matches played: [fill]

Pipeline:
- per_round refit fired at R32 boundary? Y/N. New run_id:
- Bracket configuration: which 8 third-placers advanced? [fill] — note if any
  unusual configuration affects the bracket unpredictably.
- Any failures:

Observations:
-

---

## Round of 16 (~Jul 4–7)

Matches played: [fill]

Pipeline:
- per_round refit fired? Y/N. New run_id:
- Any failures:

Observations:
-

---

## Quarter-finals (~Jul 10–11)

Matches played: [fill]

Pipeline:
- per_round refit fired? Y/N. New run_id:

Observations:
-

---

## Semi-finals (~Jul 14–15)

Matches played: [fill]

Pipeline:
- per_round refit fired? Y/N. New run_id:

Observations:
-

---

## Final + Third-place (~Jul 18–19)

Pipeline:
- Total refit events: [fill] (expected: 7)
- Total inference cycles logged: [fill]
- Total monitoring rows: [fill]
- Any unresolved failures:

---

## Post-tournament summary

Overall pipeline reliability: [fill]
Biggest failure / surprise: [fill]
Data completeness: [fill]% of WC matches have settled scores in Bronze
Both modes produced complete snapshot sets? Y/N
