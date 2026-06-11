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

Matches played: [fill]

Pipeline:
- Both modes logged inference artifacts? **Y** (pre-kickoff C.9 pass; Gold 6945 rows).
- `champion_frozen` run_id used: `4f6ce4f0808b43fcb68d4acf10c5f1a8` (per_round inference: `5272ce72261043228b1455b0a06ed5cb`).
- `champion_per_round` refit fired at MD1 boundary? [fill — fires on first settled results, 0 → 1 boundary]
  - New `champion_per_round` run_id after refit: [fill]
  - Gold rows at refit time: [fill]
- Any failures or anomalies: **DagsHub maintenance outage on go-live day** (~09:46–~19:5x UTC). Git + artifact service down: crashed the first C.9 `inference_only` run at `git push`, then blocked all artifact reads/writes (HTTP 500 on `artifacts/list` and download, for every run). Re-ran C.9 successfully after git recovered; dashboard/artifact access restored once the `mlflow-artifacts` proxy came back. Freeze stayed intact throughout (Gold unchanged at 6945 rows, run metadata healthy, aliases correct).

Observations:
-

---

## Matchday 2 — Group Stage Round 2 (~Jun 17–18)

Matches played: [fill]

Pipeline:
- Both modes logged? Y/N
- per_round refit fired? Y/N. New run_id:
- Any failures:

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
