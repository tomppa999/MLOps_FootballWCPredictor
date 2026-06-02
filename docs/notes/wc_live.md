# WC 2026 Live Log

Daily notes during the tournament (June 11 – July 19, 2026).
Maps to thesis Chapter 5 (RQ1–RQ3 results) and Chapter 3 (threats to validity).

Format: one entry per meaningful event (pipeline run, refit, failure, anomaly).
Granularity: don't log every 30-min inference cycle — log deviations, failures,
and end-of-matchday summaries.

---

## Pre-tournament deployment (Jun 1–10)

- [date]: GCP test run 1 — result:
- [date]: GCP test run 2 — result:
- [date]: GCP test run 3 — result:
- [date]: Both MLflow aliases resolve (`champion_frozen`, `champion_per_round`)? Y/N
- [date]: DVC push from container succeeded? Y/N
- [date]: Monitoring runs appear empty pre-WC? Y/N

---

## Matchday 1 — Group Stage Round 1 (Jun 11–12)

Matches played: [fill]

Pipeline:
- Both modes logged inference artifacts? Y/N
- `champion_frozen` run_id used:
- `champion_per_round` refit fired at MD1 boundary? Y/N
  - New `champion_per_round` run_id after refit:
  - Gold rows at refit time:
- Any failures or anomalies:

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
