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

MD1 leaderboard — all 24 matches. `xgboost` shows both cadences (true frozen via `champion_frozen` alias; per-round carries the premature v16 residue). `poisson_glm` / `bayesian_poisson` now show both cadences too: **frozen** rows are the D.1 reconstructed true-frozen shadows; **per-round** rows are the live premature-refit artifact. Never-refit shadows remain cadence-invariant:

| Model                       | Mean RPS ↑ | Mean RMSE | Mean NLL |
|-----------------------------|------------|-----------|----------|
| ridge                       | **0.2074** | 0.9218    | 3.027    |
| xgboost (frozen)            | 0.2075     | 0.9796    | 3.108    |
| xgboost (per-round)         | 0.2077     | 0.9812    | 3.111    |
| random_forest               | 0.2089     | 0.9789    | 3.089    |
| poisson_glm (frozen)        | 0.2105     | 0.9393    | 3.011    |
| bayesian_poisson (frozen)   | 0.2115     | 0.9429    | 3.012    |
| poisson_glm (per-round)     | 0.2131     | 0.9576    | 3.061    |
| bayesian_poisson (per-round) | 0.2142     | 0.9625    | 3.063    |
| negbin_glm                  | 0.2170     | 0.9699    | 3.074    |
| **mean_rate (floor)**       | 0.2181     | 0.9901    | 3.253    |
| sarimax                     | 0.2208     | 0.9530    | 3.034    |

- No alert window breach — all 7 team-aware models finished below the naive floor (0.218). First 24-match alert window now active.
- RPS spread is narrow (ridge 0.207 to sarimax 0.221); ridge leads on RPS but has the best RMSE too. Sarimax trails on RPS despite a low NLL — the draw penalty lands harder for it.
- Worst match: CIV 1–0 ECU — models had Ivory Coast as heavy underdogs (p_home ~0.13); sarimax RPS 0.663, negbin 0.596. GHA 1–0 PAN similar (RF/negbin ~0.54–0.55). Both genuine upsets for the WC format.
- Best match: GER 7–1 CUR — negbin RPS 0.008, poisson_glm 0.012; correctly priced dominant favorite at p_home ~0.85–0.88.
- SARIMAX anomaly: λ_away = 1e-06 for ESP vs CPV — degenerate near-zero rate (numerically clipped). No crash but flagged as reliability concern for high-asymmetry fixtures.
- *Reconstructed offline (D.1 Strand 1): `poisson_glm` / `bayesian_poisson` **frozen** rows from pinned `wc_shadow` v88 / v90 on pre-kickoff Gold — see `data/reconstruction/strand1_frozen_shadow/`. Per-round rows for those models are the live artifact. `mean_rate_poisson` frozen rebuild was dropped by decision (λ spread ~2e-3). True frozen MD1 RPS: `poisson_glm` 0.2105 / `bayesian_poisson` 0.2115 (both beat their live per-round rows of 0.2131 / 0.2142). As-logged, both cadences had loaded the premature-refit shadow for these two models (shadow-resolution bug).
- Divergence (intended) begins MD2 (correctly-gated per_round refit fired Jun 18, `champion_per_round` run_id `c522bd72...`). xgboost also differs across MD1 as a residue of the premature refit, but that divergence is an artefact, not a designed refit.

---

## Matchday 2 — Group Stage Round 2 (Jun 18–24)

Matches played: 24 (Czechia 1–1 South Africa through Colombia 1–0 DR Congo)

Pipeline:
- Both modes logged? **Y** — frozen @ 14:01 UTC, per_round @ 14:09 UTC; Gold 6969 rows, 24 MD1 matches settled.
- per_round refit fired? **N/A for the MD2 boundary** — that refit fires only after all 24 MD2 fixtures settle and feeds MD3. The MD2 *predictions* are backed by the **MD1-boundary** champion refit (`c522bd72cd274889829d06f10165b76e`, fired Jun 18). MD2 is the first matchday where per_round is *intended* to diverge from frozen via a correctly-gated refit (xgboost also diverged across MD1, but as a residue of the premature v16 refit — see below).
- **CAVEAT — frozen vs per_round is only cleanly separable for the champion.** In the artifacts only `xgboost` differs (46/48 matches); all 7 shadows are byte-identical across modes. This is *not* because per_round refits only the champion — the Jun 18 refit fitted all 4 roster models (xgboost → `wc_production` v17; poisson_glm/mean_rate/bayesian → `wc_shadow` v98/v99/v100, run-tagged `cadence_mode=per_round`, log confirms "4/4 models fitted"). The real cause is a **shadow-resolution bug**: `wc_production` separates cadences via dedicated aliases (`champion_frozen` v15 / `champion_per_round` v17), but `wc_shadow` has no alias and frozen shadow refits write **no `cadence_mode` tag**, so `load_shadow_model` resolves *both* cadences to the same newest version. Verified live: `resolve poisson_glm cadence='frozen' → v101` = identical to `per_round`. Consequence for the 3 per_round-refit roster shadows: the logged **"frozen" rows actually carry the per_round-refit shadow artifact**, not the true pre-tournament frozen model. So for MD2 only the champion's frozen↔per_round contrast is real; the **per_round side is correct for every model**, the frozen side is correct only for the champion (alias) and the 4 never-refit shadows (negbin_glm, ridge, random_forest, sarimax — identical across modes by design). **Do not patch the resolver mid-tournament** — it risks corrupting the per_round path, which is the hard one to recreate. Fix offline post-tournament (see Post-tournament). Same contamination applies to MD1's frozen shadow rows (premature per_round shadow refit ~Jun 12), so the MD1 leaderboard's non-champion rows are also "frozen-in-name-only".
- **Note correction (MD1):** the per_round artifact shows `xgboost` already differing from frozen on 22/24 MD1 matches (only Jun 11–12 identical), a residue of the premature v16 refit. The earlier "frozen ≡ per_round throughout MD1" entry is an *analysis convention* (treat v16 as a no-op), not literally true of the logged per_round predictions.
- **Data completeness anomaly (resolved):** the frozen monitoring artifact dropped one `mean_rate_poisson` row — Portugal 1–1 DR Congo (MD1, Jun 17) — leaving frozen mean_rate at 47 rows vs the per_round artifact's 48. Since `mean_rate_poisson` is byte-identical across cadence modes (0/48 rows differ), the frozen overall was back-filled from the per_round row; the table below now reflects the full 48. Root cause is a single dropped model-match row in that cycle's frozen `predictions_all_models.csv` lookup; it never touched the team-aware leaderboard.
- Any failures: None beyond the above. First clean dual-mode run since `20260618b` deployment.

Observations:

MD2 leaderboard — single table (all 48 = cumulative MD1+MD2; MD2 = last 24), sorted by MD2 mean RPS. After D.1, `xgboost`, `poisson_glm` and `bayesian_poisson` each show both cadences (true frozen vs live per_round). Never-refit shadows remain cadence-invariant; `mean_rate_poisson` is cadence-invariant and was not rebuilt.

| Model                       | Overall RPS | MD2 RPS    | Overall RMSE | MD2 RMSE   | Overall NLL | MD2 NLL    |
|-----------------------------|-------------|------------|--------------|------------|-------------|-----------|
| xgboost (per-round)         | **0.1683**  | **0.1290** | 0.9852       | 0.9892     | 2.983       | 2.854     |
| random_forest               | 0.1701      | 0.1313     | 0.9795       | 0.9802     | 2.956       | 2.822     |
| ridge                       | 0.1700      | 0.1327     | 0.9491       | 0.9764     | 2.921       | 2.816     |
| sarimax                     | 0.1769      | 0.1330     | 0.9528       | **0.9527** | **2.902**   | **2.769** |
| xgboost (frozen)            | 0.1717      | 0.1359     | 0.9831       | 0.9867     | 2.980       | 2.851     |
| poisson_glm (per-round)     | 0.1760      | 0.1389     | 0.9886       | 1.0196     | 2.999       | 2.937     |
| bayesian_poisson (per-round) | 0.1771      | 0.1400     | 0.9752       | 0.9880     | 2.967       | 2.871     |
| bayesian_poisson (frozen)   | 0.1783      | 0.1452     | **0.9490**   | 0.9550     | 2.918       | 2.824     |
| poisson_glm (frozen)        | 0.1781      | 0.1457     | 0.9502       | 0.9610     | 2.923       | 2.835     |
| negbin_glm                  | 0.1818      | 0.1467     | 0.9741       | 0.9782     | 2.976       | 2.878     |
| mean_rate (floor)           | 0.2290      | 0.2398     | 1.1256       | 1.2612     | 3.350       | 3.447     |

- **MD2 was a "chalk" matchday.** Every team-aware model's RPS roughly halved vs MD1 (≈0.21 → ≈0.13). Outcome mix was 13 home / 6 away / 5 draws — far fewer upsets than MD1, and favorites delivered (Canada 6–0, Portugal 5–0, Netherlands 5–1, Spain 4–0, France 3–0, Brazil 3–0). Lower RPS reflects an easier slate, not a model improvement per se.
- **The refit paid off, narrowly and only on ranking.** Retraining the champion on the 24 settled MD1 matches moved xgboost from 4th (frozen, among the 8 models) to **1st** on MD2 RPS (0.1359 → 0.1290, ~5% better) and to best overall (0.1717 → 0.1683). But MD2 RMSE (0.9867 → 0.9892) and NLL (2.851 → 2.854) were flat-to-marginally-worse: the refit sharpened outcome calibration without improving the goal-rate point fit. A modest, real win for per_round — on one matchday, on one metric. Live-era caveat: only `xgboost` was a clean contrast in the as-logged artifacts; D.1 now also supplies true frozen rows for `poisson_glm` / `bayesian_poisson`.
- *Reconstructed offline (D.1 Strand 1): `poisson_glm` / `bayesian_poisson` **frozen** rows from pinned `wc_shadow` v88 / v90 on pre-kickoff Gold — see `data/reconstruction/strand1_frozen_shadow/`. Per-round rows for those models are the live artifact. `mean_rate_poisson` frozen rebuild was dropped by decision (λ spread ~2e-3). On MD2 RPS the reconstructed frozen shadows (0.1452 / 0.1457) are *worse* than their live per_round rows (0.1400 / 0.1389) — the MD1-boundary refit helped these two on this slate. Among never-refit shadows, sarimax still leads on NLL (2.769); reconstructed `bayesian_poisson` (frozen) now edges overall RMSE (0.9490 vs ridge 0.9491). `random_forest` remains the best non-champion on MD2 RPS (0.1313), edging frozen xgboost.
- **Alert window (rolling 24 = exactly MD2):** all 7 team-aware models sit well under the 0.235 static naive floor in both modes — no breach. `mean_rate_poisson` itself printed 0.2398 on MD2, *above* the static floor (and above its 0.229 holdout baseline) — the naive floor behaved as designed, and the gap to the team-aware models stayed healthy (~0.10 RPS).
- **Worst matches (mean team-aware RPS):** England 0–0 Ghana (0.402) and Ecuador 0–0 Curaçao (0.350) — two goalless draws where models priced the favorite heavily; USA 2–0 Australia (0.308) and Turkey 0–1 Paraguay (0.301) rounded out the misses. The recurring theme is the unpriced low-scoring draw, same failure mode flagged in MD1.
- **Best matches:** France 3–0 Iraq (0.012), Spain 4–0 Saudi Arabia (0.013), Brazil 3–0 Haiti (0.014) — dominant favorites priced correctly across the board.
- **SARIMAX degenerate-λ recurrence:** λ_away = 1e-06 again, this time Spain vs Saudi Arabia (clipped near-zero rate). It happened to land (Spain won 4–0, sarimax RPS 0.004 — best single score of MD2), but it's the same numerical-clipping reliability concern as MD1's ESP–CPV. Now seen on 2 of 2 high-asymmetry Spain fixtures; track whether it generalizes.

---

## Matchday 3 — Group Stage Round 3 (Jun 24–28)

Matches played: 24 (Switzerland 2–1 Canada through Algeria 3–3 Austria / Jordan 1–3 Argentina, Jun 24–28).

Pipeline:
- Both modes logged? **Y** — all 72 settled matches scored in both artifacts. As-logged: frozen 575 rows, per_round 574 (isolated dropped rows, see below). **After D.1 Strand 3 backfill: 576/576** (72 × 8) per cadence.
- per_round refit fired? **N/A for the MD3 boundary** — the MD3 *predictions* are backed by the **MD2-boundary** champion refit. The MD3-boundary refit (after all 24 MD3 fixtures settle, last kickoff Jun 28) feeds **R32**, not MD3. New run_id: `4dbbaec7d20a4de0a1306b9785f86eeb` (`wc_production` v19, fired Jun 28 04:06 UTC).
- AFCON/format effect: **6 of 24 draws** (Japan 1–1 Sweden, Paraguay 0–0 Australia, Cape Verde 0–0 Saudi Arabia, Egypt 1–1 Iran, Colombia 0–0 Portugal, Algeria 3–3 Austria) — exactly the ~1/4 base rate for 24 games, so **no evidence of final-day draw-gaming / third-place hedging** this round (the three 0–0s notwithstanding). Outcome mix 9 home / 9 away / 6 draw, more balanced than MD2's 13/6/5.
- **Any failures: `ridge` dropped from one per_round inference cycle (data-completeness anomaly).** The per_round `ridge` row is missing for Colombia 0–0 Portugal (`1489419`) and DR Congo 3–1 Uzbekistan (`1539013`), both Jun 27, both served by the single per_round inference run `ed64fbfac5ed487ca222cf0a9d2c938d`. Every other model is present for those matches, and the frozen run for the same fixtures (`0facdde2…`) has `ridge` fine — so `ridge` itself is healthy (present for the other 70 per_round matches). Root cause: `run_prediction_all_models` runs each shadow in an isolated child process with a 120 s timeout (`_safe_shadow_predict`); on timeout / non-zero exit / no output it logs a warning and **silently drops the model** from that cycle's `predictions_all_models.csv`. The child's `load_shadow_model` does a DagsHub MLflow resolve+download, so a transient DagsHub stall in `ridge`'s load window blows the budget for `ridge` alone. This is the *only* code path that can omit a model from the artifact — `logging.py` writes `predictions_all_models.csv` verbatim (no dropna / dedup). Same anomaly *class* as MD2's dropped `mean_rate_poisson` row (Portugal 1–1 DR Congo) — single-cycle, single-model holes from the silent shadow-skip path, not corruption. Consequence as-logged: `ridge` MD3 = 22 matches (overall 70), everyone else 24/72. **D.1 Strand 3 copied the two missing per_round `ridge` rows from their frozen twins** (`data/reconstruction/strand3_backfill/backfill_rows.csv`) → MD3 = 24/24, overall = 72/72. `ridge` is never refitted, so both cadences must carry identical predictions; the first version of the backfill re-predicted these two rows offline instead and drifted by ~0.0003 mean RPS on the feature snapshot, which made a never-refit model look like it had a cadence effect (fixed Aug 15). Trigger confirmed from Cloud Logging: `WARNING Shadow prediction: ridge exceeded 120s — skipping` — a transient DagsHub stall during the ridge shadow load (setup_mlflow + model download) blew the child-process budget in that one per-round cycle. Visibility fix (post-tournament, not mid-flight): raise these skips to ERROR / emit a per-cycle model-count so a missing shadow alerts instead of going unnoticed.

Observations:

MD3 leaderboard — single table sorted by MD3 mean RPS. After D.1, `xgboost`, `poisson_glm` and `bayesian_poisson` each show both cadences; never-refit shadows are cadence-invariant. `ridge` is restored to full MD3 coverage (24/24; was 22/24 as-logged — Strand 3 copied the two holes from its frozen twin). "Overall" = cumulative across all 72 settled matches.

| Model                       | Overall RPS | MD3 RPS    | Overall RMSE | MD3 RMSE   | Overall NLL | MD3 NLL    |
|-----------------------------|-------------|------------|--------------|------------|-------------|-----------|
| sarimax                     | 0.1620      | **0.1322** | 0.9447       | 0.9285     | 2.917       | 2.949     |
| bayesian_poisson (per-round) | 0.1622      | 0.1323     | 0.9571       | 0.9207     | 2.954       | 2.927     |
| bayesian_poisson (frozen)   | 0.1630      | 0.1323     | **0.9316**   | **0.8968** | **2.908**   | **2.888** |
| poisson_glm (frozen)        | 0.1631      | 0.1330     | 0.9326       | 0.8975     | 2.915       | 2.898     |
| negbin_glm                  | 0.1656      | 0.1331     | 0.9581       | 0.9262     | 2.965       | 2.943     |
| poisson_glm (per-round)     | 0.1618      | 0.1335     | 0.9663       | 0.9217     | 2.978       | 2.937     |
| xgboost (per-round)         | **0.1583**  | 0.1381     | 0.9669       | 0.9303     | 2.970       | 2.944     |
| random_forest               | 0.1595      | 0.1382     | 0.9681       | 0.9451     | 2.962       | 2.975     |
| xgboost (frozen)            | 0.1609      | 0.1395     | 0.9648       | 0.9282     | 2.973       | 2.959     |
| ridge                       | 0.1602      | 0.1406     | 0.9467       | 0.9418     | 2.938       | 2.970     |
| mean_rate (floor)           | 0.2308      | 0.2344     | 1.1295       | 1.1374     | 3.340       | 3.320     |
*D.1 Strand 1: `poisson_glm` / `bayesian_poisson` **frozen** rows reconstructed (see `data/reconstruction/strand1_frozen_shadow/`); per-round rows are live.

- **The refit's edge shrank to ~1%.** The only valid frozen↔per_round contrast (xgboost): per_round beat frozen on RPS (0.1381 vs 0.1395, ~1.0%) and on NLL (2.944 vs 2.959, ~0.5%), but was marginally worse on RMSE (0.9303 vs 0.9282, ~0.2%). All three gaps are tiny over just 24 matches — within MD3 noise. The metrics are not an outcome-vs-rate split: **RPS** scores the W/D/L outcome, while **NLL and RMSE are both goal-count metrics**. NLL (Poisson log-score) and RMSE (point error of λ vs realized goals) need not move together — NLL is convex and asymmetric (penalizes under-pricing high-scoring games steeply), RMSE is symmetric and linear — so a refit can win the high-information matches on NLL while slightly overshooting λ elsewhere on RMSE. This differs from MD2, where only RPS improved and both NLL and RMSE were flat-to-worse; the RPS gain itself fell from ~5% (MD2) to ~1% (MD3).
- **Champion did not lead MD3.** On per_round MD3 RPS, four cheaper models (sarimax, bayesian_poisson, negbin_glm, poisson_glm) still beat xgboost; with full `ridge` coverage its MD3 RPS rises to 0.1406 (was 0.1377 on n=22) and no longer beats the champion. xgboost still leads cumulative Overall RPS (0.1583). The cadence advantage is matchday-dependent and small — a useful RQ1 nuance.
- **MD3 was another predictable slate on average.** Round mean RPS (per_round, team-aware): 0.214 (MD1) → 0.149 (MD2) → 0.148 (MD3). Despite genuine upsets, favorites mostly delivered.
- **Worst matches (mean team-aware RPS, per_round):** South Africa 1–0 South Korea (0.512), Ecuador 2–1 Germany (0.307), DR Congo 3–1 Uzbekistan (0.252), Turkey 3–2 USA (0.234), Japan 1–1 Sweden (0.203). Two heavy-favorite upsets (Germany, South Korea both lost) plus the recurring unpriced low-scoring/level games — same failure mode flagged in MD1 and MD2.
- **Best matches (per_round):** Jordan 1–3 Argentina (0.007), Tunisia 1–3 Netherlands (0.020), Panama 0–2 England (0.026), New Zealand 1–5 Belgium (0.028), Croatia 2–1 Ghana (0.036) — dominant favorites priced correctly.
- **Alert window (rolling 24 = exactly MD3):** all 7 team-aware models well under the 0.235 naive floor in both modes — no breach. `mean_rate_poisson` printed 0.2344, essentially *on* the floor (and above its 0.229 holdout baseline); the floor behaved as designed with a healthy ~0.10 RPS gap to the team-aware models.
- **SARIMAX degenerate-λ streak broke.** No near-zero (≤1e-5) `lambda` in any MD3 fixture — the first matchday without the clipping anomaly after MD1 (ESP–CPV) and MD2 (ESP–KSA). No high-asymmetry Spain fixture this round, consistent with the earlier "2 of 2 Spain games" pattern; keep tracking.
- **Jun 27: host-advantage scramble bug in tournament simulation (forecast-only).**
  Home advantage is already baked into predictions: `generate_all_wc_pairings` puts the
  host in `home_team` and `override_neutral_for_2026_hosts` sets `is_neutral=False`, so
  `rate_lookup` holds the host's own home rate (verified: pair Spain/US stored as
  home=United States, lambda_h=0.665, lambda_a=1.968, P(US win)=11.6% — correctly an
  underdog). But `simulate_tournament` (L516-521) SWAPS the rates (`lh, la = la, lh`) for
  host KO matches, handing the opponent's rate to the host and flipping favorite<->underdog
  180 degrees. Since `is_neutral` is positional, this only stays correct if the host is
  always the `home_team` — true for single-host pairs, NOT guaranteed for host-vs-host KO.
  - **Fix:** delete the swap; resolve KO matches from the stored host-home rate via a
    venue-aware orientation helper (`_ko_match_rates`, used by both R32 and R16-to-Final
    loops); emit both orientations for host-vs-host pairs at predict time. Applied from R32 onward.
  - **Affected outputs:** `tournament_probabilities.csv`, `ko_pairings.csv`,
    `ko_fixtures.csv` (host KO paths only). Per-match predictions, monitoring, MLflow,
    Gold all clean. RQ2 entropy was contaminated live → **regenerated in D.1 Strand 2**
    (`data/reconstruction/strand2_brackets/entropy_trajectory.csv`, 3,376 cycle snapshots);
    RQ1/RQ3 untouched.

---

## Round of 32 (Jun 28 – Jul 4)

Matches played: 16 (South Africa 0–1 Canada through Colombia 1–0 Ghana, Jun 28 – Jul 4).

Pipeline:
- Both modes logged? **Y** — all 16 R32 matches scored in both artifacts. As-logged: per_round 128 rows = 8 × 16; frozen 127 (single dropped `random_forest` row, see below). **After D.1 Strand 3: 128/128 per cadence.**
- per_round refit fired at R32 boundary? **N/A for R32 predictions** — R32 predictions are backed by the **MD3-boundary** champion refit (fires after all 24 MD3 fixtures settle). The R32-boundary refit fires after all 16 R32 matches settle (last kickoff Jul 4 01:30 UTC) and feeds **R16**, not R32. New run_id: `5e404f0cdbb94bda9d67657d02a6ead1` (`wc_production` v20, fired Jul 4 04:06 UTC).
- Bracket configuration: 16 winners advance to R16 as expected; no unusual routing observed in the settled results.
- **Any failures: `random_forest` dropped from one frozen inference cycle (data-completeness anomaly, silent-shadow-skip class).** Missing frozen row: Ivory Coast 1–2 Norway (`1564789`, Jun 30). Every other model is present for that match, and the per_round row for `random_forest` on the same fixture is fine (present for all 16 R32 matches). Same failure mode and root cause as MD2's `mean_rate_poisson` (Portugal 1–1 DR Congo) and MD3's `ridge` (Colombia 0–0 Portugal, DR Congo 3–1 Uzbekistan) — a transient DagsHub stall in one child-process shadow-load window blew the 120 s budget in `_safe_shadow_predict`, and the artifact silently omits that model for that cycle. Never-refit shadow ⇒ cadence-invariant on the 15 common R32 matches (0/15 RPS or λ differ vs per_round), so the frozen row is back-fillable from per_round. Consequence as-logged: `random_forest` frozen R32 = 15 matches, per_round = 16; cumulative overall frozen = 87, per_round = 88. **D.1 Strand 3 copied the byte-identical per_round row** → frozen R32 = 16/16, overall = 88/88. No mid-flight fix (see "Decision — silent shadow-skip" in Post-tournament).
- **Jun 29: KO results never locked + KO refits mislabeled "R32" (fix, image `20260629a`).**
  Root cause: API-Football labels KO rounds without an in-round number ("Round of 32",
  not "Round of 32 - 1"), but `parse_wc_results`'s KO branch needed a numeric suffix to
  build a slot key, so `ko_results` stayed empty the whole KO stage. Two symptoms:
  (1) `simulate_tournament` never pinned real KO winners → bracket re-simulated from the
  post-group state every cycle (RQ2); (2) `next_matchday` (derived from `ko_results`)
  stuck at "R32" → every KO per_round refit was tagged "R32" and reused the R32 seed.
  - **Fix:** key locked KO results by team-set (`frozenset({home, away})`) + stage label;
    derive `next_matchday` from `last_completed_matchday` (R32→R16→…→Final); sim and
    `_build_ko_fixtures` look up by resolved team-set (fall back to simulation on a miss);
    dashboard shows the locked next round + locked/score badges. Refit gate, monitoring,
    Gold unaffected; only the one already-logged mid-R32 cycle stays unlocked (regenerable
    offline from DVC). Deployed before the next R32 match so remaining KO rounds lock live.
- **Jun 30: WC scheduler left paused → 3 R32 inference cycles missed (RQ2 gap, no code fix).**
  Root cause: operational, not code. `wc-pipeline-trigger` (the hourly WC Cloud Scheduler
  job) was paused around the Jun 29 `20260629a` deploy/verification and **not resumed**, so
  the every-hour `auto` cadence stopped firing. (Initial `gcloud scheduler jobs resume
  daily-pipeline-trigger …` failed `NOT_FOUND` — wrong job name; the live WC job is
  `wc-pipeline-trigger`, the daily one stays paused during the tournament.) Discovered Jun 30
  ~09:0x UTC: the dashboard match view + tournament probabilities did not reflect the prior
  evening's R32 results.
  - **Matches settled during the off-window (3):** Japan–Brazil, Germany–Paraguay (4–5),
    Netherlands–Morocco (3–4). The 3 missed cycles map to 3 intermediate locked states the
    live cadence would have snapshotted:
    1. pre-Japan–Brazil (none of the three locked);
    2. Japan–Brazil locked / pre-Germany–Paraguay;
    3. Germany–Paraguay locked / pre-Netherlands–Morocco.
  - **RQ2 cost (as-logged):** the catch-up cycle locks all three results at once, so three
    per-match entropy-resolution steps collapse into a single jump in the advancement-entropy
    trajectory. It is a **uniform gap across every (model × cadence_mode) trajectory** (not a
    per-model contamination): the curve loses the points that isolate each match's information.
    **D.1 Strand 4 reconstructed the 3 intermediate snapshots** (plus the R16 gap below) with
    synthetic timestamps; merged series in `data/analysis/rq2_entropy.csv` (3,532 rows as of
    the Strand-4 merge = 3,500 Strand-2 cycles + 32 Strand-4 synthetic rows). Strand 2 is now
    3,376 rows after dropping pre-`ANALYSIS_START` dry runs; RQ2 will be rebuilt in Strand 5.
  - **Not affected:** RQ1/monitoring — `_select_pre_kickoff_run` just falls back to the most
    recent snapshot strictly before each kickoff (no leakage; all-pairs λ barely depend on
    other teams' results). The R32 per-round refit gate is unaffected (it only fires after
    *all* R32 fixtures settle). Because the pipeline was off, **Gold/Bronze never mutated in
    the window** — the current DVC-versioned Gold is the exact pre-off-window base state, a
    clean no-op that makes reconstruction faithful.
  - **Resolution (Jun 30):** `gcloud scheduler jobs resume wc-pipeline-trigger --location
    europe-west1`, then one manual catch-up execution (`gcloud scheduler jobs run
    wc-pipeline-trigger …` / `gcloud run jobs execute wc-mlops-trigger …`) to lock all three
    results and refresh predictions/monitoring/probabilities live.
  - **Offline reconstruction (D.1, DONE):** Strand 4 rebuilt the 3 intermediate locked sets by
    kickoff cutoff and re-ran both cadence modes with `simulation_seed = _seed_from_string("R32")`;
    outputs in `data/reconstruction/strand4_entropy/reconstructed_entropy_snapshots.csv`.

Observations:

R32 leaderboard — single table sorted by R32 mean RPS. After D.1, `xgboost`, `poisson_glm` and `bayesian_poisson` each show both cadences; never-refit shadows are cadence-invariant. `random_forest` frozen R32 coverage restored to 16/16 (Strand 3 copy from per_round); `ridge` overall restored to 88/88. "Overall" = cumulative across all 88 settled matches.

| Model                       | Overall RPS | R32 RPS    | Overall RMSE | R32 RMSE   | Overall NLL | R32 NLL    |
|-----------------------------|-------------|------------|--------------|------------|-------------|-----------|
| random_forest               | 0.1501      | **0.1077** | 0.8949       | **0.5655** | 2.865       | **2.427** |
| xgboost (per-round)         | **0.1496**  | 0.1105     | 0.8977       | 0.5864     | 2.881       | 2.483     |
| sarimax                     | 0.1534      | 0.1144     | 0.8869       | 0.6266     | 2.874       | 2.681     |
| xgboost (frozen)            | 0.1528      | 0.1163     | 0.8977       | 0.5958     | 2.883       | 2.477     |
| negbin_glm                  | 0.1570      | 0.1186     | 0.8905       | 0.5864     | 2.881       | 2.500     |
| bayesian_poisson (per-round) | 0.1543      | 0.1190     | 0.8903       | 0.5896     | 2.874       | 2.512     |
| poisson_glm (per-round)     | 0.1543      | 0.1204     | 0.8979       | 0.5905     | 2.893       | 2.508     |
| bayesian_poisson (frozen)   | 0.1555      | 0.1218     | **0.8695**   | 0.5904     | **2.833**   | 2.492     |
| poisson_glm (frozen)        | 0.1558      | 0.1233     | 0.8708       | 0.5924     | 2.838       | 2.491     |
| ridge                       | 0.1539      | 0.1256     | 0.8909       | 0.6399     | 2.874       | 2.589     |
| mean_rate (floor)           | 0.2329      | 0.2426     | 1.0716       | 0.8109     | 3.233       | 2.749     |
*D.1 Strand 1: `poisson_glm` / `bayesian_poisson` **frozen** rows reconstructed (see `data/reconstruction/strand1_frozen_shadow/`); per-round rows are live.

- **R32 was extremely chalky** Scored against the model's own implied favorite (mean team-aware `p_home`/`p_draw`/`p_away` argmax vs actual result): the model correctly picked the winning side in **all 13 non-draw R32 matches (13/13)**, and the 3 "misses" — Germany 1–1 Paraguay, Netherlands 1–1 Morocco, Australia 1–1 Egypt — were all draws against a *mild* implied favorite (p_fav 0.43–0.59), not a big underdog winning outright. **Zero genuine "underdog beats favorite" upsets in R32.** Round mean RPS (per_round, team-aware): 0.219 (MD1) → 0.137 (MD2) → 0.131 (MD3) → **0.117 (R32)**, the lowest of the tournament so far. Interpret cautiously: the KO seeding compresses the field to broadly asymmetric pairings (Argentina–Cape Verde, France–Sweden, Colombia–Ghana), and only 3 draws is well below the ~25% base rate — a fortunate slate for team-aware models, not a step-change in skill.
- **Refit stayed net-positive on RPS, closer to a wash overall.** The only valid frozen↔per_round contrast (xgboost, 16/16 R32 matches differ): per_round beat frozen on RPS (0.1105 vs 0.1163, ~5.0%) and RMSE (0.5864 vs 0.5958, ~1.6%), but was fractionally worse on NLL (2.4829 vs 2.4771, ~0.2%). Direction reversed vs MD3, where per_round was slightly worse on RMSE and better on both RPS and NLL — consistent with the picture that outcome-calibration (RPS) is where the refit reliably wins by 1–5% per round, while the goal-rate metrics (NLL / RMSE) trade blows within a few tenths of a percent, i.e. within round-level noise on 16–24 matches.
- **Champion did not lead R32.** On R32 RPS, `random_forest` led (0.1077, per_round) with per_round xgboost 2nd (0.1105) and frozen xgboost 4th (0.1163); `random_forest` also led on R32 RMSE (0.5655) and NLL (2.427). Same pattern as MD3 (cheaper models beat champion on the round), but xgboost still leads **cumulative Overall RPS** (per_round 0.1496 vs random_forest 0.1501, ridge 0.1539, sarimax 0.1534). Two matchdays in a row where the champion is not the round leader but retains the cumulative lead — RQ1 evidence that per_round retraining pays off *across the tournament*, not necessarily on any given round.
- **Alert window (rolling 24 = last 8 MD3 + all 16 R32, spanning Jun 27–Jul 4):** all 7 team-aware models sit at 0.105–0.126 RPS, well under the 0.235 static naive floor — no breach. `mean_rate_poisson` printed 0.2344 on the rolling window and 0.2426 on R32-only, essentially on/above the floor (and above its 0.229 holdout baseline); the gap between the floor and the mean team-aware model widened to ~0.13 RPS on R32, the largest all tournament — cleanly reflects the "chalk-slate" effect.
- **Worst matches (mean team-aware RPS, per_round):** Portugal 2–1 Croatia (0.205), Switzerland 2–0 Algeria (0.202), Germany 1–1 Paraguay (0.195), Mexico 2–0 Ecuador (0.188), Brazil 2–1 Japan (0.163). The recurring theme is the **narrow-favorite / close KO match** — models were fairly confident but the pairings were genuinely close (Portugal vs Croatia, Brazil vs Japan) so scores still landed far from λ. Germany 1–1 Paraguay is the same unpriced-low-scoring-draw failure mode flagged in MD1–MD3.
- **Best matches (per_round):** Argentina 3–2 Cape Verde (0.006), Colombia 1–0 Ghana (0.015), France 3–0 Sweden (0.023), England 2–1 DR Congo (0.053), Spain 3–0 Austria (0.073), South Africa 0–1 Canada (0.094) — dominant / lopsided pairings priced correctly.
- **SARIMAX degenerate-λ streak stays broken.** 0/16 R32 fixtures had λ ≤ 1e-5 — second consecutive round without the near-zero clipping anomaly (MD3, R32). No high-asymmetry Spain fixture on R32 either (Spain–Austria was the closest and behaved normally, λ_h ≈ 2.6, λ_a ≈ 0.4), so the "2 of 2 Spain games trigger the clip" pattern is still the operative one — keep tracking on R16.
- **KO sample-size caveat.** R32 has 16 matches — the smallest matchday so far — so RPS/NLL/RMSE deltas of ≲0.005 (roughly all the frozen↔per_round gaps except R32 RPS) are inside round noise. Trust the sign more than the magnitude until R16 is in.

---

## Round of 16 (Jul 4–7)

Matches played: 8 (Canada 0–3 Morocco through Switzerland 0–0 Colombia, Jul 4–7). All 8 R16 fixtures are settled; both monitoring artifacts now cover 96 cumulative matches (72 group + 16 R32 + 8 R16). Analyzed from the two exported monitoring snapshots (`wc2026_monitoring (16).csv` = frozen, 766 rows; `wc2026_monitoring (15).csv` = per_round, 766 rows).

Pipeline:
- Both modes logged? **Y** — frozen 96 matches × up to 8 models, per_round 96 matches × up to 8 models; no new completeness gaps introduced this round (see anomaly check below).
- per_round refit fired at the R32→R16 boundary (feeding R16 predictions)? **Y** — confirmed live in the artifacts, not just inferred: `xgboost` differs between frozen and per_round on **8/8 R16 matches** (both λ and RPS move on every fixture), the same clean champion-only contrast pattern as MD2/MD3/R32. Run_id: `5e404f0cdbb94bda9d67657d02a6ead1` (`wc_production` v20, fired Jul 4 04:06 UTC). Not recoverable from the monitoring CSV export — that only carries `inference_run_id`, i.e. the scoring cycle, not the champion model version — so this came from the registry.
- per_round refit fired at the R16→QF boundary (feeding QF predictions)? **Y** — fired successfully after all 8 R16 fixtures settled. New run_id: `5032d46f410f473c9a1cc06baf145c80` (`wc_production` v21, fired Jul 8 00:10 UTC).
- **Data completeness: no new anomalies.** As-logged, the only missing rows across all 96 cumulative matches were the three already-documented silent-shadow-skip holes (`_safe_shadow_predict` timeout class), all outside R16: frozen `mean_rate_poisson` missing MD1 Portugal 1–1 DR Congo (95/96); frozen `random_forest` missing R32 Ivory Coast 1–2 Norway (95/96); per_round `ridge` missing MD3 Colombia 0–0 Portugal and DR Congo 3–1 Uzbekistan (94/96). **After D.1 Strand 3 backfill: 96/96 × 8 models = 768/768 per cadence.** All 8 R16 matches have full 8/8 model coverage in both cadence artifacts — clean round.
- **SARIMAX degenerate-λ streak stays broken.** 0/8 R16 fixtures had λ ≤ 1e-5 — third consecutive round (MD3, R32, R16) without the near-zero clipping anomaly. Portugal vs Spain (the round's most lopsided pairing on paper) behaved normally (λ_h ≈ 1.01, λ_a ≈ 1.53) — no extreme-asymmetry Spain fixture this round to re-test the earlier "2 of 2" pattern.
- The Jul 5–6 IPv6 ELO-freshness stall (documented below) is the only pipeline failure this round; nothing else surfaced in the monitoring data itself.
- **Jul 5–6: every trigger run timed out at ELO freshness check (IPv6 stall, fix image `20260706a`).**
  Root cause: `eloratings.net` began serving an AAAA record (`2602:faa9:1008:1661:379d:50ec:ecd1:7b1a`)
  around Jul 5. Cloud Run has no working IPv6 egress; the v6 SYN is silently dropped. Python
  `requests` (via `urllib3.util.connection.create_connection`) tries v6 first per `getaddrinfo`,
  waits up to the 30 s socket timeout, then falls back to v4 — so every `_download_to` in
  `check_elo_freshness` took ~30 s instead of ~0.5 s.
  - **Symptom:** ~244 slugs × ~30 s > 90 min Cloud Run task timeout. Every `wc-pipeline-trigger`
    run from Jul 5 09:33 UTC onward was terminated with `Terminating task because it has reached
    the maximum timeout of 5400 seconds.` Log fingerprint: steady 30 s cadence between
    `INFO ELO data changed: <country>` lines; every country flagged as changed (all TSVs were
    fetched but bytes differed from the manifest each time under the timeout-then-fallback path).
  - **Diagnosis:** `curl` from laptop and Cloud Shell was fast (~1.0 s), ruling out site/global-GCP
    blocking. `gcloud run jobs describe` showed no VPC connector / custom egress, ruling out routing
    config. `curl -4` = 1.0 s vs `curl -6` = fail from Cloud Shell confirmed the AAAA is dead.
    First slug (Afghanistan) already stalled ~30 s → not rate-limiting, matches IPv6 connect-stall
    signature.
  - **Fix:** module-level `urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET`
    in `src/pipeline/trigger.py`. Restores ~0.5 s per slug → full ELO freshness back to ~2 min.
    Deployed as image `20260706a`.
  - **Scheduler:** `wc-pipeline-trigger` was paused during debugging (`gcloud scheduler jobs pause
    wc-pipeline-trigger --location europe-west1`); resume after the redeploy passes one successful
    catch-up execution.
  - **What got lost (as-logged):** the Brazil–Norway ↔ Mexico–England intermediate R16 snapshot.
    **D.1 Strand 4 reconstructed it** (1 state × 2 cadences × 4 models) alongside the 3 R32
    gaps. RQ1/monitoring unaffected (`_select_pre_kickoff_run` falls back to the
    most recent snapshot strictly before each kickoff). R16 refit gate unaffected (fires only after
    all 8 R16 fixtures settle). Gold/Bronze did not mutate during the failed windows (each run
    died mid-freshness check, before any ingestion) — no data corruption risk; the post-fix run
    picks up cleanly.
  - **Thesis note (not code):** the trigger trusts Python's default `getaddrinfo` ordering and has
    no per-request instrumentation, so a silent v6-vs-v4 issue took hours to isolate. If the
    pipeline weren't retiring at end of tournament, hardening would be: log `response.elapsed` +
    resolved family per HTTP call, shorter per-request timeout with concurrency, or a pre-flight
    HEAD / `Last-Modified` fast path.

Observations:

R16 leaderboard (n=8 matches), sorted by R16 mean RPS. After D.1, `xgboost`, `poisson_glm` and `bayesian_poisson` each show both cadences — reconstructed frozen shadows separate cleanly from per_round on this round (R16 RPS 0.1472 / 0.1474 frozen vs 0.1603 / 0.1600 per_round). Never-refit shadows remain cadence-invariant. `ridge` overall restored to 96/96. "Overall" = cumulative across all 96 settled matches.

| Model                       | Overall RPS | R16 RPS    | Overall RMSE | R16 RMSE   | Overall NLL | R16 NLL    |
|-----------------------------|-------------|------------|--------------|------------|-------------|-----------|
| negbin_glm                  | 0.1559      | **0.1437** | 0.8996       | 0.9998     | 2.886       | 2.947     |
| poisson_glm (frozen)        | 0.1551      | 0.1472     | 0.8798       | **0.9797** | 2.842       | **2.885** |
| bayesian_poisson (frozen)   | 0.1548      | 0.1474     | **0.8788**   | 0.9811     | **2.837**   | 2.890     |
| bayesian_poisson (per-round) | 0.1548      | 0.1600     | 0.9022       | 1.0335     | 2.882       | 2.980     |
| poisson_glm (per-round)     | 0.1548      | 0.1603     | 0.9090       | 1.0311     | 2.899       | 2.972     |
| sarimax                     | 0.1545      | 0.1673     | 0.9039       | 1.0911     | 2.892       | 3.090     |
| xgboost (frozen)            | 0.1547      | 0.1758     | 0.9149       | 1.1035     | 2.905       | 3.153     |
| xgboost (per-round)         | **0.1523**  | 0.1821     | 0.9183       | 1.1450     | 2.908       | 3.197     |
| ridge                       | 0.1564      | 0.1831     | 0.9076       | 1.0910     | 2.893       | 3.097     |
| random_forest               | 0.1534      | 0.1898     | 0.9150       | 1.1360     | 2.893       | 3.201     |
| mean_rate (floor)           | 0.2344      | 0.2507     | 1.0742       | 1.1031     | 3.228       | 3.170     |
*D.1 Strand 1: `poisson_glm` / `bayesian_poisson` **frozen** rows reconstructed (see `data/reconstruction/strand1_frozen_shadow/`); per-round rows are live.

- **R16 had only one genuine upset: Norway 2–1 Brazil.** Judged by each model's own implied favorite (highest of mean `p_home`/`p_draw`/`p_away`), Brazil was the clear favorite (p_home 0.54) and lost outright. The other winners (Morocco, England, Belgium, Spain, France, Argentina) were already the model's favorite going in — three of them (Morocco, England, Belgium) beat the tournament's co-host nation. Switzerland 0–0 Colombia was a near-even three-way call, not a favorite losing. The model picked the right side in 7 of 8 matches.
- **The moderately elevated R16 RPS values are a symptom of tighter matchups, not more upsets.** Mean model confidence (average of the round's max(p_home, p_draw, p_away) per match) was **0.562 for R16 — the lowest of any round so far** (MD1 0.609, MD2 0.648, MD3 0.575, R32 0.592). With the eight strongest-surviving teams now paired off, several fixtures (Canada–Morocco, Mexico–England, Portugal–Spain, USA–Belgium, Switzerland–Colombia) had no dominant favorite (implied favorite probability rarely above ~0.49), so even a *correctly called* outcome scores a non-trivial RPS — there is no low-RPS outcome available when the pre-match probabilities are close to a 3-way split. Round mean RPS (per_round, team-aware, 7 models): 0.219 (MD1) → 0.149 (MD2) → 0.148 (MD3) → 0.117 (R32) → **0.169 (R16)** reflects that compression in favorite strength, with exactly one real upset behind it, not a return to an upset-prone slate.
- **All three co-host nations were eliminated in R16** — Canada (0–3 Morocco), Mexico (2–3 England), USA (1–4 Belgium) — but, per the correction above, none of these were upsets: the models had all three as underdogs beforehand (λ favored the visitor in all three fixtures, e.g. Canada λ_h≈0.86–0.90 vs Morocco λ_a≈1.3–1.4; USA λ_h≈1.08–1.17 vs Belgium λ_a≈1.36–1.37). This is a clean data point that the per-match prediction/monitoring path is unaffected by the host-advantage **simulation** scramble bug documented under MD3 (that bug lives in `simulate_tournament`'s bracket projection, not in per-match λ or scoring).
- **The refit's edge reversed sign on RPS this round, but the champion still leads cumulatively.** The only valid frozen↔per_round contrast (xgboost, 8/8 R16 matches differ): per_round was *worse* than frozen on R16 RPS (0.1821 vs 0.1758, **+3.6% worse**) and RMSE (1.1450 vs 1.1035, +3.8% worse) and NLL (3.197 vs 3.153, +1.4% worse) — the first round where per_round loses on every metric simultaneously. Despite that, xgboost per_round still leads cumulative **Overall RPS** (0.1523, best of all 8 models) because the MD2/MD3/R32 gains outweigh this round's dip (frozen overall is 0.1547, 4th-best). One bad round doesn't erase three good ones, but it's a genuine RQ1 data point that per-round retraining is not uniformly beneficial.
- **Champion did not lead R16** — on reconstructed frozen RPS, `poisson_glm` / `bayesian_poisson` now sit 2nd/3rd behind `negbin_glm` (so frozen xgboost is further back); on per_round RPS it remains 6th of 8 (negbin_glm, bayesian_poisson, poisson_glm and sarimax all beat it). This is the third consecutive round (MD3, R32, R16) where the champion is not the round leader, reinforcing the same RQ1 nuance: per_round retraining's payoff shows up in the cumulative trend, not reliably on any single round.
- **Worst matches (mean team-aware RPS, per_round):** Brazil 1–2 Norway (0.446, by far the round's biggest miss — models had Brazil as a heavy favorite, λ_h≈1.7–1.9 vs λ_a≈0.9–1.0, and Norway won anyway), Mexico 2–3 England (0.199), United States 1–4 Belgium (0.176), Portugal 0–1 Spain (0.168), Canada 0–3 Morocco (0.159), Switzerland 0–0 Colombia (0.155, another unpriced low-scoring draw — same recurring failure mode as every prior round).
- **Best matches (per_round):** Argentina 3–2 Egypt (0.018), Paraguay 0–1 France (0.036) — both correctly priced favorites, though Argentina 3–2 was a closer scoreline than the low RPS implies (outcome-only scoring rewards getting the W/D/L right regardless of margin).
- **Alert window (rolling 24 = all 16 R32 + all 8 R16, spanning Jun 28–Jul 7):** all 7 team-aware models sit at 0.127–0.145 RPS in both modes, well under the 0.235 static naive floor — no breach. `mean_rate_poisson` printed 0.2453 on the window, comfortably above the floor and its 0.229 holdout baseline; the floor continues to behave as designed.
- **KO sample-size caveat, now more acute.** R16 has only 8 matches — the smallest matchday yet (half of R32's 16) — so the ~3.6% frozen↔per_round RPS gap and all the round-only rankings above are well within noise for a single-round read. Trust the sign and the qualitative pattern (champion not leading recent rounds; refit's cumulative edge holding) more than the exact magnitudes until QF/SF pool more matches.

---

## Quarter-finals (Jul 9–12)

Matches played: 4 (France 2–0 Morocco, Spain 2–1 Belgium, Norway 1–2 England, Argentina 3–1
Switzerland; kickoffs Jul 9 20:00 – Jul 12 01:00 UTC). Cumulative settled matches: 100.
Analyzed from the end-of-tournament monitoring snapshots (`wc2026_monitoring (19).csv` =
per_round, `(20).csv` = frozen; **830 rows each as-logged; 832/832 after D.1 Strand 3**).

Pipeline:
- Both modes logged? **Y** — all 4 QF matches have full 8/8 model coverage in both cadence
  artifacts. No new completeness holes this round.
- per_round refit fired at the R16→QF boundary (feeding QF predictions)? **Y** — confirmed in
  the artifacts: `xgboost` differs between frozen and per_round on **4/4** QF matches (λ and RPS
  both move), the same champion-only contrast as MD2/MD3/R32/R16. Run_id:
  `5032d46f410f473c9a1cc06baf145c80` (`wc_production` v21, fired Jul 8 00:10 UTC).
- per_round refit fired at the QF→SF boundary (feeding SF predictions)? **Y** — fired Jul 12
  04:06–04:26 UTC, after the last QF fixture (Argentina–Switzerland, Jul 12 01:00) settled.
  Log confirms "4/4 models fitted (matchday=SF)": `xgboost` → `wc_production` **v22**
  (`champion_per_round`), `poisson_glm` → `wc_shadow` **v113**, `mean_rate_poisson` → **v114**,
  `bayesian_poisson` → **v115**. Champion run_id: `e0a1f34833a2434c98e0e68dc828ccc1`.
- QF inference cycles (pre-kickoff run used for scoring): France–Morocco
  `c94370d334604a25a56e04c3951de312` (frozen `c4c3f5650c804fb5afcf638cd3b442e0`); Spain–Belgium
  `227d95ec019742b3a12e3692230f713d` (frozen `9ed6687a284b448e839c7a8606bbfb34`); Norway–England
  `96ba2f43ee594447ae2afca1c42858d8` (frozen `0d57f911e6674eb1a8fd69ae14decf9c`);
  Argentina–Switzerland `036cdf1e19344c7f867fcb0cfd53af4f` (frozen
  `94d03309c191452a8bc37c361cf3816a`).
- **QF matchday application logs are permanently lost (observability gap, not a data loss).**
  Cloud Logging's `_Default` bucket has 30-day retention, so everything before Jul 11 06:00 UTC
  had aged out by the time the export was taken — the Jul 10 QF matchday cycles included. Audit
  logs survive in the 400-day `_Required` bucket, and Gold/DVC/MLflow are unaffected, so this
  costs narrative detail only: the QF-boundary refit and the first two QF cycles cannot be
  narrated from application logs, only inferred from the artifacts. Retention export is now
  taken proactively (`logs/logs_qf_final.txt`, Jul 11 06:08 – Jul 20 10:31).
- **Jul 12 02:05 UTC: hard crash — `dvc push` timed out.** "ERROR: failed to push data to the
  cloud - 2 files failed to upload", raising `CalledProcessError` from `_run(["dvc", "push"])`
  at `src/pipeline/trigger.py:280`, i.e. *before* the `git add` / commit gate at 281–286. That
  cycle therefore produced no Gold snapshot on the remote and no commit. **Self-healed:** the
  retry cycle at 02:12 pushed 9 files and committed `8df3d581` at 02:15, and because Gold is
  cumulative (one row per match) the retry snapshot is a superset of the failed one. No match
  rows lost; the only casualty is the point-in-time pointer for the 02:05 cycle.
- **Jul 12 10:51 UTC: hard crash — `git push` rejected (non-fast-forward).** Different failure
  mode, later in the same function: `dvc push` succeeded ("10 files pushed") and the local
  commit `8d219b1` was created, then the push was rejected (`! [rejected] thesis -> thesis
  (fetch first)` — the remote had moved ahead), raising `CalledProcessError` from
  `_run(["git", "push"])` at `trigger.py:287`. The next cycle at 10:52 hard-resets to
  `origin/thesis`, discarding `8d219b1`. **Net effect: the data blobs from that cycle are on the
  DVC remote but no `dvc.lock` anywhere references them — orphaned, not missing.** Also
  self-healed: the 11:09 retry committed `7254a5f` at 11:10. Do **not** run `dvc gc` if those
  orphaned blobs are wanted for D.1. Hardening (post-tournament, see Post-tournament): pull
  --rebase / retry around the push step.
- **Jul 12: locked KO pen-winner bug (fix, image `20260712a`).**
  Root cause: `simulate_tournament`'s locked-result branches resolved winners by
  regulation/ET goals only; on a level score the away team always advanced.
  `parse_wc_results` stored `decided_by: "PEN"` but never read
  `teams.*.winner` or `score.penalty` (available in Bronze; Silver already
  captures them). Switzerland 0–0 Colombia (PEN, Switzerland won) therefore
  advanced Colombia into QF match 100 (Argentina vs Colombia), and the real
  Argentina–Switzerland QF result could never lock (team-set key mismatch).
  - **Why undetected until QF:** all three R32 shootouts (Germany–Paraguay,
    Netherlands–Morocco, Australia–Egypt) were won by the away team, so the
    away-bias tie-break happened to pick the correct winner each time.
  - **Symptoms:** dashboard showed Colombia as Argentina's QF opponent; QF 100
    stayed `predicted` instead of `locked`; "next round" stuck at QF; Colombia
    wrongly alive in advancement probabilities.
  - **Fix:** `_resolve_ko_winner` in `parse_wc_results` (goals → winner flags →
    penalty score); `_locked_ko_winner` in `simulate_tournament` uses the
    stored `winner` field. Regression tests cover home-side pen winner on level
    goals.
  - **Affected outputs:** `tournament_probabilities.csv`, `ko_pairings.csv`,
    `ko_fixtures.csv` — **RQ2 only** (uniform across all models × both cadences
    from first post-Switzerland–Colombia cycle through this deploy). RQ1/RQ3,
    monitoring, Gold, refit gate clean (verified: Argentina–Switzerland QF row
    present in monitoring artifact with correct actual score).

Observations:

QF leaderboard (n=4 matches), sorted by QF mean RPS. After D.1, `xgboost`, `poisson_glm` and `bayesian_poisson` each show both cadences; never-refit shadows are cadence-invariant. `ridge` overall restored to 100/100. "Overall" = cumulative across all 100 settled matches.

| Model                       | Overall RPS | QF RPS     | Overall RMSE | QF RMSE    | Overall NLL | QF NLL     |
|-----------------------------|-------------|------------|--------------|------------|-------------|-----------|
| sarimax                     | 0.1519      | **0.0889** | 0.8800       | **0.3056** | 2.871       | 2.369     |
| bayesian_poisson (per-round) | 0.1522      | 0.0902     | 0.8795       | 0.3356     | 2.862       | **2.369** |
| xgboost (per-round)         | **0.1498**  | 0.0910     | 0.8949       | 0.3319     | 2.887       | 2.393     |
| random_forest               | 0.1510      | 0.0931     | 0.8932       | 0.3718     | 2.874       | 2.425     |
| negbin_glm                  | 0.1534      | 0.0938     | 0.8781       | 0.3623     | 2.866       | 2.385     |
| bayesian_poisson (frozen)   | 0.1524      | 0.0943     | **0.8575**   | 0.3461     | **2.819**   | 2.375     |
| poisson_glm (per-round)     | 0.1524      | 0.0951     | 0.8861       | 0.3355     | 2.878       | 2.372     |
| poisson_glm (frozen)        | 0.1529      | 0.0985     | 0.8586       | 0.3498     | 2.823       | 2.382     |
| xgboost (frozen)            | 0.1529      | 0.1099     | 0.8943       | 0.3994     | 2.886       | 2.435     |
| ridge                       | 0.1548      | 0.1161     | 0.8860       | 0.3670     | 2.874       | 2.416     |
| mean_rate (floor)           | 0.2357      | 0.2670     | 1.0613       | 0.7500     | 3.209       | 2.774     |
*D.1 Strand 1: `poisson_glm` / `bayesian_poisson` **frozen** rows reconstructed (see `data/reconstruction/strand1_frozen_shadow/`); per-round rows are live.

- **The QF was the best-predicted round of the entire tournament.** Round mean RPS (per_round,
  team-aware): 0.161 (Group) → 0.117 (R32) → 0.169 (R16) → **0.0955 (QF)**. Every model except
  `ridge` and the floor landed in 0.089–0.095. The models picked the winning side in **4 of 4**
  matches, there were **zero draws**, and mean confidence recovered to 0.605 (vs R16's 0.562) —
  the eight remaining teams re-separated into clear favorites and clear underdogs after R16's
  compressed pairings. This is a chalk round, not a skill jump; the same caveat as R32 applies.
- **The refit's biggest win of the tournament.** The only valid frozen↔per_round contrast
  (xgboost, 4/4 QF matches differ): per_round beat frozen on QF RPS (0.0910 vs 0.1099,
  **~17% better**), RMSE (0.3319 vs 0.3994, ~17%) and NLL (2.393 vs 2.435, ~1.7%) — the first
  round where per_round wins on all three metrics simultaneously, and by the largest margin
  seen. It reverses R16, where per_round lost on all three. On 4 matches this is noise-dominated
  (see caveat), but it restored per_round's cumulative lead: xgboost per_round Overall RPS
  0.1498, best of all 8 models, vs frozen 0.1529.
- **Champion did not lead the round — again.** On QF RPS `sarimax` led (0.0889) with
  per_round `bayesian_poisson` 2nd (0.0902) and per_round `xgboost` 3rd (0.0910); reconstructed frozen `bayesian_poisson` (0.0943) / `poisson_glm` (0.0985) sit behind the live per_round rows on this chalk slate. Fourth consecutive round
  (MD3, R32, R16, QF) where the champion is not the round leader while retaining the cumulative
  lead. This is now a stable pattern rather than a run of noise, and the central RQ1 nuance:
  **per-round retraining's payoff is cumulative, not per-round.**
- **Worst match:** Spain 2–1 Belgium (mean team-aware RPS 0.124) — the closest QF on paper
  (p_home 0.551) and the only one decided by a single goal. **Best match:** France 2–0 Morocco
  (0.068, p_home 0.660), with Argentina 3–1 Switzerland close behind (0.080). Norway 1–2 England
  (0.110) is notable as the one QF where the models favored the *away* side (p_away 0.574) and
  were right.
- **Alert window (rolling 24 = last 4 R32 + all 8 R16 + 8 group/other, ending at the QF):** all
  7 team-aware models sit at 0.116–0.139 RPS, well under the 0.235 static naive floor — no
  breach. `mean_rate_poisson` printed **0.2562**, above the floor and above its 0.229 holdout
  baseline, and the ALERT fired on every cycle in both cadences (see the standing
  naive-floor-breach note under the Final).
- **SARIMAX degenerate-λ streak stays broken.** 0/4 QF fixtures had λ ≤ 1e-5 — fourth
  consecutive round (MD3, R32, R16, QF) without the near-zero clipping anomaly. No
  extreme-asymmetry Spain fixture (Spain–Belgium was λ_h ≈ 1.72 vs λ_a ≈ 1.00), so the earlier
  "2 of 2 Spain games" pattern is still untested since MD2.
- **KO sample-size caveat, at its most acute.** 4 matches. Treat every QF-only figure above,
  including the headline 17% refit win, as directional only.

---

## Semi-finals (Jul 14–15)

Matches played: 2 (France 0–2 Spain, Jul 14; England 1–2 Argentina, Jul 15). Cumulative settled
matches: 102.

Pipeline:
- Both modes logged? **Y** — both SF matches have full 8/8 model coverage in both cadence
  artifacts. No new completeness holes.
- per_round refit fired at the QF→SF boundary (feeding SF predictions)? **Y** — the Jul 12
  04:06 UTC refit documented in the QF section (`wc_production` v22; shadows v113/v114/v115).
  Confirmed in the artifacts: `xgboost` differs across cadences on **2/2** SF matches.
  Champion run_id: `e0a1f34833a2434c98e0e68dc828ccc1`.
- per_round refit fired at the SF→Final boundary (feeding the Final and third-place match)?
  **Y** — fired Jul 15 22:06–22:30 UTC, after the last SF fixture (England–Argentina, Jul 15
  19:00) settled. Log confirms "4/4 models fitted (matchday=Final)": `xgboost` →
  `wc_production` **v23** (`champion_per_round`), `poisson_glm` → `wc_shadow` **v116**,
  `mean_rate_poisson` → **v117**, `bayesian_poisson` → **v118**. Shadow metadata resolved from
  `wc_staging` at refit time: poisson_glm v49 (run `25bcb1d4e7874f16bced9a3778cc77fe`),
  mean_rate_poisson v57 (run `bfdc135ee168442e9c31b391e38702dc`), bayesian_poisson v48 (run
  `3ffc2cd425de4678b3888c573d9185cd`). Champion run_id:
  `cc0d9d8626c042b9a8ba33eb0491799c`.
- SF inference cycles: France–Spain `5f6af99760c94929bbaa3ecfb3da0427` (frozen
  `8c7c9d5a6f4a427892fe300a409f8055`); England–Argentina `5ece23f7faa343bcb2027139f35b75e5`
  (frozen `090b616173c745e18f512b6301a9bb28`).
- **Three new silent shadow-skips (`_safe_shadow_predict` 120 s timeout class) — none reached
  the artifacts.** Cloud Logging shows `WARNING Shadow prediction: <model> exceeded 120s —
  skipping` for `negbin_glm` (Jul 13 08:15:36), `ridge` (Jul 13 08:17:36) and `random_forest`
  (Jul 14 20:21:09). Unlike the MD2/MD3/R32 instances, **all three self-healed**: each skipped
  cycle was followed by a successful cycle before the next kickoff, so the final monitoring
  artifacts have full coverage for those models on every SF fixture. Verified end-to-end — the
  only holes in either end-of-tournament artifact are the four already-documented pre-QF ones.
  **The D.1 backfill list is therefore *not* extended by these three.** They matter as evidence
  of frequency (6 skip events across the tournament, ~1 per round) rather than as data loss;
  see "Decision — silent shadow-skip" below.
- Any other failures: none. Jul 13–15 ran a clean 12 cycles/day.

Observations:

SF leaderboard (n=2 matches), sorted by SF mean RPS. After D.1, `xgboost`, `poisson_glm` and `bayesian_poisson` each show both cadences; never-refit shadows are cadence-invariant. `ridge` overall restored to 102/102. "Overall" = cumulative across all 102 settled matches.

| Model                       | Overall RPS | SF RPS     | Overall RMSE | SF RMSE    | Overall NLL | SF NLL     |
|-----------------------------|-------------|------------|--------------|------------|-------------|-----------|
| random_forest               | 0.1517      | **0.1883** | 0.8865       | **0.5511** | 2.867       | **2.514** |
| xgboost (per-round)         | **0.1507**  | 0.1920     | 0.8885       | 0.5704     | 2.880       | 2.531     |
| sarimax                     | 0.1529      | 0.2022     | 0.8740       | 0.5740     | 2.866       | 2.569     |
| xgboost (frozen)            | 0.1539      | 0.2030     | 0.8889       | 0.6215     | 2.880       | 2.579     |
| negbin_glm                  | 0.1544      | 0.2039     | 0.8725       | 0.5934     | 2.859       | 2.517     |
| bayesian_poisson (per-round) | 0.1532      | 0.2052     | 0.8738       | 0.5891     | 2.856       | 2.542     |
| bayesian_poisson (frozen)   | 0.1535      | 0.2054     | **0.8524**   | 0.5958     | **2.813**   | 2.516     |
| poisson_glm (frozen)        | 0.1539      | 0.2081     | 0.8536       | 0.6027     | 2.818       | 2.528     |
| poisson_glm (per-round)     | 0.1535      | 0.2091     | 0.8805       | 0.5981     | 2.872       | 2.558     |
| ridge                       | 0.1559      | 0.2129     | 0.8803       | 0.5981     | 2.868       | 2.591     |
| mean_rate (floor)           | 0.2363      | 0.2670     | 1.0552       | 0.7500     | 3.198       | 2.640     |
*D.1 Strand 1: `poisson_glm` / `bayesian_poisson` **frozen** rows reconstructed (see `data/reconstruction/strand1_frozen_shadow/`); per-round rows are live.

- **Both semi-finals were won by the side the models did not favor at home, and RPS roughly
  doubled.** Round mean RPS jumped from 0.0955 (QF) to **0.2019** — the second-worst round of
  the tournament after the third-place match. France 0–2 Spain was the miss (mean RPS 0.274):
  the models had it as a near-coin-flip leaning France (p_home 0.370 vs p_away 0.360, λ_h 1.258
  vs λ_a 1.236) and Spain won comfortably. England 1–2 Argentina was called correctly (p_away
  0.542, RPS 0.130). Picks: **1 of 2**.
- **Confidence collapsed to 0.456** — the lowest of any round to that point (QF 0.605, R16
  0.562). With four elite teams left the models had essentially no strong opinion, so even the
  correctly-called match scored a mediocre RPS. Same structural effect flagged in R16: when the
  predictive distribution is near a three-way split there is no low-RPS outcome available.
- **The refit held its edge, narrowly.** xgboost per_round vs frozen (2/2 matches differ): RPS
  0.1920 vs 0.2030 (~5.4% better), RMSE 0.5704 vs 0.6215 (~8.2%), NLL 2.531 vs 2.579 (~1.9%) —
  per_round wins all three for the second consecutive round. Cumulative Overall RPS: per_round
  0.1507 (best of all models) vs frozen 0.1539.
- **Champion did not lead the round — fifth in a row.** `random_forest` led SF RPS (0.1883) and
  also SF RMSE and NLL; per_round `xgboost` was 2nd. `random_forest` has now led three rounds
  (R32, SF) or come top-two in most KO rounds while never leading cumulatively — worth a
  sentence in the RQ1 write-up about cheap models being competitive round-to-round.
- **Alert window (rolling 24, ending at the SF):** team-aware models at 0.128–0.148 RPS, all
  well under the 0.235 floor. `mean_rate_poisson` unchanged at **0.2562** — still breaching,
  alert still firing every cycle in both cadences.
- **SARIMAX degenerate-λ streak stays broken** — 0/2 SF fixtures with λ ≤ 1e-5 (France–Spain
  λ 1.364/1.324; England–Argentina λ 1.135/1.879).
- **Sample size: 2 matches.** Every SF-only number is illustrative, not evidential.

---

## Final + Third-place (Jul 18–19)

Matches played: 2 (third place: France 4–6 England, Jul 18; **Final: Spain 1–0 Argentina,
Jul 19**). Cumulative settled matches: **104 of 104 — the full tournament**.

Pipeline:
- Both modes logged? **Y** — both matches have full 8/8 model coverage in both cadence
  artifacts. Final monitoring cycle: Jul 20 10:27 UTC (frozen) / 10:31 UTC (per_round), both
  reporting "Monitoring scored 830 match-model rows across 8 models" as-logged
  (**832/832 after D.1 Strand 3 backfill**).
- per_round refit fired at the SF→Final boundary? **Y** — the Jul 15 22:06 UTC refit documented
  in the SF section (`wc_production` v23; shadows v116/v117/v118). Confirmed in the artifacts:
  `xgboost` differs across cadences on **2/2** matches. Champion run_id:
  `cc0d9d8626c042b9a8ba33eb0491799c`.
- No refit fired after the Final (no next round to feed) — v23 / v116–v118 are the terminal
  model versions of the tournament.
- Inference cycles: third place `bb5bd46ad52e4f289937e559d901f158` (frozen
  `9f925cd4645f4e408265d0aa88399acd`); Final `e2c6069cd06140cdbe28b6d73320b04d` (frozen
  `ed701cc605fa4d86a44b5f2a4447b1c8`).
- **Total refit events: 8** (expected 7). Reconstructed from the `mean_rate_poisson` λ
  trajectory, which changes only on a refit and covers all four roster models since the gate
  fits them in one call: 9 distinct λ regimes = 1 pre-tournament fit + 8 refits. Seven are the
  legitimate round-boundary refits (MD1→MD2, MD2→MD3, MD3→R32, R32→R16, R16→QF, QF→SF,
  SF→Final); the eighth is the **premature MD1 refit of Jun 12** (documented under MD1). So
  "expected: 7" was right for correctly-gated refits, and the surplus event is the known bug.

  | Regime | λ | Matches covered | Feeds | Registry |
  |--------|---|-----------------|-------|----------|
  | 1 | 1.323199 | Jun 11 19:00 – Jun 12 02:00 (2) | MD1 opening | pre-tournament fit — prod v15, shadow v88–v94 |
  | 2 | 1.323151 | Jun 12 19:00 – Jun 18 02:00 (22) | — | **premature MD1 refit** — prod v16, shadow v95–v97 |
  | 3 | 1.324056 | Jun 18 16:00 – Jun 24 02:00 (24) | MD2 | prod v17, shadow v98–v100 |
  | 4 | 1.324238 | Jun 24 19:00 – Jun 28 02:00 (24) | MD3 | prod v18, shadow v101–v103 |
  | 5 | 1.325011 | Jun 28 19:00 – Jul 4 01:30 (16) | R32 | prod v19, shadow v104–v106 |
  | 6 | 1.324982 | Jul 4 17:00 – Jul 7 20:00 (8) | R16 | prod v20, shadow v107–v109 |
  | 7 | 1.325114 | Jul 9 20:00 – Jul 12 01:00 (4) | QF | prod v21, shadow v110–v112 |
  | 8 | 1.325218 | Jul 14 – Jul 15 (2) | SF | prod v22, shadow v113–v115 |
  | 9 | 1.325195 | Jul 18 – Jul 19 (2) | 3rd + Final | prod v23, shadow v116–v118 |

  Champion run_ids behind each `wc_production` version: v15 `5f5a313ad5c14199aef0a791d2e4041a`,
  v16 `b667c176418b400996e004febb9beda8`, v17 `c522bd72cd274889829d06f10165b76e`,
  v18 `bf1a5437fe26466cb43ef59e42898f69`, v19 `4dbbaec7d20a4de0a1306b9785f86eeb`,
  v20 `5e404f0cdbb94bda9d67657d02a6ead1`, v21 `5032d46f410f473c9a1cc06baf145c80`,
  v22 `e0a1f34833a2434c98e0e68dc828ccc1`, v23 `cc0d9d8626c042b9a8ba33eb0491799c`.

  Mapping method: no registry version carries a `matchday` tag and no aliases survive on any
  version (`champion_per_round` included), so versions were matched to regimes by run start time
  against each regime's match window — every refit fired hours before its window opened, and the
  v22 (Jul 12 04:06) and v23 (Jul 15 22:06) timestamps independently match the SF and Final
  refits already logged above. The `cadence_mode=per_round` tag first appears at prod v16 /
  shadow v95, which is the same boundary the shadow-resolution bug turns on.

- **Total inference cycles logged: 87 distinct pre-kickoff scoring cycles per cadence** (87 in
  each artifact, i.e. 174 across both modes) across 104 matches. The Cloud Logging export
  (Jul 11 06:08 – Jul 20 10:31) records 109 trigger cycles over its 10-day window: 8 (Jul 11,
  partial), 11 (Jul 12, two crash-and-retry cycles), 12/day Jul 13–19, and 6 on Jul 20 before
  the scheduler was paused.
- **Total monitoring rows: 830 per cadence as-logged (1,660 total)** = 104 matches × 8 models − 2
  dropped rows per artifact. Frozen was missing `mean_rate_poisson` on MD1 Portugal 1–1 DR Congo
  and `random_forest` on R32 Ivory Coast 1–2 Norway; per_round was missing `ridge` on the two MD3
  Jun 27 fixtures. All four holes are pre-QF. **After D.1 Strand 3: 832/832 per cadence
  (1,664 total)** — see `data/reconstruction/strand3_backfill/backfill_rows.csv` and
  `data/analysis/rq1_matches.csv`.
- **Standing alert: `mean_rate_poisson` breached the naive floor for the rest of the
  tournament.** The rolling-24 RPS first crossed 0.235 at match 40 (Jun 22, printing 0.2398)
  and, after dipping back under during the chalky R32 window (0.2235), climbed monotonically
  through the knockouts: 0.2344 → 0.2453 (R16) → **0.2562** from the QF to the end, against the
  0.2350 static floor and a 0.2287 holdout baseline. The export alone contains **198 ALERT
  lines** (both cadences, every cycle). **Root cause is structural, not drift:**
  `mean_rate_poisson` predicts `lambda_h == lambda_a` for **100% of rows** (verified across all
  104 matches, both cadences), so `p_home == p_away` always and its RPS takes only two values
  in the whole tournament — ≈0.1365 when the match is drawn, ≈0.2670 otherwise. It carries no
  home/host advantage and no team effects at all, so it converges on the naive predictor by
  construction. The alert behaved exactly as designed; the finding is that the floor model and
  the naive floor are the same thing, which is the point of having it. **No investigation
  needed beyond this note** — but see the host-advantage TODO, since the missing home term is
  the same defect class.
- **Any unresolved failures: none.** Both Jul 12 crashes self-healed via retry (see QF), and no
  crash, timeout, or shadow-skip occurred anywhere in the Jul 15–20 window. Jul 18's seven
  quiet cycles (00:00–12:00, no monitoring output) were the rest-day early exit — "No source
  has new data. elo_fresh=False, api_fresh=False" — not a fault.
- **End state:** last pipeline commit `91cd6f57` ("data: auto-update pipeline 2026-07-20",
  Jul 20 10:06:12 UTC), tagged **`wc2026-end-of-tournament`** as the fixed D.1 replay endpoint.
  Scheduler `wc-pipeline-trigger` paused after the Jul 20 10:00 cycle.

Observations:

Final + third-place leaderboard (n=2 matches, treated as one block), sorted by round mean RPS. After D.1, `xgboost`, `poisson_glm` and `bayesian_poisson` each show both cadences; never-refit shadows are cadence-invariant and fully covered (832/832). "Overall" = the **complete tournament**, all 104 matches.

| Model                       | Overall RPS | Rnd RPS    | Overall RMSE | Rnd RMSE   | Overall NLL | Rnd NLL    |
|-----------------------------|-------------|------------|--------------|------------|-------------|-----------|
| mean_rate (floor)           | 0.2369      | 0.2670     | 1.0781       | 2.2500     | 3.252       | 5.980     |
| random_forest               | 0.1542      | **0.2813** | 0.9131       | 2.2661     | 2.935       | 6.385     |
| poisson_glm (frozen)        | 0.1584      | 0.2873     | **0.8982**   | 2.2782     | 2.927       | 6.327     |
| poisson_glm (per-round)     | 0.1561      | 0.2875     | 0.9074       | 2.2820     | 2.936       | 6.206     |
| bayesian_poisson (per-round) | 0.1558      | 0.2879     | 0.9009       | 2.2830     | **2.922**   | 6.294     |
| bayesian_poisson (frozen)   | 0.1579      | 0.2880     | **0.8982**   | 2.2821     | 2.926       | 6.420     |
| xgboost (per-round)         | **0.1534**  | 0.2936     | 0.9145       | **2.2402** | 2.945       | 6.247     |
| sarimax                     | 0.1556      | 0.2951     | 0.9008       | 2.2687     | 2.926       | **5.987** |
| negbin_glm                  | 0.1572      | 0.2964     | 0.8995       | 2.2731     | 2.927       | 6.401     |
| xgboost (frozen)            | 0.1567      | 0.2985     | 0.9157       | 2.2822     | 2.949       | 6.439     |
| ridge                       | 0.1589      | 0.3112     | 0.9070       | 2.2697     | 2.932       | 6.193     |

- **`xgboost` (per-round) is the tournament champion on the primary metric: Overall RPS 0.1534
  across all 104 matches**, ahead of `random_forest` (0.1542), `sarimax` (0.1556),
  `bayesian_poisson` per_round (0.1558), `poisson_glm` per_round (0.1561), `xgboost` frozen
  (0.1567), `negbin_glm` (0.1572), and the reconstructed frozen shadows
  `bayesian_poisson` (0.1579) / `poisson_glm` (0.1584). The spread across the
  seven team-aware models remains remarkably tight — 0.1534 to 0.1589, about 3% end to end —
  while the naive floor sits at 0.2369, ~54% worse than the best model. **The headline RQ1
  result is that every team-aware model comfortably beat the naive floor over a full
  tournament, and that the differences between them are small relative to that gap.**
- **All three refit-eligible models now favour per-round retraining**, which was not true of the
  first reconstruction pass: `xgboost` −0.0033, `poisson_glm` −0.0023, `bayesian_poisson`
  −0.0021 RPS. The gain concentrates in the group stage (−0.0027 to −0.0036) and is roughly
  neutral in the knockout rounds, consistent with refits having more new data to absorb early.
  Effect sizes stay small against n=104 — a consistent direction, not a decisive one.
- *Reconstructed offline (D.1 Strand 1): `poisson_glm` / `bayesian_poisson` **frozen** rows from pinned `wc_shadow` v88 / v90 on pre-kickoff Gold — see `data/reconstruction/strand1_frozen_shadow/`. Per-round rows for those models are the live artifact. `mean_rate_poisson` frozen rebuild was dropped by decision (λ spread ~2e-3). Frozen figures above are the **corrected** Strand 1 rerun (Aug 15, settle-delta + batch-feature fix); the first pass leaked each match's own result and overstated the frozen arm by ~0.002 RPS.
- **Per-round retraining won the full-tournament comparison: 0.1534 (per_round) vs 0.1567
  (frozen), ~2.1% better on RPS** — still the cleanest champion cadence contrast; D.1 additionally restores frozen shadows for `poisson_glm` / `bayesian_poisson` (see Strand 1). Frozen was
  also marginally worse on RMSE (0.9157 vs 0.9145) and NLL (2.949 vs 2.945). Round by round the
  refit's RPS edge was +5% (MD2), +1% (MD3), +5% (R32), **−3.6% (R16)**, +17% (QF), +5.4% (SF),
  +1.7% (3rd/Final): positive in six of seven rounds, but with one clear reversal and a
  magnitude that swings wildly on small rounds. The honest summary is *a small, consistent,
  cumulative gain, not a decisive one.*
- **Both final-weekend matches were mispredicted, and the models were at their least confident
  all tournament.** Third place (France 4–6 England, mean RPS 0.294): the models leaned France
  (p_home 0.390) and it finished 4–6 — the highest-scoring match of the tournament and, at NLL
  ≈6.0–6.4, by far the worst-fit goal count for every model (λ ≈ 1.3 per side against 10 actual
  goals). **The Final (Spain 1–0 Argentina, mean RPS 0.293): the models marginally favored
  Argentina (p_away 0.392 vs p_home 0.343, λ_a 1.487 vs λ_h 1.313) and Spain won.** Mean
  confidence was 0.390 (third place) and 0.392 (Final), the two lowest of the tournament.
  Round picks: **0 of 2**.
- **Knockout-stage pick record: 5 of 8** (QF 4/4, SF 1/2, third place 0/1, Final 0/1). The
  models' accuracy declined monotonically as the field narrowed and pairings tightened — which
  is the expected behavior of a well-calibrated system, not a failure: mean confidence fell
  0.605 → 0.456 → 0.390, so the model was *telling* us it did not know.
- **Round mean RPS trajectory (per_round, team-aware), whole tournament:** 0.219 (MD1) → 0.149
  (MD2) → 0.148 (MD3) → 0.117 (R32) → 0.169 (R16) → **0.0955 (QF)** → 0.202 (SF) → 0.294 (3rd)
  → 0.293 (Final). The two extremes are both knockout rounds, which reinforces the
  sample-size caveat: round-level RPS tracks slate difficulty (favorite/underdog separation)
  far more than it tracks model quality.
- **The knockout run was goal-heavy:** 28 goals in the 8 matches from the QF onward (mean 3.5
  per match) vs a tournament mean of 2.96, with **zero draws in those 8 matches** — every KO tie
  from the QF on was settled in regulation. This is why the goal-count metrics (NLL, RMSE)
  degrade so sharply in the last three rounds while the outcome metric (RPS) degrades more
  gently.
- **SARIMAX degenerate-λ anomaly never recurred.** 0 of the 8 QF-onward fixtures had λ ≤ 1e-5;
  the clipping was confined to MD1 (ESP–CPV) and MD2 (ESP–KSA), i.e. the two extreme-asymmetry
  Spain group fixtures, and did not appear in any of the five subsequent rounds (MD3 through
  the Final). Final assessment: a numerical edge case tied to very high λ asymmetry, not a
  progressive reliability problem.

---

## Post-tournament summary

**Overall pipeline reliability: high, with three multi-hour outages and two self-healed
crashes.** The pipeline ran an unattended every-1–2h cadence from Jun 11 to Jul 20 and delivered
a scored, pre-kickoff prediction for **all 104 matches in both cadence modes** — no match was
ever missed. Downtime came from four incidents: the Jun 29–30 scheduler pause (3 missed cycles),
the Jul 5–6 IPv6 ELO stall (at least 25.8 h of killed runs, 1 missed snapshot), and the two
Jul 12 push failures (each recovered by the next retry cycle, ~10 and ~19 minutes). None of them cost
a prediction or a Gold row; all four cost only RQ2 entropy-trajectory resolution or point-in-time
snapshot pointers. Six silent shadow-skips occurred across the tournament (~1 per round), of
which three left holes in the artifacts and three self-healed.

**Biggest failure / surprise:** the **shadow-resolution bug** — the highest-cost defect of the
project, because it is the only one that silently invalidated a *result* rather than an
operation. `wc_shadow` has no cadence alias, so the "frozen" column for the three refit-eligible
roster shadows silently carried the per_round artifact for the entire tournament, leaving `xgboost` as the study's only clean frozen↔per_round contrast in the *as-logged*
artifacts — one model instead of four (D.1 Strand 1 later restored true frozen rows for
`poisson_glm` / `bayesian_poisson`). It
produced no error, no alert, and no missing data; it was found only by noticing that two columns
that should differ were byte-identical. The runner-up surprise is analytical rather than
operational: **the champion did not lead five of the last six rounds** while still winning
cumulatively, and the final-weekend matches were both mispredicted at the lowest confidence of
the tournament.

**Data completeness: 100%** — 104 of 104 WC 2026 matches have settled scores in Bronze, and all
104 are scored in both monitoring artifacts. Model-level completeness was 830/832 rows per
cadence as-logged (99.76%): four dropped model-match rows, all pre-QF, all from the silent
shadow-skip path. **After D.1 Strand 3: 832/832 (100%) per cadence.**

**Both modes produced complete snapshot sets? Y, with two documented caveats (both now
addressed by D.1).** Every match has a pre-kickoff snapshot in both modes. The caveats were
(1) the frozen mode's snapshots for `poisson_glm` / `bayesian_poisson` / `mean_rate_poisson`
were not genuinely frozen (shadow-resolution bug — Strand 1 rebuilt the first two; mean_rate
dropped by decision), and (2) four intermediate *entropy-trajectory* snapshots were missing
from both modes (3 from the R32 scheduler pause, 1 from the R16 IPv6 stall) — a uniform gap
across every model × cadence, affecting RQ2 only (Strand 2 regenerated curves; Strand 4
inserted the four snapshots).

### DONE — offline frozen-shadow reconstruction (shadow-resolution bug)

**STATUS: DONE (Aug 15) — D.1 Strand 1.** Driver: `src/analysis/strand1_frozen_shadow.py`.
Pinned `wc_shadow` versions: `poisson_glm` v88, `bayesian_poisson` v90 (untagged
`stage=shadow-refit`). Outputs: `data/reconstruction/strand1_frozen_shadow/`
(`frozen_shadow_combined.csv` 208 rows; `frozen_shadow_leaderboard.csv`). Tournament frozen
means **after the Aug 15 correction rerun** (see "Strand 1 correction" below): bayesian RPS
0.15788 / NLL 2.92649 / RMSE 0.89823; poisson_glm 0.15845 / 2.92722 / 0.89821 (n=104).
`mean_rate_poisson` rebuild dropped by decision (λ spread ~2e-3). Code fix
also landed: `run_shadow_refit` tags `cadence_mode=frozen`; `_latest_version_with_tags`
fallback is cadence-aware; regression tests in `tests/models/`. Audit: Strand 1 — 20 pass, 0 warn.

Why (record): `wc_shadow` has no cadence alias and frozen shadow refits write no `cadence_mode`
tag, so `load_shadow_model` resolved BOTH cadences to the newest (per_round) version.
The logged "frozen" rows for the per_round-refit roster shadows (`poisson_glm`,
`bayesian_poisson`, `mean_rate_poisson`) are therefore the per_round artifact, not the
true frozen model. Affects every matchday's frozen shadow rows (MD1 onward), not just MD2.

What's already correct (no rework): the champion both cadences (alias-separated), and the
4 never-refit shadows (`negbin_glm`, `ridge`, `random_forest`, `sarimax`); and the entire
**per_round** column for all models.

Reconstruction steps (completed):
1. Identified genuinely-frozen shadow versions in `wc_shadow` (latest `stage=shadow-refit`,
   no `cadence_mode` tag — poisson_glm v88, bayesian_poisson v90).
2. For each settled match, rebuilt its pre-kickoff feature row from the DVC-versioned Gold
   snapshot of the corresponding pre-kickoff inference cycle (strict `inference_timestamp < kickoff`).
3. Predicted with the frozen shadow versions, recomputed RPS / NLL / RMSE_h / RMSE_a, and
   backfilled the frozen values for poisson_glm / bayesian_poisson in the tables above.
4. Prereq verified (see prerequisites section below).
5. Code fix landed (not mid-tournament): tag `run_shadow_refit` runs with
   `cadence_mode=frozen` AND make `_latest_version_with_tags` fallback cadence-aware.

### DONE — Strand 1 correction: own-result leakage + single-row features

**STATUS: DONE (Aug 15).** The first Strand 1 pass carried two defects of its own; both are
fixed and all 208 rows were regenerated. See `errors_overview.md` §2.2 for the full write-up.

1. **Own-result leakage.** `parse_wc_results_before_kickoff` counts a match as settled when
   `kickoff <= cutoff`, and Strand 1 passed each match's own kickoff as the cutoff without
   `settle_delta`. Every replayed match therefore had its own final score appended to Gold
   before its features were built, with `reference_date` pushed to kickoff + 1 day so the
   rolling window actually read it. Confirmed on the tournament opener: augmented Gold held
   6946 rows when nothing should yet have been appended.
2. **Single-row feature construction.** `build_inference_features` derives
   `days_since_last_match` from Gold concatenated with the *upcoming* rows. Live inference
   passes all 1131 pairings at once; the replay passed one fixture. The two paths therefore
   disagreed even with identical Gold and identical model.

**Effect.** The leak flattered the frozen arm by **+0.0018 (bayesian) / +0.0019 (poisson_glm)
mean RPS** — the same order as the RQ1 cadence effect itself, and in the direction that
reversed the finding. Pre-correction, frozen appeared to beat per-round for both models; after
correction all three refit-eligible models favour per-round.

**Fix.** Strand 1 passes `SETTLE_DELTA` (2 h) and predicts through the full pairing batch via
new `replay_common` helpers (`snapshot_key`, `batch_lambdas`, `predict_fixture_from_batch`).
Validated end to end: the opener now reproduces the live λ exactly (2.083377 / 0.656733) with
zero results known and Gold at 6945 rows.

**Guard.** New `s1.leakage` audit check fails if any match sees its own or an unsettled result;
`tests/analysis/test_strand1_frozen_shadow.py` pins the settle-delta pass-through, the
self-leakage invariant and the batch shape. Audit after rerun: **83 pass / 7 warn / 0 fail**.

### DONE — host-advantage fix regeneration (RQ2 entropy)

**STATUS: DONE (Aug 15) — D.1 Strand 2.** Driver: `src/analysis/strand2_brackets.py`.
Replayed each cycle's DVC-versioned per-match predictions through corrected
`simulate_tournament` (same seeds, locked KO results). Outputs:
`data/reconstruction/strand2_brackets/` (per-cycle advancement + ko_pairings) and
`entropy_trajectory.csv` (3,376 rows after `ANALYSIS_START`). Merged with Strand 4 into
`data/analysis/rq2_entropy.csv` (3,532 rows as of that merge). Also regenerates the mid-R32 unlocked cycle and
the Jul 7–`20260712a` pen-winner contamination uniformly. Audit caveats (Strand 2): 5 warns —
cadence-asymmetric cycle deltas under silent skips; entropy monotonicity violations on
577/3500 rows of the pre-window trajectory (worst upward jump 0.0048). 0 fail.

Why (record): the sim swapped already-correct host-home rates (scramble bug, see MD3 Jun 27),
invalidating host-path advancement probabilities and the RQ2 Shannon-entropy trajectories
built from them. Per-match predictions, RQ1/RQ3, monitoring, Gold are clean. Also folds in
the one already-logged mid-R32 cycle that stayed unlocked under the Jun 29 KO-results
locking bug (fixed live in `20260629a`), and the pen-winner bracket bug (fixed live in
`20260712a`).

### Analysis window — drop pre-tournament dry runs

Replay drivers share `ANALYSIS_START = 2026-06-11 16:27:37 UTC` (the last frozen
cycle before first kickoff at 19:00). That timestamp, defined in
`src/analysis/rq_datasets/paths.py` and applied by `inference_cycles_for`, drops 34 of
the 879 logged cycles (20 frozen, 14 per_round). They are dry runs against an empty
tournament and answer none of the RQs; three of them also logged neither a
`simulation_seed` nor a `matchday_label`, so `simulate_tournament(seed=None)` drew from
OS entropy and could never be reproduced. Keeping the 16:27 frozen cycle
(`4f6ce4f0…`) and its 16:35 per_round twin (`5272ce72…`) leaves one pre-tournament
baseline per cadence. Cycle indexes in `rq2_entropy.csv` are dense ranks *within this
window*, not the live logged sequence — any earlier citation of a cycle number is
stale. Strand 2's `entropy_trajectory.csv` is 3,376 rows (423 frozen + 422 per_round
cycles × up to four roster models).

### DONE — backfill dropped never-refit shadow rows (data completeness)

**STATUS: DONE (Aug 15) — D.1 Strand 3.** Driver: `src/analysis/strand3_backfill.py`.
Output: `data/reconstruction/strand3_backfill/backfill_rows.csv` (4 rows). Achieved
**832/832 per cadence** in `data/analysis/rq1_matches.csv` (1,664 total). All three models
involved are never refitted, so every row is copied byte-for-byte from its twin in the other
cadence: 2 MD3 per_round `ridge` rows from frozen; R32 frozen `random_forest` from per_round;
MD1 frozen `mean_rate_poisson` from per_round. Audit (Strand 3, against the
`data/analysis/live/monitoring_*.csv` exports): 14 pass, 0 warn, 0 fail — coverage 832/832 and
all four copy byte-identity checks verified.

Revision (Aug 15): the first version re-predicted the two `ridge` rows offline from pinned
`wc_shadow` + a DVC Gold snapshot rather than copying them. That drifted on the feature
snapshot and left `ridge` — a never-refit model — showing a spurious frozen↔per_round delta
of ~0.0003 mean RPS, the same order of magnitude as the real RQ1 cadence effect. Strand 3 is
now a uniform copy step over `BACKFILL_FIXTURES` (each spec carries `copy_from_cadence`), and
`audit_reconstruction` fails if any backfill row is not a cadence copy (`s3.all_copied`).

Why (record): the silent shadow-skip path (`_safe_shadow_predict` → DagsHub load timeout) fired
**six times** across the tournament (~1 per round). Three of those skips landed on the last
pre-kickoff cycle for the affected fixture and therefore left permanent holes in the artifacts:

- **MD3 per_round `ridge`** — cycle `ed64fbfac5ed487ca222cf0a9d2c938d`; 2 holes: Colombia
  0–0 Portugal (`1489419`) and DR Congo 3–1 Uzbekistan (`1539013`), both Jun 27. Never-refit
  ⇒ cadence-invariant, so both rows are copied from the frozen cycle `0facdde2…`, which has
  `ridge` intact for the same two fixtures.
- **R32 frozen `random_forest`** — 1 hole: Ivory Coast 1–2 Norway (`1564789`, Jun 30).
  Never-refit ⇒ cadence-invariant, so the per_round row on the same fixture is byte-identical
  to what frozen would have logged (verified: 0/15 RPS or λ diffs across cadences on the
  common R32 `random_forest` rows).
- **MD1 frozen `mean_rate_poisson`** — 1 hole: Portugal 1–1 DR Congo (`1539003`, Jun 17).
  Cadence-invariant, already back-filled from per_round in the MD2 table.

**LIST IS FINAL (verified against the end-of-tournament artifacts, Jul 20).** Exactly **four**
model-match rows were missing across both cadences. Three *further* silent shadow-skips fired
later (`negbin_glm` Jul 13 08:15, `ridge` Jul 13 08:17, `random_forest` Jul 14 20:21) but
**left no holes**. Do **not** extend this list for them.

### DONE — reconstruct missed entropy-trajectory snapshots (pipeline outages)

**STATUS: DONE (Aug 15) — D.1 Strand 4.** Driver: `src/analysis/strand4_entropy.py`.
Output: `data/reconstruction/strand4_entropy/reconstructed_entropy_snapshots.csv`
(32 rows = 4 snapshots × 2 cadences × 4 models; `synthetic=True`). Inserted into
`data/analysis/rq2_entropy.csv` alongside Strand 2's cycle rows (3,376 after the
analysis-window cut). Seeds:
`_seed_from_string("R32")` / `"R16"`. Audit caveats (Strand 4): 2 warns — entropy ordering
7/32 rows; continuity jumps (45 neighbour jumps exceed real p95). 0 fail.

Why (record): a pipeline outage collapses per-match resolution steps into a single jump in the
RQ2 advancement-entropy trajectory. Affects **every (model × cadence_mode) trajectory
uniformly**; RQ1/monitoring and refit gates are untouched. Two cases:

1. **R32 scheduler-pause gap (Jun 29–30):** 3 intermediate locked states —
   pre-Japan–Brazil; Japan–Brazil locked / pre-Germany–Paraguay; Germany–Paraguay locked /
   pre-Netherlands–Morocco.
2. **R16 IPv6-stall gap (Jul 5–6):** 1 missed snapshot between Brazil–Norway and
   Mexico–England (at least 25.8 h gap).

### DONE — D.1 replay prerequisites (Gold / DVC history integrity)

All four reconstruction strands above share one prerequisite: per-cycle Gold/DVC history must
be intact enough to rebuild each match's pre-kickoff feature row (strict
`inference_timestamp < kickoff`).

**STATUS: VERIFIED (Aug 10) — the prerequisite holds; D.1 is unblocked.**
**STATUS: CONSUMED (Aug 15) — D.1 replay completed successfully against this prerequisite.**
Two independent checks against the tag:
- **Remote completeness.** `dvc status --cloud --all-commits data/gold` reported `0 files` to
  transfer on every Gold snapshot walked (500+) — no object missing from the DagsHub remote. The
  run was stopped near the end once the pattern was uniform; no missing object was ever reported.
- **Pre-kickoff coverage.** All **104 of 104** matches have a Gold commit strictly before
  kickoff. Median lead time 1.89 h, min 1.08 h (measured against the date-only `date_utc`, so
  these are conservative). Gold history is rich: **437 distinct `data/gold` hashes across the 454
  commits** touching `dvc.lock`.
- **The two Jul 12 gaps are confirmed harmless** — they do not appear among the worst lead times,
  because the retry cycles restored normal cadence before the next kickoff. The rebuild tolerates
  the two missing point-in-time snapshots; no recovery attempt is needed.

Lead-time findings from that check, using real kickoff times (see the R32 and R16 notes):
- **The R32 pause degraded one match, not three.** Only Netherlands–Morocco (Jun 30 01:00 UTC)
  sat inside the off-window, at a 15.63 h lead time. Ivory Coast–Norway (17:00) and France–Sweden
  (21:00) kicked off after the catch-up cycle and had normal ~0.90 h lead times. This concerns
  *feature freshness* only; the 3 missed intermediate **entropy** snapshots stand as documented.
- **Portugal–Spain (Jul 6 19:00 UTC) had a 19-minute pre-kickoff margin** — the tightest of the
  tournament, and a better illustration of cadence risk than either outage, since it had no
  cause beyond the 1–2 h cycle landing awkwardly against the kickoff.
- Mexico–England (Jul 6 01:00 UTC) ran on an 18.91 h old snapshot that predated Brazil–Norway.
  Harmless for RQ1: both teams last played Jul 1, so their features were current, and stale Gold
  is the leakage-safe direction. The cost is the missing R16 bracket snapshot already logged.

Status as of Jul 20:

- **Replay endpoint pinned.** Last pipeline commit `91cd6f57` ("data: auto-update pipeline
  2026-07-20", Jul 20 10:06:12 UTC) is tagged **`wc2026-end-of-tournament`** (annotated, pushed).
  Every D.1 run starts from `git checkout wc2026-end-of-tournament && dvc pull` so the
  reconstruction replays against a fixed end-of-tournament data state.
- **Do NOT run `dvc repro` during D.1.** It would rebuild Silver and Gold and produce new
  hashes, breaking the tagged state. (`dvc status` reports "changed deps" on `src/silver` and
  `src/gold` at the tag — this is `__pycache__` noise, not source drift; the tracked outputs
  match.)
- **Two known gaps in the per-cycle commit history, both benign.** The Jul 12 02:05 cycle
  (`dvc push` crash) produced no snapshot and no commit; the Jul 12 10:51 cycle (`git push`
  rejected) pushed its data but had its pointer commit discarded by the next cycle's hard reset,
  leaving orphaned blobs on the remote. Both were superseded by retry cycles within the same
  two-hour slot (`8df3d581` at 02:15, `7254a5f` at 11:10), and because Gold is cumulative each
  retry snapshot is a superset of the failed one — **no match rows are unrecoverable.** What is
  gone is only the point-in-time view at those two timestamps. Confirm the rebuild tolerates two
  missing point-in-time snapshots rather than attempting to recover them, and do **not** run
  `dvc gc` while the 10:51 orphans may still be wanted.
- **Application logs before Jul 11 06:00 UTC are gone** (Cloud Logging `_Default`, 30-day
  retention), including the Jul 10 QF matchday. Audit logs survive in `_Required` (400 days).
  The exported window Jul 11 06:08 – Jul 20 10:31 is preserved in `logs/logs_qf_final.txt`. This
  limits narration, not reconstruction — Gold, DVC and MLflow carry everything D.1 needs.

### Decision — pipeline push failures: document, do NOT build a fix

Two hard crashes hit `run_dvc_pipeline` on Jul 12 (02:05 `dvc push` timeout at
`trigger.py:280`; 10:51 non-fast-forward `git push` rejection at `trigger.py:287`). Both are
transient-remote failures against DagsHub, and both self-healed on the next scheduled cycle
because the pipeline is idempotent and Gold is cumulative. The natural hardening — a
`git pull --rebase` before push, and retry-with-backoff around both push steps — is **preventive
only**, and the pipeline retires at the end of the tournament, so **no code change will be
shipped**, consistent with the silent-shadow-skip decision below. Capture as thesis prose:
- Chapter 3 / Chapter 6: an unattended CT pipeline that commits its own data state needs the
  push step to be idempotent and retry-safe, because the remote is the one dependency it cannot
  control. The crash-and-retry pattern worked here only because the cadence was frequent
  (every 1–2 h) and the data model cumulative; a daily cadence or a mutable Gold would have
  turned the same two failures into real snapshot loss.

### Decision — silent shadow-skip: document, do NOT build a fix

The holes above come from `run_prediction_all_models` silently dropping a shadow on
transient DagsHub load failure (warning only). The natural mitigations — in-cycle retry on
the shadow load/predict child, ERROR-level logging, and a per-cycle model-count completeness
check — are **preventive only**: they pay off on *future* inference cycles. The pipeline
retires at the end of the tournament, so there are no future cycles to protect and **no code
change will be shipped** (not mid-tournament, not after). Capture it instead as thesis prose:
- Chapter 3 (threats to validity) / Chapter 6 (limitations + further work): the dual-mode
  inference layer trades completeness for resilience (one bad shadow can't kill a cycle), at
  the cost of silent per-cycle holes; recommended hardening = retry + loud logging + a
  per-cycle completeness alert. Converts the finding into write-up value without engineering
  on a shelved pipeline.
