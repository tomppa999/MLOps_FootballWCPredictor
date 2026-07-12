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

MD1 leaderboard — all 24 matches. `xgboost` is the true **frozen** champion (from the frozen artifact); the four never-refit shadows are cadence-invariant; `poisson_glm` and `bayesian_poisson` (marked `*`) carry the **premature-refit** shadow version for *both* cadences (see caveat below), not the true frozen model:

| Model               | Mean RPS ↑ | Mean RMSE | Mean NLL |
|---------------------|------------|-----------|----------|
| ridge               | **0.2074** | 0.9218    | 3.027    |
| xgboost (frozen)    | 0.2075     | 0.9796    | 3.108    |
| random_forest       | 0.2089     | 0.9789    | 3.089    |
| poisson_glm *       | 0.2131     | 0.9576    | 3.061    |
| bayesian_poisson *  | 0.2142     | 0.9625    | 3.063    |
| negbin_glm          | 0.2170     | 0.9699    | 3.074    |
| **mean_rate (floor)** | 0.2181 | 0.9901 | 3.253    |
| sarimax             | 0.2208     | 0.9530    | 3.034    |

- No alert window breach — all 7 team-aware models finished below the naive floor (0.218). First 24-match alert window now active.
- RPS spread is narrow (ridge 0.207 to sarimax 0.221); ridge leads on RPS but has the best RMSE too. Sarimax trails on RPS despite a low NLL — the draw penalty lands harder for it.
- Worst match: CIV 1–0 ECU — models had Ivory Coast as heavy underdogs (p_home ~0.13); sarimax RPS 0.663, negbin 0.596. GHA 1–0 PAN similar (RF/negbin ~0.54–0.55). Both genuine upsets for the WC format.
- Best match: GER 7–1 CUR — negbin RPS 0.008, poisson_glm 0.012; correctly priced dominant favorite at p_home ~0.85–0.88.
- SARIMAX anomaly: λ_away = 1e-06 for ESP vs CPV — degenerate near-zero rate (numerically clipped). No crash but flagged as reliability concern for high-asymmetry fixtures.
- **`*` caveat (poisson_glm, bayesian_poisson):** these MD1 rows are *not* the true frozen model. The premature MD1 per_round refit (~Jun 12, v16-era) registered new `wc_shadow` versions for the experiment-roster shadows, and the shadow-resolution bug (no cadence alias/tag — see MD2 pipeline note) makes *both* frozen and per_round inference load that same newest version. So the frozen and per_round MD1 values for these two are identical and both reflect the prematurely-refit artifact. `xgboost` is unaffected here because it loads the true frozen champion via the `champion_frozen` alias (frozen artifact: RPS 0.2075 vs the contaminated per_round 0.2077). True frozen values for `*` rows are deferred to the post-tournament offline reconstruction.
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

MD2 leaderboard — single table (all 48 = cumulative MD1+MD2; MD2 = last 24), sorted by MD2 mean RPS. `xgboost` is shown for both cadences (the only clean contrast); `poisson_glm` and `bayesian_poisson` are tagged **(per round)** because their frozen value is not separable from per_round (shadow-resolution bug — see caveat above); `mean_rate_poisson` is cadence-invariant here anyway; the other four are identical across modes by design.

| Model                       | Overall RPS | MD2 RPS    | Overall RMSE | MD2 RMSE   | Overall NLL | MD2 NLL   |
|-----------------------------|-------------|------------|--------------|------------|-------------|-----------|
| xgboost (per-round)         | **0.1683**  | **0.1290** | 0.9852       | 0.9892     | 2.983       | 2.854     |
| random_forest               | 0.1701      | 0.1313     | 0.9795       | 0.9802     | 2.956       | 2.822     |
| ridge                       | 0.1700      | 0.1327     | **0.9491**   | 0.9764     | 2.921       | 2.816     |
| sarimax                     | 0.1769      | 0.1330     | 0.9528       | **0.9527** | **2.902**   | **2.769** |
| xgboost (frozen)            | 0.1717      | 0.1359     | 0.9831       | 0.9867     | 2.980       | 2.851     |
| poisson_glm (per round)     | 0.1760      | 0.1389     | 0.9886       | 1.0196     | 2.999       | 2.937     |
| bayesian_poisson (per round)| 0.1771      | 0.1400     | 0.9752       | 0.9880     | 2.967       | 2.871     |
| negbin_glm                  | 0.1818      | 0.1467     | 0.9741       | 0.9782     | 2.976       | 2.878     |
| mean_rate (floor)           | 0.2290      | 0.2398     | 1.1256       | 1.2612     | 3.350       | 3.447     |

- **MD2 was a "chalk" matchday.** Every team-aware model's RPS roughly halved vs MD1 (≈0.21 → ≈0.13). Outcome mix was 13 home / 6 away / 5 draws — far fewer upsets than MD1, and favorites delivered (Canada 6–0, Portugal 5–0, Netherlands 5–1, Spain 4–0, France 3–0, Brazil 3–0). Lower RPS reflects an easier slate, not a model improvement per se.
- **The refit paid off, narrowly and only on ranking.** Retraining the champion on the 24 settled MD1 matches moved xgboost from 4th (frozen, among the 8 models) to **1st** on MD2 RPS (0.1359 → 0.1290, ~5% better) and to best overall (0.1717 → 0.1683). But MD2 RMSE (0.9867 → 0.9892) and NLL (2.851 → 2.854) were flat-to-marginally-worse: the refit sharpened outcome calibration without improving the goal-rate point fit. A modest, real win for per_round — on one matchday, on one metric. This is the **only** valid frozen↔per_round model contrast for MD2 (see caveat).
- **Frozen shadow values pending.** For `poisson_glm` and `bayesian_poisson` the rows above are the per_round-refit shadow models; their true frozen counterparts can't be read from the logs (the resolver collapses both cadences). The genuinely-frozen contrast for these is deferred to the post-tournament offline reconstruction. Among the never-refit shadows, ridge leads on RMSE (0.949) and sarimax on NLL (2.769); random_forest is the best non-champion on RPS (0.1313), edging frozen xgboost.
- **Alert window (rolling 24 = exactly MD2):** all 7 team-aware models sit well under the 0.235 static naive floor in both modes — no breach. `mean_rate_poisson` itself printed 0.2398 on MD2, *above* the static floor (and above its 0.229 holdout baseline) — the naive floor behaved as designed, and the gap to the team-aware models stayed healthy (~0.10 RPS).
- **Worst matches (mean team-aware RPS):** England 0–0 Ghana (0.402) and Ecuador 0–0 Curaçao (0.350) — two goalless draws where models priced the favorite heavily; USA 2–0 Australia (0.308) and Turkey 0–1 Paraguay (0.301) rounded out the misses. The recurring theme is the unpriced low-scoring draw, same failure mode flagged in MD1.
- **Best matches:** France 3–0 Iraq (0.012), Spain 4–0 Saudi Arabia (0.013), Brazil 3–0 Haiti (0.014) — dominant favorites priced correctly across the board.
- **SARIMAX degenerate-λ recurrence:** λ_away = 1e-06 again, this time Spain vs Saudi Arabia (clipped near-zero rate). It happened to land (Spain won 4–0, sarimax RPS 0.004 — best single score of MD2), but it's the same numerical-clipping reliability concern as MD1's ESP–CPV. Now seen on 2 of 2 high-asymmetry Spain fixtures; track whether it generalizes.

---

## Matchday 3 — Group Stage Round 3 (Jun 24–28)

Matches played: 24 (Switzerland 2–1 Canada through Algeria 3–3 Austria / Jordan 1–3 Argentina, Jun 24–28).

Pipeline:
- Both modes logged? **Y** — all 72 settled matches scored in both artifacts (frozen 575 rows, per_round 574; the deficits are isolated dropped rows, see below).
- per_round refit fired? **N/A for the MD3 boundary** — the MD3 *predictions* are backed by the **MD2-boundary** champion refit. The MD3-boundary refit (after all 24 MD3 fixtures settle, last kickoff Jun 28) feeds **R32**, not MD3. New run_id: [fill from MLflow once R32 inference runs].
- AFCON/format effect: **6 of 24 draws** (Japan 1–1 Sweden, Paraguay 0–0 Australia, Cape Verde 0–0 Saudi Arabia, Egypt 1–1 Iran, Colombia 0–0 Portugal, Algeria 3–3 Austria) — exactly the ~1/4 base rate for 24 games, so **no evidence of final-day draw-gaming / third-place hedging** this round (the three 0–0s notwithstanding). Outcome mix 9 home / 9 away / 6 draw, more balanced than MD2's 13/6/5.
- **Any failures: `ridge` dropped from one per_round inference cycle (data-completeness anomaly).** The per_round `ridge` row is missing for Colombia 0–0 Portugal (`1489419`) and DR Congo 3–1 Uzbekistan (`1539013`), both Jun 27, both served by the single per_round inference run `ed64fbfac5ed487ca222cf0a9d2c938d`. Every other model is present for those matches, and the frozen run for the same fixtures (`0facdde2…`) has `ridge` fine — so `ridge` itself is healthy (present for the other 70 per_round matches). Root cause: `run_prediction_all_models` runs each shadow in an isolated child process with a 120 s timeout (`_safe_shadow_predict`); on timeout / non-zero exit / no output it logs a warning and **silently drops the model** from that cycle's `predictions_all_models.csv`. The child's `load_shadow_model` does a DagsHub MLflow resolve+download, so a transient DagsHub stall in `ridge`'s load window blows the budget for `ridge` alone. This is the *only* code path that can omit a model from the artifact — `logging.py` writes `predictions_all_models.csv` verbatim (no dropna / dedup). Same anomaly *class* as MD2's dropped `mean_rate_poisson` row (Portugal 1–1 DR Congo) — single-cycle, single-model holes from the silent shadow-skip path, not corruption. Consequence: `ridge` MD3 = 22 matches (overall 70), everyone else 24/72; never-refit shadow so still identical across modes on the 22 common matches. Trigger confirmed from Cloud Logging: `WARNING Shadow prediction: ridge exceeded 120s — skipping` — a transient DagsHub stall during the ridge shadow load (setup_mlflow + model download) blew the child-process budget in that one per-round cycle. Visibility fix (post-tournament, not mid-flight): raise these skips to ERROR / emit a per-cycle model-count so a missing shadow alerts instead of going unnoticed.

Observations:

MD3 leaderboard — single table sorted by MD3 mean RPS. Convention as in MD2: **per_round shown for all models** (the per_round column is correct for every model); `xgboost` carries a separate **frozen** line (the only clean frozen↔per_round contrast, via the `champion_*` aliases). Frozen values for the other roster shadows (`poisson_glm`, `bayesian_poisson`, `mean_rate_poisson`) are still collapsed onto the per_round artifact by the shadow-resolution bug and are deferred to the post-tournament offline reconstruction; the four never-refit shadows (`sarimax`, `negbin_glm`, `ridge`, `random_forest`) are identical across modes by design. "Overall" = cumulative across all 72 settled matches (MD1 per_round shadow rows carry the premature-v16 residue, so cumulative shadow numbers are slightly soft; MD3-only columns are clean).

| Model                       | Overall RPS | MD3 RPS    | Overall RMSE | MD3 RMSE   | Overall NLL | MD3 NLL   |
|-----------------------------|-------------|------------|--------------|------------|-------------|-----------|
| sarimax                     | 0.1620      | **0.1322** | 0.9447       | 0.9285     | 2.917       | 2.949     |
| bayesian_poisson            | 0.1622      | 0.1323     | 0.9571       | **0.9207** | 2.954       | **2.927** |
| negbin_glm                  | 0.1656      | 0.1331     | 0.9581       | 0.9262     | 2.965       | 2.943     |
| poisson_glm                 | 0.1618      | 0.1335     | 0.9663       | 0.9217     | 2.978       | 2.937     |
| ridge (n=22)                | 0.1599      | 0.1377     | **0.9429**   | 0.9295     | 2.938       | 2.975     |
| xgboost (per-round)         | **0.1583**  | 0.1381     | 0.9669       | 0.9303     | 2.970       | 2.944     |
| random_forest               | 0.1595      | 0.1382     | 0.9681       | 0.9451     | 2.962       | 2.975     |
| xgboost (frozen)            | 0.1609      | 0.1395     | 0.9648       | 0.9282     | 2.973       | 2.959     |
| mean_rate (floor)           | 0.2308      | 0.2344     | 1.1295       | 1.1374     | 3.340       | 3.320     |

- **The refit's edge shrank to ~1%.** The only valid frozen↔per_round contrast (xgboost): per_round beat frozen on RPS (0.1381 vs 0.1395, ~1.0%) and on NLL (2.944 vs 2.959, ~0.5%), but was marginally worse on RMSE (0.9303 vs 0.9282, ~0.2%). All three gaps are tiny over just 24 matches — within MD3 noise. The metrics are not an outcome-vs-rate split: **RPS** scores the W/D/L outcome, while **NLL and RMSE are both goal-count metrics**. NLL (Poisson log-score) and RMSE (point error of λ vs realized goals) need not move together — NLL is convex and asymmetric (penalizes under-pricing high-scoring games steeply), RMSE is symmetric and linear — so a refit can win the high-information matches on NLL while slightly overshooting λ elsewhere on RMSE. This differs from MD2, where only RPS improved and both NLL and RMSE were flat-to-worse; the RPS gain itself fell from ~5% (MD2) to ~1% (MD3).
- **Champion did not lead MD3.** On per_round MD3 RPS, four cheaper models (sarimax, bayesian_poisson, negbin_glm, poisson_glm) beat xgboost, reversing MD2 where the refit pushed xgboost to 1st. xgboost still leads cumulative Overall RPS (0.1583). The cadence advantage is matchday-dependent and small — a useful RQ1 nuance.
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
    Gold all clean. RQ2 entropy contaminated -> regenerate post-tournament; RQ1/RQ3 untouched.

---

## Round of 32 (Jun 28 – Jul 4)

Matches played: 16 (South Africa 0–1 Canada through Colombia 1–0 Ghana, Jun 28 – Jul 4).

Pipeline:
- Both modes logged? **Y** — all 16 R32 matches scored in both artifacts (per_round 128 rows = 8 × 16; frozen 127; deficit is a single dropped `random_forest` row, see below).
- per_round refit fired at R32 boundary? **N/A for R32 predictions** — R32 predictions are backed by the **MD3-boundary** champion refit (fires after all 24 MD3 fixtures settle). The R32-boundary refit fires after all 16 R32 matches settle (last kickoff Jul 4 01:30 UTC) and feeds **R16**, not R32. New run_id: [fill from MLflow].
- Bracket configuration: 16 winners advance to R16 as expected; no unusual routing observed in the settled results.
- **Any failures: `random_forest` dropped from one frozen inference cycle (data-completeness anomaly, silent-shadow-skip class).** Missing frozen row: Ivory Coast 1–2 Norway (`1564789`, Jun 30). Every other model is present for that match, and the per_round row for `random_forest` on the same fixture is fine (present for all 16 R32 matches). Same failure mode and root cause as MD2's `mean_rate_poisson` (Portugal 1–1 DR Congo) and MD3's `ridge` (Colombia 0–0 Portugal, DR Congo 3–1 Uzbekistan) — a transient DagsHub stall in one child-process shadow-load window blew the 120 s budget in `_safe_shadow_predict`, and the artifact silently omits that model for that cycle. Never-refit shadow ⇒ cadence-invariant on the 15 common R32 matches (0/15 RPS or λ differ vs per_round), so the frozen row is back-fillable from per_round; add to the D.1 backfill list. Consequence: `random_forest` frozen R32 = 15 matches, per_round = 16; cumulative overall frozen = 87, per_round = 88. No mid-flight fix (see "Decision — silent shadow-skip" in Post-tournament).
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
  - **RQ2 cost:** the catch-up cycle locks all three results at once, so three per-match
    entropy-resolution steps collapse into a single jump in the advancement-entropy
    trajectory. It is a **uniform gap across every (model × cadence_mode) trajectory** (not a
    per-model contamination): the curve loses the points that isolate each match's information.
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
  - **Offline reconstruction (D.1, NOT mid-tournament):** see Post-tournament TODO. Mid-R32
    cycles are deterministic (`simulation_seed = _seed_from_string("R32")` for every cycle
    while R32 is in progress), so rebuilding the 3 intermediate locked sets by kickoff cutoff
    and re-running both cadence modes reproduces the missing snapshots bit-identically.

Observations:

R32 leaderboard — single table sorted by R32 mean RPS. Same convention as MD2/MD3: **per_round shown for all models** (the per_round column is correct for every model); `xgboost` carries a separate **frozen** line (the only clean frozen↔per_round contrast, via the `champion_*` aliases — 16/16 R32 matches differ on both RPS and λ). Frozen values for the 3 refit-eligible roster shadows (`poisson_glm`, `bayesian_poisson`, `mean_rate_poisson`) are still collapsed onto the per_round artifact by the shadow-resolution bug and are deferred to the post-tournament offline reconstruction; the four never-refit shadows (`sarimax`, `negbin_glm`, `ridge`, `random_forest`) are byte-identical across modes by design (verified: 0/16 RPS or λ diffs across cadences for all seven shadows). "Overall" = cumulative across all 88 settled matches (per_round column; `ridge` overall = 86, all others 88 in per_round). MD1 shadow rows still carry the premature-v16 residue so cumulative shadow numbers are slightly soft; R32-only columns are clean.

| Model                       | Overall RPS | R32 RPS    | Overall RMSE | R32 RMSE   | Overall NLL | R32 NLL   |
|-----------------------------|-------------|------------|--------------|------------|-------------|-----------|
| random_forest               | 0.1501      | **0.1077** | 0.8949       | **0.5655** | 2.865       | **2.427** |
| xgboost (per-round)         | **0.1496**  | 0.1105     | 0.8977       | 0.5864     | 2.881       | 2.483     |
| sarimax                     | 0.1534      | 0.1144     | **0.8869**   | 0.6266     | 2.874       | 2.682     |
| xgboost (frozen)            | 0.1528      | 0.1163     | 0.8977       | 0.5958     | 2.883       | 2.477     |
| negbin_glm                  | 0.1570      | 0.1186     | 0.8905       | 0.5864     | 2.881       | 2.500     |
| bayesian_poisson            | 0.1543      | 0.1190     | 0.8903       | 0.5896     | 2.874       | 2.512     |
| poisson_glm                 | 0.1543      | 0.1204     | 0.8979       | 0.5905     | 2.893       | 2.508     |
| ridge (n=86 / n=16)         | 0.1535      | 0.1256     | 0.8866       | 0.6399     | **2.873**   | 2.589     |
| mean_rate (floor)           | 0.2329      | 0.2426     | 1.0716       | 0.8109     | 3.233       | 2.749     |

- **R32 was extremely chalky** Scored against the model's own implied favorite (mean team-aware `p_home`/`p_draw`/`p_away` argmax vs actual result): the model correctly picked the winning side in **all 13 non-draw R32 matches (13/13)**, and the 3 "misses" — Germany 1–1 Paraguay, Netherlands 1–1 Morocco, Australia 1–1 Egypt — were all draws against a *mild* implied favorite (p_fav 0.43–0.59), not a big underdog winning outright. **Zero genuine "underdog beats favorite" upsets in R32.** Round mean RPS (per_round, team-aware): 0.219 (MD1) → 0.137 (MD2) → 0.131 (MD3) → **0.117 (R32)**, the lowest of the tournament so far. Interpret cautiously: the KO seeding compresses the field to broadly asymmetric pairings (Argentina–Cape Verde, France–Sweden, Colombia–Ghana), and only 3 draws is well below the ~25% base rate — a fortunate slate for team-aware models, not a step-change in skill.
- **Refit stayed net-positive on RPS, closer to a wash overall.** The only valid frozen↔per_round contrast (xgboost, 16/16 R32 matches differ): per_round beat frozen on RPS (0.1105 vs 0.1163, ~5.0%) and RMSE (0.5864 vs 0.5958, ~1.6%), but was fractionally worse on NLL (2.4829 vs 2.4771, ~0.2%). Direction reversed vs MD3, where per_round was slightly worse on RMSE and better on both RPS and NLL — consistent with the picture that outcome-calibration (RPS) is where the refit reliably wins by 1–5% per round, while the goal-rate metrics (NLL / RMSE) trade blows within a few tenths of a percent, i.e. within round-level noise on 16–24 matches.
- **Champion did not lead R32.** On R32 RPS, `random_forest` led (0.1077, per_round) with per_round xgboost 2nd (0.1105) and frozen xgboost 4th (0.1163); `random_forest` also led on R32 RMSE (0.5655) and NLL (2.427). Same pattern as MD3 (cheaper models beat champion on the round), but xgboost still leads **cumulative Overall RPS** (per_round 0.1496 vs random_forest 0.1501, ridge 0.1535, sarimax 0.1534). Two matchdays in a row where the champion is not the round leader but retains the cumulative lead — RQ1 evidence that per_round retraining pays off *across the tournament*, not necessarily on any given round.
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
- per_round refit fired at the R32→R16 boundary (feeding R16 predictions)? **Y** — confirmed live in the artifacts, not just inferred: `xgboost` differs between frozen and per_round on **8/8 R16 matches** (both λ and RPS move on every fixture), the same clean champion-only contrast pattern as MD2/MD3/R32. Run_id: **not recoverable from the monitoring CSV export** (it only carries `inference_run_id`, i.e. the scoring cycle, not the champion model version) — pull `champion_per_round`'s version/run_id from MLflow and fill in here.
- per_round refit fired at the R16→QF boundary (feeding QF predictions)? **Y** — fired successfully after all 8 R16 fixtures settled. New run_id: [fill from MLflow].
- **Data completeness: no new anomalies.** The only missing rows across all 96 cumulative matches are the three already-documented silent-shadow-skip holes (`_safe_shadow_predict` timeout class), and all three are confirmed outside R16 in this export: frozen `mean_rate_poisson` missing MD1 Portugal 1–1 DR Congo (95/96); frozen `random_forest` missing R32 Ivory Coast 1–2 Norway (95/96); per_round `ridge` missing MD3 Colombia 0–0 Portugal and DR Congo 3–1 Uzbekistan (94/96). All 8 R16 matches have full 8/8 model coverage in both cadence artifacts — clean round.
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
  - **What got lost:** the Brazil–Norway ↔ Mexico–England intermediate R16 snapshot (see
    Post-tournament TODO). RQ1/monitoring unaffected (`_select_pre_kickoff_run` falls back to the
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

R16 leaderboard (n=8 matches), sorted by R16 mean RPS. Same convention as MD2/MD3/R32: **per_round shown for all models** (correct for every model on this scoring path); `xgboost` carries a separate **frozen** line (the only clean frozen↔per_round contrast, via the `champion_*` aliases — 8/8 R16 matches differ on both RPS and λ). `poisson_glm`/`bayesian_poisson` frozen and per_round values are still byte-identical here too (shadow-resolution bug persists unchanged — 0/8 R16 rows differ for either), so their row is shown once, unlabeled, per the established convention; the three never-refit shadows still fully cadence-invariant. "Overall" = cumulative across all 96 settled matches.

| Model                       | Overall RPS | R16 RPS    | Overall RMSE | R16 RMSE   | Overall NLL | R16 NLL   |
|-----------------------------|-------------|------------|---------------|------------|--------------|-----------|
| negbin_glm                  | 0.1559      | **0.1437** | 0.8996        | **0.9998** | 2.886        | **2.947** |
| bayesian_poisson            | 0.1548      | 0.1600     | 0.9022        | 1.0335     | **2.882**    | 2.980     |
| poisson_glm                 | 0.1548      | 0.1603     | 0.9090        | 1.0311     | 2.899        | 2.972     |
| sarimax                     | 0.1545      | 0.1673     | 0.9039        | 1.0911     | 2.892        | 3.090     |
| xgboost (frozen)            | 0.1547      | 0.1758     | 0.9149        | 1.1035     | 2.905        | 3.153     |
| xgboost (per-round)         | **0.1523**  | 0.1821     | 0.9183        | 1.1450     | 2.908        | 3.197     |
| ridge (n=94)                | 0.1560      | 0.1831     | 0.9040        | 1.0910     | 2.892        | 3.097     |
| random_forest               | 0.1534      | 0.1898     | **0.9150**    | 1.1360     | 2.893        | 3.201     |
| mean_rate (floor)           | 0.2344      | 0.2507     | 1.0742        | 1.1031     | 3.228        | 3.170     |

- **R16 had only one genuine upset: Norway 2–1 Brazil.** Judged by each model's own implied favorite (highest of mean `p_home`/`p_draw`/`p_away`), Brazil was the clear favorite (p_home 0.54) and lost outright. The other winners (Morocco, England, Belgium, Spain, France, Argentina) were already the model's favorite going in — three of them (Morocco, England, Belgium) beat the tournament's co-host nation. Switzerland 0–0 Colombia was a near-even three-way call, not a favorite losing. The model picked the right side in 7 of 8 matches.
- **The moderately elevated R16 RPS values are a symptom of tighter matchups, not more upsets.** Mean model confidence (average of the round's max(p_home, p_draw, p_away) per match) was **0.562 for R16 — the lowest of any round so far** (MD1 0.609, MD2 0.648, MD3 0.575, R32 0.592). With the eight strongest-surviving teams now paired off, several fixtures (Canada–Morocco, Mexico–England, Portugal–Spain, USA–Belgium, Switzerland–Colombia) had no dominant favorite (implied favorite probability rarely above ~0.49), so even a *correctly called* outcome scores a non-trivial RPS — there is no low-RPS outcome available when the pre-match probabilities are close to a 3-way split. Round mean RPS (per_round, team-aware, 7 models): 0.219 (MD1) → 0.149 (MD2) → 0.148 (MD3) → 0.117 (R32) → **0.169 (R16)** reflects that compression in favorite strength, with exactly one real upset behind it, not a return to an upset-prone slate.
- **All three co-host nations were eliminated in R16** — Canada (0–3 Morocco), Mexico (2–3 England), USA (1–4 Belgium) — but, per the correction above, none of these were upsets: the models had all three as underdogs beforehand (λ favored the visitor in all three fixtures, e.g. Canada λ_h≈0.86–0.90 vs Morocco λ_a≈1.3–1.4; USA λ_h≈1.08–1.17 vs Belgium λ_a≈1.36–1.37). This is a clean data point that the per-match prediction/monitoring path is unaffected by the host-advantage **simulation** scramble bug documented under MD3 (that bug lives in `simulate_tournament`'s bracket projection, not in per-match λ or scoring).
- **The refit's edge reversed sign on RPS this round, but the champion still leads cumulatively.** The only valid frozen↔per_round contrast (xgboost, 8/8 R16 matches differ): per_round was *worse* than frozen on R16 RPS (0.1821 vs 0.1758, **+3.6% worse**) and RMSE (1.1450 vs 1.1035, +3.8% worse) and NLL (3.197 vs 3.153, +1.4% worse) — the first round where per_round loses on every metric simultaneously. Despite that, xgboost per_round still leads cumulative **Overall RPS** (0.1523, best of all 8 models) because the MD2/MD3/R32 gains outweigh this round's dip (frozen overall is 0.1547, 4th-best). One bad round doesn't erase three good ones, but it's a genuine RQ1 data point that per-round retraining is not uniformly beneficial.
- **Champion did not lead R16** — 4th of 8 models on frozen RPS, 6th of 8 on per_round RPS (negbin_glm, bayesian_poisson, poisson_glm and sarimax all beat it on R16-only RPS). This is the third consecutive round (MD3, R32, R16) where the champion is not the round leader, reinforcing the same RQ1 nuance: per_round retraining's payoff shows up in the cumulative trend, not reliably on any single round.
- **Worst matches (mean team-aware RPS, per_round):** Brazil 1–2 Norway (0.446, by far the round's biggest miss — models had Brazil as a heavy favorite, λ_h≈1.7–1.9 vs λ_a≈0.9–1.0, and Norway won anyway), Mexico 2–3 England (0.199), United States 1–4 Belgium (0.176), Portugal 0–1 Spain (0.168), Canada 0–3 Morocco (0.159), Switzerland 0–0 Colombia (0.155, another unpriced low-scoring draw — same recurring failure mode as every prior round).
- **Best matches (per_round):** Argentina 3–2 Egypt (0.018), Paraguay 0–1 France (0.036) — both correctly priced favorites, though Argentina 3–2 was a closer scoreline than the low RPS implies (outcome-only scoring rewards getting the W/D/L right regardless of margin).
- **Alert window (rolling 24 = all 16 R32 + all 8 R16, spanning Jun 28–Jul 7):** all 7 team-aware models sit at 0.127–0.145 RPS in both modes, well under the 0.235 static naive floor — no breach. `mean_rate_poisson` printed 0.2453 on the window, comfortably above the floor and its 0.229 holdout baseline; the floor continues to behave as designed.
- **KO sample-size caveat, now more acute.** R16 has only 8 matches — the smallest matchday yet (half of R32's 16) — so the ~3.6% frozen↔per_round RPS gap and all the round-only rankings above are well within noise for a single-round read. Trust the sign and the qualitative pattern (champion not leading recent rounds; refit's cumulative edge holding) more than the exact magnitudes until QF/SF pool more matches.

---

## Quarter-finals (~Jul 10–11)

Matches played: [fill]

Pipeline:
- per_round refit fired? Y/N. New run_id:
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

### TODO — offline frozen-shadow reconstruction (shadow-resolution bug)

Why: `wc_shadow` has no cadence alias and frozen shadow refits write no `cadence_mode`
tag, so `load_shadow_model` resolved BOTH cadences to the newest (per_round) version.
The logged "frozen" rows for the per_round-refit roster shadows (`poisson_glm`,
`bayesian_poisson`, `mean_rate_poisson`) are therefore the per_round artifact, not the
true frozen model. Affects every matchday's frozen shadow rows (MD1 onward), not just MD2.

What's already correct (no rework): the champion both cadences (alias-separated), and the
4 never-refit shadows (`negbin_glm`, `ridge`, `random_forest`, `sarimax`); and the entire
**per_round** column for all models.

Reconstruction steps (run after the Final, do NOT touch the live pipeline mid-tournament):
1. Identify the genuinely-frozen shadow versions in `wc_shadow` (latest `stage=shadow-refit`,
   no `cadence_mode` tag — e.g. poisson_glm v88) for poisson_glm / bayesian_poisson / mean_rate.
2. For each settled match, rebuild its pre-kickoff feature row from the DVC-versioned Gold
   snapshot of the corresponding pre-kickoff inference cycle (strict `inference_timestamp < kickoff`).
3. Predict with the frozen shadow versions, recompute RPS / NLL / RMSE_h / RMSE_a, and emit a
   corrected frozen leaderboard; backfill the (frozen) values for those 3 models in the tables above.
4. Prereq: ensure Gold/DVC history for every cycle is retained so step 2 is reproducible — verify
   before relying on it.
5. Separately, land the code fix (NOT mid-tournament): tag `run_shadow_refit` runs with
   `cadence_mode=frozen` AND make `_latest_version_with_tags` fallback cadence-aware (skip versions
   whose run carries a *different* explicit `cadence_mode`), or add per-cadence shadow aliases
   mirroring the champion. Add a regression test under `tests/models/`.

### TODO — host-advantage fix regeneration (RQ2 entropy)

Why: the sim swapped already-correct host-home rates (scramble bug, see MD3 Jun 27),
invalidating host-path advancement probabilities and the RQ2 Shannon-entropy trajectories
built from them. Per-match predictions, RQ1/RQ3, monitoring, Gold are clean. Also folds in
the one already-logged mid-R32 cycle that stayed unlocked under the Jun 29 KO-results
locking bug (fixed live in `20260629a`): replaying that cycle through the corrected sim
with locked KO results in step 2 regenerates its bracket bit-identically from DVC. **Also
folds in the pen-winner bracket bug (QF pipeline note, fixed live in `20260712a`):** every
snapshot from the first post-Switzerland–Colombia cycle (Jul 7) through the fix deploy
contaminated `tournament_probabilities.csv`, `ko_pairings.csv`, and `ko_fixtures.csv`
uniformly across all models × both cadences (Colombia wrongly advanced, real QF result
unlockable). The same D.1 replay through the corrected `simulate_tournament` (same seeds,
locked KO results) regenerates these bit-faithfully since the pen-winner fix will be in the
replay code.

Steps (run with D.1, after the Final, alongside the frozen-shadow rebuild):
1. Confirm swap deletion + venue-aware orientation + host-vs-host dual-orientation merged
   and regression tests pass.
2. Replay each cycle's DVC-versioned per-match predictions through the corrected
   `simulate_tournament` (same seed) -> regenerate advancement / ko_pairings / ko_fixtures.
3. Recompute Shannon entropy per snapshot (normalise p_i/32, H = -sum p_i log p_i); rebuild
   frozen-vs-per-round entropy resolution curves for all roster models.
4. Discard pre-fix host-path bracket/advancement figures; regenerate thesis plots.
5. Combine into the single D.1 pass with the frozen-shadow reconstruction.
6. Prereq: per-cycle prediction DVC history retained (shared with frozen-shadow TODO).

### TODO — backfill dropped never-refit shadow rows (data completeness)

Why: the silent shadow-skip path (`_safe_shadow_predict` → DagsHub load timeout) has now
dropped a never-refit shadow from **three** inference cycles across the tournament, one per
matchday since MD2:

- **MD3 per_round `ridge`** — cycle `ed64fbfac5ed487ca222cf0a9d2c938d`; 2 holes: Colombia
  0–0 Portugal (`1489419`) and DR Congo 3–1 Uzbekistan (`1539013`), both Jun 27.
- **R32 frozen `random_forest`** — 1 hole: Ivory Coast 1–2 Norway (`1564789`, Jun 30).
  Never-refit ⇒ cadence-invariant, so the per_round row on the same fixture is byte-identical
  to what frozen would have logged (verified: 0/15 RPS or λ diffs across cadences on the
  common R32 `random_forest` rows).

**Not** covered by the frozen-shadow reconstruction — that rebuilds *frozen* rows for the 3
contaminated *refit-eligible* shadows, whereas these are dropped rows on *never-refit* shadows
the reconstruction explicitly leaves untouched. List all instances here so nothing is missed.

Steps (fold into the single D.1 pass — same mechanics and prereq as the frozen-shadow rebuild):
1. For each dropped row, resolve the shadow version that the affected cycle would have loaded
   (never-refit ⇒ same version as every other row for that model, so deterministic).
2. Rebuild the pre-kickoff feature row from the DVC-versioned Gold snapshot of that cycle
   (strict `inference_timestamp < kickoff`).
3. Predict, recompute RPS / NLL / RMSE_h / RMSE_a, and backfill:
   - `ridge` per_round → +2 rows (MD3 = 24/24, overall = 88/88).
   - `random_forest` frozen → +1 row (R32 = 16/16, overall = 88/88). Alternatively, copy
     the byte-identical per_round row for the same fixture (both paths give the same result).
   (The frozen `mean_rate_poisson` MD1 hole — Portugal 1–1 DR Congo — is already handled:
   cadence-invariant, back-filled from per_round, and recomputed anyway by the frozen-shadow
   rebuild.)
4. Prereq: per-cycle Gold/DVC history retained (shared with the two TODOs above).

### TODO — reconstruct missed entropy-trajectory snapshots (pipeline outages)

Why: a pipeline outage collapses per-match resolution steps into a single jump in the RQ2
advancement-entropy trajectory. Affects **every (model × cadence_mode) trajectory uniformly**;
RQ1/monitoring and refit gates are untouched. Two cases:

1. **R32 scheduler-pause gap (Jun 29–30):** `wc-pipeline-trigger` was left paused over the
   Jun 29–30 R32 evening (see R32 pipeline note), so the hourly cadence missed 3 intermediate
   locked states — pre-Japan–Brazil; Japan–Brazil locked / pre-Germany–Paraguay; Germany–Paraguay
   locked / pre-Netherlands–Morocco. The Jun 30 catch-up cycle locked all three results
   (Japan–Brazil, Germany–Paraguay 4–5, Netherlands–Morocco 3–4) at once, collapsing three
   per-match resolution steps into one jump.
2. **R16 IPv6-stall gap (Jul 5–6):** every trigger run from Jul 5 was killed by the 90 min task
   timeout at the ELO freshness check (see R16 pipeline note above), so the hourly cadence
   produced no inference cycle in the window bounded by Brazil–Norway and Mexico–England. One
   missed snapshot: lock all matches with `kickoff <= Brazil–Norway`, re-run with
   `simulation_seed = _seed_from_string("R16")`, insert one point at a synthetic
   `inference_timestamp` between the two kickoffs.

Reconstruction steps (fold into the single D.1 pass; do NOT touch the live pipeline). Steps
1–3 apply per case — use each round's own KO seed and kickoff cutoffs:
1. Establish the intermediate locked states by kickoff cutoff against the now-complete Bronze.
   KO results are keyed by `frozenset({home, away})`, so each state is just adding/removing one
   team-set entry from `ko_results` + `finished_fixtures`.
   - **R32 (3 states):** (a) lock all matches with `kickoff < Japan–Brazil`; (b) add
     Japan–Brazil; (c) add Germany–Paraguay (NED–Morocco still open).
   - **R16 (1 state):** lock all matches with `kickoff <= Brazil–Norway` (Mexico–England still
     open).
2. For each state, re-run `run_inference_and_simulation(cadence_mode=...)` for both modes on
   current code with the round's fixed seed (`_seed_from_string("R32")` or `"R16"`). R32 base
   Gold needs no special checkout — the pipeline was off, so current DVC-versioned Gold is the
   exact pre-off-window state. R16 base Gold is likewise unchanged (failed runs died before
   ingestion).
3. Take each `tournament_probabilities.csv` (per model × mode), normalise the advancement
   vector (`p_i = adv_i / 32`, `H = -Σ p_i log p_i`), and insert the points into the trajectory
   with synthetic `inference_timestamp`s between the respective kickoffs so they order correctly
   in the entropy resolution curve.
4. Before relying on this, confirm in MLflow exactly which inference cycles are missing across
   each outage window (compare `inference_timestamp`s to the relevant kickoff times) so the
   snapshot counts are exact and no additional gap is overlooked (R32: 3 snapshots; R16: 1
   snapshot between Brazil–Norway and Mexico–England).
5. Prereq: per-cycle Gold/DVC history retained (shared with the TODOs above).

### Decision — silent shadow-skip: document, do NOT build a fix

The hole above comes from `run_prediction_all_models` silently dropping a shadow on
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
