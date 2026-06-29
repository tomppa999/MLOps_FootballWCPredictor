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

## Round of 32 (~Jun 27 – Jul 2)

Matches played: [fill]

Pipeline:
- per_round refit fired at R32 boundary? Y/N. New run_id:
- Bracket configuration: which 8 third-placers advanced? [fill] — note if any
  unusual configuration affects the bracket unpredictably.
- Any failures:
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
built from them. Per-match predictions, RQ1/RQ3, monitoring, Gold are clean.

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

### TODO — backfill dropped per-round `ridge` rows (data completeness)

Why: the silent shadow-skip path (`_safe_shadow_predict` → DagsHub load timeout) dropped
`ridge` from one per-round inference cycle (`ed64fbfac5ed487ca222cf0a9d2c938d`), leaving 2
holes in the per_round monitoring artifact: Colombia 0–0 Portugal (`1489419`) and DR Congo
3–1 Uzbekistan (`1539013`), both Jun 27 (see MD3 pipeline note). **Not** covered by the
frozen-shadow reconstruction above — that rebuilds *frozen* rows for the 3 contaminated
roster shadows, whereas these are *per_round* rows for `ridge`, a never-refit shadow the
reconstruction explicitly leaves untouched. List separately so it isn't missed.

Steps (fold into the single D.1 pass — same mechanics and prereq as the frozen-shadow rebuild):
1. Resolve the per_round `ridge` shadow version that cycle `ed64fbfac5…` would have loaded
   (never-refit ⇒ same version as every other `ridge` row, so deterministic).
2. Rebuild the pre-kickoff feature rows for the 2 fixtures from the DVC-versioned Gold
   snapshot of that cycle (strict `inference_timestamp < kickoff`).
3. Predict, recompute RPS / NLL / RMSE_h / RMSE_a, backfill the 2 rows; `ridge` then reads
   24/72 like the others. (The frozen `mean_rate_poisson` MD1 hole — Portugal 1–1 DR Congo —
   is already handled: cadence-invariant, back-filled from per_round, and recomputed anyway
   by the frozen-shadow rebuild.)
4. Prereq: per-cycle Gold/DVC history retained (shared with the two TODOs above).

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
