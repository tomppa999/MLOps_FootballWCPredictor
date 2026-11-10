# Threats to Validity

This document consolidates known threats to the validity of the pipeline, models, and results.
It serves as a reference for the Threats to Validity section of the project report.

---

## Data quality threats

**Sparse match statistics coverage**
Approximately 51% of matches in the Bronze dataset have any statistics (shots, fouls, etc.).
Coverage is significantly lower for CAF, AFC, and OFC confederation matches compared to UEFA and CONMEBOL.
The `stats_tier` column encodes coverage level, but models that rely on statistics-based features will
have reduced effective training data for non-European and non-South-American teams.

**Inconsistent statistics reporting**
Red Cards are reported as `null` rather than `0` when no cards were issued by the API provider.
Offsides and goalkeeper saves are occasionally null even when other stats are present — these are
provider-side omissions, not true zeros. Handled via imputation rules, but residual noise remains.

**Elo TSV data quality**
Some historical Elo rows contain malformed dates (`YYYY 00 00`). Mitigated by dropping pre-2018 rows
and applying a minimum year filter, but any future extension of the training window may re-expose this.

**Namibia `NA` country code**
The ISO country code `NA` (Namibia) is interpreted as a missing value by pandas by default.
Fixed via `keep_default_na=False`, but illustrates the fragility of external data contracts.
Any new country code that collides with a pandas NA alias would require the same fix.

**Residual Elo gaps**
A small number of fixtures (~3–8 depending on run) remain without an exact Elo match and fall back
to the last known Elo value via `merge_asof`. The fallback introduces a small approximation for
teams with infrequent Elo updates.

**North Macedonia code ambiguity**
The country code changed historically (`MK` vs `NM`). Handled via date-aware Elo code resolution,
but any future code changes would require a manual update to `data/mappings/elo_code_map.csv`.

---

## Pipeline and data architecture threats

**Dual-source team identity mismatch**
API-Football and Elo use different team names and country codes. Resolution relies on a manually
curated `team_mapping_master_merged.csv`. Every renamed nation, newly promoted team, or API name
change must be added manually. This is a persistent maintenance liability with no automated validation.

**`is_neutral` flag reliability**
The primary source for the neutral venue flag is Elo's venue column. When no exact Elo match exists,
a heuristic fallback is applied. A non-trivial share of matches — particularly in confederation
qualifiers — rely on the heuristic, which may misclassify some venues.

**Home/away swap in source data**
Some raw API-Football fixtures have home and away sides inverted relative to the Elo record.
Identified and handled during Silver transformation, but this remains a latent risk for future data pulls
if the API changes its reporting convention.

**±1 day timezone misalignment**
Matches scheduled near midnight UTC can appear on different calendar dates across sources.
Mitigated but not fully eliminated; affects Elo join accuracy for a small number of edge cases.

**Home advantage approximation for 2026 KO rounds**
The group-stage home advantage correction for USA, Canada, and Mexico is implemented using
pre-assigned venue data. For knockout rounds, venue assignment depends on the bracket outcome and
is only approximately modeled. Monte Carlo paths do not dynamically assign KO venues.

**Competition heterogeneity within the training set**
The Gold dataset spans both qualification matches (low-stakes, rotated squads) and tournament finals
(peak effort, first-choice XI) under a shared feature schema. The `competition_tier` and `is_knockout`
flags partially control for this, but residual heterogeneity in team motivation and effort is unmodeled.

**EURO/AFCON qualifier bundling**
Some tournament qualification campaigns share a `league_id` and `season` in the API with the
corresponding finals. Required date-based tier overrides during Silver transformation. Equivalent
quirks in future data would require the same manual fix.

---

## Football domain threats

**Sparse national team calendar**
National teams play 6–12 matches per year. Rolling form windows, Elo stability, and statistics-based
features are all computed over a much smaller per-team sample than club-football models, amplifying
noise and limiting the effective training signal.

**Long gaps between international windows**
Matches within a qualification window may be spaced only 3–4 days apart, while gaps between
windows can exceed 6 months. SARIMAX's integer-indexed time dimension treats all consecutive
observations as equally spaced, which simplifies the temporal structure. All models are affected
by the lack of continuity; rolling features computed across long breaks may include stale signal.

**Squad and coaching composition changes**
International squads rotate significantly between matches. Coaching changes can fundamentally alter
a team's tactical identity but are not modeled directly. Rolling form and Elo are the only proxies.
A coach transition close to a tournament is a known model weakness.

**2026 FIFA bracket format volatility**
The 2026 World Cup uses 12 groups of 4, with the top 2 from each group plus the best 8 of 12
third-place finishers advancing (32 total). The Round of 32 matchups depend on which specific
groups' third-placers qualify, mapped via a FIFA-defined lookup table — there are C(12,8) = 495
possible bracket configurations. A single goal difference in any group can restructure the entire
knockout tree, introducing significant volatility. EURO evidence since introducing the best-third
format shows a meaningful increase in draws on matchday 3, consistent with teams gaming the format
once qualification mathematics are known. There is insufficient data under this specific format to
model the behaviour reliably. The Monte Carlo simulation handles the bracket mapping correctly but
the underlying goal distribution models were trained on simpler tournament structures.

**Small overall sample size**
The combination of sparse per-team calendars and the national-team context means the effective
dataset is substantially smaller than comparable club football datasets. Models prone to overfitting
(LSTM, 1D CNN, XGBoost) require careful regularization.

---

## Evaluation threats

**Model selection under frozen conditions**
Model selection used pre-tournament holdout performance under frozen fitted parameters,
together with model-family diversity. Strong performance under that scheme does not imply
the greatest benefit from per-round refitting. The experiment measures the effect of
refitting for the selected models with fixed hyperparameters; it does not identify the
model class or hyperparameters best suited to repeated refitting.

**Dependence in accuracy and entropy comparisons**
RQ1 pairs the two forecasts on each match. RQ2 pairs snapshots and gives
each changed state one observation. Individual-observation bootstrap draws
and independent sign flips rely on assumptions about the joint distribution
of the paired observations. Matches can share teams and model fits.
Successive entropy states share tournament results and fitted parameters
even after deduplication. Pointwise interval coverage and permutation
p-values are conditional on the validity of those resampling assumptions.

**Post-result correction families**
The four [Holm families](notes/multiple_comparisons.md) were revised after
inspection of the results. The reporting hierarchy is exploratory. Holm's
within-family error control assumes valid individual p-values and a fixed
family. It addresses dependence among tests within such a family, while
dependence within the original resampling units remains a separate issue.

**Calibration trends in one tournament**
The analysis has 104 total matches, 80 treated matches, and four or five
trend periods, with small late-round cells. Shuffling period labels assumes
exchangeability across periods despite changing opponents, outcomes, and
forecast difficulty. Forecast arm swaps also require exchangeability under
their null. Binned ECE is sensitive to sample size and bin composition.
Observed calibration changes may reflect case mix or sampling variation.
These constraints limit inference about concept drift and whether
refitting mitigated it. Results concern two policies in one tournament,
with generalisation to other tournaments and policies left open.

## Limits of interpretation and reporting

- Separate comparisons within group and knockout phases leave the
  between-phase difference untested.
- RQ2 entropy describes the concentration of team probabilities. The
  comparison of levels leaves resolution speed and forecast quality open.
- Pointwise bootstrap intervals and Holm-adjusted p-values use different
  error criteria. For XGBoost knockout ECE the raw arm-swap p-value and
  bootstrap CI also disagree. They use different resampling procedures.
  Report both and qualify the inference as in the
  [findings memo](notes/findings_memo.md#levels-and-the-knockout-ece-result).
- The audited reconstruction and its identity checks belong in Methods
  as forecast provenance. See the
  [reconstruction record](notes/wc_live.md#done--strand-5-cadence-correct-frozen-entropy-supersedes-strand-4).

---

## Monitoring and deployment threats

**Alert threshold may be too permissive**
`evaluate_alert_threshold` runs automatically on every monitoring cycle and compares rolling-mean
RPS (last 24 matches) to each model's WC 2022 holdout baseline multiplied by `ALERT_FACTOR` (1.3).
That threshold can permit RPS worse than a uniform-random predictor (~0.235). A fix to use a
naive-baseline floor is planned (thesis Phase 1a).

**Alerts are log-only**
Breaches emit `logger.warning` only. There is no email, push, or Cloud Monitoring integration;
degradation is easy to miss unless logs are actively monitored during the tournament.

**Elo source reliability**
Elo ratings are fetched from a third-party website (eloratings.net). If the source changes its
structure, becomes unavailable, or delays updates during an active tournament, the fallback is
the last validated snapshot. This introduces staleness risk precisely when current ratings matter most.

**No automated data drift detection**
The pipeline does not implement statistical drift detection (e.g. PSI, KS tests, or KL divergence
monitors) on incoming features. The live monitoring measures forecast performance
on settled matches. Team strength and the mix of context features can change as
the field narrows. Over the short tournament, expected changes in fixture mix and
small period samples complicate interpretation of shifts. The offline RQ3
analysis therefore reports calibration diagnostics and their limits as described
under [evaluation threats](#evaluation-threats).

**API-Football rate limits and silent failures**
The ingestion pipeline is rate-limited by the API-Football subscription tier. Silent failures
(API returns HTTP 200 with an `errors` field rather than a non-2xx status) were identified during
development. The pipeline surfaces API errors explicitly, but a network outage or plan change
could result in incomplete data pulls that are difficult to detect without downstream validation.
