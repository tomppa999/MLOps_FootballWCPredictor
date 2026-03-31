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
a team's tactical identity. Coach identity was considered as a feature but excluded because tactical
patterns are better captured through unsupervised clustering of play-style features. This simplification
should be acknowledged; a coach transition close to a tournament is a known model weakness.

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

## Monitoring and deployment threats

**No automated alerting**
MLflow logs all training metrics and run artifacts, but there is no automated alert if model
performance degrades after deployment. Monitoring is passive and requires manual inspection
of the MLflow dashboard.

**Elo source reliability**
Elo ratings are fetched from a third-party website (eloratings.net). If the source changes its
structure, becomes unavailable, or delays updates during an active tournament, the fallback is
the last validated snapshot. This introduces staleness risk precisely when current ratings matter most.

**API-Football rate limits and silent failures**
The ingestion pipeline is rate-limited by the API-Football subscription tier. Silent failures
(API returns HTTP 200 with an `errors` field rather than a non-2xx status) were identified during
development. The pipeline surfaces API errors explicitly, but a network outage or plan change
could result in incomplete data pulls that are difficult to detect without downstream validation.
