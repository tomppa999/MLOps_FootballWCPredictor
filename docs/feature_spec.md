# Feature Specification

## Principles

- All features must use only information available before the target match.
- No future leakage.
- Stable training/inference schema.
- Gold is one row per match.
- The modeling target is goals / score-distribution behavior, not direct W/D/L.

## Gold Row Structure

Each Gold row represents one historical match and includes:

- Match identifiers and metadata.
- Home-team pre-match features.
- Away-team pre-match features.
- Selected difference/context features.
- Targets (`home_goals`, `away_goals`) for historical rows.
- Retained audit columns that are useful for transparency but not necessarily
  part of `FEATURE_COLUMNS`.

## Modeling Feature Set

The thesis feature set is intentionally compact. It keeps full-coverage,
time-aware strength, context, and form signals, and excludes sparse in-game
statistics from the active modeling feature set.

### Strength Features

- `home_elo_pre`
- `away_elo_pre`
- `elo_diff` (= `home_elo_pre - away_elo_pre`): which side is favoured.
- `elo_sum` (= `home_elo_pre + away_elo_pre`): overall match quality / strength
  composition (top-vs-top vs top-vs-bottom). Added in the Gold v2 iteration;
  consistently top-5 in importance for tree-based and additive baseline models.

Elo values are sourced from EloRatings snapshots. The project does not
reimplement Elo update formulas.

### Goal-Form Features

Rolling goal statistics are computed from each team's prior matches only:

- `home_team_rolling_goals_for`
- `home_team_rolling_goals_against`
- `away_team_rolling_goals_for`
- `away_team_rolling_goals_against`

These use all prior matches, regardless of `stats_tier`, because goals are
available for every settled fixture.

### Rolling Elo-Change Features

Rolling Elo-change captures opponent-quality-adjusted recent form:

- `home_team_rolling_elo_change`
- `away_team_rolling_elo_change`

For each team, the feature is computed as the sum of `(elo_post - elo_pre)` over
the last `N` prior matches. The default window is 5 matches; candidate windows
3, 5, and 10 are evaluated during the thesis refit cycle. The selected window
is treated as a feature-engineering hyperparameter and logged with the model
selection results.

These features have full coverage apart from each team's first appearances and
do not require a new data source. They are strictly time-aware: the target
match's own post-match Elo is never used.

### Squad-Cohesion Features

These columns proxy national-team-level squad cohesion (*Eingespieltheit*) —
the time the unit has had since its last international match. They are not a
player-rest signal: at international cadence, players play club matches between
national-team windows, so individual fatigue is not what is measured.

- `home_days_since_last_match`: days since the home team's previous
  international match. NaN on first appearance in the dataset.
- `away_days_since_last_match`: same for the away team.
- `rest_diff` (= home - away): signed gap difference. NaN propagates if either
  side is NaN. Variable name retained from the initial design where player-rest
  was the hypothesised mechanism.

Per-team gaps are computed via team-centric reshape and
`groupby("team").shift(1)` on chronologically sorted match history. Each match's
feature uses only the team's previous match date (`< current match date`).

### Context Features

- `competition_tier`
- `is_knockout`
- `is_neutral`

`competition_tier` is a compact ordinal feature derived from API-Football
competition metadata:

| Tier | Meaning |
|---:|---|
| 1 | FIFA World Cup |
| 2 | Continental final tournament |
| 3 | World Cup qualification, continental qualification, or Nations League |
| 4 | Friendly or other |

The exact mapping logic is defined centrally in the Silver transformation and
applied consistently before Gold is built.

`is_knockout` indicates whether the match belongs to a knockout stage rather
than a league/group/qualification stage. It is derived from match stage or round
metadata, not guessed only from competition name.

`is_neutral` is the venue neutrality flag. For 2026 World Cup host nations
(USA, Canada, Mexico), inference overrides `is_neutral` to `False` when a host
plays in its own country, so the model applies the home-advantage effect learned
from historical data. No separate `is_home_advantage_2026` feature exists.

#### 2026 KO-Round Home Advantage

For knockout rounds, venue assignment depends on simulated bracket outcomes.
The tournament simulation carries a `venue_country` mapping (venue city to host
country) in `data/tournament/wc2026.json`, not in Gold. When a simulated path
places a host nation into a knockout match at a venue in its own country,
`is_neutral` is overridden to `False` for that simulated match. This works for
both deterministic and Monte Carlo inference.

### Time-Series Support Columns

- `home_team_match_index`
- `away_team_match_index`

These are 1-indexed ordinal counts of each team's match appearances in
chronological order. They support SARIMAX as the time axis and are not ordinary
predictor features for every model family.

## Retained Non-Feature Columns

Gold may retain columns that are useful for audit, analysis, and transparency
but are excluded from `FEATURE_COLUMNS`.

### Rolling Shot Columns

The following shot-derived rolling columns are retained in Gold but excluded
from the thesis modeling feature set:

- `home_team_rolling_shots`
- `away_team_rolling_shots`
- `home_team_rolling_shot_accuracy`
- `away_team_rolling_shot_accuracy`
- `home_team_rolling_conversion`
- `away_team_rolling_conversion`

Where used for audit:

- `shot_accuracy = shots_on_target / total_shots`
- `conversion = goals / shots_on_target`

Ratios use safe handling for zero or very small denominators.

Reason for exclusion: permutation importance was negligible, and the underlying
statistics have a large coverage gap that biases training toward UEFA/CONMEBOL.

### Rolling Tactical Profile Columns

The following tactical profile columns are retained in Gold but excluded from
the thesis modeling feature set:

| Column | Tactical proxy |
|---|---|
| `{side}_team_rolling_tac_total_shots` | Shot volume / attacking directness |
| `{side}_team_rolling_tac_shot_precision` | Shot accuracy (`shots_on_goal / total_shots`) |
| `{side}_team_rolling_tac_fouls` | Aggression / defensive behavior |
| `{side}_team_rolling_tac_corner_kicks` | Set-piece tendency |
| `{side}_team_rolling_tac_possession_pct` | Ball control / possession style |

These columns are computed from prior matches with `stats_tier != "none"` when
available. They are no longer used as model inputs in the thesis feature set.

### Dropped Feature Experiments

- `is_cross_confederation` was added in Gold v2 but removed after importance
  analysis showed it invisible in all nine feature-based models. It was heavily
  confounded with `competition_tier` because most cross-confederation matches
  are friendlies.
- Tactical clustering was investigated and dropped. Silhouette analysis across
  five column-subset and PCA variants showed all scores below 0.40 and no
  interpretable cluster structure.

## Sample Weights (Training Only)

Sample weights are not Gold features. They are computed in the training harness
and passed to model fitting.

`competition_tier` therefore has a dual role:

- As a predictor feature: lets the model adjust predicted goal rates for the
  match type being predicted.
- As a training-weight input: tells the optimizer that higher-stakes matches
  should contribute more strongly to parameter estimation.

These roles are complementary, not redundant.

### Time Decay

`w_time = 0.5 ** (days_ago / (half_period_years * 365.25))`

`days_ago` is computed relative to the training cutoff. `half_period_years`
defaults to 3 years following Ley et al. (2019) and is tuned via Optuna in
`[1.0, 5.0]` for models that support sample weights. The mean-rate Poisson
baseline fixes the value at 3 years.

### Match Importance

Weights follow Ley et al. (2019), based on the pre-2018 FIFA ranking
methodology:

| `competition_tier` | Weight |
|---:|---:|
| 1 (World Cup) | 4 |
| 2 (continental final) | 3 |
| 3 (qualifier, Nations League) | 2.5 |
| 4 (friendly) | 1 |

Final sample weight:

`sample_weight = w_time * w_importance`

## Targets

The base supervised targets are:

- `home_goals`
- `away_goals`

Models predict goals, expected goals, or goal-distribution parameters first.
Home/draw/away probabilities are derived downstream from score distributions
and used for RPS evaluation and Monte Carlo tournament simulation.

## Feature Availability and Leakage Rules

All rolling features use prior matches only. The `stats_tier` of prior matches
determines which retained audit statistics can be computed, but stat-derived
shot/tactical columns are no longer active model features in the thesis setup.

| `stats_tier` | Retained audit columns contributed |
|---|---|
| `full` | Rolling shot and tactical columns where available |
| `partial` | Rolling shot/card-style columns where available |
| `cards_only` | No active model features; limited audit value |
| `none` | No stat-derived audit columns |

Goals-based rolling features and rolling Elo-change are independent of
`stats_tier` and use all prior matches.

The target match's own goals, cards, shots, possession, Elo change, or final
result are never used to construct its features.

## Explicit Non-Features

- No xG dependency (not offered by the data source).
- No manually reimplemented Elo calculation (Elo values are read from TSV
  snapshots).
- No direct W/D/L target as the main modeling objective.
- No `rest_days` feature in the player-fatigue sense; `days_since_last_match`
  and `rest_diff` are squad-cohesion proxies.
- No rolling shot or tactical columns in `FEATURE_COLUMNS` for the thesis model
  set; they are retained only for audit/transparency.