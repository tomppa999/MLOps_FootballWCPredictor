# Feature Specification

## Principles
- all features must use only information available before the target match
- no future leakage
- stable train/inference schema
- Gold is one row per match

## Gold row structure
Each Gold row represents one match and should include:
- match identifiers and metadata
- home team pre-match features
- away team pre-match features
- selected difference features

## Feature groups

### Strength features
- home_elo_pre
- away_elo_pre
- elo_diff

### Form features
Weighted rolling statistics from prior matches for each team:
- home_team_rolling_goals_for
- home_team_rolling_goals_against
- away_team_rolling_goals_for
- away_team_rolling_goals_against
- home_team_rolling_shots
- away_team_rolling_shots
- home_team_rolling_shot_accuracy
- away_team_rolling_shot_accuracy
- home_team_rolling_conversion
- away_team_rolling_conversion

Where useful:
- shot_accuracy = shots_on_target / total_shots
- conversion = goals / shots_on_target

Ratio features must use safe handling for zero or very small denominators.

### Tactical profile features
Use rolling team-stat summaries to represent tactical approach. Available statistics may include:
- Shots on Goal
- Shots off Goal
- Shots insidebox
- Shots outsidebox
- Total Shots
- Blocked Shots
- Fouls
- Corner Kicks
- Offsides
- Ball Possession
- Yellow Cards
- Red Cards
- Goalkeeper Saves
- Total passes
- Passes accurate
- Passes %

These should be aggregated into rolling pre-match summaries for each team.

### Tactical clustering
Use unsupervised learning on rolling tactical/statistical summaries to assign style clusters or style indicators.

Examples of broad tactical tendencies that may later be interpreted from the learned clusters:
- ball control / possession-oriented play
- more direct or riskier attacking patterns
- aggressive or foul-heavy defensive behavior
- inside-box vs outside-box shooting tendencies

Requirements:
- time-aware fitting
- deterministic where feasible
- no future-data leakage

Cluster outputs may be represented in Gold as:
- home_tactical_cluster
- away_tactical_cluster
- or cluster-derived numeric summaries

The clustering step does not need to predefine exact tactical dimensions perfectly in advance. It is acceptable to cluster on the statistical profile first and interpret the resulting clusters afterward.

### Context features
- neutral indicator or venue type
- competition_tier
- is_knockout
- confederation / continent indicators

competition_tier should be a compact ordinal feature derived from API-Football competition metadata:

- 1 = FIFA World Cup
- 2 = continental final tournament
- 3 = World Cup qualification, continental qualification, or Nations League
- 4 = friendly or other

The exact mapping logic should be defined centrally and applied consistently in the raw-to-Silver transformation.

is_knockout should indicate whether the match is part of a knockout stage rather than a league/group/qualification stage.
This should be derived from match stage or round metadata from API-Football, not guessed only from competition name.

## Targets
The modeling target should be based on goals / score-distribution behavior, not direct W-D-L outcomes as the main target.

Likely base targets:
- home_goals
- away_goals

More advanced count-model parameters may be estimated in model-specific logic if needed.

## Feature availability by stats_tier

Rolling form and tactical features derived from match statistics are only
contributed by prior matches that have sufficient statistics.  The `stats_tier`
column in Silver records what was available for each historical match.

| `stats_tier` | Features contributed to rolling windows |
|---|---|
| `full` | all rolling shot, possession, pass, and card features |
| `partial` | shot and card features; possession and pass features may be null |
| `cards_only` | card features only |
| `none` | no statistical features; only goals-based form features contributed |

Concretely, when computing a team's rolling feature vector entering match M,
only prior matches with `stats_tier != 'none'` contribute to shot/possession/
pass rolling averages.  Goals-based form features (rolling goals for/against,
conversion rate) use all prior matches regardless of tier.

The `stats_tier` of the **prior matches** — not the current match — determines
what was available at feature-build time.  This is the correct time-aware
interpretation: we never look at the target match's own statistics as a feature.

In practice, `stats_tier='full'` coverage is sparse before 2020 and reliable
from 2020 onwards.  Tactical and possession features will therefore be missing
for older training rows.  Models must handle this gracefully (e.g. missing
indicator, imputation from team mean, or conditional feature set).

## Explicit non-features
- no xG dependency (not offered by our data source)
- no manually reimplemented Elo calculation (they are taken from the .tsv files only)
- no rest-days feature in the initial version (does not make sense for national team match calendar)