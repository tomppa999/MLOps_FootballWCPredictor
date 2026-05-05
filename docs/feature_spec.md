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
- elo_diff (= home_elo_pre − away_elo_pre): which side is favoured.
- elo_sum (= home_elo_pre + away_elo_pre): overall match quality / strength
  composition (top-vs-top vs top-vs-bottom). Added in the Gold v2 iteration;
  consistently top-5 in importance for tree-based and additive baseline models
  (see report Section 7.2).

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

### Tactical profile features (rolling)

The 5 rolling tactical profile columns per side are computed from prior matches
with `stats_tier != 'none'` (same filter as shot features):

| Column | Tactical proxy |
|---|---|
| `{side}_team_rolling_tac_total_shots` | Shot volume / attacking directness |
| `{side}_team_rolling_tac_shot_precision` | Shot accuracy (shots\_on\_goal / total\_shots) |
| `{side}_team_rolling_tac_fouls` | Aggression / defensive behavior |
| `{side}_team_rolling_tac_corner_kicks` | Set-piece tendency |
| `{side}_team_rolling_tac_possession_pct` | Ball control / possession style |

The original 6-column design (`shots_on_goal`, `shots_off_goal`, `total_shots`,
`fouls`, `corner_kicks`, `possession_pct`) was consolidated to 5 columns by
replacing `shots_on_goal` and `shots_off_goal` with `total_shots` +
`shot_precision` (= shots\_on\_goal / total\_shots), reducing redundancy while
preserving the same information (see Section 5.2 of the report). Detailed
columns (`shots_insidebox`, `shots_outsidebox`, `total_passes`,
`passes_accurate`, `pass_accuracy_pct`, `blocked_shots`) are excluded due to
poor availability in the `partial` tier (~1.6% non-null).

### Squad cohesion features (time since last national-team match)

These three columns proxy national-team-level squad cohesion
(*Eingespieltheit*) — the time the unit has been training and playing
together since their last international match. They are **not** a player-rest
signal: at international cadence, players play club matches between
national-team windows, so individual fatigue is not what is measured. What is
measured is *time-as-a-unit*, which proxies tactical familiarity, set-piece
coordination, and roster stability.

- `home_days_since_last_match` — gap (in days) between the home team's
  current match and its previous international match. NaN on first appearance
  in the dataset.
- `away_days_since_last_match` — same for the away team.
- `rest_diff` (= home − away) — signed gap difference. NaN propagates if
  either side is NaN. Variable name retained from the initial design where
  player-rest was the hypothesised mechanism.

Per-team gap is computed via team-centric reshape and
`groupby('team').shift(1)` on chronologically sorted match history (mirroring
the pattern in `_build_team_history`). All three columns are strictly
time-aware: each match's feature uses only the team's previous match date
(strictly < current match date).

In the v2 importance analysis these features only surfaced in Bayesian
Poisson's top-5 (`rest_diff` #2, `away_days_since_last_match` #4): the
prior-driven sampler can absorb a weak-but-real signal that other model
families suppress as noise. See report Section 7.2 for the full
interpretation.

### Tactical clustering (dropped)

KMeans clustering on rolling tactical profiles was investigated but dropped.
Silhouette analysis across five column-subset and PCA variants (see
`src/gold/cluster_experiment_v2.py`, `docs/figures/cluster_experiment_v2.png`)
showed all silhouette scores below 0.40 and no variant producing well-separated
clusters at any interpretable k. The rolling tactical profile columns are used
directly as continuous features instead.

### Time-series support

- `home_team_match_index` / `away_team_match_index` — 1-indexed ordinal count
  of each team's match appearances in chronological order. Used by SARIMAX as
  the time axis (not a predictor feature). Counts across both home and away
  appearances.

### Context features
- `is_neutral` — venue neutrality flag. For 2026 WC group-stage matches where a
  host nation (USA, Canada, Mexico) plays at home, `is_neutral` is overridden to
  `False` so the model applies the home-advantage effect it learned from
  historical data. No separate `is_home_advantage_2026` column exists.
- `competition_tier`
- `is_knockout`
- `is_cross_confederation` — nullable boolean, True if the two teams'
  confederations differ. NaN if either confederation is missing (never
  silently coerced to True/False). Added in the Gold v2 iteration; in
  practice contributed near-zero importance in all nine models because it
  is heavily confounded with `competition_tier` (most cross-confederation
  matches are friendlies).
- confederation / continent indicators

#### 2026 KO-round home advantage (inference-time only)

For knockout rounds, venue assignment depends on bracket outcomes. The tournament
simulation carries a `venue_country` mapping (venue city → host country) in
`data/tournament/wc2026.json` — **not** Gold features. When the simulation
places a host nation into a KO match at a venue in their country, `is_neutral`
is overridden to `False` for that simulated match. This works for both
deterministic and probabilistic (Monte Carlo) inference. The mapping is generated
once by `scripts/scrape_wc2026_bracket.py` and committed to the repo.

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
- no `rest_days` feature in the player-fatigue sense: at international
  cadence, players play club matches between national-team windows, so a
  literal player-rest feature is not meaningful. The Gold v2 iteration
  introduced `days_since_last_match` (per side) and `rest_diff` as squad-
  cohesion proxies instead — see "Squad cohesion features" above. Variable
  name `rest_diff` is retained from the initial design hypothesis.