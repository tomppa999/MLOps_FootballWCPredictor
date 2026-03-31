"""Time-aware rolling feature computation for the Gold layer.

For each match M and each side (home/away), we look at the team's last N
prior matches (strictly date < M.date) and compute rolling averages from the
*team's perspective* (goals for/against, shots, etc.), regardless of whether
the team was home or away in those prior matches.

Feature availability respects ``stats_tier``:
- Goals-based features use all prior matches regardless of tier.
- Shot/tactical features use only prior matches with ``stats_tier != 'none'``.

Also computes per-team ordinal match indices for SARIMAX time-series support.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WINDOW: Final[int] = 10

# Raw stats extracted from Silver into per-team history rows.
_RAW_TACTICAL_STATS: Final[list[str]] = [
    "shots_on_goal",
    "total_shots",
    "fouls",
    "corner_kicks",
    "possession_pct",
]

# Output tactical profile suffixes used for clustering and Gold columns.
# shot_precision = shots_on_goal / total_shots (NaN when total_shots == 0).
TACTICAL_STAT_SUFFIXES: Final[list[str]] = [
    "total_shots",
    "shot_precision",
    "fouls",
    "corner_kicks",
    "possession_pct",
]


def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Element-wise division that returns NaN where denominator is zero."""
    return num / denom.replace(0, np.nan)


def _build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape Silver into a team-centric history with one row per (match, team).

    Each row records the team's goals_for, goals_against, the 6 tactical stats,
    and the match's stats_tier / date.
    """
    records = []
    for prefix, goals_for_col, goals_against_col in [
        ("home", "home_goals", "away_goals"),
        ("away", "away_goals", "home_goals"),
    ]:
        team_col = f"{prefix}_team"
        cols: dict = {
            "fixture_id": df["fixture_id"],
            "date_utc": df["date_utc"],
            "team": df[team_col],
            "goals_for": pd.to_numeric(df[goals_for_col], errors="coerce"),
            "goals_against": pd.to_numeric(df[goals_against_col], errors="coerce"),
            "stats_tier": df["stats_tier"],
        }
        for suffix in _RAW_TACTICAL_STATS:
            cols[suffix] = pd.to_numeric(
                df[f"{prefix}_{suffix}"], errors="coerce"
            )

        records.append(pd.DataFrame(cols))

    history = pd.concat(records, ignore_index=True)
    history = history.sort_values(["team", "date_utc", "fixture_id"]).reset_index(
        drop=True
    )
    return history


def _rolling_for_team(
    team_history: pd.DataFrame, n: int
) -> dict[str, float | None]:
    """Compute rolling averages from a team's prior-match slice.

    ``team_history`` must already be filtered to strictly-prior matches and
    sorted chronologically. We take the last ``n`` rows.
    """
    nan_result: dict[str, float] = {
        "rolling_goals_for": np.nan,
        "rolling_goals_against": np.nan,
        "rolling_shots": np.nan,
        "rolling_shot_accuracy": np.nan,
        "rolling_conversion": np.nan,
    }
    for suffix in TACTICAL_STAT_SUFFIXES:
        nan_result[f"rolling_tac_{suffix}"] = np.nan

    if team_history.empty:
        return nan_result

    last_n = team_history.tail(n)

    result: dict[str, float] = {
        "rolling_goals_for": last_n["goals_for"].mean(),
        "rolling_goals_against": last_n["goals_against"].mean(),
    }

    # Shot + tactical features: only matches with actual stats
    with_stats = last_n[last_n["stats_tier"] != "none"]
    if with_stats.empty:
        result["rolling_shots"] = np.nan
        result["rolling_shot_accuracy"] = np.nan
        result["rolling_conversion"] = np.nan
        for suffix in TACTICAL_STAT_SUFFIXES:
            result[f"rolling_tac_{suffix}"] = np.nan
    else:
        total_shots_sum = with_stats["total_shots"].sum()
        shots_on_sum = with_stats["shots_on_goal"].sum()
        goals_sum = with_stats["goals_for"].sum()

        result["rolling_shots"] = with_stats["total_shots"].mean()
        result["rolling_shot_accuracy"] = _safe_div(
            pd.Series([shots_on_sum]), pd.Series([total_shots_sum])
        ).iloc[0]
        result["rolling_conversion"] = _safe_div(
            pd.Series([goals_sum]), pd.Series([shots_on_sum])
        ).iloc[0]

        for suffix in ("total_shots", "fouls", "corner_kicks", "possession_pct"):
            result[f"rolling_tac_{suffix}"] = with_stats[suffix].mean()

        result["rolling_tac_shot_precision"] = _safe_div(
            pd.Series([shots_on_sum]), pd.Series([total_shots_sum])
        ).iloc[0]

    return result


def compute_rolling_features(
    df: pd.DataFrame, n_matches: int = DEFAULT_WINDOW
) -> pd.DataFrame:
    """Compute rolling pre-match features and match indices for a Silver DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Silver-schema DataFrame. Must contain ``date_utc``, ``home_team``,
        ``away_team``, goals columns, stat columns, and ``stats_tier``.
    n_matches : int
        Number of prior matches to consider for rolling averages.

    Returns
    -------
    pd.DataFrame
        Same index as *df*, with rolling feature columns and match indices appended.
    """
    df = df.copy()
    df["date_utc"] = pd.to_datetime(df["date_utc"])
    df = df.sort_values(["date_utc", "fixture_id"]).reset_index(drop=True)

    history = _build_team_history(df)

    team_groups: dict[str, pd.DataFrame] = {
        team: grp for team, grp in history.groupby("team")
    }

    rolling_rows: list[dict] = []

    for idx, row in df.iterrows():
        match_date = row["date_utc"]
        result: dict = {}

        for side in ("home", "away"):
            team = row[f"{side}_team"]
            prefix = f"{side}_team"

            team_hist = team_groups.get(team)
            if team_hist is not None:
                prior = team_hist[team_hist["date_utc"] < match_date]
            else:
                prior = pd.DataFrame()

            # Match index: 1-indexed ordinal count of this team's appearances
            result[f"{prefix}_match_index"] = len(prior) + 1

            feats = _rolling_for_team(prior, n_matches)
            for key, val in feats.items():
                result[f"{prefix}_{key}"] = val

        rolling_rows.append(result)

    rolling_df = pd.DataFrame(rolling_rows, index=df.index)
    return pd.concat([df, rolling_df], axis=1)
