"""Time-aware temporal features for the Gold layer.

For each match M and each side (home/away) we compute the number of days
since that team's previous match (strictly date < M.date) regardless of
which side it played. The first appearance of any team is NaN.

We also derive ``rest_diff = home_days_since_last_match -
away_days_since_last_match`` as a leakage-safe relative-rest signal. Both
inputs are NaN-propagating, so ``rest_diff`` is NaN whenever either side
is NaN.

This mirrors the team-centric reshape pattern used in
``src/gold/rolling_features.py::_build_team_history``.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _build_team_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape Gold-shape rows into one row per (fixture, team, side)."""
    home_records = pd.DataFrame({
        "fixture_id": df["fixture_id"],
        "date_utc": df["date_utc"],
        "team": df["home_team"],
        "side": "home",
    })
    away_records = pd.DataFrame({
        "fixture_id": df["fixture_id"],
        "date_utc": df["date_utc"],
        "team": df["away_team"],
        "side": "away",
    })
    return pd.concat([home_records, away_records], ignore_index=True)


def add_days_since_last_match(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-side ``days_since_last_match`` and ``rest_diff`` columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``fixture_id``, ``date_utc``, ``home_team``, ``away_team``.
        The input is not mutated.

    Returns
    -------
    pd.DataFrame
        Copy of the input with three new ``Float64`` columns:
        ``home_days_since_last_match``, ``away_days_since_last_match``,
        ``rest_diff`` (= home - away). NaN on first appearance, and NaN
        propagates into ``rest_diff`` whenever either side is NaN.
    """
    df = df.copy()

    history = _build_team_match_log(df)
    history["date_utc"] = pd.to_datetime(history["date_utc"])
    history = history.sort_values(
        ["team", "date_utc", "fixture_id"]
    ).reset_index(drop=True)

    # Per-team previous-match date — first appearance becomes NaT.
    history["prior_date"] = history.groupby("team")["date_utc"].shift(1)
    gap = (history["date_utc"] - history["prior_date"]).dt.days
    history["days_since_last_match"] = gap.astype("Float64")

    home_log = history[history["side"] == "home"]
    away_log = history[history["side"] == "away"]
    home_map = home_log.set_index("fixture_id")["days_since_last_match"]
    away_map = away_log.set_index("fixture_id")["days_since_last_match"]

    df["home_days_since_last_match"] = (
        df["fixture_id"].map(home_map).astype("Float64")
    )
    df["away_days_since_last_match"] = (
        df["fixture_id"].map(away_map).astype("Float64")
    )
    df["rest_diff"] = (
        df["home_days_since_last_match"] - df["away_days_since_last_match"]
    )
    return df
