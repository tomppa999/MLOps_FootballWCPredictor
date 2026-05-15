"""Context feature helpers for the Gold layer."""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

WC_2026_HOSTS: Final[frozenset[str]] = frozenset(
    {"United States", "Canada", "Mexico"}
)
WC_2026_COMPETITION_TIER: Final[int] = 1


def add_elo_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``elo_diff`` = ``home_elo_pre`` - ``away_elo_pre``."""
    df = df.copy()
    df["elo_diff"] = df["home_elo_pre"] - df["away_elo_pre"]
    return df


def add_elo_sum(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``elo_sum`` = ``home_elo_pre`` + ``away_elo_pre``.

    NaN propagates: if either pre-match Elo is missing the sum is NaN.
    """
    df = df.copy()
    df["elo_sum"] = df["home_elo_pre"] + df["away_elo_pre"]
    return df


def override_neutral_for_2026_hosts(df: pd.DataFrame) -> pd.DataFrame:
    """Set ``is_neutral = False`` for 2026 WC group-stage matches involving a host nation.

    World Cup matches are normally neutral, but for the 2026 tri-hosted
    tournament the host nations play group-stage matches in their own country.
    This override lets the model apply the home-advantage effect it already
    learned from historical ``is_neutral`` data.

    The check covers both ``home_team`` and ``away_team`` because
    ``generate_all_wc_pairings`` creates alphabetically-ordered pairs where a
    host nation may end up in either column.
    """
    df = df.copy()
    wc_2026_group = (
        (df["season"].astype(int) == 2026)
        & (df["competition_tier"].astype(int) == WC_2026_COMPETITION_TIER)
        & (~df["is_knockout"].astype(bool))
    )
    host_involved = df["home_team"].isin(WC_2026_HOSTS) | df["away_team"].isin(WC_2026_HOSTS)
    mask = wc_2026_group & host_involved
    n_overridden = mask.sum()
    if n_overridden:
        logger.info("Overriding is_neutral=False for %d 2026 WC host matches", n_overridden)
    df.loc[mask, "is_neutral"] = False
    return df
