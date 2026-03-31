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


def override_neutral_for_2026_hosts(df: pd.DataFrame) -> pd.DataFrame:
    """Set ``is_neutral = False`` for 2026 WC group-stage matches where a host plays at home.

    World Cup matches are normally neutral, but for the 2026 tri-hosted
    tournament the host nations play group-stage matches in their own country.
    This override lets the model apply the home-advantage effect it already
    learned from historical ``is_neutral`` data.
    """
    df = df.copy()
    mask = (
        (df["season"].astype(int) == 2026)
        & (df["competition_tier"].astype(int) == WC_2026_COMPETITION_TIER)
        & (~df["is_knockout"].astype(bool))
        & (df["home_team"].isin(WC_2026_HOSTS))
    )
    n_overridden = mask.sum()
    if n_overridden:
        logger.info("Overriding is_neutral=False for %d 2026 WC host matches", n_overridden)
    df.loc[mask, "is_neutral"] = False
    return df
