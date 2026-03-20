"""Competition tier and knockout detection for the Silver layer.

Tier definitions (from feature_spec.md):
  1 = FIFA World Cup
  2 = continental final tournament
  3 = WC qualification, continental qualification, Nations League
  4 = friendly or other
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Competition tier by API-Football league_id
# (source: data/mappings/api_football_tournaments.csv)
# ---------------------------------------------------------------------------

TIER_MAP: Final[dict[int, int]] = {
    # Tier 1 — World Cup
    1: 1,
    # Tier 2 — Continental final tournaments
    6: 2,    # Africa Cup of Nations
    7: 2,    # Asian Cup
    22: 2,   # CONCACAF Gold Cup
    4: 2,    # EURO Championship
    9: 2,    # Copa America
    806: 2,  # OFC Nations Cup
    860: 2,  # Arab Cup
    # Tier 3 — Qualifiers and Nations Leagues
    29: 3,   # WC Qualification Africa
    30: 3,   # WC Qualification Asia
    31: 3,   # WC Qualification CONCACAF
    32: 3,   # WC Qualification Europe
    33: 3,   # WC Qualification Oceania
    34: 3,   # WC Qualification South America
    37: 3,   # WC Qualification Intercontinental Play-offs
    36: 3,   # AFCON Qualification
    35: 3,   # Asian Cup Qualification
    858: 3,  # CONCACAF Gold Cup Qualification
    960: 3,  # EURO Qualification
    536: 3,  # CONCACAF Nations League
    808: 3,  # CONCACAF Nations League Qualification
    5: 3,    # UEFA Nations League
    # Tier 4 — Friendlies
    10: 4,
}

DEFAULT_TIER: Final[int] = 4

# API-Football lumped qualifying matches under the main tournament league_id
# for certain seasons. These overrides reclassify pre-tournament matches as
# tier 3 (qualification) using the actual tournament start date.
_TOURNAMENT_QUALIFIER_OVERRIDES: Final[
    list[tuple[int, int, datetime.date]]
] = [
    # (league_id, season, tournament_start_date)
    (4, 2020, datetime.date(2021, 6, 11)),   # Euro 2020 qualifiers under league_id=4
    (6, 2019, datetime.date(2019, 6, 21)),   # AFCON 2019 qualifiers under league_id=6
]

# ---------------------------------------------------------------------------
# Knockout detection via round string
# ---------------------------------------------------------------------------

_KNOCKOUT_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"final",
        r"semi.?final",
        r"quarter.?final",
        r"round of \d+",
        r"8th finals",
        r"play.?off",
        r"3rd place",
    ]
]


def competition_tier(league_id: int) -> int:
    """Return the competition tier (1-4) for a given API-Football league_id."""
    tier = TIER_MAP.get(league_id)
    if tier is None:
        logger.warning("Unmapped league_id=%d, defaulting to tier %d", league_id, DEFAULT_TIER)
        return DEFAULT_TIER
    return tier


def is_knockout(round_str: str | None) -> bool:
    """Return True if the round string indicates a knockout stage."""
    if not round_str:
        return False
    return any(pat.search(round_str) for pat in _KNOCKOUT_PATTERNS)


def assign_competition_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add competition_tier and is_knockout columns to a fixtures DataFrame.

    Expects columns: league_id, round.
    Optionally uses date_utc and season to apply qualifier overrides for
    competitions where API-Football tagged qualifying matches under the main
    tournament league_id (Euro 2020, AFCON 2019).
    """
    df = df.copy()
    df["competition_tier"] = df["league_id"].map(TIER_MAP).fillna(DEFAULT_TIER).astype("Int8")
    df["is_knockout"] = df["round"].apply(is_knockout)

    if "date_utc" in df.columns and "season" in df.columns:
        dates = pd.to_datetime(df["date_utc"]).dt.date
        for league_id, season, tournament_start in _TOURNAMENT_QUALIFIER_OVERRIDES:
            mask = (
                (df["league_id"] == league_id)
                & (df["season"] == season)
                & (dates < tournament_start)
            )
            n = mask.sum()
            if n:
                logger.info(
                    "Reclassified %d pre-tournament matches as tier 3 "
                    "(league_id=%d season=%d, before %s)",
                    n, league_id, season, tournament_start,
                )
            df.loc[mask, "competition_tier"] = 3

    return df
