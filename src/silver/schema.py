"""Silver schema definition: column names, dtypes, stat-type mapping, validation."""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API-Football stat type → Silver column suffix mapping
# ---------------------------------------------------------------------------

STAT_TYPE_TO_COLUMN: Final[dict[str, str]] = {
    "Shots on Goal": "shots_on_goal",
    "Shots off Goal": "shots_off_goal",
    "Total Shots": "total_shots",
    "Blocked Shots": "blocked_shots",
    "Shots insidebox": "shots_insidebox",
    "Shots outsidebox": "shots_outsidebox",
    "Fouls": "fouls",
    "Corner Kicks": "corner_kicks",
    "Offsides": "offsides",
    "Ball Possession": "possession_pct",
    "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards",
    "Goalkeeper Saves": "goalkeeper_saves",
    "Total passes": "total_passes",
    "Passes accurate": "passes_accurate",
    "Passes %": "pass_accuracy_pct",
}

# Stat columns that represent percentages (stored as float, "39%" → 39.0)
PCT_STAT_COLUMNS: Final[frozenset[str]] = frozenset({"possession_pct", "pass_accuracy_pct"})

# Stat columns with larger values that need Int32
WIDE_INT_STAT_COLUMNS: Final[frozenset[str]] = frozenset({"total_passes", "passes_accurate"})

# ---------------------------------------------------------------------------
# Ordered column list (canonical Silver schema)
# ---------------------------------------------------------------------------

MATCH_ID_COLUMNS: Final[list[str]] = [
    "fixture_id",
    "date_utc",
    "kickoff_utc",
    "league_id",
    "league_name",
    "season",
    "round",
    "competition_tier",
    "is_knockout",
    "venue_name",
    "venue_city",
    "is_neutral",
    "neutral_source",
    "match_status",
]

TEAM_COLUMNS: Final[list[str]] = [
    "home_team_api_id",
    "home_team_api_name",
    "home_team",
    "home_country_code",
    "home_confederation",
    "away_team_api_id",
    "away_team_api_name",
    "away_team",
    "away_country_code",
    "away_confederation",
]

SCORE_COLUMNS: Final[list[str]] = [
    "home_goals",
    "away_goals",
    "home_goals_ht",
    "away_goals_ht",
    "home_goals_ft",
    "away_goals_ft",
    "home_goals_et",
    "away_goals_et",
    "home_goals_pen",
    "away_goals_pen",
]

ELO_COLUMNS: Final[list[str]] = [
    "home_elo_pre",
    "away_elo_pre",
    "home_elo_post",
    "away_elo_post",
]

_STAT_SUFFIXES = list(STAT_TYPE_TO_COLUMN.values())

STAT_COLUMNS: Final[list[str]] = [
    f"{side}_{suffix}" for side in ("home", "away") for suffix in _STAT_SUFFIXES
]

QUALITY_COLUMNS: Final[list[str]] = [
    "has_statistics",
    "stats_tier",
]

# Core stat suffixes used to determine stats_tier (excludes red_cards, offsides, goalkeeper_saves)
CORE_STAT_SUFFIXES: Final[list[str]] = [
    "shots_on_goal",
    "shots_off_goal",
    "total_shots",
    "blocked_shots",
    "shots_insidebox",
    "shots_outsidebox",
    "fouls",
    "corner_kicks",
    "possession_pct",
    "total_passes",
    "passes_accurate",
    "pass_accuracy_pct",
]

PARTIAL_STAT_SUFFIXES: Final[list[str]] = [
    "possession_pct",
    "shots_on_goal",
    "total_shots",
]

SILVER_COLUMNS: Final[list[str]] = (
    MATCH_ID_COLUMNS + TEAM_COLUMNS + SCORE_COLUMNS + ELO_COLUMNS + STAT_COLUMNS + QUALITY_COLUMNS
)

# ---------------------------------------------------------------------------
# Dtype specification for the Silver DataFrame / Parquet
# ---------------------------------------------------------------------------


def _build_dtypes() -> dict[str, str]:
    dtypes: dict[str, str] = {}

    # Match identity
    dtypes["fixture_id"] = "int64"
    dtypes["league_id"] = "int64"
    dtypes["season"] = "int64"
    dtypes["competition_tier"] = "Int8"
    dtypes["is_knockout"] = "boolean"
    dtypes["is_neutral"] = "boolean"

    # Teams
    for side in ("home", "away"):
        dtypes[f"{side}_team_api_id"] = "int64"

    # Scores
    for col in SCORE_COLUMNS:
        dtypes[col] = "Int16"

    # Elo
    for col in ELO_COLUMNS:
        dtypes[col] = "Float32"

    # Stats
    for suffix in _STAT_SUFFIXES:
        for side in ("home", "away"):
            col = f"{side}_{suffix}"
            if suffix in PCT_STAT_COLUMNS:
                dtypes[col] = "Float32"
            elif suffix in WIDE_INT_STAT_COLUMNS:
                dtypes[col] = "Int32"
            else:
                dtypes[col] = "Int16"

    # Quality flags
    dtypes["has_statistics"] = "boolean"
    dtypes["stats_tier"] = "string"

    return dtypes


SILVER_DTYPES: Final[dict[str, str]] = _build_dtypes()

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

NON_NULLABLE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "fixture_id",
        "date_utc",
        "kickoff_utc",
        "league_id",
        "league_name",
        "season",
        "competition_tier",
        "is_knockout",
        "is_neutral",
        "neutral_source",
        "match_status",
        "home_team_api_id",
        "home_team_api_name",
        "away_team_api_id",
        "away_team_api_name",
        "has_statistics",
        "stats_tier",
    }
)


def compute_stats_tier(row: dict, has_statistics: bool) -> str:
    """Determine the stats_tier for a single fixture's parsed statistics.

    Levels: "full", "partial", "cards_only", "none".
    """
    if not has_statistics:
        return "none"

    def _both_non_null(suffix: str) -> bool:
        return row.get(f"home_{suffix}") is not None and row.get(f"away_{suffix}") is not None

    all_core = all(_both_non_null(s) for s in CORE_STAT_SUFFIXES)
    if all_core:
        return "full"

    has_partial = all(_both_non_null(s) for s in PARTIAL_STAT_SUFFIXES)
    if has_partial:
        return "partial"

    return "cards_only"


def validate_silver(df: pd.DataFrame) -> list[str]:
    """Return a list of validation error messages. Empty list means valid."""
    errors: list[str] = []

    missing = set(SILVER_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")

    extra = set(df.columns) - set(SILVER_COLUMNS)
    if extra:
        errors.append(f"Unexpected columns: {sorted(extra)}")

    if "fixture_id" in df.columns and df.duplicated(subset=["fixture_id"]).any():
        n_dup = df.duplicated(subset=["fixture_id"]).sum()
        errors.append(f"Duplicate fixture_id rows: {n_dup}")

    for col in NON_NULLABLE_COLUMNS:
        if col in df.columns and df[col].isna().any():
            n_null = df[col].isna().sum()
            errors.append(f"Non-nullable column '{col}' has {n_null} nulls")

    return errors


def apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast Silver DataFrame columns to canonical dtypes (in-place safe)."""
    for col, dtype in SILVER_DTYPES.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                logger.warning("Could not cast column '%s' to %s", col, dtype)
    return df
