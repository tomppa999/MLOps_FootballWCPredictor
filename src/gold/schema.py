"""Gold schema definition: column names, dtypes, and validation."""

from __future__ import annotations

from typing import Final

import pandas as pd

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------

MATCH_ID_COLUMNS: Final[list[str]] = [
    "fixture_id",
    "date_utc",
    "season",
    "league_id",
    "league_name",
]

TEAM_COLUMNS: Final[list[str]] = [
    "home_team",
    "home_country_code",
    "home_confederation",
    "away_team",
    "away_country_code",
    "away_confederation",
]

MATCH_INDEX_COLUMNS: Final[list[str]] = [
    "home_team_match_index",
    "away_team_match_index",
]

CONTEXT_COLUMNS: Final[list[str]] = [
    "competition_tier",
    "is_knockout",
    "is_neutral",
]

STRENGTH_COLUMNS: Final[list[str]] = [
    "home_elo_pre",
    "away_elo_pre",
    "elo_diff",
]

ROLLING_GOALS_COLUMNS: Final[list[str]] = [
    "home_team_rolling_goals_for",
    "home_team_rolling_goals_against",
    "away_team_rolling_goals_for",
    "away_team_rolling_goals_against",
]

ROLLING_SHOT_COLUMNS: Final[list[str]] = [
    "home_team_rolling_shots",
    "home_team_rolling_shot_accuracy",
    "home_team_rolling_conversion",
    "away_team_rolling_shots",
    "away_team_rolling_shot_accuracy",
    "away_team_rolling_conversion",
]

# Rolling tactical profile columns (5 stats x 2 sides) — direct model features
ROLLING_TACTICAL_COLUMNS: Final[list[str]] = [
    f"{side}_team_rolling_tac_{suffix}"
    for side in ("home", "away")
    for suffix in (
        "total_shots",
        "shot_precision",
        "fouls",
        "corner_kicks",
        "possession_pct",
    )
]

TARGET_COLUMNS: Final[list[str]] = [
    "home_goals",
    "away_goals",
]

GOLD_COLUMNS: Final[list[str]] = (
    MATCH_ID_COLUMNS
    + TEAM_COLUMNS
    + MATCH_INDEX_COLUMNS
    + CONTEXT_COLUMNS
    + STRENGTH_COLUMNS
    + ROLLING_GOALS_COLUMNS
    + ROLLING_SHOT_COLUMNS
    + ROLLING_TACTICAL_COLUMNS
    + TARGET_COLUMNS
)

# Features available for modeling (excludes identifiers, metadata, and targets)
FEATURE_COLUMNS: Final[list[str]] = (
    CONTEXT_COLUMNS
    + STRENGTH_COLUMNS
    + ROLLING_GOALS_COLUMNS
    + ROLLING_SHOT_COLUMNS
    + ROLLING_TACTICAL_COLUMNS
)

# ---------------------------------------------------------------------------
# Dtypes
# ---------------------------------------------------------------------------

GOLD_DTYPES: Final[dict[str, str]] = {
    "fixture_id": "int64",
    "season": "int64",
    "league_id": "int64",
    "home_team_match_index": "Int32",
    "away_team_match_index": "Int32",
    "competition_tier": "Int8",
    "is_knockout": "boolean",
    "is_neutral": "boolean",
    "home_elo_pre": "Float64",
    "away_elo_pre": "Float64",
    "elo_diff": "Float64",
    "home_team_rolling_goals_for": "Float64",
    "home_team_rolling_goals_against": "Float64",
    "away_team_rolling_goals_for": "Float64",
    "away_team_rolling_goals_against": "Float64",
    "home_team_rolling_shots": "Float64",
    "home_team_rolling_shot_accuracy": "Float64",
    "home_team_rolling_conversion": "Float64",
    "away_team_rolling_shots": "Float64",
    "away_team_rolling_shot_accuracy": "Float64",
    "away_team_rolling_conversion": "Float64",
    **{col: "Float64" for col in ROLLING_TACTICAL_COLUMNS},
    "home_goals": "Int16",
    "away_goals": "Int16",
}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

NON_NULLABLE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "fixture_id",
        "date_utc",
        "season",
        "league_id",
        "home_team",
        "away_team",
        "home_team_match_index",
        "away_team_match_index",
        "competition_tier",
        "is_knockout",
        "is_neutral",
        "home_goals",
        "away_goals",
    }
)


def validate_gold(df: pd.DataFrame) -> list[str]:
    """Return a list of validation error messages. Empty list means valid."""
    errors: list[str] = []

    missing = set(GOLD_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")

    extra = set(df.columns) - set(GOLD_COLUMNS)
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


def apply_gold_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast Gold DataFrame columns to canonical dtypes."""
    for col, dtype in GOLD_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df
