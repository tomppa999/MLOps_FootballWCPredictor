"""Build the Silver matches table from Bronze API-Football data + Elo.

Usage:
    python -m src.silver.build_silver [--data-root DATA_ROOT]

Reads:
    data/raw/api_football/fixtures/*/fixtures.json
    data/raw/api_football/statistics/*/*.json
    data/raw/elo/*.tsv
    data/mappings/team_mapping_master_merged.csv
    data/mappings/elo_code_map.csv
    data/mappings/api_football_tournaments.csv

Writes:
    data/silver/matches/season=YYYY/part-0.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.silver.competition_mapping import assign_competition_columns
from src.silver.elo_history import join_elo_to_fixtures
from src.silver.schema import (
    SILVER_COLUMNS,
    STAT_TYPE_TO_COLUMN,
    PCT_STAT_COLUMNS,
    apply_dtypes,
    compute_stats_tier,
    validate_silver,
)

logger = logging.getLogger(__name__)

FINISHED_STATUSES = frozenset({"FT", "AET", "PEN"})


# ---------------------------------------------------------------------------
# Fixture parsing
# ---------------------------------------------------------------------------


def parse_fixtures_file(path: Path) -> list[dict]:
    """Parse a single fixtures.json into a list of flat row dicts."""
    with open(path) as f:
        data = json.load(f)

    response = data.get("response", [])
    rows = []

    for entry in response:
        fixture = entry.get("fixture", {})
        league = entry.get("league", {})
        teams = entry.get("teams", {})
        goals = entry.get("goals", {})
        score = entry.get("score", {})

        status_short = fixture.get("status", {}).get("short", "")
        if status_short not in FINISHED_STATUSES:
            continue

        row = {
            "fixture_id": fixture.get("id"),
            "kickoff_utc": fixture.get("date"),
            "league_id": league.get("id"),
            "league_name": league.get("name"),
            "season": league.get("season"),
            "round": league.get("round"),
            "venue_name": fixture.get("venue", {}).get("name"),
            "venue_city": fixture.get("venue", {}).get("city"),
            "match_status": status_short,
            # Teams
            "home_team_api_id": teams.get("home", {}).get("id"),
            "home_team_api_name": teams.get("home", {}).get("name"),
            "away_team_api_id": teams.get("away", {}).get("id"),
            "away_team_api_name": teams.get("away", {}).get("name"),
            # Goals (final incl. ET, excl. penalties)
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            # Score breakdown
            "home_goals_ht": score.get("halftime", {}).get("home"),
            "away_goals_ht": score.get("halftime", {}).get("away"),
            "home_goals_ft": score.get("fulltime", {}).get("home"),
            "away_goals_ft": score.get("fulltime", {}).get("away"),
            "home_goals_et": score.get("extratime", {}).get("home"),
            "away_goals_et": score.get("extratime", {}).get("away"),
            "home_goals_pen": score.get("penalty", {}).get("home"),
            "away_goals_pen": score.get("penalty", {}).get("away"),
        }
        rows.append(row)

    return rows


def load_all_fixtures(fixtures_dir: Path) -> pd.DataFrame:
    """Load and concatenate fixtures from all date-window subdirectories."""
    fixture_files = sorted(fixtures_dir.glob("*/fixtures.json"))
    if not fixture_files:
        raise FileNotFoundError(f"No fixtures.json files found under {fixtures_dir}")

    all_rows: list[dict] = []
    for fp in fixture_files:
        rows = parse_fixtures_file(fp)
        logger.info("Parsed %d finished fixtures from %s", len(rows), fp.parent.name)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df["date_utc"] = pd.to_datetime(df["kickoff_utc"]).dt.date
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"])

    # Dedup on fixture_id
    n_before = len(df)
    df = df.drop_duplicates(subset=["fixture_id"], keep="first")
    if len(df) < n_before:
        logger.warning("Dropped %d duplicate fixture_id rows", n_before - len(df))

    logger.info("Total unique finished fixtures: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Youth team filter
# ---------------------------------------------------------------------------

_YOUTH_RE = re.compile(r"U\d{2}$")


def filter_youth_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where either team name matches a youth pattern like 'U18', 'U23'."""
    mask = df["home_team_api_name"].str.contains(_YOUTH_RE, na=False) | df[
        "away_team_api_name"
    ].str.contains(_YOUTH_RE, na=False)
    n_dropped = mask.sum()
    if n_dropped:
        logger.info("Dropped %d youth-team matches", n_dropped)
    return df[~mask].reset_index(drop=True)


def filter_unmapped_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Drop matches where either team has no country_code in the team mapping.

    This removes B-squads (Algeria B, Egypt B…), club teams that leaked
    through the tournament filter (Hull City, Alanyaspor…), and non-FIFA
    representative sides (Basque Country, Catalonia…), all of which lack a
    canonical FIFA country code.
    """
    mask = df["home_country_code"].isna() | df["away_country_code"].isna()
    n_dropped = mask.sum()
    if n_dropped:
        logger.info("Dropped %d matches with unmapped teams (no country_code)", n_dropped)
    return df[~mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics parsing
# ---------------------------------------------------------------------------


def _build_stats_index(stats_dir: Path) -> dict[int, Path]:
    """Build a fixture_id → file_path index across all date-window directories."""
    index: dict[int, Path] = {}
    for json_path in stats_dir.glob("*/*.json"):
        try:
            fixture_id = int(json_path.stem)
            index[fixture_id] = json_path
        except ValueError:
            continue
    logger.info("Statistics index: %d fixture files across %s", len(index), stats_dir)
    return index


def _parse_pct(value: str | int | float | None) -> float | None:
    """Parse a percentage string like '39%' into 39.0."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().rstrip("%")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _parse_stat_value(value, suffix: str):
    """Parse a stat value into the appropriate type."""
    if suffix in PCT_STAT_COLUMNS:
        return _parse_pct(value)
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def parse_statistics_for_fixture(stats_path: Path) -> dict:
    """Parse a fixture statistics JSON into flat {column: value} dict."""
    with open(stats_path) as f:
        data = json.load(f)

    response = data.get("response", [])
    if not response:
        return {"has_statistics": False, "stats_tier": "none"}

    result: dict = {"has_statistics": True}

    for idx, side in enumerate(("home", "away")):
        if idx >= len(response):
            for suffix in STAT_TYPE_TO_COLUMN.values():
                result[f"{side}_{suffix}"] = None
            continue

        team_stats = response[idx].get("statistics", [])
        stat_map = {s["type"]: s["value"] for s in team_stats if "type" in s}

        for api_type, suffix in STAT_TYPE_TO_COLUMN.items():
            raw = stat_map.get(api_type)
            parsed = _parse_stat_value(raw, suffix)
            result[f"{side}_{suffix}"] = parsed

        # Red Cards null → 0 when other stats are present
        if result[f"{side}_red_cards"] is None:
            has_other = any(
                result.get(f"{side}_{s}") is not None
                for s in STAT_TYPE_TO_COLUMN.values()
                if s != "red_cards"
            )
            if has_other:
                result[f"{side}_red_cards"] = 0

    result["stats_tier"] = compute_stats_tier(result, has_statistics=True)
    return result


def join_statistics(
    fixtures: pd.DataFrame,
    stats_dir: Path,
) -> pd.DataFrame:
    """Join match statistics to fixtures using the fixture_id → file index."""
    stats_index = _build_stats_index(stats_dir)

    stat_rows = []
    for fid in fixtures["fixture_id"]:
        stats_path = stats_index.get(fid)
        if stats_path is not None:
            stat_rows.append({"fixture_id": fid, **parse_statistics_for_fixture(stats_path)})
        else:
            stat_rows.append({"fixture_id": fid, "has_statistics": False, "stats_tier": "none"})

    stats_df = pd.DataFrame(stat_rows)
    merged = fixtures.merge(stats_df, on="fixture_id", how="left")

    merged["has_statistics"] = merged["has_statistics"].fillna(False)
    merged["stats_tier"] = merged["stats_tier"].fillna("none")

    n_with_stats = merged["has_statistics"].sum()
    tier_counts = merged["stats_tier"].value_counts()
    logger.info(
        "Statistics join: %d/%d have stats. stats_tier distribution:\n%s",
        n_with_stats, len(merged), tier_counts.to_string(),
    )
    return merged


# ---------------------------------------------------------------------------
# Team mapping
# ---------------------------------------------------------------------------


def load_team_mapping(mapping_path: Path) -> pd.DataFrame:
    """Load the team mapping master table."""
    return pd.read_csv(mapping_path)


def apply_team_mapping(fixtures: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    """Join canonical team names, country codes, and confederations."""
    tm = team_map[
        ["api_football_team_id", "canonical_team_name", "country_code", "confederation"]
    ].copy()
    tm = tm.rename(columns={"api_football_team_id": "team_id"})

    for side in ("home", "away"):
        fixtures = fixtures.merge(
            tm.rename(
                columns={
                    "team_id": f"{side}_team_api_id",
                    "canonical_team_name": f"{side}_team",
                    "country_code": f"{side}_country_code",
                    "confederation": f"{side}_confederation",
                }
            ),
            on=f"{side}_team_api_id",
            how="left",
        )

    n_unmapped_home = fixtures["home_team"].isna().sum()
    n_unmapped_away = fixtures["away_team"].isna().sum()
    if n_unmapped_home or n_unmapped_away:
        logger.warning(
            "Unmapped teams: %d home, %d away. Using API names as fallback.",
            n_unmapped_home,
            n_unmapped_away,
        )
        fixtures["home_team"] = fixtures["home_team"].fillna(fixtures["home_team_api_name"])
        fixtures["away_team"] = fixtures["away_team"].fillna(fixtures["away_team_api_name"])

    return fixtures


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_silver(data_root: Path) -> pd.DataFrame:
    """Build the complete Silver DataFrame from Bronze data."""
    fixtures_dir = data_root / "raw" / "api_football" / "fixtures"
    stats_dir = data_root / "raw" / "api_football" / "statistics"
    elo_dir = data_root / "raw" / "elo"
    team_map_path = data_root / "mappings" / "team_mapping_master_merged.csv"
    elo_code_map_path = data_root / "mappings" / "elo_code_map.csv"

    # 1. Parse fixtures
    logger.info("--- Step 1: Loading fixtures ---")
    df = load_all_fixtures(fixtures_dir)

    # 2. Join statistics
    logger.info("--- Step 2: Joining statistics ---")
    df = join_statistics(df, stats_dir)

    # 3. Apply team mapping
    logger.info("--- Step 3: Applying team mapping ---")
    team_map = load_team_mapping(team_map_path)
    df = apply_team_mapping(df, team_map)

    # 3b. Filter youth teams
    logger.info("--- Step 3b: Filtering youth teams ---")
    df = filter_youth_teams(df)

    # 3c. Filter unmapped teams (B-squads, clubs, non-FIFA entities)
    logger.info("--- Step 3c: Filtering unmapped teams ---")
    df = filter_unmapped_teams(df)

    # 4. Competition tier and knockout detection
    logger.info("--- Step 4: Competition mapping ---")
    df = assign_competition_columns(df)

    # 5. Join Elo and neutral flag
    logger.info("--- Step 5: Joining Elo ratings ---")
    df = join_elo_to_fixtures(df, elo_dir, elo_code_map_path)

    # 6. Reorder columns and apply dtypes
    logger.info("--- Step 6: Final schema alignment ---")
    for col in SILVER_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[SILVER_COLUMNS]
    df = apply_dtypes(df)

    # 7. Sort
    df = df.sort_values(["date_utc", "fixture_id"]).reset_index(drop=True)

    # 8. Validate
    errors = validate_silver(df)
    if errors:
        for err in errors:
            logger.error("Schema validation: %s", err)
    else:
        logger.info("Schema validation passed")

    logger.info("Silver table: %d rows, %d columns", len(df), len(df.columns))
    return df


def write_silver_parquet(df: pd.DataFrame, output_dir: Path) -> None:
    """Write Silver DataFrame as season-partitioned Parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df["_season"] = df["season"].astype(int)
    for season, group in df.groupby("_season"):
        season_dir = output_dir / f"season={season}"
        season_dir.mkdir(parents=True, exist_ok=True)
        out_path = season_dir / "part-0.parquet"
        group.drop(columns=["_season"]).to_parquet(out_path, index=False, engine="pyarrow")
        logger.info("Wrote %d rows to %s", len(group), out_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build Silver matches table")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-root>/silver/matches)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.data_root / "silver" / "matches")

    df = build_silver(args.data_root)
    write_silver_parquet(df, output_dir)

    logger.info("Done. Silver table written to %s", output_dir)


if __name__ == "__main__":
    main()
