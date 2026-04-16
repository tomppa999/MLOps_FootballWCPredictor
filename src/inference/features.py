"""Build inference-ready feature rows from upcoming (NS/TBD) fixtures.

Parses upcoming fixtures from Bronze, maps team names to canonical form,
and computes rolling features using the same machinery as the Gold layer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.gold.context_features import WC_2026_HOSTS, override_neutral_for_2026_hosts
from src.gold.rolling_features import (
    DEFAULT_WINDOW,
    _build_team_history,
    _rolling_for_team,
)
from src.silver.competition_mapping import TIER_MAP, is_knockout

logger = logging.getLogger(__name__)

_UPCOMING_STATUSES: Final[frozenset[str]] = frozenset({"NS", "TBD"})
FINISHED_STATUSES: Final[frozenset[str]] = frozenset({"FT", "AET", "PEN"})

# Maps lowercase API-Football round prefix → offset to add to the in-round match number
# to get the internal wc2026.json match number.
# R32: matches 73–88, R16: 89–96, QF: 97–100, SF: 101–102, Final: 103
_KO_ROUND_TO_OFFSET: Final[dict[str, int]] = {
    "round of 32": 72,
    "round of 16": 88,
    "quarter-finals": 96,
    "semi-finals": 100,
    "final": 102,
}

_FIXTURES_DIR: Final[Path] = Path("data/raw/api_football/fixtures")
_TEAM_MAPPING_PATH: Final[Path] = Path("data/mappings/team_mapping_master_merged.csv")
_WC2026_CONFIG_PATH: Final[Path] = Path("data/tournament/wc2026.json")
_WC_SEASONS: Final[frozenset[int]] = frozenset({2025, 2026})


def _load_api_id_to_canonical(mapping_path: Path = _TEAM_MAPPING_PATH) -> dict[int, str]:
    """Build a lookup from api_football_team_id → canonical_team_name."""
    tm = pd.read_csv(mapping_path)
    tm = tm.dropna(subset=["api_football_team_id", "canonical_team_name"])
    return dict(zip(tm["api_football_team_id"].astype(int), tm["canonical_team_name"]))


def parse_upcoming_fixtures(
    fixtures_dir: Path = _FIXTURES_DIR,
    mapping_path: Path = _TEAM_MAPPING_PATH,
) -> pd.DataFrame:
    """Read Bronze fixtures and extract upcoming (NS/TBD) matches.

    Returns a DataFrame with columns matching the Gold identifiers and
    context columns: fixture_id, date_utc, home_team, away_team, league_id,
    league_name, season, is_neutral, is_knockout, competition_tier.
    """
    fixture_files = sorted(fixtures_dir.glob("*/fixtures.json"))
    if not fixture_files:
        logger.warning("No fixtures.json files found under %s", fixtures_dir)
        return pd.DataFrame()

    id_to_name = _load_api_id_to_canonical(mapping_path)
    rows: list[dict] = []

    for fp in fixture_files:
        with open(fp) as f:
            data = json.load(f)

        for entry in data.get("response", []):
            fixture = entry.get("fixture", {})
            status_short = fixture.get("status", {}).get("short", "")
            if status_short not in _UPCOMING_STATUSES:
                continue

            league = entry.get("league", {})
            teams = entry.get("teams", {})

            home_api_id = teams.get("home", {}).get("id")
            away_api_id = teams.get("away", {}).get("id")
            home_name = id_to_name.get(home_api_id, teams.get("home", {}).get("name"))
            away_name = id_to_name.get(away_api_id, teams.get("away", {}).get("name"))

            league_id = league.get("id")
            comp_tier = TIER_MAP.get(league_id, 4)
            round_str = league.get("round", "")

            rows.append({
                "fixture_id": fixture.get("id"),
                "date_utc": pd.to_datetime(fixture.get("date")).date() if fixture.get("date") else None,
                "home_team": home_name,
                "away_team": away_name,
                "league_id": league_id,
                "league_name": league.get("name"),
                "season": league.get("season"),
                "competition_tier": comp_tier,
                "is_knockout": is_knockout(round_str),
                "is_neutral": comp_tier in (1, 2),
            })

    if not rows:
        logger.info("No upcoming fixtures found in Bronze.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["fixture_id"], keep="first")
    df["date_utc"] = pd.to_datetime(df["date_utc"])
    df = df.sort_values(["date_utc", "fixture_id"]).reset_index(drop=True)

    df = override_neutral_for_2026_hosts(df)

    logger.info("Parsed %d upcoming fixtures from Bronze.", len(df))
    return df


def _load_wc_teams(config_path: Path = _WC2026_CONFIG_PATH) -> list[str]:
    """Return the flat list of 48 WC 2026 teams from the tournament config."""
    with open(config_path) as f:
        config = json.load(f)
    return [t for teams in config.get("groups", {}).values() for t in teams]


def generate_all_wc_pairings(config_path: Path = _WC2026_CONFIG_PATH) -> pd.DataFrame:
    """Generate all C(48,2) = 1128 ordered team pairings for rate prediction.

    Every pair appears once as (home, away); the simulation mirrors rates for
    the reverse direction automatically.  All matches use WC-level context
    features (neutral venue, tier 1).
    """
    from itertools import combinations

    teams = _load_wc_teams(config_path)
    if len(teams) != 48:
        logger.warning("Expected 48 WC teams, got %d", len(teams))

    base_date = pd.Timestamp("2026-06-11")
    rows: list[dict] = []
    for home, away in combinations(sorted(teams), 2):
        rows.append({
            "fixture_id": f"wc2026_pair_{home}_{away}",
            "date_utc": base_date,
            "home_team": home,
            "away_team": away,
            "league_id": 1,
            "league_name": "World Cup",
            "season": 2025,
            "competition_tier": 1,
            "is_knockout": False,
            "is_neutral": True,
        })

    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"])
    df = override_neutral_for_2026_hosts(df)

    logger.info("Generated %d WC all-pairs fixtures for rate prediction.", len(df))
    return df


def generate_wc_group_fixtures(config_path: Path = _WC2026_CONFIG_PATH) -> pd.DataFrame:
    """Generate deterministic 2026 World Cup group-stage fixtures.

    Uses ``groups`` and ``group_matchdays`` from wc2026.json and emits one row
    per group match (72 rows total for 12 groups x 6 matches).
    """
    with open(config_path) as f:
        config = json.load(f)

    groups: dict[str, list[str]] = config.get("groups", {})
    matchdays_cfg = config.get("group_matchdays", [])
    if not groups or not matchdays_cfg:
        logger.warning("Tournament config missing groups/group_matchdays at %s", config_path)
        return pd.DataFrame()

    rows: list[dict] = []
    base_date = pd.Timestamp("2026-06-11", tz="UTC")

    for group_letter, teams in groups.items():
        if len(teams) != 4:
            logger.warning("Skipping group %s with invalid team count %d", group_letter, len(teams))
            continue

        for md in matchdays_cfg:
            matchday = int(md["matchday"])
            md_date = base_date + pd.Timedelta(days=matchday - 1)
            for pair_idx, (home_idx, away_idx) in enumerate(md["pairs"]):
                home_team = teams[home_idx]
                away_team = teams[away_idx]
                rows.append({
                    "fixture_id": f"wc2026_{group_letter}_md{matchday}_{pair_idx}",
                    "date_utc": md_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "league_id": 1,
                    "league_name": "World Cup",
                    "season": 2025,
                    "competition_tier": 1,
                    "is_knockout": False,
                    "is_neutral": True,
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["date_utc", "fixture_id"]).reset_index(drop=True)
    df = override_neutral_for_2026_hosts(df)

    logger.info("Generated %d WC 2026 group fixtures from config.", len(df))
    return df


def parse_wc_results(
    fixtures_dir: Path = _FIXTURES_DIR,
    mapping_path: Path = _TEAM_MAPPING_PATH,
) -> dict:
    """Parse finished WC 2026 fixtures from Bronze and return locked results.

    Scans all fixtures.json files, filters for league_id==1 (FIFA World Cup)
    and statuses in FINISHED_STATUSES, then separates group-stage matches
    from KO matches.

    Returns a dict with:
      - group_results: dict[(home, away), (home_goals, away_goals)]
      - ko_results:    dict[match_num, {home, away, home_goals, away_goals, decided_by}]
      - next_matchday: int (1/2/3 for group stage) or str stage name for KO
    """
    fixture_files = sorted(fixtures_dir.glob("*/fixtures.json"))
    if not fixture_files:
        logger.warning("No fixtures.json files found for WC results parsing.")
        return {"group_results": {}, "ko_results": {}, "next_matchday": 1}

    id_to_name = _load_api_id_to_canonical(mapping_path)
    group_results: dict[tuple[str, str], tuple[int, int]] = {}
    ko_results: dict[int, dict] = {}
    max_group_matchday: int = 0

    for fp in fixture_files:
        with open(fp) as f:
            data = json.load(f)

        for entry in data.get("response", []):
            fixture = entry.get("fixture", {})
            status = fixture.get("status", {}).get("short", "")
            if status not in FINISHED_STATUSES:
                continue

            league = entry.get("league", {})
            if league.get("id") != 1:
                continue
            season = league.get("season")
            if season not in _WC_SEASONS:
                continue

            teams = entry.get("teams", {})
            goals_raw = entry.get("goals", {})

            home_api_id = teams.get("home", {}).get("id")
            away_api_id = teams.get("away", {}).get("id")
            home_name = id_to_name.get(home_api_id, teams.get("home", {}).get("name"))
            away_name = id_to_name.get(away_api_id, teams.get("away", {}).get("name"))

            hg_raw = goals_raw.get("home")
            ag_raw = goals_raw.get("away")
            if hg_raw is None or ag_raw is None:
                continue
            home_goals, away_goals = int(hg_raw), int(ag_raw)

            round_str: str = league.get("round", "")
            round_lower = round_str.lower()

            if round_lower.startswith("group"):
                # e.g. "Group A - 2"
                if (home_name, away_name) not in group_results:
                    group_results[(home_name, away_name)] = (home_goals, away_goals)
                parts = round_str.rsplit(" - ", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    max_group_matchday = max(max_group_matchday, int(parts[1]))
            else:
                # KO match — map round string to internal match number
                for stage_key, offset in _KO_ROUND_TO_OFFSET.items():
                    if round_lower.startswith(stage_key):
                        parts = round_str.rsplit(" - ", 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            match_num = offset + int(parts[1])
                            if match_num not in ko_results:
                                ko_results[match_num] = {
                                    "home": home_name,
                                    "away": away_name,
                                    "home_goals": home_goals,
                                    "away_goals": away_goals,
                                    "decided_by": status,
                                }
                        break

    # Derive what stage comes next (for logging/display)
    if ko_results:
        ko_nums = set(ko_results.keys())
        if 103 in ko_nums:
            next_matchday: int | str = "Complete"
        elif ko_nums & {101, 102}:
            next_matchday = "Final"
        elif ko_nums & set(range(97, 101)):
            next_matchday = "SF"
        elif ko_nums & set(range(89, 97)):
            next_matchday = "QF"
        else:
            next_matchday = "R16"
    elif max_group_matchday >= 3:
        next_matchday = "R32"
    elif max_group_matchday > 0:
        next_matchday = max_group_matchday + 1
    else:
        next_matchday = 1

    logger.info(
        "WC results locked: %d group matches, %d KO matches, next stage: %s",
        len(group_results),
        len(ko_results),
        next_matchday,
    )
    return {
        "group_results": group_results,
        "ko_results": ko_results,
        "next_matchday": next_matchday,
    }


def _get_latest_elo(gold_df: pd.DataFrame, team: str) -> float:
    """Look up the most recent Elo for a team from Gold history."""
    team_home = gold_df.loc[gold_df["home_team"] == team, ["date_utc", "home_elo_pre"]]
    team_away = gold_df.loc[gold_df["away_team"] == team, ["date_utc", "away_elo_pre"]]

    latest = pd.NaT
    elo = np.nan

    if not team_home.empty:
        row = team_home.sort_values("date_utc").iloc[-1]
        latest = row["date_utc"]
        elo = row["home_elo_pre"]

    if not team_away.empty:
        row = team_away.sort_values("date_utc").iloc[-1]
        if pd.isna(latest) or row["date_utc"] > latest:
            elo = row["away_elo_pre"]

    return elo


def build_inference_features(
    upcoming_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    n_matches: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Compute rolling features for upcoming fixtures using Gold history.

    Uses the same ``_build_team_history`` / ``_rolling_for_team`` machinery
    as Gold-layer feature computation, ensuring feature parity.

    Returns a DataFrame with the same feature columns as Gold (minus targets).
    """
    if upcoming_df.empty:
        return upcoming_df

    gold_df = gold_df.copy()
    gold_df["date_utc"] = pd.to_datetime(gold_df["date_utc"])

    history = _build_team_history(gold_df)
    team_groups: dict[str, pd.DataFrame] = {
        team: grp for team, grp in history.groupby("team")
    }

    feature_rows: list[dict] = []
    for _, row in upcoming_df.iterrows():
        match_date = pd.to_datetime(row["date_utc"])
        result: dict = {
            "fixture_id": row["fixture_id"],
            "date_utc": row["date_utc"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "league_id": row.get("league_id"),
            "league_name": row.get("league_name"),
            "season": row.get("season"),
            "competition_tier": row["competition_tier"],
            "is_knockout": row["is_knockout"],
            "is_neutral": row["is_neutral"],
        }

        for side in ("home", "away"):
            team = row[f"{side}_team"]
            prefix = f"{side}_team"

            team_hist = team_groups.get(team)
            if team_hist is not None:
                prior = team_hist[team_hist["date_utc"] < match_date]
            else:
                prior = pd.DataFrame()

            result[f"{prefix}_match_index"] = len(prior) + 1
            feats = _rolling_for_team(prior, n_matches)
            for key, val in feats.items():
                result[f"{prefix}_{key}"] = val

        home_elo = _get_latest_elo(gold_df, row["home_team"])
        away_elo = _get_latest_elo(gold_df, row["away_team"])
        result["home_elo_pre"] = home_elo
        result["away_elo_pre"] = away_elo
        result["elo_diff"] = home_elo - away_elo

        feature_rows.append(result)

    features_df = pd.DataFrame(feature_rows)
    logger.info(
        "Built inference features: %d rows, %d columns",
        len(features_df),
        len(features_df.columns),
    )
    return features_df
