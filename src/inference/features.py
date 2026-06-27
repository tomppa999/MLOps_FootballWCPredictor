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
from src.gold.temporal_features import add_days_since_last_match
from src.silver.competition_mapping import TIER_MAP, is_knockout

logger = logging.getLogger(__name__)

_UPCOMING_STATUSES: Final[frozenset[str]] = frozenset({"NS", "TBD"})
FINISHED_STATUSES: Final[frozenset[str]] = frozenset({"FT", "AET", "PEN"})

# Ordered sequence of matchday labels used to determine the last completed round.
# Ordered from earliest to latest; "0" sentinel is never returned by _round_to_matchday_label.
_MATCHDAY_ORDER: Final[tuple[str, ...]] = ("1", "2", "3", "R32", "R16", "QF", "SF", "Final")

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


def generate_all_wc_pairings(
    config_path: Path = _WC2026_CONFIG_PATH,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate the C(48,2) = 1128 team pairings for rate prediction.

    Most pairs appear once as (home, away) and the simulation mirrors rates
    for the reverse direction automatically.  Host-vs-host pairs are emitted
    in BOTH orientations (each host as ``home_team``) so that the venue host's
    home-advantaged rate is available regardless of bracket slot — with the
    three 2026 hosts that adds 3 extra rows (1131 total).  All matches use
    WC-level context features (neutral venue, tier 1).

    When *reference_date* is given the pairings use that date so that
    ``build_inference_features`` rolling lookups (``date_utc < match_date``)
    include any WC result rows appended to Gold that are dated before it.
    Defaults to ``2026-06-11`` (tournament opening day).
    """
    from itertools import combinations

    teams = _load_wc_teams(config_path)
    if len(teams) != 48:
        logger.warning("Expected 48 WC teams, got %d", len(teams))

    base_date = reference_date if reference_date is not None else pd.Timestamp("2026-06-11")

    def _row(home: str, away: str) -> dict:
        return {
            "fixture_id": f"wc2026_pair_{home}_{away}",
            "date_utc": base_date,
            "home_team": home,
            "away_team": away,
            "league_id": 1,
            "league_name": "World Cup",
            "season": 2026,
            "competition_tier": 1,
            "is_knockout": False,
            "is_neutral": True,
        }

    rows: list[dict] = []
    for team_a, team_b in combinations(sorted(teams), 2):
        a_host = team_a in WC_2026_HOSTS
        b_host = team_b in WC_2026_HOSTS

        if a_host and b_host:
            # Host-vs-host: home advantage is positional, so a single stored
            # row would only carry one host's home-oriented rate. Emit BOTH
            # orientations (each host as home_team) so the simulation can
            # recover the venue host's true home rate. The is_neutral=False
            # override (applied below) attaches the boost to each home_team.
            rows.append(_row(team_a, team_b))
            rows.append(_row(team_b, team_a))
        elif b_host:
            # Exactly one host (team_b): place it in home_team so the
            # is_neutral=False override gives the host its home advantage.
            rows.append(_row(team_b, team_a))
        else:
            # Non-host pair, or host is already team_a — keep alphabetical.
            rows.append(_row(team_a, team_b))

    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"])
    df = override_neutral_for_2026_hosts(df)

    logger.info("Generated %d WC all-pairs fixtures for rate prediction.", len(df))
    return df


def generate_wc_group_fixtures(config_path: Path = _WC2026_CONFIG_PATH) -> pd.DataFrame:
    """Generate deterministic 2026 World Cup group-stage fixtures.

    Uses ``groups``, ``group_matchdays``, and ``group_schedule`` from
    wc2026.json and emits one row per group match (72 rows total for
    12 groups x 6 matches).  Each match gets its real calendar date from
    ``group_schedule[group][matchday]``.
    """
    with open(config_path) as f:
        config = json.load(f)

    groups: dict[str, list[str]] = config.get("groups", {})
    matchdays_cfg = config.get("group_matchdays", [])
    group_schedule: dict[str, dict[str, str]] = config.get("group_schedule", {})
    if not groups or not matchdays_cfg:
        logger.warning("Tournament config missing groups/group_matchdays at %s", config_path)
        return pd.DataFrame()

    rows: list[dict] = []

    for group_letter, teams in groups.items():
        if len(teams) != 4:
            logger.warning("Skipping group %s with invalid team count %d", group_letter, len(teams))
            continue

        schedule = group_schedule.get(group_letter, {})

        for md in matchdays_cfg:
            matchday = int(md["matchday"])
            date_str = schedule.get(str(matchday))
            if date_str:
                md_date = pd.Timestamp(date_str)
            else:
                md_date = pd.Timestamp("2026-06-11") + pd.Timedelta(days=matchday - 1)
                logger.warning(
                    "No date in group_schedule for group %s MD%d, falling back to %s",
                    group_letter, matchday, md_date.date(),
                )

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
                    "season": 2026,
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


def _load_expected_matches_per_round(
    config_path: Path = _WC2026_CONFIG_PATH,
) -> dict[str, int]:
    """Return the expected number of matches per round label from the tournament config.

    Used instead of counting ``scheduled`` fixtures from the API, which is
    unreliable because API-Football returns stale NS entries for the same
    match under both season 2025 and season 2026 with *different* fixture_ids
    — making fixture_id deduplication insufficient.
    """
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Could not load tournament config from %s — expected counts unavailable.", config_path)
        return {}

    expected: dict[str, int] = {}

    n_groups = len(config.get("groups", {}))
    for md in config.get("group_matchdays", []):
        label = str(md["matchday"])
        expected[label] = n_groups * len(md["pairs"])

    r32_count = len(config.get("r32_matches", []))
    if r32_count:
        expected["R32"] = r32_count

    for match in config.get("ko_bracket", []):
        stage = match.get("stage", "")
        if stage in ("R16", "QF", "SF", "Final"):
            expected[stage] = expected.get(stage, 0) + 1

    return expected


def parse_wc_results(
    fixtures_dir: Path = _FIXTURES_DIR,
    mapping_path: Path = _TEAM_MAPPING_PATH,
    _expected_per_round: dict[str, int] | None = None,
) -> dict:
    """Parse finished WC 2026 fixtures from Bronze and return locked results.

    Scans all fixtures.json files, filters for league_id==1 (FIFA World Cup)
    and statuses in FINISHED_STATUSES, then separates group-stage matches
    from KO matches.

    Args:
        _expected_per_round: override expected match counts per round label.
            Only for testing — production uses counts from wc2026.json.

    Returns a dict with:
      - group_results: dict[(home, away), (home_goals, away_goals)]
      - ko_results:    dict[match_num, {home, away, home_goals, away_goals, decided_by}]
      - next_matchday: int (1/2/3 for group stage) or str stage name for KO
      - last_completed_matchday: str label of the last fully-played round ("0" if none)
      - finished_fixtures: list of dicts for all finished WC fixtures
    """
    # Newest window first so that when the same fixture_id appears in multiple
    # windows (e.g. status "1H" in an older window, "FT" in the latest),
    # the most-recent status wins and the stale entry is skipped by seen_fixture_ids.
    fixture_files = sorted(fixtures_dir.glob("*/fixtures.json"), reverse=True)
    if not fixture_files:
        logger.warning("No fixtures.json files found for WC results parsing.")
        return {
            "group_results": {},
            "ko_results": {},
            "next_matchday": 1,
            "last_completed_matchday": "0",
            "finished_fixtures": [],
        }

    if _expected_per_round is None:
        _expected_per_round = _load_expected_matches_per_round()

    id_to_name = _load_api_id_to_canonical(mapping_path)
    group_results: dict[tuple[str, str], tuple[int, int]] = {}
    ko_results: dict[int, dict] = {}
    finished_fixtures: list[dict] = []
    max_group_matchday: int = 0
    # Counts only finished fixtures per round (for completion detection).
    finished_per_round: dict[str, int] = {}
    # Deduplicate by fixture_id across fixture files (API-Football returns WC
    # fixtures under both season 2025 and season 2026 for cross-year tournaments).
    seen_fixture_ids: set[int] = set()

    for fp in fixture_files:
        with open(fp) as f:
            data = json.load(f)

        for entry in data.get("response", []):
            fixture = entry.get("fixture", {})
            status = fixture.get("status", {}).get("short", "")

            # League/season filter applies to all statuses.
            league = entry.get("league", {})
            if league.get("id") != 1:
                continue
            season = league.get("season")
            if season not in _WC_SEASONS:
                continue

            fixture_id = fixture.get("id")
            if fixture_id is not None and fixture_id in seen_fixture_ids:
                continue
            if fixture_id is not None:
                seen_fixture_ids.add(fixture_id)

            round_str: str = league.get("round", "")
            round_label = _round_to_matchday_label(round_str)

            # Only finished fixtures contribute to results and rolling features.
            if status not in FINISHED_STATUSES:
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

            is_ko = not round_str.lower().startswith("group")
            finished_fixtures.append({
                "fixture_id": fixture.get("id"),
                "date_utc": fixture.get("date"),
                "home_team": home_name,
                "away_team": away_name,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "is_knockout": is_ko,
                "round": round_str,
            })
            round_lower = round_str.lower()

            if round_label is not None:
                finished_per_round[round_label] = finished_per_round.get(round_label, 0) + 1

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

    # Derive what stage comes next (for logging/display).
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

    # Derive last fully-completed matchday: highest round where finished count
    # meets the expected count from the tournament config.  Using config-derived
    # expected counts instead of API scheduled counts avoids inflation from stale
    # NS entries that API-Football returns with different fixture_ids across
    # season 2025 and 2026, which fixture_id deduplication cannot catch.
    last_completed_matchday: str = "0"
    for label in _MATCHDAY_ORDER:
        expected = _expected_per_round.get(label, 0)
        if expected == 0:
            break  # round not in config or not started
        done = finished_per_round.get(label, 0)
        if done >= expected:
            last_completed_matchday = label
        else:
            break  # round in progress

    logger.info(
        "WC results locked: %d group matches, %d KO matches, next stage: %s, "
        "last completed round: %s",
        len(group_results),
        len(ko_results),
        next_matchday,
        last_completed_matchday,
    )
    return {
        "group_results": group_results,
        "ko_results": ko_results,
        "next_matchday": next_matchday,
        "last_completed_matchday": last_completed_matchday,
        "finished_fixtures": finished_fixtures,
    }


def _round_to_matchday_label(round_str: str) -> str | None:
    """Map an API-Football round string to a snapshot ``matchday_label``.

    Labels align with ``parse_wc_results``'s ``next_matchday`` values:
    group matchdays ``"1"``/``"2"``/``"3"``, then ``"R32"``, ``"R16"``,
    ``"QF"``, ``"SF"``, ``"Final"``.
    """
    if not round_str:
        return None
    round_lower = round_str.lower()
    if round_lower.startswith("group"):
        parts = round_str.rsplit(" - ", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[1]
        return None
    if round_lower.startswith("round of 32"):
        return "R32"
    if round_lower.startswith("round of 16"):
        return "R16"
    if round_lower.startswith("quarter-final"):
        return "QF"
    if round_lower.startswith("semi-final"):
        return "SF"
    if round_lower.startswith("final"):
        return "Final"
    return None


def derive_snapshot_metadata(wc_results: dict) -> dict[str, str | int]:
    """Derive snapshot sequence metadata from ``parse_wc_results`` output.

    Returns:
        matchday_label: round currently being predicted (from ``next_matchday``).
        matches_completed_in_matchday: finished fixtures in the currently
            in-progress round (the round immediately after
            ``last_completed_matchday``). This differs from ``matchday_label``
            while a round is partially done: e.g. with 16/24 MD1 matches
            settled, ``matchday_label="2"`` but the in-progress round is "1"
            and ``matches_completed_in_matchday=16``.
        total_matches_completed: monotonic count of all locked group + KO matches.
    """
    matchday_label = str(wc_results.get("next_matchday", 1))
    group_results = wc_results.get("group_results", {})
    ko_results = wc_results.get("ko_results", {})
    total_matches_completed = len(group_results) + len(ko_results)

    # Determine the in-progress round: immediately after last_completed_matchday
    # in _MATCHDAY_ORDER.  Falls back to matchday_label when the sentinel "0"
    # maps to an unknown position or the order is exhausted.
    last_completed = str(wc_results.get("last_completed_matchday", "0"))
    if last_completed == "0":
        in_progress_label: str = "1"
    elif last_completed in _MATCHDAY_ORDER:
        idx = _MATCHDAY_ORDER.index(last_completed)
        in_progress_label = (
            _MATCHDAY_ORDER[idx + 1] if idx + 1 < len(_MATCHDAY_ORDER) else matchday_label
        )
    else:
        in_progress_label = matchday_label

    matches_completed_in_matchday = 0
    for fixture in wc_results.get("finished_fixtures", []):
        label = _round_to_matchday_label(fixture.get("round", ""))
        if label == in_progress_label:
            matches_completed_in_matchday += 1

    return {
        "matchday_label": matchday_label,
        "matches_completed_in_matchday": matches_completed_in_matchday,
        "total_matches_completed": total_matches_completed,
    }


def wc_results_to_gold_rows(wc_results: dict) -> pd.DataFrame:
    """Convert finished WC fixtures into Gold-compatible rows for rolling features.

    The returned DataFrame carries ``stats_tier="none"`` so that
    ``_rolling_for_team`` includes these rows for goal rolling averages
    but skips them for shot/tactical features (which require real stats).
    """
    rows = wc_results.get("finished_fixtures", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).dt.tz_localize(None)
    df["stats_tier"] = "none"
    df["competition_tier"] = 1
    df["is_neutral"] = True
    df["league_id"] = 1
    df["league_name"] = "World Cup"
    df["season"] = 2026
    df = df.sort_values("date_utc").reset_index(drop=True)

    logger.info("Built %d Gold-compatible WC result rows.", len(df))
    return df


def _get_latest_elo(
    gold_df: pd.DataFrame,
    team: str,
    match_date: pd.Timestamp | None = None,
) -> float:
    """Look up the most recent Elo for *team* from Gold history.

    When *match_date* is given, only rows with ``date_utc < match_date``
    are considered — consistent with the rolling-feature date guard.
    """
    df = gold_df
    if match_date is not None:
        df = df[df["date_utc"] < match_date]

    team_home = df.loc[df["home_team"] == team, ["date_utc", "home_elo_pre"]].dropna(
        subset=["home_elo_pre"]
    )
    team_away = df.loc[df["away_team"] == team, ["date_utc", "away_elo_pre"]].dropna(
        subset=["away_elo_pre"]
    )

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

        home_elo = _get_latest_elo(gold_df, row["home_team"], match_date)
        away_elo = _get_latest_elo(gold_df, row["away_team"], match_date)
        result["home_elo_pre"] = home_elo
        result["away_elo_pre"] = away_elo
        result["elo_diff"] = home_elo - away_elo
        result["elo_sum"] = home_elo + away_elo

        feature_rows.append(result)

    features_df = pd.DataFrame(feature_rows)

    # Temporal features: days_since_last_match per side + rest_diff.
    # Combine Gold history with upcoming rows so the function can see each
    # team's last historical match and compute the gap to the upcoming date.
    temporal_cols = ["fixture_id", "date_utc", "home_team", "away_team"]
    gold_stub = gold_df[temporal_cols].copy()
    upcoming_stub = features_df[temporal_cols].copy()
    combined = pd.concat([gold_stub, upcoming_stub], ignore_index=True)
    combined["date_utc"] = pd.to_datetime(combined["date_utc"])
    combined = combined.sort_values(["date_utc", "fixture_id"]).reset_index(drop=True)
    combined = add_days_since_last_match(combined)

    # Map temporal results back to upcoming fixture_ids
    temporal_map = combined.set_index("fixture_id")[
        ["home_days_since_last_match", "away_days_since_last_match", "rest_diff"]
    ]
    upcoming_ids = features_df["fixture_id"]
    for col in ("home_days_since_last_match", "away_days_since_last_match", "rest_diff"):
        features_df[col] = upcoming_ids.map(temporal_map[col]).astype("Float64")

    logger.info(
        "Built inference features: %d rows, %d columns",
        len(features_df),
        len(features_df.columns),
    )
    return features_df
