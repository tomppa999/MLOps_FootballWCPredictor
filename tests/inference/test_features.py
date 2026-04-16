"""Tests for src.inference.features."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference.features import (
    build_inference_features,
    generate_all_wc_pairings,
    generate_wc_group_fixtures,
    parse_upcoming_fixtures,
    parse_wc_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fixtures_json(
    fixtures_dir: Path,
    entries: list[dict],
    window: str = "2026-06-11_2026-06-15",
) -> None:
    """Write a minimal fixtures.json with given entries."""
    window_dir = fixtures_dir / window
    window_dir.mkdir(parents=True, exist_ok=True)
    response = []
    for e in entries:
        response.append({
            "fixture": {
                "id": e["id"],
                "date": e.get("date", "2026-06-11T20:00:00+00:00"),
                "status": {"short": e["status"]},
            },
            "league": {
                "id": e.get("league_id", 1),
                "name": e.get("league_name", "World Cup"),
                "season": e.get("season", 2026),
                "round": e.get("round", "Group A - 1"),
            },
            "teams": {
                "home": {"id": e.get("home_id", 100), "name": e.get("home_name", "Team A")},
                "away": {"id": e.get("away_id", 200), "name": e.get("away_name", "Team B")},
            },
            "goals": {"home": None, "away": None},
        })
    (window_dir / "fixtures.json").write_text(json.dumps({"response": response}))


def _write_team_mapping(mapping_path: Path, teams: list[dict]) -> None:
    """Write a minimal team mapping CSV."""
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["canonical_team_name,country_code,confederation,api_football_team_name,api_football_team_id,elo_display_name,elo_slug"]
    for t in teams:
        lines.append(
            f"{t['name']},{t.get('cc', 'XX')},{t.get('conf', 'UEFA')},"
            f"{t.get('api_name', t['name'])},{t['api_id']},{t['name']},{t['name']}"
        )
    mapping_path.write_text("\n".join(lines))


def _make_gold_df() -> pd.DataFrame:
    """Create a minimal Gold DataFrame for testing rolling features."""
    rows = []
    for i in range(15):
        rows.append({
            "fixture_id": 1000 + i,
            "date_utc": f"2026-0{1 + i // 10}-{10 + i % 28}",
            "season": 2025,
            "league_id": 10,
            "league_name": "Friendly",
            "home_team": "France",
            "home_country_code": "FRA",
            "home_confederation": "UEFA",
            "away_team": "Germany",
            "away_country_code": "DEU",
            "away_confederation": "UEFA",
            "home_team_match_index": i + 1,
            "away_team_match_index": i + 1,
            "competition_tier": 4,
            "is_knockout": False,
            "is_neutral": False,
            "home_elo_pre": 2000 + i,
            "away_elo_pre": 1980 + i,
            "elo_diff": 20.0,
            "home_goals": 2,
            "away_goals": 1,
            "home_team_rolling_goals_for": 1.5,
            "home_team_rolling_goals_against": 1.0,
            "away_team_rolling_goals_for": 1.0,
            "away_team_rolling_goals_against": 1.5,
            "home_team_rolling_shots": 5.0,
            "home_team_rolling_shot_accuracy": 0.4,
            "home_team_rolling_conversion": 0.2,
            "away_team_rolling_shots": 4.0,
            "away_team_rolling_shot_accuracy": 0.35,
            "away_team_rolling_conversion": 0.18,
            **{f"home_team_rolling_tac_{s}": 1.0 for s in ["total_shots", "shot_precision", "fouls", "corner_kicks", "possession_pct"]},
            **{f"away_team_rolling_tac_{s}": 1.0 for s in ["total_shots", "shot_precision", "fouls", "corner_kicks", "possession_pct"]},
            "stats_tier": "full",
            "home_shots_on_goal": 4,
            "home_total_shots": 10,
            "home_fouls": 12,
            "home_corner_kicks": 5,
            "home_possession_pct": 55.0,
            "away_shots_on_goal": 3,
            "away_total_shots": 8,
            "away_fouls": 14,
            "away_corner_kicks": 3,
            "away_possession_pct": 45.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# parse_upcoming_fixtures
# ---------------------------------------------------------------------------


class TestParseUpcomingFixtures:
    def test_extracts_ns_and_tbd_only(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        mapping = tmp_path / "mapping.csv"
        _write_team_mapping(mapping, [
            {"name": "France", "api_id": 2},
            {"name": "Germany", "api_id": 25},
        ])
        _write_fixtures_json(fixtures_dir, [
            {"id": 1, "status": "NS", "home_id": 2, "home_name": "France", "away_id": 25, "away_name": "Germany"},
            {"id": 2, "status": "FT", "home_id": 2, "home_name": "France", "away_id": 25, "away_name": "Germany"},
            {"id": 3, "status": "TBD", "home_id": 25, "home_name": "Germany", "away_id": 2, "away_name": "France"},
        ])
        result = parse_upcoming_fixtures(fixtures_dir, mapping)
        assert len(result) == 2
        assert set(result["fixture_id"]) == {1, 3}

    def test_maps_team_names_via_mapping(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        mapping = tmp_path / "mapping.csv"
        _write_team_mapping(mapping, [
            {"name": "France", "api_id": 2, "api_name": "France"},
            {"name": "Germany", "api_id": 25, "api_name": "Germany"},
        ])
        _write_fixtures_json(fixtures_dir, [
            {"id": 1, "status": "NS", "home_id": 2, "home_name": "France", "away_id": 25, "away_name": "Germany"},
        ])
        result = parse_upcoming_fixtures(fixtures_dir, mapping)
        assert result.iloc[0]["home_team"] == "France"
        assert result.iloc[0]["away_team"] == "Germany"

    def test_returns_empty_when_no_upcoming(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        mapping = tmp_path / "mapping.csv"
        _write_team_mapping(mapping, [{"name": "France", "api_id": 2}])
        _write_fixtures_json(fixtures_dir, [
            {"id": 1, "status": "FT", "home_id": 2, "away_id": 2},
        ])
        result = parse_upcoming_fixtures(fixtures_dir, mapping)
        assert result.empty

    def test_returns_empty_when_no_fixtures_dir(self, tmp_path):
        mapping = tmp_path / "mapping.csv"
        _write_team_mapping(mapping, [{"name": "X", "api_id": 1}])
        result = parse_upcoming_fixtures(tmp_path / "nonexistent", mapping)
        assert result.empty


# ---------------------------------------------------------------------------
# generate_wc_group_fixtures
# ---------------------------------------------------------------------------


class TestGenerateWcGroupFixtures:
    def test_generates_72_group_fixtures(self):
        result = generate_wc_group_fixtures()
        assert len(result) == 72
        assert set(["fixture_id", "home_team", "away_team", "is_knockout"]).issubset(result.columns)
        assert result["is_knockout"].eq(False).all()

    def test_uses_group_matchday_pairs_from_config(self, tmp_path):
        config_path = tmp_path / "wc2026.json"
        config_path.write_text(json.dumps({
            "groups": {"A": ["Team1", "Team2", "Team3", "Team4"]},
            "group_matchdays": [
                {"matchday": 1, "pairs": [[0, 1], [2, 3]]},
                {"matchday": 2, "pairs": [[0, 2], [3, 1]]},
                {"matchday": 3, "pairs": [[3, 0], [1, 2]]},
            ],
        }))

        result = generate_wc_group_fixtures(config_path=config_path)
        assert len(result) == 6
        pairings = set(zip(result["home_team"], result["away_team"]))
        assert ("Team1", "Team2") in pairings
        assert ("Team4", "Team1") in pairings


# ---------------------------------------------------------------------------
# generate_all_wc_pairings
# ---------------------------------------------------------------------------


class TestGenerateAllWcPairings:
    def test_generates_1128_pairings(self):
        result = generate_all_wc_pairings()
        assert len(result) == 1128

    def test_all_48_teams_present(self):
        result = generate_all_wc_pairings()
        all_teams = set(result["home_team"]) | set(result["away_team"])
        assert len(all_teams) == 48

    def test_no_self_matches(self):
        result = generate_all_wc_pairings()
        self_matches = result[result["home_team"] == result["away_team"]]
        assert self_matches.empty

    def test_group_fixtures_are_subset_of_pairings(self):
        group_df = generate_wc_group_fixtures()
        pairs_df = generate_all_wc_pairings()
        pair_set = set(zip(pairs_df["home_team"], pairs_df["away_team"]))
        reverse_set = {(a, h) for h, a in pair_set}
        full_set = pair_set | reverse_set
        for _, row in group_df.iterrows():
            assert (row["home_team"], row["away_team"]) in full_set

    def test_custom_config(self, tmp_path):
        config_path = tmp_path / "wc2026.json"
        config_path.write_text(json.dumps({
            "groups": {"A": ["T1", "T2", "T3", "T4"]},
        }))
        result = generate_all_wc_pairings(config_path=config_path)
        assert len(result) == 6  # C(4, 2) = 6


# ---------------------------------------------------------------------------
# build_inference_features
# ---------------------------------------------------------------------------


class TestBuildInferenceFeatures:
    def test_produces_rolling_features_for_known_teams(self):
        gold_df = _make_gold_df()
        upcoming = pd.DataFrame([{
            "fixture_id": 9999,
            "date_utc": "2026-07-01",
            "home_team": "France",
            "away_team": "Germany",
            "league_id": 1,
            "league_name": "World Cup",
            "season": 2026,
            "competition_tier": 1,
            "is_knockout": False,
            "is_neutral": True,
        }])
        upcoming["date_utc"] = pd.to_datetime(upcoming["date_utc"])

        result = build_inference_features(upcoming, gold_df)
        assert len(result) == 1
        assert "home_team_rolling_goals_for" in result.columns
        assert "away_team_rolling_goals_for" in result.columns
        assert "elo_diff" in result.columns
        assert not np.isnan(result.iloc[0]["elo_diff"])

    def test_returns_empty_for_empty_upcoming(self):
        gold_df = _make_gold_df()
        result = build_inference_features(pd.DataFrame(), gold_df)
        assert result.empty

    def test_unknown_team_gets_nan_rolling(self):
        gold_df = _make_gold_df()
        upcoming = pd.DataFrame([{
            "fixture_id": 8888,
            "date_utc": "2026-07-01",
            "home_team": "Atlantis",
            "away_team": "Germany",
            "league_id": 10,
            "league_name": "Friendly",
            "season": 2026,
            "competition_tier": 4,
            "is_knockout": False,
            "is_neutral": False,
        }])
        upcoming["date_utc"] = pd.to_datetime(upcoming["date_utc"])

        result = build_inference_features(upcoming, gold_df)
        assert np.isnan(result.iloc[0]["home_team_rolling_goals_for"])
        assert np.isnan(result.iloc[0]["home_elo_pre"])


# ---------------------------------------------------------------------------
# Helpers for parse_wc_results
# ---------------------------------------------------------------------------


def _write_finished_fixture(
    fixtures_dir: Path,
    entry: dict,
    window: str = "2026-06-11_2026-06-15",
) -> None:
    """Append one entry to an existing (or new) fixtures.json in the window dir."""
    window_dir = fixtures_dir / window
    window_dir.mkdir(parents=True, exist_ok=True)
    fp = window_dir / "fixtures.json"

    if fp.exists():
        existing = json.loads(fp.read_text())
        response = existing.get("response", [])
    else:
        response = []

    response.append({
        "fixture": {
            "id": entry["id"],
            "date": entry.get("date", "2026-06-11T20:00:00+00:00"),
            "status": {"short": entry["status"]},
        },
        "league": {
            "id": entry.get("league_id", 1),
            "name": entry.get("league_name", "World Cup"),
            "season": entry.get("season", 2026),
            "round": entry.get("round", "Group A - 1"),
        },
        "teams": {
            "home": {"id": entry.get("home_id", 100), "name": entry.get("home_name", "France")},
            "away": {"id": entry.get("away_id", 200), "name": entry.get("away_name", "Germany")},
        },
        "goals": {
            "home": entry.get("home_goals"),
            "away": entry.get("away_goals"),
        },
    })
    fp.write_text(json.dumps({"response": response}))


# ---------------------------------------------------------------------------
# parse_wc_results
# ---------------------------------------------------------------------------


class TestParseWcResults:
    def test_finished_wc_group_match_is_captured(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        mapping = tmp_path / "mapping.csv"
        _write_team_mapping(mapping, [
            {"name": "France", "api_id": 2},
            {"name": "Germany", "api_id": 25},
        ])
        _write_finished_fixture(fixtures_dir, {
            "id": 1, "status": "FT",
            "league_id": 1, "round": "Group A - 1",
            "home_id": 2, "home_name": "France",
            "away_id": 25, "away_name": "Germany",
            "home_goals": 2, "away_goals": 1,
        })
        result = parse_wc_results(fixtures_dir, mapping)
        assert ("France", "Germany") in result["group_results"]
        assert result["group_results"][("France", "Germany")] == (2, 1)

    def test_upcoming_match_is_excluded(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        mapping = tmp_path / "mapping.csv"
        _write_team_mapping(mapping, [
            {"name": "France", "api_id": 2},
            {"name": "Germany", "api_id": 25},
        ])
        _write_finished_fixture(fixtures_dir, {
            "id": 2, "status": "NS",
            "league_id": 1, "round": "Group A - 2",
            "home_id": 2, "home_goals": None, "away_id": 25, "away_goals": None,
        })
        result = parse_wc_results(fixtures_dir, mapping)
        assert len(result["group_results"]) == 0

    def test_historical_wc_season_is_excluded(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        mapping = tmp_path / "mapping.csv"
        _write_team_mapping(mapping, [
            {"name": "France", "api_id": 2},
            {"name": "Germany", "api_id": 25},
        ])
        _write_finished_fixture(fixtures_dir, {
            "id": 4, "status": "FT",
            "league_id": 1, "season": 2022, "round": "Group A - 1",
            "home_id": 2, "home_goals": 1, "away_id": 25, "away_goals": 0,
        })
        result = parse_wc_results(fixtures_dir, mapping)
        assert len(result["group_results"]) == 0

    def test_non_wc_league_is_excluded(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        mapping = tmp_path / "mapping.csv"
        _write_team_mapping(mapping, [
            {"name": "France", "api_id": 2},
            {"name": "Germany", "api_id": 25},
        ])
        _write_finished_fixture(fixtures_dir, {
            "id": 3, "status": "FT",
            "league_id": 39,  # Premier League, not WC
            "round": "Regular Season - 1",
            "home_id": 2, "home_goals": 1, "away_id": 25, "away_goals": 0,
        })
        result = parse_wc_results(fixtures_dir, mapping)
        assert len(result["group_results"]) == 0

    def test_ko_match_maps_to_correct_match_number(self, tmp_path):
        """'Round of 32 - 1' should map to internal match_num 73 (offset 72 + 1)."""
        fixtures_dir = tmp_path / "fixtures"
        mapping = tmp_path / "mapping.csv"
        _write_team_mapping(mapping, [
            {"name": "France", "api_id": 2},
            {"name": "Germany", "api_id": 25},
        ])
        _write_finished_fixture(fixtures_dir, {
            "id": 10, "status": "FT",
            "league_id": 1, "round": "Round of 32 - 1",
            "home_id": 2, "home_goals": 2,
            "away_id": 25, "away_goals": 0,
        })
        result = parse_wc_results(fixtures_dir, mapping)
        assert 73 in result["ko_results"]
        assert result["ko_results"][73]["home"] == "France"
        assert result["ko_results"][73]["away_goals"] == 0

    def test_mixed_group_and_ko_results(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        mapping = tmp_path / "mapping.csv"
        _write_team_mapping(mapping, [
            {"name": "France", "api_id": 2},
            {"name": "Germany", "api_id": 25},
            {"name": "Brazil", "api_id": 6},
            {"name": "Argentina", "api_id": 26},
        ])
        # Two group matches
        _write_finished_fixture(fixtures_dir, {
            "id": 1, "status": "FT", "league_id": 1, "round": "Group A - 1",
            "home_id": 2, "home_goals": 1, "away_id": 25, "away_goals": 1,
        })
        _write_finished_fixture(fixtures_dir, {
            "id": 2, "status": "AET", "league_id": 1, "round": "Round of 16 - 2",
            "home_id": 6, "home_goals": 2, "away_id": 26, "away_goals": 1,
        })
        result = parse_wc_results(fixtures_dir, mapping)
        assert len(result["group_results"]) == 1
        assert 90 in result["ko_results"]   # offset 88 + 2
        assert result["ko_results"][90]["decided_by"] == "AET"

    def test_next_matchday_after_group_md1(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        mapping = tmp_path / "mapping.csv"
        _write_team_mapping(mapping, [
            {"name": "France", "api_id": 2},
            {"name": "Germany", "api_id": 25},
        ])
        _write_finished_fixture(fixtures_dir, {
            "id": 1, "status": "FT", "league_id": 1, "round": "Group A - 1",
            "home_id": 2, "home_goals": 1, "away_id": 25, "away_goals": 0,
        })
        result = parse_wc_results(fixtures_dir, mapping)
        assert result["next_matchday"] == 2

    def test_next_matchday_after_r32(self, tmp_path):
        fixtures_dir = tmp_path / "fixtures"
        mapping = tmp_path / "mapping.csv"
        _write_team_mapping(mapping, [
            {"name": "France", "api_id": 2},
            {"name": "Germany", "api_id": 25},
        ])
        _write_finished_fixture(fixtures_dir, {
            "id": 5, "status": "FT", "league_id": 1, "round": "Round of 32 - 3",
            "home_id": 2, "home_goals": 1, "away_id": 25, "away_goals": 0,
        })
        result = parse_wc_results(fixtures_dir, mapping)
        assert result["next_matchday"] == "R16"

    def test_no_fixtures_dir_returns_empty(self, tmp_path):
        result = parse_wc_results(tmp_path / "nonexistent", tmp_path / "mapping.csv")
        assert result["group_results"] == {}
        assert result["ko_results"] == {}
        assert result["next_matchday"] == 1
