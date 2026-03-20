"""Tests for src.silver.build_silver — fixture and statistics parsing."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.silver.build_silver import (
    apply_team_mapping,
    filter_unmapped_teams,
    filter_youth_teams,
    load_team_mapping,
    parse_fixtures_file,
    parse_statistics_for_fixture,
)

FIXTURE_SAMPLE = Path("data_samples/raw/api_football/fixture_sample_1.json")
STATS_SAMPLE = Path("data_samples/raw/api_football/statistics_sample_1.json")


class TestParseFixtures:
    def test_sample_fixture_parses(self):
        rows = parse_fixtures_file(FIXTURE_SAMPLE)
        assert len(rows) == 1
        row = rows[0]
        assert row["fixture_id"] == 978279
        assert row["home_team_api_name"] == "Argentina"
        assert row["away_team_api_name"] == "Croatia"
        assert row["home_goals"] == 3
        assert row["away_goals"] == 0
        assert row["match_status"] == "FT"

    def test_score_breakdown(self):
        rows = parse_fixtures_file(FIXTURE_SAMPLE)
        row = rows[0]
        assert row["home_goals_ht"] == 2
        assert row["away_goals_ht"] == 0
        assert row["home_goals_ft"] == 3
        assert row["away_goals_ft"] == 0
        assert row["home_goals_et"] is None
        assert row["home_goals_pen"] is None

    def test_league_metadata(self):
        rows = parse_fixtures_file(FIXTURE_SAMPLE)
        row = rows[0]
        assert row["league_id"] == 1
        assert row["league_name"] == "World Cup"
        assert row["season"] == 2022
        assert row["round"] == "Semi-finals"


class TestParseStatistics:
    def test_sample_stats_parse(self):
        result = parse_statistics_for_fixture(STATS_SAMPLE)
        assert result["has_statistics"] is True
        assert result["home_shots_on_goal"] == 7
        assert result["away_shots_on_goal"] == 2
        assert result["home_total_shots"] == 9
        assert result["away_total_shots"] == 12

    def test_possession_pct_parsed(self):
        result = parse_statistics_for_fixture(STATS_SAMPLE)
        assert result["home_possession_pct"] == 39.0
        assert result["away_possession_pct"] == 61.0

    def test_pass_accuracy_pct_parsed(self):
        result = parse_statistics_for_fixture(STATS_SAMPLE)
        assert result["home_pass_accuracy_pct"] == 83.0
        assert result["away_pass_accuracy_pct"] == 88.0

    def test_red_cards_null_becomes_zero(self):
        result = parse_statistics_for_fixture(STATS_SAMPLE)
        assert result["home_red_cards"] == 0
        assert result["away_red_cards"] == 0

    def test_stats_tier_full_when_red_cards_fixed(self):
        result = parse_statistics_for_fixture(STATS_SAMPLE)
        assert result["stats_tier"] == "full"


class TestStatsTierLevels:
    """Test stats_tier assignment for each of the four levels."""

    def test_tier_none_for_empty_response(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"response": []}))
        result = parse_statistics_for_fixture(p)
        assert result["stats_tier"] == "none"
        assert result["has_statistics"] is False

    def test_tier_full_when_all_core_present(self):
        result = parse_statistics_for_fixture(STATS_SAMPLE)
        assert result["stats_tier"] == "full"

    def test_tier_partial_when_key_stats_present(self, tmp_path):
        stats = {
            "response": [
                {
                    "team": {"id": 1},
                    "statistics": [
                        {"type": "Shots on Goal", "value": 5},
                        {"type": "Total Shots", "value": 10},
                        {"type": "Ball Possession", "value": "55%"},
                    ],
                },
                {
                    "team": {"id": 2},
                    "statistics": [
                        {"type": "Shots on Goal", "value": 3},
                        {"type": "Total Shots", "value": 8},
                        {"type": "Ball Possession", "value": "45%"},
                    ],
                },
            ]
        }
        p = tmp_path / "partial.json"
        p.write_text(json.dumps(stats))
        result = parse_statistics_for_fixture(p)
        assert result["stats_tier"] == "partial"

    def test_tier_cards_only_when_minimal_stats(self, tmp_path):
        stats = {
            "response": [
                {
                    "team": {"id": 1},
                    "statistics": [
                        {"type": "Yellow Cards", "value": 2},
                        {"type": "Red Cards", "value": 0},
                    ],
                },
                {
                    "team": {"id": 2},
                    "statistics": [
                        {"type": "Yellow Cards", "value": 1},
                        {"type": "Red Cards", "value": 0},
                    ],
                },
            ]
        }
        p = tmp_path / "cards_only.json"
        p.write_text(json.dumps(stats))
        result = parse_statistics_for_fixture(p)
        assert result["stats_tier"] == "cards_only"


class TestUnmappedTeamFilter:
    def test_null_country_code_dropped(self):
        df = pd.DataFrame(
            {
                "home_team_api_name": ["Algeria B", "France"],
                "away_team_api_name": ["Morocco", "Germany"],
                "home_country_code": [None, "FRA"],
                "away_country_code": ["MAR", "DEU"],
            }
        )
        result = filter_unmapped_teams(df)
        assert len(result) == 1
        assert result["home_team_api_name"].iloc[0] == "France"

    def test_both_mapped_kept(self):
        df = pd.DataFrame(
            {
                "home_team_api_name": ["Spain"],
                "away_team_api_name": ["Italy"],
                "home_country_code": ["ESP"],
                "away_country_code": ["ITA"],
            }
        )
        result = filter_unmapped_teams(df)
        assert len(result) == 1

    def test_away_null_dropped(self):
        df = pd.DataFrame(
            {
                "home_team_api_name": ["France"],
                "away_team_api_name": ["Basque Country"],
                "home_country_code": ["FRA"],
                "away_country_code": [None],
            }
        )
        result = filter_unmapped_teams(df)
        assert len(result) == 0


class TestYouthFilter:
    def test_u18_teams_dropped(self):
        df = pd.DataFrame(
            {
                "home_team_api_name": ["Portugal U18", "France"],
                "away_team_api_name": ["Spain U18", "Germany"],
            }
        )
        result = filter_youth_teams(df)
        assert len(result) == 1
        assert result["home_team_api_name"].iloc[0] == "France"

    def test_u23_teams_dropped(self):
        df = pd.DataFrame(
            {
                "home_team_api_name": ["Brazil", "Tajikistan U23"],
                "away_team_api_name": ["Argentina U23", "Japan"],
            }
        )
        result = filter_youth_teams(df)
        assert len(result) == 0

    def test_senior_teams_kept(self):
        df = pd.DataFrame(
            {
                "home_team_api_name": ["France", "Spain"],
                "away_team_api_name": ["Germany", "Italy"],
            }
        )
        result = filter_youth_teams(df)
        assert len(result) == 2

    def test_u_in_name_but_not_youth_kept(self):
        df = pd.DataFrame(
            {
                "home_team_api_name": ["Uruguay"],
                "away_team_api_name": ["United States"],
            }
        )
        result = filter_youth_teams(df)
        assert len(result) == 1


class TestTeamMapping:
    def test_apply_mapping_adds_canonical_names(self):
        team_map = load_team_mapping(Path("data/mappings/team_mapping_master_merged.csv"))
        fixtures = pd.DataFrame(
            {
                "home_team_api_id": [26],
                "home_team_api_name": ["Argentina"],
                "away_team_api_id": [3],
                "away_team_api_name": ["Croatia"],
            }
        )
        result = apply_team_mapping(fixtures, team_map)
        assert result["home_team"].iloc[0] == "Argentina"
        assert result["away_team"].iloc[0] == "Croatia"
        assert result["home_country_code"].iloc[0] == "ARG"
        assert result["away_country_code"].iloc[0] == "HRV"
        assert result["home_confederation"].iloc[0] == "CONMEBOL"
        assert result["away_confederation"].iloc[0] == "UEFA"

    def test_fyr_macedonia_maps_correctly(self):
        team_map = load_team_mapping(Path("data/mappings/team_mapping_master_merged.csv"))
        fixtures = pd.DataFrame(
            {
                "home_team_api_id": [1105],
                "home_team_api_name": ["FYR Macedonia"],
                "away_team_api_id": [26],
                "away_team_api_name": ["Argentina"],
            }
        )
        result = apply_team_mapping(fixtures, team_map)
        assert result["home_team"].iloc[0] == "North Macedonia"
        assert result["home_country_code"].iloc[0] == "MKD"
        assert result["home_confederation"].iloc[0] == "UEFA"

    def test_unmapped_team_uses_api_name_fallback(self):
        """Teams missing from the mapping get api_name as fallback but null country_code."""
        team_map = load_team_mapping(Path("data/mappings/team_mapping_master_merged.csv"))
        fixtures = pd.DataFrame(
            {
                "home_team_api_id": [999999],
                "home_team_api_name": ["Hull City"],
                "away_team_api_id": [26],
                "away_team_api_name": ["Argentina"],
            }
        )
        result = apply_team_mapping(fixtures, team_map)
        assert result["home_team"].iloc[0] == "Hull City"
        assert pd.isna(result["home_country_code"].iloc[0])

    def test_swaziland_maps_correctly(self):
        team_map = load_team_mapping(Path("data/mappings/team_mapping_master_merged.csv"))
        fixtures = pd.DataFrame(
            {
                "home_team_api_id": [1488],
                "home_team_api_name": ["Swaziland"],
                "away_team_api_id": [26],
                "away_team_api_name": ["Argentina"],
            }
        )
        result = apply_team_mapping(fixtures, team_map)
        assert result["home_team"].iloc[0] == "Eswatini"
        assert result["home_country_code"].iloc[0] == "SWZ"
        assert result["home_confederation"].iloc[0] == "CAF"
