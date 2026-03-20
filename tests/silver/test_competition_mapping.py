"""Tests for src.silver.competition_mapping."""

import datetime

import pandas as pd
import pytest

from src.silver.competition_mapping import (
    TIER_MAP,
    assign_competition_columns,
    competition_tier,
    is_knockout,
)


class TestCompetitionTier:
    def test_world_cup_is_tier_1(self):
        assert competition_tier(1) == 1

    def test_euro_is_tier_2(self):
        assert competition_tier(4) == 2

    def test_copa_america_is_tier_2(self):
        assert competition_tier(9) == 2

    def test_wc_qualification_is_tier_3(self):
        assert competition_tier(32) == 3

    def test_uefa_nations_league_is_tier_3(self):
        assert competition_tier(5) == 3

    def test_friendlies_is_tier_4(self):
        assert competition_tier(10) == 4

    def test_unknown_league_defaults_to_4(self):
        assert competition_tier(99999) == 4


class TestIsKnockout:
    @pytest.mark.parametrize(
        "round_str",
        [
            "Semi-finals",
            "Quarter-finals",
            "Round of 16",
            "Final",
            "3rd Place Final",
            "Round of 32",
            "Play-offs - Path A",
            "8th Finals",
        ],
    )
    def test_knockout_rounds(self, round_str: str):
        assert is_knockout(round_str) is True

    @pytest.mark.parametrize(
        "round_str",
        [
            "Group A",
            "Group B - 1",
            "Friendlies 1",
            "Qualification Round 3",
            "Regular Season - 5",
            "",
            None,
        ],
    )
    def test_non_knockout_rounds(self, round_str: str | None):
        assert is_knockout(round_str) is False


class TestAssignCompetitionColumns:
    def test_adds_columns(self):
        df = pd.DataFrame(
            {
                "league_id": [1, 10, 5],
                "round": ["Semi-finals", "Friendlies 1", "Group A - 3"],
            }
        )
        result = assign_competition_columns(df)
        assert list(result["competition_tier"]) == [1, 4, 3]
        assert list(result["is_knockout"]) == [True, False, False]

    def test_euro_2020_qualifiers_reclassified_to_tier_3(self):
        """Matches in Euro 2020 season before 2021-06-11 are qualifiers → tier 3."""
        df = pd.DataFrame(
            {
                "league_id": [4, 4, 4],
                "season": [2020, 2020, 2020],
                "round": ["Group A", "Group B", "Quarter-finals"],
                "date_utc": [
                    datetime.date(2019, 6, 10),   # qualifier
                    datetime.date(2021, 6, 15),   # tournament
                    datetime.date(2021, 7, 3),    # tournament
                ],
            }
        )
        result = assign_competition_columns(df)
        assert list(result["competition_tier"]) == [3, 2, 2]

    def test_afcon_2019_qualifiers_reclassified_to_tier_3(self):
        """Matches in AFCON 2019 season before 2019-06-21 are qualifiers → tier 3."""
        df = pd.DataFrame(
            {
                "league_id": [6, 6],
                "season": [2019, 2019],
                "round": ["Qualifying Round 1", "Group A"],
                "date_utc": [
                    datetime.date(2018, 10, 12),  # qualifier
                    datetime.date(2019, 6, 25),   # tournament
                ],
            }
        )
        result = assign_competition_columns(df)
        assert list(result["competition_tier"]) == [3, 2]

    def test_other_euro_seasons_unaffected(self):
        """Euro 2024 should remain tier 2 regardless of date."""
        df = pd.DataFrame(
            {
                "league_id": [4],
                "season": [2024],
                "round": ["Group A"],
                "date_utc": [datetime.date(2024, 6, 15)],
            }
        )
        result = assign_competition_columns(df)
        assert result["competition_tier"].iloc[0] == 2

    def test_no_date_utc_column_still_works(self):
        """When date_utc is absent overrides are skipped without error."""
        df = pd.DataFrame({"league_id": [4], "round": ["Group A"]})
        result = assign_competition_columns(df)
        assert result["competition_tier"].iloc[0] == 2
