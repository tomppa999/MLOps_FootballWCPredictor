"""Tests for helper functions in src.dashboard.app."""

import pandas as pd
import pytest

from src.dashboard.app import resolve_ko_fixtures


def _make_ko_fixtures() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "match_num": 49,
            "stage": "R32",
            "home_team": "France",
            "away_team": "Morocco",
            "status": "locked",
            "home_goals": 2,
            "away_goals": 0,
            "decided_by": "FT",
            "pairing_frequency": 1.0,
        },
        {
            "match_num": 65,
            "stage": "QF",
            "home_team": "Brazil",
            "away_team": "Argentina",
            "status": "predicted",
            "home_goals": None,
            "away_goals": None,
            "decided_by": "",
            "pairing_frequency": 0.38,
        },
        {
            "match_num": 57,
            "stage": "R16",
            "home_team": "Spain",
            "away_team": "England",
            "status": "predicted",
            "home_goals": None,
            "away_goals": None,
            "decided_by": "",
            "pairing_frequency": 0.24,
        },
    ])


class TestResolveKoFixtures:
    def test_returns_list_of_dicts(self):
        result = resolve_ko_fixtures(_make_ko_fixtures())
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_empty_df_returns_empty_list(self):
        assert resolve_ko_fixtures(pd.DataFrame()) == []

    def test_none_returns_empty_list(self):
        assert resolve_ko_fixtures(None) == []

    def test_ordered_by_stage_then_match_num(self):
        result = resolve_ko_fixtures(_make_ko_fixtures())
        stages = [r["stage"] for r in result]
        # R32 < R16 < QF
        assert stages == ["R32", "R16", "QF"]

    def test_match_num_ordering_within_stage(self):
        df = pd.DataFrame([
            {"match_num": 52, "stage": "R32", "home_team": "A", "away_team": "B",
             "status": "predicted", "home_goals": None, "away_goals": None,
             "decided_by": "", "pairing_frequency": 0.5},
            {"match_num": 49, "stage": "R32", "home_team": "C", "away_team": "D",
             "status": "predicted", "home_goals": None, "away_goals": None,
             "decided_by": "", "pairing_frequency": 0.4},
        ])
        result = resolve_ko_fixtures(df)
        assert result[0]["match_num"] == 49
        assert result[1]["match_num"] == 52

    def test_status_fields_preserved(self):
        result = resolve_ko_fixtures(_make_ko_fixtures())
        locked = next(r for r in result if r["status"] == "locked")
        predicted = next(r for r in result if r["status"] == "predicted")
        assert locked["home_goals"] == 2
        # pandas converts None to NaN for numeric-mixed columns
        import math
        assert predicted["home_goals"] is None or (
            isinstance(predicted["home_goals"], float) and math.isnan(predicted["home_goals"])
        )

    def test_pairing_frequency_preserved(self):
        result = resolve_ko_fixtures(_make_ko_fixtures())
        by_match = {r["match_num"]: r for r in result}
        assert by_match[49]["pairing_frequency"] == pytest.approx(1.0)
        assert by_match[65]["pairing_frequency"] == pytest.approx(0.38)
