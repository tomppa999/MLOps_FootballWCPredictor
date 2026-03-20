"""Tests for src.silver.schema."""

import pandas as pd
import pytest

from src.silver.schema import (
    CORE_STAT_SUFFIXES,
    PARTIAL_STAT_SUFFIXES,
    SILVER_COLUMNS,
    SILVER_DTYPES,
    STAT_TYPE_TO_COLUMN,
    apply_dtypes,
    compute_stats_tier,
    validate_silver,
)


class TestColumnConsistency:
    def test_all_stat_columns_have_both_sides(self):
        for suffix in STAT_TYPE_TO_COLUMN.values():
            assert f"home_{suffix}" in SILVER_COLUMNS
            assert f"away_{suffix}" in SILVER_COLUMNS

    def test_no_duplicate_columns(self):
        assert len(SILVER_COLUMNS) == len(set(SILVER_COLUMNS))

    def test_dtypes_cover_typed_columns(self):
        skip = {
            "date_utc", "kickoff_utc", "league_name", "round",
            "venue_name", "venue_city", "match_status", "neutral_source",
            "home_team_api_name", "away_team_api_name",
            "home_team", "away_team",
            "home_country_code", "away_country_code",
            "home_confederation", "away_confederation",
        }
        for col in SILVER_COLUMNS:
            if col in skip:
                continue
            assert col in SILVER_DTYPES, f"Missing dtype for {col}"

    def test_stats_tier_is_string_dtype(self):
        assert SILVER_DTYPES["stats_tier"] == "string"

    def test_column_count(self):
        assert len(SILVER_COLUMNS) >= 68


class TestValidation:
    def _make_valid_row(self) -> dict:
        row = {col: None for col in SILVER_COLUMNS}
        row["fixture_id"] = 12345
        row["date_utc"] = pd.Timestamp("2022-12-13").date()
        row["kickoff_utc"] = pd.Timestamp("2022-12-13T19:00:00")
        row["league_id"] = 1
        row["league_name"] = "World Cup"
        row["season"] = 2022
        row["round"] = "Semi-finals"
        row["competition_tier"] = 1
        row["is_knockout"] = True
        row["is_neutral"] = True
        row["neutral_source"] = "elo"
        row["match_status"] = "FT"
        row["home_team_api_id"] = 26
        row["home_team_api_name"] = "Argentina"
        row["home_team"] = "Argentina"
        row["home_country_code"] = "ARG"
        row["home_confederation"] = "CONMEBOL"
        row["away_team_api_id"] = 3
        row["away_team_api_name"] = "Croatia"
        row["away_team"] = "Croatia"
        row["away_country_code"] = "HRV"
        row["away_confederation"] = "UEFA"
        row["has_statistics"] = True
        row["stats_tier"] = "full"
        return row

    def test_valid_row_passes(self):
        df = pd.DataFrame([self._make_valid_row()])
        errors = validate_silver(df)
        assert errors == []

    def test_missing_column_detected(self):
        df = pd.DataFrame([self._make_valid_row()])
        df = df.drop(columns=["fixture_id"])
        errors = validate_silver(df)
        assert any("Missing columns" in e for e in errors)

    def test_duplicate_fixture_id_detected(self):
        row = self._make_valid_row()
        df = pd.DataFrame([row, row])
        errors = validate_silver(df)
        assert any("Duplicate" in e for e in errors)

    def test_null_in_non_nullable_detected(self):
        row = self._make_valid_row()
        row["league_name"] = None
        df = pd.DataFrame([row])
        errors = validate_silver(df)
        assert any("league_name" in e for e in errors)


class TestComputeStatsTier:
    def _make_full_row(self) -> dict:
        row = {}
        for suffix in CORE_STAT_SUFFIXES:
            row[f"home_{suffix}"] = 10
            row[f"away_{suffix}"] = 10
        return row

    def test_full_when_all_core_present(self):
        row = self._make_full_row()
        assert compute_stats_tier(row, has_statistics=True) == "full"

    def test_none_when_no_statistics(self):
        assert compute_stats_tier({}, has_statistics=False) == "none"

    def test_partial_when_key_stats_present(self):
        row = {}
        for suffix in PARTIAL_STAT_SUFFIXES:
            row[f"home_{suffix}"] = 5
            row[f"away_{suffix}"] = 5
        assert compute_stats_tier(row, has_statistics=True) == "partial"

    def test_cards_only_when_insufficient(self):
        row = {"home_yellow_cards": 2, "away_yellow_cards": 1}
        assert compute_stats_tier(row, has_statistics=True) == "cards_only"


class TestApplyDtypes:
    def test_integer_columns_become_nullable(self):
        df = pd.DataFrame({"fixture_id": [1], "home_goals": [3], "away_goals": [None]})
        df = apply_dtypes(df)
        assert df["home_goals"].dtype.name == "Int16"
        assert df["away_goals"].dtype.name == "Int16"
        assert pd.isna(df["away_goals"].iloc[0])
