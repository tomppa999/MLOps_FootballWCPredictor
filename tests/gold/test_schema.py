"""Tests for src.gold.schema."""

import pandas as pd
import pytest

from src.gold.schema import (
    GOLD_COLUMNS,
    GOLD_DTYPES,
    NON_NULLABLE_COLUMNS,
    apply_gold_dtypes,
    validate_gold,
)


def _make_valid_gold_row() -> dict:
    row = {col: None for col in GOLD_COLUMNS}
    row["fixture_id"] = 99999
    row["date_utc"] = "2022-12-13"
    row["season"] = 2022
    row["league_id"] = 1
    row["league_name"] = "World Cup"
    row["home_team"] = "Argentina"
    row["home_country_code"] = "ARG"
    row["home_confederation"] = "CONMEBOL"
    row["away_team"] = "Croatia"
    row["away_country_code"] = "HRV"
    row["away_confederation"] = "UEFA"
    row["home_team_match_index"] = 42
    row["away_team_match_index"] = 38
    row["competition_tier"] = 1
    row["is_knockout"] = True
    row["is_neutral"] = True
    row["home_elo_pre"] = 1800.0
    row["away_elo_pre"] = 1700.0
    row["elo_diff"] = 100.0
    row["home_goals"] = 3
    row["away_goals"] = 0
    return row


class TestColumnConsistency:
    def test_no_duplicate_columns(self):
        assert len(GOLD_COLUMNS) == len(set(GOLD_COLUMNS))

    def test_targets_present(self):
        assert "home_goals" in GOLD_COLUMNS
        assert "away_goals" in GOLD_COLUMNS

    def test_elo_diff_present(self):
        assert "elo_diff" in GOLD_COLUMNS

    def test_match_index_present(self):
        assert "home_team_match_index" in GOLD_COLUMNS
        assert "away_team_match_index" in GOLD_COLUMNS



class TestValidation:
    def test_valid_row_passes(self):
        df = pd.DataFrame([_make_valid_gold_row()])
        errors = validate_gold(df)
        assert errors == []

    def test_missing_column_detected(self):
        df = pd.DataFrame([_make_valid_gold_row()])
        df = df.drop(columns=["elo_diff"])
        errors = validate_gold(df)
        assert any("Missing columns" in e for e in errors)

    def test_extra_column_detected(self):
        df = pd.DataFrame([_make_valid_gold_row()])
        df["extra_col"] = 42
        errors = validate_gold(df)
        assert any("Unexpected columns" in e for e in errors)

    def test_duplicate_fixture_id_detected(self):
        row = _make_valid_gold_row()
        df = pd.DataFrame([row, row])
        errors = validate_gold(df)
        assert any("Duplicate" in e for e in errors)

    def test_null_target_detected(self):
        row = _make_valid_gold_row()
        row["home_goals"] = None
        df = pd.DataFrame([row])
        errors = validate_gold(df)
        assert any("home_goals" in e for e in errors)

    def test_null_fixture_id_detected(self):
        row = _make_valid_gold_row()
        row["fixture_id"] = None
        df = pd.DataFrame([row])
        errors = validate_gold(df)
        assert any("fixture_id" in e for e in errors)


class TestApplyDtypes:
    def test_integer_targets_nullable(self):
        df = pd.DataFrame([_make_valid_gold_row()])
        df = apply_gold_dtypes(df)
        assert df["home_goals"].dtype.name == "Int16"

    def test_float_elo_columns(self):
        df = pd.DataFrame([_make_valid_gold_row()])
        df = apply_gold_dtypes(df)
        assert df["elo_diff"].dtype.name == "Float64"

    def test_match_index_dtype(self):
        df = pd.DataFrame([_make_valid_gold_row()])
        df = apply_gold_dtypes(df)
        assert df["home_team_match_index"].dtype.name == "Int32"
