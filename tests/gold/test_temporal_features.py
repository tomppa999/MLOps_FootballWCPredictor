"""Tests for src.gold.temporal_features.

Covers ``days_since_last_match`` (per-team, time-aware) and ``rest_diff``.
The function reshapes Gold-shape rows into a team-centric history, sorts by
team and date, and uses ``shift(1)`` per team to compute the gap to that
team's previous match. First appearances must be NaN.

These columns proxy national-team squad cohesion (*Eingespieltheit*),
not player rest -- at international cadence players play 4-6 club matches
between national-team windows. The variable name ``rest_diff`` is retained
from the initial design hypothesis. See ``docs/feature_spec.md`` and report
Section 7.2 for the framing.
"""

import numpy as np
import pandas as pd
import pytest

from src.gold.temporal_features import add_days_since_last_match


def _make_row(
    fixture_id: int, date: str, home_team: str, away_team: str
) -> dict:
    return {
        "fixture_id": fixture_id,
        "date_utc": date,
        "home_team": home_team,
        "away_team": away_team,
    }


class TestDaysSinceLastMatch:
    """Per-team gap to previous match, regardless of home/away side."""

    def test_first_appearance_is_nan(self):
        df = pd.DataFrame([_make_row(1, "2022-01-01", "A", "B")])
        result = add_days_since_last_match(df)
        assert pd.isna(result["home_days_since_last_match"].iloc[0])
        assert pd.isna(result["away_days_since_last_match"].iloc[0])

    def test_second_match_uses_prior_date(self):
        df = pd.DataFrame([
            _make_row(1, "2022-01-01", "A", "B"),
            _make_row(2, "2022-01-08", "A", "C"),
        ])
        result = add_days_since_last_match(df)
        # Team A played 7 days ago; team C is new.
        assert result["home_days_since_last_match"].iloc[1] == pytest.approx(7.0)
        assert pd.isna(result["away_days_since_last_match"].iloc[1])

    def test_team_appears_as_away_then_home(self):
        """Prior-match lookup is per-team, regardless of side it played."""
        df = pd.DataFrame([
            _make_row(1, "2022-01-01", "X", "A"),
            _make_row(2, "2022-01-11", "A", "Y"),
        ])
        result = add_days_since_last_match(df)
        assert result["home_days_since_last_match"].iloc[1] == pytest.approx(10.0)

    def test_uses_only_strictly_prior_matches(self):
        """The current match must never count as its own prior."""
        df = pd.DataFrame([
            _make_row(1, "2022-01-01", "A", "B"),
            _make_row(2, "2022-01-08", "A", "B"),
        ])
        result = add_days_since_last_match(df)
        assert result["home_days_since_last_match"].iloc[0] is pd.NA or pd.isna(
            result["home_days_since_last_match"].iloc[0]
        )
        assert result["home_days_since_last_match"].iloc[1] == pytest.approx(7.0)
        assert result["away_days_since_last_match"].iloc[1] == pytest.approx(7.0)

    def test_handles_string_dates(self):
        """Function should accept string date_utc values (matches build_gold flow)."""
        df = pd.DataFrame([
            _make_row(1, "2022-01-01", "A", "B"),
            _make_row(2, "2022-01-15", "A", "C"),
        ])
        result = add_days_since_last_match(df)
        assert result["home_days_since_last_match"].iloc[1] == pytest.approx(14.0)

    def test_no_input_mutation(self):
        df = pd.DataFrame([
            _make_row(1, "2022-01-01", "A", "B"),
            _make_row(2, "2022-01-08", "A", "C"),
        ])
        original_cols = list(df.columns)
        original_values = df.copy(deep=True)
        _ = add_days_since_last_match(df)
        assert list(df.columns) == original_cols
        pd.testing.assert_frame_equal(df, original_values)

    def test_columns_added(self):
        df = pd.DataFrame([_make_row(1, "2022-01-01", "A", "B")])
        result = add_days_since_last_match(df)
        for col in (
            "home_days_since_last_match",
            "away_days_since_last_match",
            "rest_diff",
        ):
            assert col in result.columns


class TestRestDiff:
    """rest_diff = home_days_since_last_match - away_days_since_last_match."""

    def test_sign_convention_home_more_rested(self):
        """If home rested 14 days and away rested 7, rest_diff = +7."""
        df = pd.DataFrame([
            _make_row(1, "2022-01-01", "A", "X"),  # A first appearance
            _make_row(2, "2022-01-08", "B", "C"),  # C first appearance
            _make_row(3, "2022-01-15", "A", "C"),  # A: 14d rest, C: 7d rest
        ])
        result = add_days_since_last_match(df)
        assert result["home_days_since_last_match"].iloc[2] == pytest.approx(14.0)
        assert result["away_days_since_last_match"].iloc[2] == pytest.approx(7.0)
        assert result["rest_diff"].iloc[2] == pytest.approx(7.0)

    def test_sign_convention_away_more_rested(self):
        """If away rested longer than home, rest_diff is negative."""
        df = pd.DataFrame([
            _make_row(1, "2022-01-01", "A", "C"),
            _make_row(2, "2022-01-04", "A", "X"),  # A plays again 3d later
            _make_row(3, "2022-01-15", "A", "C"),  # A: 11d, C: 14d → rest_diff = -3
        ])
        result = add_days_since_last_match(df)
        assert result["rest_diff"].iloc[2] == pytest.approx(-3.0)

    def test_nan_propagates_when_both_missing(self):
        df = pd.DataFrame([_make_row(1, "2022-01-01", "A", "B")])
        result = add_days_since_last_match(df)
        assert pd.isna(result["rest_diff"].iloc[0])

    def test_nan_propagates_when_home_missing(self):
        df = pd.DataFrame([
            _make_row(1, "2022-01-01", "Z", "A"),
            _make_row(2, "2022-01-10", "B", "A"),  # B first appearance, A: 9d
        ])
        result = add_days_since_last_match(df)
        assert pd.isna(result["home_days_since_last_match"].iloc[1])
        assert result["away_days_since_last_match"].iloc[1] == pytest.approx(9.0)
        assert pd.isna(result["rest_diff"].iloc[1])

    def test_nan_propagates_when_away_missing(self):
        df = pd.DataFrame([
            _make_row(1, "2022-01-01", "A", "Z"),
            _make_row(2, "2022-01-10", "A", "B"),  # B first appearance, A: 9d
        ])
        result = add_days_since_last_match(df)
        assert result["home_days_since_last_match"].iloc[1] == pytest.approx(9.0)
        assert pd.isna(result["away_days_since_last_match"].iloc[1])
        assert pd.isna(result["rest_diff"].iloc[1])
