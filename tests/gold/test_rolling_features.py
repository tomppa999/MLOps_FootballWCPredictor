"""Tests for src.gold.rolling_features — time-awareness, stats_tier filtering, match indices."""

import numpy as np
import pandas as pd
import pytest

from src.gold.rolling_features import compute_rolling_features


def _make_silver_row(
    fixture_id: int,
    date: str,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    stats_tier: str = "full",
    home_shots_on_goal: int | None = 5,
    home_total_shots: int | None = 10,
    away_shots_on_goal: int | None = 4,
    away_total_shots: int | None = 8,
    home_shots_off_goal: int | None = 3,
    away_shots_off_goal: int | None = 2,
    home_fouls: int | None = 12,
    away_fouls: int | None = 10,
    home_corner_kicks: int | None = 6,
    away_corner_kicks: int | None = 4,
    home_possession_pct: float | None = 55.0,
    away_possession_pct: float | None = 45.0,
) -> dict:
    return {
        "fixture_id": fixture_id,
        "date_utc": date,
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_shots_on_goal": home_shots_on_goal,
        "home_total_shots": home_total_shots,
        "away_shots_on_goal": away_shots_on_goal,
        "away_total_shots": away_total_shots,
        "home_shots_off_goal": home_shots_off_goal,
        "away_shots_off_goal": away_shots_off_goal,
        "home_fouls": home_fouls,
        "away_fouls": away_fouls,
        "home_corner_kicks": home_corner_kicks,
        "away_corner_kicks": away_corner_kicks,
        "home_possession_pct": home_possession_pct,
        "away_possession_pct": away_possession_pct,
        "stats_tier": stats_tier,
        "match_status": "FT",
    }


def _build_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestNoLeakage:
    """The first match for a team must have NaN rolling features."""

    def test_first_match_has_nan_rolling(self):
        rows = [
            _make_silver_row(1, "2022-01-01", "A", "B", 2, 1),
            _make_silver_row(2, "2022-02-01", "A", "C", 1, 0),
        ]
        df = _build_df(rows)
        result = compute_rolling_features(df, n_matches=10)

        first = result.iloc[0]
        assert pd.isna(first["home_team_rolling_goals_for"])
        assert pd.isna(first["away_team_rolling_goals_for"])

    def test_current_match_not_in_own_rolling(self):
        """Match at t=1 should only see t=0, never itself."""
        rows = [
            _make_silver_row(1, "2022-01-01", "A", "B", 3, 0),
            _make_silver_row(2, "2022-02-01", "A", "C", 0, 5),
        ]
        df = _build_df(rows)
        result = compute_rolling_features(df, n_matches=10)

        second = result.iloc[1]
        assert second["home_team_rolling_goals_for"] == pytest.approx(3.0)
        assert second["home_team_rolling_goals_against"] == pytest.approx(0.0)


class TestStatsTierFiltering:
    """Shot features must exclude matches with stats_tier='none'."""

    def test_shot_features_nan_when_all_priors_none(self):
        rows = [
            _make_silver_row(
                1, "2022-01-01", "A", "B", 2, 1,
                stats_tier="none",
                home_shots_on_goal=None, home_total_shots=None,
                away_shots_on_goal=None, away_total_shots=None,
                home_shots_off_goal=None, away_shots_off_goal=None,
                home_fouls=None, away_fouls=None,
                home_corner_kicks=None, away_corner_kicks=None,
                home_possession_pct=None, away_possession_pct=None,
            ),
            _make_silver_row(2, "2022-02-01", "A", "C", 1, 0),
        ]
        df = _build_df(rows)
        result = compute_rolling_features(df, n_matches=10)

        second = result.iloc[1]
        assert pd.isna(second["home_team_rolling_shots"])
        assert pd.isna(second["home_team_rolling_shot_accuracy"])
        assert pd.isna(second["home_team_rolling_conversion"])

    def test_tactical_features_nan_when_all_priors_none(self):
        rows = [
            _make_silver_row(
                1, "2022-01-01", "A", "B", 2, 1,
                stats_tier="none",
                home_shots_on_goal=None, home_total_shots=None,
                away_shots_on_goal=None, away_total_shots=None,
                home_shots_off_goal=None, away_shots_off_goal=None,
                home_fouls=None, away_fouls=None,
                home_corner_kicks=None, away_corner_kicks=None,
                home_possession_pct=None, away_possession_pct=None,
            ),
            _make_silver_row(2, "2022-02-01", "A", "C", 1, 0),
        ]
        df = _build_df(rows)
        result = compute_rolling_features(df, n_matches=10)

        second = result.iloc[1]
        assert pd.isna(second["home_team_rolling_tac_fouls"])
        assert pd.isna(second["home_team_rolling_tac_possession_pct"])

    def test_goal_features_present_even_when_stats_none(self):
        rows = [
            _make_silver_row(
                1, "2022-01-01", "A", "B", 2, 1,
                stats_tier="none",
                home_shots_on_goal=None, home_total_shots=None,
                away_shots_on_goal=None, away_total_shots=None,
                home_shots_off_goal=None, away_shots_off_goal=None,
                home_fouls=None, away_fouls=None,
                home_corner_kicks=None, away_corner_kicks=None,
                home_possession_pct=None, away_possession_pct=None,
            ),
            _make_silver_row(2, "2022-02-01", "A", "C", 1, 0),
        ]
        df = _build_df(rows)
        result = compute_rolling_features(df, n_matches=10)

        second = result.iloc[1]
        assert second["home_team_rolling_goals_for"] == pytest.approx(2.0)
        assert second["home_team_rolling_goals_against"] == pytest.approx(1.0)


class TestTeamPerspective:
    """Goals for/against should be from the team's perspective regardless of side."""

    def test_away_team_goals_perspective(self):
        rows = [
            _make_silver_row(1, "2022-01-01", "X", "A", 0, 3),
            _make_silver_row(2, "2022-02-01", "B", "A", 1, 2),
        ]
        df = _build_df(rows)
        result = compute_rolling_features(df, n_matches=10)

        second = result.iloc[1]
        assert second["away_team_rolling_goals_for"] == pytest.approx(3.0)
        assert second["away_team_rolling_goals_against"] == pytest.approx(0.0)

    def test_home_then_away_perspective(self):
        """Team alternating home/away should accumulate both perspectives."""
        rows = [
            _make_silver_row(1, "2022-01-01", "A", "B", 2, 1),
            _make_silver_row(2, "2022-02-01", "C", "A", 0, 4),
            _make_silver_row(3, "2022-03-01", "A", "D", 0, 0),
        ]
        df = _build_df(rows)
        result = compute_rolling_features(df, n_matches=10)

        third = result.iloc[2]
        assert third["home_team_rolling_goals_for"] == pytest.approx(3.0)
        assert third["home_team_rolling_goals_against"] == pytest.approx(0.5)


class TestWindowSize:
    def test_window_limits_lookback(self):
        """Only the last n_matches should be used."""
        rows = [
            _make_silver_row(1, "2022-01-01", "A", "B", 10, 0),
            _make_silver_row(2, "2022-02-01", "A", "C", 0, 0),
            _make_silver_row(3, "2022-03-01", "A", "D", 0, 0),
            _make_silver_row(4, "2022-04-01", "A", "E", 0, 0),
        ]
        df = _build_df(rows)
        result = compute_rolling_features(df, n_matches=2)

        fourth = result.iloc[3]
        assert fourth["home_team_rolling_goals_for"] == pytest.approx(0.0)


class TestMatchIndex:
    """Per-team ordinal match index (1-indexed)."""

    def test_first_match_index_is_one(self):
        rows = [
            _make_silver_row(1, "2022-01-01", "A", "B", 1, 0),
        ]
        result = compute_rolling_features(_build_df(rows))
        assert result.iloc[0]["home_team_match_index"] == 1
        assert result.iloc[0]["away_team_match_index"] == 1

    def test_index_increments(self):
        rows = [
            _make_silver_row(1, "2022-01-01", "A", "B", 1, 0),
            _make_silver_row(2, "2022-02-01", "A", "C", 2, 0),
            _make_silver_row(3, "2022-03-01", "A", "D", 3, 0),
        ]
        result = compute_rolling_features(_build_df(rows))
        assert result.iloc[0]["home_team_match_index"] == 1
        assert result.iloc[1]["home_team_match_index"] == 2
        assert result.iloc[2]["home_team_match_index"] == 3

    def test_index_counts_across_home_and_away(self):
        rows = [
            _make_silver_row(1, "2022-01-01", "A", "B", 1, 0),
            _make_silver_row(2, "2022-02-01", "C", "A", 0, 2),
            _make_silver_row(3, "2022-03-01", "A", "D", 3, 0),
        ]
        result = compute_rolling_features(_build_df(rows))
        # match 1: A home (idx 1), match 2: A away (idx 2), match 3: A home (idx 3)
        assert result.iloc[0]["home_team_match_index"] == 1
        assert result.iloc[1]["away_team_match_index"] == 2
        assert result.iloc[2]["home_team_match_index"] == 3


class TestTacticalRollingFeatures:
    """Rolling tactical profile features are computed from prior matches."""

    def test_tactical_features_computed(self):
        rows = [
            _make_silver_row(1, "2022-01-01", "A", "B", 2, 1,
                             home_fouls=14, home_possession_pct=60.0),
            _make_silver_row(2, "2022-02-01", "A", "C", 1, 0),
        ]
        result = compute_rolling_features(_build_df(rows))
        second = result.iloc[1]
        assert second["home_team_rolling_tac_fouls"] == pytest.approx(14.0)
        assert second["home_team_rolling_tac_possession_pct"] == pytest.approx(60.0)

    def test_shot_precision_computed(self):
        rows = [
            _make_silver_row(1, "2022-01-01", "A", "B", 2, 1,
                             home_shots_on_goal=4, home_total_shots=10),
            _make_silver_row(2, "2022-02-01", "A", "C", 1, 0),
        ]
        result = compute_rolling_features(_build_df(rows))
        second = result.iloc[1]
        assert second["home_team_rolling_tac_shot_precision"] == pytest.approx(0.4)
        assert second["home_team_rolling_tac_total_shots"] == pytest.approx(10.0)

    def test_shot_precision_nan_when_zero_total_shots(self):
        rows = [
            _make_silver_row(1, "2022-01-01", "A", "B", 0, 0,
                             home_shots_on_goal=0, home_total_shots=0),
            _make_silver_row(2, "2022-02-01", "A", "C", 1, 0),
        ]
        result = compute_rolling_features(_build_df(rows))
        second = result.iloc[1]
        assert pd.isna(second["home_team_rolling_tac_shot_precision"])
