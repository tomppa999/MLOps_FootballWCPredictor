"""Tests for helper functions in src.dashboard.app."""

import math

import pandas as pd
import pytest

from src.dashboard.app import (
    _apply_completed_result,
    _completed_results_lookup,
    resolve_ko_fixtures,
)


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


def _monitoring_df() -> pd.DataFrame:
    """Minimal wc2026_monitoring.csv fixture (two settled matches, two models)."""
    return pd.DataFrame([
        {
            "match_id": 1,
            "kickoff_utc": "2026-06-12T15:00:00+00:00",
            "home": "Mexico",
            "away": "Saudi Arabia",
            "actual_h": 2,
            "actual_a": 0,
            "actual_outcome": 0,
            "model_name": "xgboost",
            "lambda_h": 1.5,
            "lambda_a": 0.9,
            "p_home": 0.71,
            "p_draw": 0.17,
            "p_away": 0.12,
            "rps": 0.024,
            "nll": 2.1,
            "rmse_h": 0.5,
            "rmse_a": 0.9,
            "cadence_mode": "frozen",
            "inference_run_id": "abc123",
        },
        {
            "match_id": 1,
            "kickoff_utc": "2026-06-12T15:00:00+00:00",
            "home": "Mexico",
            "away": "Saudi Arabia",
            "actual_h": 2,
            "actual_a": 0,
            "actual_outcome": 0,
            "model_name": "poisson_glm",
            "lambda_h": 1.4,
            "lambda_a": 1.0,
            "p_home": 0.65,
            "p_draw": 0.20,
            "p_away": 0.15,
            "rps": 0.031,
            "nll": 2.3,
            "rmse_h": 0.6,
            "rmse_a": 1.0,
            "cadence_mode": "frozen",
            "inference_run_id": "abc123",
        },
        {
            "match_id": 2,
            "kickoff_utc": "2026-06-12T18:00:00+00:00",
            "home": "Korea Republic",
            "away": "Czech Republic",
            "actual_h": 2,
            "actual_a": 1,
            "actual_outcome": 0,
            "model_name": "xgboost",
            "lambda_h": 1.2,
            "lambda_a": 1.3,
            "p_home": 0.40,
            "p_draw": 0.28,
            "p_away": 0.32,
            "rps": 0.256,
            "nll": 2.5,
            "rmse_h": 0.8,
            "rmse_a": 0.3,
            "cadence_mode": "frozen",
            "inference_run_id": "abc123",
        },
    ])


class TestCompletedResultsLookup:
    def test_returns_empty_for_none(self):
        assert _completed_results_lookup(None, "xgboost") == {}

    def test_returns_empty_for_empty_df(self):
        assert _completed_results_lookup(pd.DataFrame(), "xgboost") == {}

    def test_filters_to_champion(self):
        lookup = _completed_results_lookup(_monitoring_df(), "xgboost")
        assert len(lookup) == 2

    def test_non_champion_excluded(self):
        lookup = _completed_results_lookup(_monitoring_df(), "poisson_glm")
        # poisson_glm only has match_id=1
        assert len(lookup) == 1

    def test_key_is_frozenset(self):
        lookup = _completed_results_lookup(_monitoring_df(), "xgboost")
        key = frozenset({"Mexico", "Saudi Arabia"})
        assert key in lookup

    def test_probs_scaled_to_percent(self):
        lookup = _completed_results_lookup(_monitoring_df(), "xgboost")
        rec = lookup[frozenset({"Mexico", "Saudi Arabia"})]
        assert rec["p_home_pct"] == pytest.approx(71.0)
        assert rec["p_draw_pct"] == pytest.approx(17.0)
        assert rec["p_away_pct"] == pytest.approx(12.0)

    def test_actual_goals_stored(self):
        lookup = _completed_results_lookup(_monitoring_df(), "xgboost")
        rec = lookup[frozenset({"Mexico", "Saudi Arabia"})]
        assert rec["actual_home_goals"] == 2
        assert rec["actual_away_goals"] == 0

    def test_none_champion_uses_all_rows(self):
        lookup = _completed_results_lookup(_monitoring_df(), None)
        # Without champion filter, last write wins per frozenset key.
        # Both matches should appear (Mexico/SA + Korea/Czech).
        assert frozenset({"Mexico", "Saudi Arabia"}) in lookup
        assert frozenset({"Korea Republic", "Czech Republic"}) in lookup

    def test_no_cadence_mode_column(self):
        df = _monitoring_df().drop(columns=["cadence_mode"])
        lookup = _completed_results_lookup(df, "xgboost")
        # Without cadence_mode column, no cadence filter is applied.
        assert len(lookup) == 2


class TestApplyCompletedResult:
    def _base_row(self, home: str = "Mexico", away: str = "Saudi Arabia") -> dict:
        return {
            "home_team": home,
            "away_team": away,
            "p_home": 72.0,
            "p_draw": 16.0,
            "p_away": 12.0,
        }

    def test_upcoming_match_flagged_not_played(self):
        row = self._base_row("France", "Argentina")
        lookup = _completed_results_lookup(_monitoring_df(), "xgboost")
        result = _apply_completed_result(row, lookup)
        assert result["played"] is False
        assert result["actual_h"] is None
        assert result["actual_a"] is None

    def test_upcoming_match_probs_unchanged(self):
        row = self._base_row("France", "Argentina")
        lookup = _completed_results_lookup(_monitoring_df(), "xgboost")
        result = _apply_completed_result(row, lookup)
        assert result["p_home"] == pytest.approx(72.0)

    def test_played_match_flagged_played(self):
        row = self._base_row("Mexico", "Saudi Arabia")
        lookup = _completed_results_lookup(_monitoring_df(), "xgboost")
        result = _apply_completed_result(row, lookup)
        assert result["played"] is True

    def test_played_match_pre_kickoff_probs_applied(self):
        row = self._base_row("Mexico", "Saudi Arabia")
        lookup = _completed_results_lookup(_monitoring_df(), "xgboost")
        result = _apply_completed_result(row, lookup)
        assert result["p_home"] == pytest.approx(71.0)
        assert result["p_draw"] == pytest.approx(17.0)
        assert result["p_away"] == pytest.approx(12.0)

    def test_played_match_actual_goals_set(self):
        row = self._base_row("Mexico", "Saudi Arabia")
        lookup = _completed_results_lookup(_monitoring_df(), "xgboost")
        result = _apply_completed_result(row, lookup)
        assert result["actual_h"] == 2
        assert result["actual_a"] == 0

    def test_orientation_flip_when_config_home_differs(self):
        # Config says Saudi Arabia is home, but monitoring recorded Mexico as home.
        row = self._base_row("Saudi Arabia", "Mexico")
        lookup = _completed_results_lookup(_monitoring_df(), "xgboost")
        result = _apply_completed_result(row, lookup)
        assert result["played"] is True
        # p_home (for Saudi Arabia) should be the monitoring p_away (for Mexico's away)
        assert result["p_home"] == pytest.approx(12.0)
        assert result["p_away"] == pytest.approx(71.0)
        # Goals also flipped
        assert result["actual_h"] == 0
        assert result["actual_a"] == 2

    def test_empty_lookup_all_upcoming(self):
        row = self._base_row("Mexico", "Saudi Arabia")
        result = _apply_completed_result(row, {})
        assert result["played"] is False


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
