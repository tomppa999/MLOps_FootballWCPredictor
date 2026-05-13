"""Tests for src.inference.run (orchestrator)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.inference.run import run_inference_and_simulation


def _make_gold_df() -> pd.DataFrame:
    """Minimal Gold DataFrame for testing."""
    rows = []
    for i in range(5):
        rows.append({
            "fixture_id": 1000 + i,
            "date_utc": f"2025-06-{10 + i}",
            "season": 2025,
            "league_id": 10,
            "league_name": "Friendly",
            "home_team": "France",
            "away_team": "Germany",
            "home_goals": 2,
            "away_goals": 1,
            "home_elo_pre": 2000.0,
            "away_elo_pre": 1980.0,
            "elo_diff": 20.0,
            "competition_tier": 4,
            "is_knockout": False,
            "is_neutral": False,
            "stats_tier": "none",
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


def _make_pairings() -> pd.DataFrame:
    return pd.DataFrame([{
        "fixture_id": "wc2026_pair_France_Germany",
        "date_utc": "2026-06-15",
        "home_team": "France",
        "away_team": "Germany",
        "competition_tier": 1,
        "is_knockout": False,
        "is_neutral": True,
    }])


def _make_predictions() -> pd.DataFrame:
    return pd.DataFrame([{
        "fixture_id": "wc2026_pair_France_Germany",
        "home_team": "France",
        "away_team": "Germany",
        "date_utc": "2026-06-15",
        "lambda_h": 1.5,
        "lambda_a": 1.2,
        "p_home": 0.45,
        "p_draw": 0.25,
        "p_away": 0.30,
    }])


class TestRunInferenceAndSimulation:
    @patch("src.inference.run.log_inference_artifacts", return_value="run_123")
    @patch("src.inference.run.simulate_tournament")
    @patch("src.inference.run.run_prediction_all_models")
    @patch("src.inference.run.run_prediction")
    @patch("src.inference.run.build_inference_features")
    @patch("src.inference.run.generate_wc_group_fixtures")
    @patch("src.inference.run.generate_all_wc_pairings")
    @patch("src.inference.run.parse_wc_results")
    @patch("src.inference.run.load_gold")
    def test_happy_path(
        self,
        mock_load_gold,
        mock_parse_results,
        mock_all_pairings,
        mock_group_fixtures,
        mock_features,
        mock_predict,
        mock_predict_all,
        mock_simulate,
        mock_log,
    ):
        mock_load_gold.return_value = _make_gold_df()
        mock_parse_results.return_value = {
            "group_results": {},
            "ko_results": {},
            "next_matchday": 1,
        }

        mock_all_pairings.return_value = _make_pairings()
        mock_group_fixtures.return_value = _make_pairings()

        features = _make_pairings().copy()
        features["elo_diff"] = 20.0
        mock_features.return_value = features

        mock_predict.return_value = _make_predictions()
        mock_predict_all.return_value = _make_predictions()

        mock_simulate.return_value = {
            "advancement": pd.DataFrame([{"team": "France", "p_winner": 0.15}]),
            "n_sims": 100,
        }

        run_id = run_inference_and_simulation(n_sims=100)

        assert run_id == "run_123"
        mock_all_pairings.assert_called_once()
        mock_features.assert_called_once()
        mock_predict.assert_called_once()
        mock_simulate.assert_called_once()
        mock_log.assert_called_once()

    @patch("src.inference.run.generate_all_wc_pairings")
    @patch("src.inference.run.parse_wc_results")
    @patch("src.inference.run.load_gold")
    def test_returns_empty_when_no_pairings(
        self, mock_load_gold, mock_parse_results, mock_all_pairings,
    ):
        mock_load_gold.return_value = _make_gold_df()
        mock_parse_results.return_value = {
            "group_results": {},
            "ko_results": {},
            "next_matchday": 1,
        }
        mock_all_pairings.return_value = pd.DataFrame()

        run_id = run_inference_and_simulation()
        assert run_id == ""
