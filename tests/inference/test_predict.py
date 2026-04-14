"""Tests for src.inference.predict."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.inference.predict import run_prediction


def _make_upcoming_features() -> pd.DataFrame:
    """Minimal feature DataFrame matching CORE_FEATURE_COLUMNS."""
    return pd.DataFrame([{
        "fixture_id": 1,
        "home_team": "France",
        "away_team": "Germany",
        "date_utc": "2026-07-01",
        "elo_diff": 20.0,
        "competition_tier": 1,
        "is_knockout": False,
        "is_neutral": True,
        "home_team_rolling_goals_for": 1.8,
        "home_team_rolling_goals_against": 0.9,
        "away_team_rolling_goals_for": 1.5,
        "away_team_rolling_goals_against": 1.1,
    }])


class TestRunPrediction:
    @patch("src.inference.predict.load_champion")
    @patch("src.inference.predict.get_champion_metadata")
    @patch("src.inference.predict.setup_mlflow")
    def test_returns_correct_schema(self, mock_setup, mock_meta, mock_load):
        mock_meta.return_value = MagicMock(model_name="poisson_glm")

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[1.5, 1.2]])
        mock_load.return_value = mock_model

        features_df = _make_upcoming_features()
        result = run_prediction(features_df)

        assert "lambda_h" in result.columns
        assert "lambda_a" in result.columns
        assert "p_home" in result.columns
        assert "p_draw" in result.columns
        assert "p_away" in result.columns
        assert len(result) == 1

    @patch("src.inference.predict.load_champion")
    @patch("src.inference.predict.get_champion_metadata")
    @patch("src.inference.predict.setup_mlflow")
    def test_probabilities_sum_to_one(self, mock_setup, mock_meta, mock_load):
        mock_meta.return_value = MagicMock(model_name="poisson_glm")

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[1.5, 1.2]])
        mock_load.return_value = mock_model

        features_df = _make_upcoming_features()
        result = run_prediction(features_df)

        prob_sum = result.iloc[0]["p_home"] + result.iloc[0]["p_draw"] + result.iloc[0]["p_away"]
        assert abs(prob_sum - 1.0) < 1e-6

    @patch("src.inference.predict.load_champion")
    @patch("src.inference.predict.get_champion_metadata")
    @patch("src.inference.predict.setup_mlflow")
    def test_raises_on_missing_features(self, mock_setup, mock_meta, mock_load):
        mock_meta.return_value = MagicMock(model_name="poisson_glm")
        mock_load.return_value = MagicMock()

        # Missing elo_diff
        features_df = pd.DataFrame([{
            "fixture_id": 1,
            "home_team": "France",
            "away_team": "Germany",
            "date_utc": "2026-07-01",
        }])

        with pytest.raises(ValueError, match="Missing feature columns"):
            run_prediction(features_df)
