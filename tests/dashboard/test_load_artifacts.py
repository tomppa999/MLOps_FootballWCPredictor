"""Tests for src.dashboard.load_artifacts — champion filtering."""

from __future__ import annotations

import pandas as pd
import pytest

from unittest.mock import MagicMock, patch

from src.dashboard.load_artifacts import _filter_to_champion, _get_latest_inference_run


def _multi_model_df() -> pd.DataFrame:
    """Simulated tournament_probabilities.csv with model_name column."""
    return pd.DataFrame([
        {"model_name": "xgboost", "team": "France", "p_winner": 0.18},
        {"model_name": "xgboost", "team": "Germany", "p_winner": 0.12},
        {"model_name": "poisson_glm", "team": "France", "p_winner": 0.17},
        {"model_name": "poisson_glm", "team": "Germany", "p_winner": 0.11},
        {"model_name": "bayesian_poisson", "team": "France", "p_winner": 0.16},
        {"model_name": "mean_rate_poisson", "team": "France", "p_winner": 0.08},
    ])


def test_filter_returns_only_champion_rows():
    df = _multi_model_df()
    result = _filter_to_champion(df, "xgboost")
    assert set(result["team"].tolist()) == {"France", "Germany"}
    assert len(result) == 2


def test_filter_drops_model_name_column():
    df = _multi_model_df()
    result = _filter_to_champion(df, "xgboost")
    assert "model_name" not in result.columns


def test_filter_backward_compat_no_model_name_column():
    """Pre-Option-A DataFrames (no model_name column) pass through unchanged."""
    df = pd.DataFrame([
        {"team": "France", "p_winner": 0.18},
        {"team": "Germany", "p_winner": 0.12},
    ])
    result = _filter_to_champion(df, "xgboost")
    assert "model_name" not in result.columns
    assert len(result) == 2
    pd.testing.assert_frame_equal(result, df)


def test_filter_returns_empty_for_unknown_champion():
    df = _multi_model_df()
    result = _filter_to_champion(df, "nonexistent_model")
    assert len(result) == 0
    assert "model_name" not in result.columns


@patch("src.dashboard.load_artifacts.setup_mlflow")
@patch("src.dashboard.load_artifacts.mlflow.tracking.MlflowClient")
def test_get_latest_inference_run_prefers_frozen_lineage(mock_client_cls, mock_setup):
    frozen_run = MagicMock()
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-1")
    mock_client.search_runs.side_effect = [
        [frozen_run],
    ]
    mock_client_cls.return_value = mock_client

    result = _get_latest_inference_run()
    assert result is frozen_run
    first_call = mock_client.search_runs.call_args_list[0]
    assert 'params.cadence_mode = "frozen"' in first_call.kwargs["filter_string"]


@patch("src.dashboard.load_artifacts.setup_mlflow")
@patch("src.dashboard.load_artifacts.mlflow.tracking.MlflowClient")
def test_get_latest_inference_run_falls_back_when_no_frozen_runs(
    mock_client_cls, mock_setup,
):
    fallback_run = MagicMock()
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-1")
    mock_client.search_runs.side_effect = [
        [],
        [fallback_run],
    ]
    mock_client_cls.return_value = mock_client

    result = _get_latest_inference_run()
    assert result is fallback_run
    assert mock_client.search_runs.call_count == 2
