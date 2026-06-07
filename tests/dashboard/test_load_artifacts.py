"""Tests for src.dashboard.load_artifacts — champion filtering."""

from __future__ import annotations

import pandas as pd
import pytest

from src.dashboard.load_artifacts import _filter_to_champion


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
