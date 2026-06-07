"""Tests for src.inference.logging — multi-model artifact stacking."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.inference.logging import _stack_per_model, log_inference_artifacts


# ---------------------------------------------------------------------------
# _stack_per_model unit tests
# ---------------------------------------------------------------------------


def _adv_df(model_name: str) -> pd.DataFrame:
    return pd.DataFrame([{"team": "France", "p_winner": 0.15}])


def test_stack_per_model_adds_model_name_column():
    per_model = {
        "xgboost": {"advancement": _adv_df("xgboost")},
        "poisson_glm": {"advancement": _adv_df("poisson_glm")},
    }
    result = _stack_per_model(per_model, "advancement")
    assert result is not None
    assert "model_name" in result.columns
    assert set(result["model_name"].unique()) == {"xgboost", "poisson_glm"}
    assert len(result) == 2


def test_stack_per_model_skips_missing_key():
    per_model = {
        "xgboost": {"advancement": _adv_df("xgboost")},
        "poisson_glm": {},  # no "advancement" key
    }
    result = _stack_per_model(per_model, "advancement")
    assert result is not None
    assert set(result["model_name"].unique()) == {"xgboost"}


def test_stack_per_model_returns_none_when_all_empty():
    per_model = {
        "xgboost": {"advancement": pd.DataFrame()},
    }
    result = _stack_per_model(per_model, "advancement")
    assert result is None


def test_stack_per_model_returns_none_for_empty_dict():
    result = _stack_per_model({}, "advancement")
    assert result is None


# ---------------------------------------------------------------------------
# log_inference_artifacts integration tests (MLflow mocked out)
# ---------------------------------------------------------------------------


def _make_predictions() -> pd.DataFrame:
    return pd.DataFrame([{
        "fixture_id": "wc2026_France_Germany",
        "home_team": "France",
        "away_team": "Germany",
        "date_utc": "2026-06-15",
        "lambda_h": 1.5,
        "lambda_a": 1.2,
        "p_home": 0.45,
        "p_draw": 0.25,
        "p_away": 0.30,
    }])


def _make_per_model_results() -> dict:
    models = ["xgboost", "poisson_glm"]
    out = {}
    for m in models:
        out[m] = {
            "advancement": pd.DataFrame([{"team": "France", "p_winner": 0.15}]),
            "group_positions": pd.DataFrame([{"team": "France", "p_1st": 0.6}]),
            "ko_pairings": pd.DataFrame([{"team_a": "France", "team_b": "Germany"}]),
            "n_sims": 100,
        }
    return out


@patch("src.inference.logging.mlflow")
@patch("src.inference.logging.start_run")
@patch("src.inference.logging.log_run")
@patch("src.inference.logging.get_latest_production_run_id", return_value="prod_run_123")
@patch("src.inference.logging.setup_mlflow")
def test_log_inference_artifacts_returns_run_id(
    mock_setup, mock_prod_id, mock_log_run, mock_start_run, mock_mlflow
):
    fake_run = MagicMock()
    fake_run.info.run_id = "test_run_id"
    mock_start_run.return_value.__enter__ = MagicMock(return_value=fake_run)
    mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

    run_id = log_inference_artifacts(
        predictions_df=_make_predictions(),
        scoreline_dist=None,
        per_model_tournament_results=_make_per_model_results(),
        n_sims=100,
        gold_row_count=6000,
        champion_model_name="xgboost",
    )

    assert run_id == "test_run_id"


@patch("src.inference.logging.mlflow")
@patch("src.inference.logging.start_run")
@patch("src.inference.logging.log_run")
@patch("src.inference.logging.get_latest_production_run_id", return_value="prod_run_123")
@patch("src.inference.logging.setup_mlflow")
def test_log_run_includes_simulated_models_param(
    mock_setup, mock_prod_id, mock_log_run, mock_start_run, mock_mlflow
):
    fake_run = MagicMock()
    fake_run.info.run_id = "test_run_id"
    mock_start_run.return_value.__enter__ = MagicMock(return_value=fake_run)
    mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

    log_inference_artifacts(
        predictions_df=_make_predictions(),
        scoreline_dist=None,
        per_model_tournament_results=_make_per_model_results(),
        n_sims=100,
        gold_row_count=6000,
        champion_model_name="xgboost",
    )

    # Verify that log_run was called with the required params
    call_kwargs = mock_log_run.call_args.kwargs
    params = call_kwargs.get("params", {})
    assert "simulated_models" in params
    assert "champion_model_name" in params
    assert params["champion_model_name"] == "xgboost"
