"""Tests for ``run_prediction_all_models``: per-model feature slicing + schema."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.inference.predict import _predict_one_model, run_prediction_all_models
from src.models.config import MODEL_FEATURE_SETS
from src.models.mlflow_utils import CHAMPION_ALIAS_PER_ROUND


def _make_features() -> pd.DataFrame:
    """Feature row covering the union of CORE + FULL columns."""
    from src.models.config import FULL_FEATURE_COLUMNS

    base = {
        "fixture_id": "wc2026_pair_France_Germany",
        "home_team": "France",
        "away_team": "Germany",
        "date_utc": "2026-06-15",
    }
    for col in FULL_FEATURE_COLUMNS:
        base[col] = 1.0
    return pd.DataFrame([base])


def _in_process_shadow(name, features_df, **_kwargs):
    """Simulate shadow prediction in-process (avoids multiprocessing in tests)."""
    model = MagicMock()
    model.predict.return_value = np.array([[1.4, 1.2]])
    return _predict_one_model(
        model, features_df, MODEL_FEATURE_SETS[name], name,
    )


@patch("src.inference.predict._safe_shadow_predict", side_effect=_in_process_shadow)
@patch("src.inference.predict.load_champion")
@patch("src.inference.predict.get_champion_metadata")
@patch("src.inference.predict.setup_mlflow")
def test_run_prediction_all_models_long_format(
    mock_setup, mock_meta, mock_load_champion, mock_safe_shadow,
):
    """Returns one row per (fixture, model) with the documented schema."""
    mock_meta.return_value = MagicMock(model_name="xgboost")

    champion = MagicMock()
    champion.predict.return_value = np.array([[1.7, 1.0]])
    mock_load_champion.return_value = champion

    features = _make_features()
    candidates = ["xgboost", "ridge", "poisson_glm"]

    result = run_prediction_all_models(features, candidate_names=candidates)

    expected_cols = {
        "fixture_id", "home_team", "away_team", "date_utc",
        "model_name", "lambda_h", "lambda_a", "p_home", "p_draw", "p_away",
    }
    assert expected_cols.issubset(result.columns)
    assert sorted(result["model_name"].unique()) == sorted(candidates)
    assert len(result) == len(candidates)

    for _, row in result.iterrows():
        s = row["p_home"] + row["p_draw"] + row["p_away"]
        assert abs(s - 1.0) < 1e-6


@patch("src.inference.predict.load_champion")
@patch("src.inference.predict.get_champion_metadata")
@patch("src.inference.predict.setup_mlflow")
def test_each_model_uses_its_own_feature_slice(
    mock_setup, mock_meta, mock_load_champion,
):
    """A CORE-only candidate must not be passed FULL columns the champion uses."""
    shadow_calls: list[pd.DataFrame] = []

    def shadow_side_effect(name, features_df, **_kwargs):
        model = MagicMock()

        def fake_predict(X):
            shadow_calls.append(X.copy() if isinstance(X, pd.DataFrame) else X)
            return np.array([[1.0, 1.0]])

        model.predict.side_effect = fake_predict
        return _predict_one_model(
            model, features_df, MODEL_FEATURE_SETS[name], name,
        )

    mock_meta.return_value = MagicMock(model_name="xgboost")

    champion = MagicMock()
    champion.predict.return_value = np.array([[1.5, 1.1]])
    mock_load_champion.return_value = champion

    features = _make_features()
    with patch(
        "src.inference.predict._safe_shadow_predict",
        side_effect=shadow_side_effect,
    ):
        run_prediction_all_models(features, candidate_names=["xgboost", "ridge"])

    from src.models.config import CORE_FEATURE_COLUMNS, FULL_FEATURE_COLUMNS

    champion_call = champion.predict.call_args.args[0]
    assert champion_call.shape[1] == len(FULL_FEATURE_COLUMNS)

    assert len(shadow_calls) == 1
    ridge_input = shadow_calls[0]
    assert ridge_input.shape[1] == len(CORE_FEATURE_COLUMNS)


@patch("src.inference.predict._safe_shadow_predict")
@patch("src.inference.predict.load_champion")
@patch("src.inference.predict.get_champion_metadata")
@patch("src.inference.predict.setup_mlflow")
def test_per_round_mode_passes_alias_and_cadence_mode(
    mock_setup, mock_meta, mock_load_champion, mock_safe_shadow,
):
    mock_meta.return_value = MagicMock(model_name="xgboost")
    champion = MagicMock()
    champion.predict.return_value = np.array([[1.7, 1.0]])
    mock_load_champion.return_value = champion
    mock_safe_shadow.return_value = None

    features = _make_features()
    run_prediction_all_models(
        features, cadence_mode="per_round", candidate_names=["xgboost"],
    )

    mock_meta.assert_called_once_with(alias=CHAMPION_ALIAS_PER_ROUND)
    mock_load_champion.assert_called_once_with(alias=CHAMPION_ALIAS_PER_ROUND)
