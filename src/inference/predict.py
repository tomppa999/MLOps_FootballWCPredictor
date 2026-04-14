"""Predict with the frozen champion model.

Loads the champion pyfunc from MLflow, extracts the correct feature columns
based on the model's identity, and returns predicted Poisson rates plus
analytical outcome probabilities.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.models.config import MODEL_FEATURE_SETS
from src.models.evaluation import compute_outcome_probs
from src.models.mlflow_utils import (
    get_champion_metadata,
    load_champion,
    setup_mlflow,
)

logger = logging.getLogger(__name__)


def run_prediction(upcoming_features_df: pd.DataFrame) -> pd.DataFrame:
    """Load the champion model and predict on upcoming fixtures.

    Returns a DataFrame with columns: fixture_id, home_team, away_team,
    date_utc, lambda_h, lambda_a, p_home, p_draw, p_away.
    """
    setup_mlflow()
    meta = get_champion_metadata()
    champion = load_champion()
    feature_cols = MODEL_FEATURE_SETS[meta.model_name]

    logger.info(
        "Predicting with champion: %s (%d features)",
        meta.model_name,
        len(feature_cols),
    )

    missing = [c for c in feature_cols if c not in upcoming_features_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for {meta.model_name}: {missing}")

    X = upcoming_features_df[feature_cols].to_numpy(dtype="float64", na_value=np.nan)
    preds = champion.predict(pd.DataFrame(X, columns=feature_cols))

    preds = np.atleast_2d(preds)
    lambda_h = preds[:, 0].clip(1e-6)
    lambda_a = preds[:, 1].clip(1e-6)

    probs = compute_outcome_probs(lambda_h, lambda_a)

    result = pd.DataFrame({
        "fixture_id": upcoming_features_df["fixture_id"].values,
        "home_team": upcoming_features_df["home_team"].values,
        "away_team": upcoming_features_df["away_team"].values,
        "date_utc": upcoming_features_df["date_utc"].values,
        "lambda_h": lambda_h,
        "lambda_a": lambda_a,
        "p_home": probs[:, 0],
        "p_draw": probs[:, 1],
        "p_away": probs[:, 2],
    })

    logger.info("Predictions generated for %d fixtures.", len(result))
    return result
