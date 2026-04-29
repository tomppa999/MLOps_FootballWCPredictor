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
    load_shadow_model,
    setup_mlflow,
)

logger = logging.getLogger(__name__)


def _predict_one_model(
    model: object,
    upcoming_features_df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Run a single model on the slice of features it expects.

    Returns a DataFrame with the standard prediction columns plus
    ``model_name``.
    """
    missing = [c for c in feature_cols if c not in upcoming_features_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for {model_name}: {missing}")

    X = upcoming_features_df[feature_cols].to_numpy(dtype="float64", na_value=np.nan)
    preds = model.predict(pd.DataFrame(X, columns=feature_cols))
    preds = np.atleast_2d(preds)
    lambda_h = preds[:, 0].clip(1e-6)
    lambda_a = preds[:, 1].clip(1e-6)
    probs = compute_outcome_probs(lambda_h, lambda_a)

    return pd.DataFrame({
        "fixture_id": upcoming_features_df["fixture_id"].values,
        "home_team": upcoming_features_df["home_team"].values,
        "away_team": upcoming_features_df["away_team"].values,
        "date_utc": upcoming_features_df["date_utc"].values,
        "model_name": model_name,
        "lambda_h": lambda_h,
        "lambda_a": lambda_a,
        "p_home": probs[:, 0],
        "p_draw": probs[:, 1],
        "p_away": probs[:, 2],
    })


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

    result = _predict_one_model(
        champion, upcoming_features_df, feature_cols, meta.model_name,
    )
    result = result.drop(columns=["model_name"])
    logger.info("Predictions generated for %d fixtures.", len(result))
    return result


def run_prediction_all_models(
    upcoming_features_df: pd.DataFrame,
    *,
    candidate_names: list[str] | None = None,
) -> pd.DataFrame:
    """Predict every fixture with the champion + every shadow model.

    Each model uses its own ``MODEL_FEATURE_SETS[model_name]`` slice — never
    the champion's feature set for the others — so a CORE-only candidate
    (e.g. ridge) does not see the FULL feature columns the champion uses.

    Returns a long-format DataFrame with one row per (fixture, model):
        fixture_id, home_team, away_team, date_utc,
        model_name, lambda_h, lambda_a, p_home, p_draw, p_away.

    The simulation path (:func:`run_prediction`) stays champion-only. This
    function exists only to feed the monitoring layer.
    """
    setup_mlflow()
    champion_meta = get_champion_metadata()

    if candidate_names is None:
        candidate_names = list(MODEL_FEATURE_SETS.keys())

    logger.info(
        "Predicting all candidates (%d models, %d fixtures): champion=%s",
        len(candidate_names),
        len(upcoming_features_df),
        champion_meta.model_name,
    )

    blocks: list[pd.DataFrame] = []

    champion = load_champion()
    blocks.append(
        _predict_one_model(
            champion,
            upcoming_features_df,
            MODEL_FEATURE_SETS[champion_meta.model_name],
            champion_meta.model_name,
        )
    )

    for name in candidate_names:
        if name == champion_meta.model_name:
            continue
        try:
            model = load_shadow_model(name)
        except ValueError:
            logger.warning(
                "No registered shadow/staging version for %s — skipping.",
                name,
            )
            continue
        try:
            block = _predict_one_model(
                model, upcoming_features_df, MODEL_FEATURE_SETS[name], name,
            )
        except Exception as exc:  # noqa: BLE001 - one bad shadow shouldn't kill the others
            logger.warning("Shadow prediction failed for %s: %s", name, exc)
            continue
        blocks.append(block)

    out = pd.concat(blocks, ignore_index=True)
    logger.info(
        "All-model predictions: %d rows across %d models",
        len(out),
        out["model_name"].nunique(),
    )
    return out
