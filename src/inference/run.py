"""Inference + simulation orchestrator.

Loads Gold, parses upcoming fixtures, builds features, predicts with the
frozen champion, runs Monte Carlo tournament simulation, and logs all
artifacts to MLflow.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.inference.features import build_inference_features, parse_upcoming_fixtures, parse_wc_results
from src.inference.logging import log_inference_artifacts
from src.inference.predict import run_prediction
from src.inference.simulation import (
    sample_scorelines,
    scoreline_distribution,
    simulate_tournament,
)
from src.models.data_split import load_gold

logger = logging.getLogger(__name__)


def run_inference_and_simulation(
    n_sims: int = 10_000,
    gold_path: Path | None = None,
) -> str:
    """End-to-end inference: features → predict → simulate → log.

    Returns the MLflow run_id of the inference run.
    """
    logger.info("=== Inference and simulation ===")

    # Load Gold history
    if gold_path is not None:
        gold_df = load_gold(gold_path)
    else:
        gold_df = load_gold()
    logger.info("Gold loaded: %d rows", len(gold_df))

    # Parse upcoming fixtures from Bronze
    upcoming_df = parse_upcoming_fixtures()
    if upcoming_df.empty:
        logger.warning("No upcoming fixtures found — skipping inference.")
        return ""

    # Build inference features
    features_df = build_inference_features(upcoming_df, gold_df)

    # Predict with champion
    predictions_df = run_prediction(features_df)

    # Per-match scoreline sampling
    samples = sample_scorelines(
        predictions_df["lambda_h"].values,
        predictions_df["lambda_a"].values,
        n_sims=n_sims,
    )
    sl_dist = scoreline_distribution(samples)

    # Parse already-played WC results from Bronze and lock them into simulation
    wc_results = parse_wc_results()
    locked_group = wc_results["group_results"] or None
    locked_ko = wc_results["ko_results"] or None
    logger.info(
        "Locked results — group matches: %d, KO matches: %d, next stage: %s",
        len(wc_results["group_results"]),
        len(wc_results["ko_results"]),
        wc_results["next_matchday"],
    )

    # Tournament simulation
    tournament_results = simulate_tournament(
        predictions_df,
        n_sims=n_sims,
        locked_group_results=locked_group,
        locked_ko_results=locked_ko,
    )

    # Log to MLflow
    run_id = log_inference_artifacts(
        predictions_df=predictions_df,
        scoreline_dist=sl_dist,
        tournament_results=tournament_results,
        n_sims=n_sims,
        gold_row_count=len(gold_df),
    )

    logger.info("=== Inference complete (run_id=%s) ===", run_id)
    return run_id
