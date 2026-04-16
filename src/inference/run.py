"""Inference + simulation orchestrator.

Loads Gold, parses upcoming fixtures, builds features, predicts with the
frozen champion, runs Monte Carlo tournament simulation, and logs all
artifacts to MLflow.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.inference.features import (
    build_inference_features,
    generate_all_wc_pairings,
    generate_wc_group_fixtures,
    parse_wc_results,
)
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

    # Predict all C(48,2) = 1128 WC pairings so the simulation has
    # model-derived rates for every possible KO matchup.
    all_pairings = generate_all_wc_pairings()
    if all_pairings.empty:
        logger.warning("No WC pairings generated — skipping inference.")
        return ""

    all_features = build_inference_features(all_pairings, gold_df)
    all_predictions_df = run_prediction(all_features)
    logger.info("All-pairs predictions: %d rows", len(all_predictions_df))

    # Identify unplayed group fixtures for scoreline sampling artifact.
    all_group_fixtures = generate_wc_group_fixtures()
    locked_group_keys = set(wc_results["group_results"].keys())
    upcoming_group_teams = set()
    for _, r in all_group_fixtures.iterrows():
        if (r["home_team"], r["away_team"]) not in locked_group_keys:
            upcoming_group_teams.add((r["home_team"], r["away_team"]))

    group_mask = all_predictions_df.apply(
        lambda r: (r["home_team"], r["away_team"]) in upcoming_group_teams,
        axis=1,
    )
    group_predictions_df = all_predictions_df.loc[group_mask]

    logger.info(
        "Group fixtures: %d total, %d locked, %d to sample scorelines",
        len(all_group_fixtures),
        len(locked_group_keys),
        len(group_predictions_df),
    )

    if not group_predictions_df.empty:
        samples = sample_scorelines(
            group_predictions_df["lambda_h"].values,
            group_predictions_df["lambda_a"].values,
            n_sims=n_sims,
        )
        sl_dist = scoreline_distribution(samples)
    else:
        sl_dist = pd.DataFrame(
            columns=["match_idx", "home_goals", "away_goals", "probability"]
        )

    # Tournament simulation with full rate coverage
    tournament_results = simulate_tournament(
        all_predictions_df,
        n_sims=n_sims,
        locked_group_results=locked_group,
        locked_ko_results=locked_ko,
    )

    # Log to MLflow
    run_id = log_inference_artifacts(
        predictions_df=all_predictions_df,
        scoreline_dist=sl_dist,
        tournament_results=tournament_results,
        n_sims=n_sims,
        gold_row_count=len(gold_df),
    )

    logger.info("=== Inference complete (run_id=%s) ===", run_id)
    return run_id
