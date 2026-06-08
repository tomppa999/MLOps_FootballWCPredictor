"""Inference + simulation orchestrator.

Loads Gold, parses upcoming fixtures, builds features, predicts with the
frozen champion, runs Monte Carlo tournament simulation for all 4 roster
models (Option A), and logs all artifacts to MLflow.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference.features import (
    build_inference_features,
    derive_snapshot_metadata,
    generate_all_wc_pairings,
    generate_wc_group_fixtures,
    parse_wc_results,
    wc_results_to_gold_rows,
)
from src.inference.logging import log_inference_artifacts
from src.inference.predict import run_prediction, run_prediction_all_models
from src.inference.simulation import (
    sample_scorelines,
    scoreline_distribution,
    simulate_tournament,
)
from src.models.config import EXPERIMENT_MODELS
from src.models.data_split import load_gold
from src.models.mlflow_utils import _alias_for_mode, get_champion_metadata

logger = logging.getLogger(__name__)


def _seed_from_timestamp(ts_iso: str) -> int:
    """Derive a deterministic 32-bit seed from an ISO timestamp string.

    Uses SHA-256 rather than built-in ``hash()`` because Python salts string
    hashing per process (``PYTHONHASHSEED``), making ``hash()`` non-reproducible
    across runs.  The result is stable for the same ``ts_iso`` regardless of
    environment, so a cycle can always be replayed given only its logged
    ``inference_timestamp``.
    """
    digest = hashlib.sha256(ts_iso.encode()).hexdigest()
    return int(digest, 16) % (2**32)


def _simulate_roster(
    all_models_predictions_df: pd.DataFrame,
    champion_predictions_df: pd.DataFrame,
    champion_model_name: str,
    n_sims: int,
    locked_group: dict | None,
    locked_ko: dict | None,
    seed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Simulate the tournament for each EXPERIMENT_MODELS roster entry.

    The champion always simulates from ``champion_predictions_df`` (the
    dedicated champion-only prediction path) for reliability.  Non-champion
    roster models simulate from rows filtered out of
    ``all_models_predictions_df``; a missing or failed model is skipped and
    logged as a warning rather than aborting the full simulation.

    All models in a cycle share the same ``seed`` so the simulations are
    individually reproducible and start from a coupled RNG state.  Note: true
    common-random-numbers variance reduction is not achieved because
    ``rng.poisson(λ)`` consumes a λ-dependent number of underlying uniforms
    (Knuth's algorithm), causing streams to desynchronise after the first
    match.  Cross-model comparability relies on ``n_sims`` being large enough
    rather than on seed coupling.

    Returns a dict mapping model_name → simulate_tournament result dict.
    """
    per_model: dict[str, dict[str, Any]] = {}

    for model_name in EXPERIMENT_MODELS:
        if model_name == champion_model_name:
            # Champion uses its own dedicated predictions for reliability.
            preds = champion_predictions_df
        else:
            mask = all_models_predictions_df["model_name"] == model_name
            preds = (
                all_models_predictions_df.loc[mask]
                .drop(columns=["model_name"])
                .reset_index(drop=True)
            )
            if preds.empty:
                logger.warning(
                    "Roster model %s not found in all-models predictions — "
                    "skipping simulation.",
                    model_name,
                )
                continue

        try:
            results = simulate_tournament(
                preds,
                n_sims=n_sims,
                locked_group_results=locked_group,
                locked_ko_results=locked_ko,
                seed=seed,
            )
            per_model[model_name] = results
            logger.info("Simulation complete for %s.", model_name)
        except Exception:
            logger.exception("Simulation failed for %s — skipping.", model_name)

    return per_model


def run_inference_and_simulation(
    n_sims: int = 10_000,
    gold_path: Path | None = None,
    *,
    cadence_mode: str = "frozen",
    matchday_label: str | None = None,
    matches_completed_in_matchday: int | None = None,
    total_matches_completed: int | None = None,
) -> str:
    """End-to-end inference: features → predict → simulate → log.

    Returns the MLflow run_id of the inference run.
    """
    logger.info("=== Inference and simulation ===")

    # Capture cycle timestamp and derive a deterministic per-cycle seed.
    # The seed is shared across all 4 roster model simulations so each cycle
    # is individually reproducible from its logged inference_timestamp.
    # A different seed per cycle keeps MC noise independent along the RQ2
    # entropy trajectory (see docs/notes/decisions.md).
    cycle_ts = datetime.now(timezone.utc).isoformat()
    simulation_seed = _seed_from_timestamp(cycle_ts)
    logger.info("Cycle timestamp: %s  simulation_seed: %d", cycle_ts, simulation_seed)

    # Load Gold history
    if gold_path is not None:
        gold_df = load_gold(gold_path)
    else:
        gold_df = load_gold()
    logger.info("Gold loaded: %d rows", len(gold_df))

    # Parse already-played WC results from Bronze and lock them into simulation
    wc_results = parse_wc_results()
    snapshot_meta = derive_snapshot_metadata(wc_results)
    if matchday_label is None:
        matchday_label = str(snapshot_meta["matchday_label"])
    if matches_completed_in_matchday is None:
        matches_completed_in_matchday = int(snapshot_meta["matches_completed_in_matchday"])
    if total_matches_completed is None:
        total_matches_completed = int(snapshot_meta["total_matches_completed"])
    locked_group = wc_results["group_results"] or None
    locked_ko = wc_results["ko_results"] or None
    logger.info(
        "Locked results — group matches: %d, KO matches: %d, next stage: %s",
        len(wc_results["group_results"]),
        len(wc_results["ko_results"]),
        wc_results["next_matchday"],
    )

    # Augment Gold with finished WC results so rolling features for later
    # tournament matches incorporate earlier WC scores (Phase 4).
    wc_gold_rows = wc_results_to_gold_rows(wc_results)
    if not wc_gold_rows.empty:
        augmented_gold = pd.concat([gold_df, wc_gold_rows], ignore_index=True)
        augmented_gold = augmented_gold.sort_values("date_utc").reset_index(drop=True)
        latest_wc_date = wc_gold_rows["date_utc"].max()
        reference_date = latest_wc_date + pd.Timedelta(days=1)
        logger.info(
            "Augmented Gold with %d WC result rows (%d total), reference_date=%s",
            len(wc_gold_rows),
            len(augmented_gold),
            reference_date.date(),
        )
    else:
        augmented_gold = gold_df
        reference_date = None

    # Predict all C(48,2) = 1128 WC pairings so the simulation has
    # model-derived rates for every possible KO matchup.
    all_pairings = generate_all_wc_pairings(reference_date=reference_date)
    if all_pairings.empty:
        logger.warning("No WC pairings generated — skipping inference.")
        return ""

    all_features = build_inference_features(all_pairings, augmented_gold)

    # Champion predictions (dedicated path; feeds scoreline sampling and
    # predictions.csv artifact).
    all_predictions_df = run_prediction(all_features, cadence_mode=cadence_mode)
    logger.info("All-pairs predictions: %d rows", len(all_predictions_df))

    # Resolve champion model name for simulation routing and artifact params.
    try:
        champion_model_name = get_champion_metadata(
            alias=_alias_for_mode(cadence_mode),
        ).model_name
    except Exception:
        logger.warning(
            "Could not resolve champion model name — defaulting to 'xgboost'.",
        )
        champion_model_name = "xgboost"

    # Long-format predictions across champion + all 9 shadow candidates.
    # Best-effort: a failure must not block the simulation pipeline.
    all_models_predictions_df: pd.DataFrame | None = None
    try:
        all_models_predictions_df = run_prediction_all_models(
            all_features, cadence_mode=cadence_mode,
        )
    except Exception:
        logger.exception(
            "All-model prediction failed — non-champion roster simulations skipped.",
        )

    # Scoreline sampling: champion only (group fixtures, unplayed only).
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
            rng=np.random.default_rng(simulation_seed),
        )
        sl_dist = scoreline_distribution(samples)
        teams = group_predictions_df[["home_team", "away_team"]].reset_index(drop=True)
        sl_dist["home_team"] = sl_dist["match_idx"].map(teams["home_team"])
        sl_dist["away_team"] = sl_dist["match_idx"].map(teams["away_team"])
    else:
        sl_dist = pd.DataFrame(
            columns=["match_idx", "home_goals", "away_goals", "probability", "home_team", "away_team"]
        )

    # Option A: simulate all 4 EXPERIMENT_MODELS roster entries.
    # Champion uses dedicated all_predictions_df for reliability; the other
    # 3 roster models use filtered rows from all_models_predictions_df.
    # Non-champion simulations are skipped gracefully if predictions are missing.
    if all_models_predictions_df is not None and not all_models_predictions_df.empty:
        per_model_tournament_results = _simulate_roster(
            all_models_predictions_df=all_models_predictions_df,
            champion_predictions_df=all_predictions_df,
            champion_model_name=champion_model_name,
            n_sims=n_sims,
            locked_group=locked_group,
            locked_ko=locked_ko,
            seed=simulation_seed,
        )
    else:
        # Fallback: only the champion simulation is available.
        logger.info(
            "All-models predictions unavailable — running champion simulation only.",
        )
        per_model_tournament_results = {}
        try:
            results = simulate_tournament(
                all_predictions_df,
                n_sims=n_sims,
                locked_group_results=locked_group,
                locked_ko_results=locked_ko,
                seed=simulation_seed,
            )
            per_model_tournament_results[champion_model_name] = results
        except Exception:
            logger.exception("Champion simulation failed.")

    logger.info(
        "Per-model simulation complete: %d models simulated (%s).",
        len(per_model_tournament_results),
        ", ".join(sorted(per_model_tournament_results.keys())),
    )

    # Log to MLflow
    run_id = log_inference_artifacts(
        predictions_df=all_predictions_df,
        scoreline_dist=sl_dist,
        per_model_tournament_results=per_model_tournament_results,
        n_sims=n_sims,
        gold_row_count=len(gold_df),
        champion_model_name=champion_model_name,
        all_models_predictions_df=all_models_predictions_df,
        inference_timestamp=cycle_ts,
        simulation_seed=simulation_seed,
        cadence_mode=cadence_mode,
        matchday_label=matchday_label,
        matches_completed_in_matchday=matches_completed_in_matchday,
        total_matches_completed=total_matches_completed,
    )

    logger.info("=== Inference complete (run_id=%s) ===", run_id)
    return run_id
