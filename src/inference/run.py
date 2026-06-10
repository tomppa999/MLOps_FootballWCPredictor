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


def _build_ko_fixtures(
    locked_ko: dict[int, dict],
    ko_slot_pairings: "pd.DataFrame | None",
) -> pd.DataFrame:
    """Build a champion-only ko_fixtures DataFrame (one row per KO match slot).

    For locked slots: status='locked', actual score, pairing_frequency=1.0.
    For predicted slots: status='predicted', modal (home, away) from
    ko_slot_pairings, with their observed frequency.
    """
    if ko_slot_pairings is None or ko_slot_pairings.empty:
        # No simulation results available; populate only locked fixtures.
        rows = []
        for match_num, res in sorted(locked_ko.items()):
            rows.append({
                "match_num": match_num,
                "stage": "Unknown",
                "home_team": res["home"],
                "away_team": res["away"],
                "status": "locked",
                "home_goals": res["home_goals"],
                "away_goals": res["away_goals"],
                "decided_by": res.get("decided_by", ""),
                "pairing_frequency": 1.0,
            })
        return pd.DataFrame(rows)

    rows = []
    # Modal predicted pair per slot (already sorted by match_num, count desc)
    modal = (
        ko_slot_pairings.groupby("match_num", sort=False)
        .first()
        .reset_index()
    )
    for _, m_row in modal.iterrows():
        match_num = int(m_row["match_num"])
        if match_num in locked_ko:
            res = locked_ko[match_num]
            rows.append({
                "match_num": match_num,
                "stage": m_row["stage"],
                "home_team": res["home"],
                "away_team": res["away"],
                "status": "locked",
                "home_goals": int(res["home_goals"]),
                "away_goals": int(res["away_goals"]),
                "decided_by": res.get("decided_by", ""),
                "pairing_frequency": 1.0,
            })
        else:
            rows.append({
                "match_num": match_num,
                "stage": m_row["stage"],
                "home_team": m_row["home_team"],
                "away_team": m_row["away_team"],
                "status": "predicted",
                "home_goals": None,
                "away_goals": None,
                "decided_by": "",
                "pairing_frequency": float(m_row["frequency"]),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("match_num").reset_index(drop=True)
    return df


def _sample_ko_scorelines(
    ko_fixtures: pd.DataFrame,
    predictions_df: pd.DataFrame,
    n_sims: int,
    seed: int | None,
) -> pd.DataFrame:
    """Sample Poisson scoreline distributions for KO fixtures.

    Uses the champion model's predicted rates for every slot, whether locked
    or predicted (the actual result for locked slots is shown separately in
    the UI; the distribution still reflects model predictions).
    """
    if ko_fixtures.empty:
        return pd.DataFrame(
            columns=["match_idx", "home_goals", "away_goals", "probability",
                     "home_team", "away_team", "stage", "match_num"]
        )

    rate_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for _, row in predictions_df.iterrows():
        rate_lookup[(row["home_team"], row["away_team"])] = (
            float(row["lambda_h"]),
            float(row["lambda_a"]),
        )
        if (row["away_team"], row["home_team"]) not in rate_lookup:
            rate_lookup[(row["away_team"], row["home_team"])] = (
                float(row["lambda_a"]),
                float(row["lambda_h"]),
            )

    lambda_h_list, lambda_a_list, valid_rows = [], [], []
    for _, fix in ko_fixtures.iterrows():
        h, a = fix["home_team"], fix["away_team"]
        rates = rate_lookup.get((h, a)) or rate_lookup.get((a, h))
        if rates is None:
            logger.warning("No rate found for KO fixture %s vs %s — skipping scoreline.", h, a)
            continue
        lambda_h_list.append(rates[0] if (h, a) in rate_lookup else rates[1])
        lambda_a_list.append(rates[1] if (h, a) in rate_lookup else rates[0])
        valid_rows.append(fix)

    if not lambda_h_list:
        return pd.DataFrame(
            columns=["match_idx", "home_goals", "away_goals", "probability",
                     "home_team", "away_team", "stage", "match_num"]
        )

    samples = sample_scorelines(
        np.array(lambda_h_list),
        np.array(lambda_a_list),
        n_sims=n_sims,
        rng=np.random.default_rng(seed),
    )
    sl = scoreline_distribution(samples)
    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    sl["home_team"] = sl["match_idx"].map(valid_df["home_team"])
    sl["away_team"] = sl["match_idx"].map(valid_df["away_team"])
    sl["stage"] = sl["match_idx"].map(valid_df["stage"].reset_index(drop=True))
    sl["match_num"] = sl["match_idx"].map(
        valid_df["match_num"].reset_index(drop=True)
    )
    return sl


def run_inference_and_simulation(
    n_sims: int = 10_000,
    gold_path: Path | None = None,
    *,
    cadence_mode: str = "frozen",
    matchday_label: str | None = None,
    matches_completed_in_matchday: int | None = None,
    total_matches_completed: int | None = None,
    simulation_seed: int | None = None,
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
    if simulation_seed is None:
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
    # Build a rate lookup keyed by (home, away) so we can look up in either orientation.
    all_group_fixtures = generate_wc_group_fixtures()
    locked_group_keys = set(wc_results["group_results"].keys())
    pred_rate_lookup: dict[tuple[str, str], pd.Series] = {
        (row["home_team"], row["away_team"]): row
        for _, row in all_predictions_df.iterrows()
    }

    group_rows: list[dict] = []
    for _, fix in all_group_fixtures.iterrows():
        fwd = (fix["home_team"], fix["away_team"])
        rev = (fix["away_team"], fix["home_team"])
        if fwd in locked_group_keys or rev in locked_group_keys:
            continue
        if fwd in pred_rate_lookup:
            group_rows.append(pred_rate_lookup[fwd].to_dict())
        elif rev in pred_rate_lookup:
            # Prediction exists with flipped orientation — reorient to the
            # canonical fixture home/away so scorelines are attributed correctly.
            pr = pred_rate_lookup[rev].to_dict()
            pr["home_team"] = fix["home_team"]
            pr["away_team"] = fix["away_team"]
            pr["lambda_h"], pr["lambda_a"] = pr["lambda_a"], pr["lambda_h"]
            pr["p_home"], pr["p_away"] = pr["p_away"], pr["p_home"]
            group_rows.append(pr)
        else:
            logger.warning(
                "No prediction found for group fixture %s vs %s — skipping scoreline.",
                fix["home_team"], fix["away_team"],
            )

    group_predictions_df = pd.DataFrame(group_rows).reset_index(drop=True)

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
        group_teams = group_predictions_df[["home_team", "away_team"]].reset_index(drop=True)
        sl_dist["home_team"] = sl_dist["match_idx"].map(group_teams["home_team"])
        sl_dist["away_team"] = sl_dist["match_idx"].map(group_teams["away_team"])
        sl_dist["stage"] = "Group"
        sl_dist["match_num"] = None
    else:
        sl_dist = pd.DataFrame(
            columns=["match_idx", "home_goals", "away_goals", "probability",
                     "home_team", "away_team", "stage", "match_num"]
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

    # Build ko_fixtures and sample KO scorelines from champion results.
    champion_sim = per_model_tournament_results.get(champion_model_name, {})
    ko_slot_pairings = champion_sim.get("ko_slot_pairings")
    ko_fixtures = _build_ko_fixtures(wc_results["ko_results"], ko_slot_pairings)
    ko_sl_dist = _sample_ko_scorelines(
        ko_fixtures, all_predictions_df, n_sims, simulation_seed
    )
    if not ko_sl_dist.empty:
        sl_dist = pd.concat([sl_dist, ko_sl_dist], ignore_index=True)

    # Log to MLflow
    run_id = log_inference_artifacts(
        predictions_df=all_predictions_df,
        scoreline_dist=sl_dist,
        ko_fixtures=ko_fixtures if not ko_fixtures.empty else None,
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
