"""Log inference and simulation artifacts to MLflow."""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.models.mlflow_utils import (
    get_latest_production_run_id,
    log_run,
    setup_mlflow,
    start_run,
)

logger = logging.getLogger(__name__)


def _stack_per_model(
    per_model_results: dict[str, dict[str, Any]],
    key: str,
) -> pd.DataFrame | None:
    """Stack one artifact key across all models, adding a model_name column.

    Returns None if no model produces a non-empty DataFrame for ``key``.
    """
    frames: list[pd.DataFrame] = []
    for model_name, results in per_model_results.items():
        df = results.get(key)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            continue
        df = df.copy()
        df.insert(0, "model_name", model_name)
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def log_inference_artifacts(
    predictions_df: pd.DataFrame,
    scoreline_dist: pd.DataFrame | None,
    per_model_tournament_results: dict[str, dict[str, Any]],
    *,
    n_sims: int,
    gold_row_count: int,
    champion_model_name: str = "unknown",
    all_models_predictions_df: pd.DataFrame | None = None,
    inference_timestamp: str | None = None,
    simulation_seed: int | None = None,
) -> str:
    """Start an MLflow run tagged stage=inference and log all artifacts.

    Logged artifacts:
      - predictions.csv: champion-only λ_h/λ_a + outcome probs (input to
        the tournament simulation).
      - predictions_all_models.csv: long-format predictions for the
        champion plus every shadow candidate — input to the monitoring
        layer.  Only logged when ``all_models_predictions_df`` is provided.
      - scoreline_distributions.csv: sampled scoreline probabilities
        (champion only).
      - tournament_probabilities.csv: per-team advancement by round,
        one row per (model_name, team).  Contains all simulated roster
        models (``per_model_tournament_results``).
      - group_positions.csv: per-team group-finish probabilities,
        stacked across roster models.
      - ko_pairings.csv: per-stage KO matchup frequencies,
        stacked across roster models.

    ``champion_model_name`` is logged as a param so the dashboard can
    filter tournament artifacts back to the champion without an extra
    MLflow call.  ``simulation_seed`` and ``inference_timestamp`` are logged
    for reproducibility — a cycle can be replayed by re-running with the same
    seed and the same Gold/predictions inputs.

    Returns the inference MLflow run_id.
    """
    setup_mlflow()

    champion_run_id = get_latest_production_run_id() or "unknown"
    ts = inference_timestamp or datetime.now(timezone.utc).isoformat()

    # Stack per-model simulation artifacts
    combined_advancement = _stack_per_model(per_model_tournament_results, "advancement")
    combined_group_positions = _stack_per_model(per_model_tournament_results, "group_positions")
    combined_ko_pairings = _stack_per_model(per_model_tournament_results, "ko_pairings")

    params: dict[str, str] = {
        "n_sims": str(n_sims),
        "champion_run_id": champion_run_id,
        "champion_model_name": champion_model_name,
        "gold_row_count": str(gold_row_count),
        "inference_timestamp": ts,
        "simulated_models": ",".join(sorted(per_model_tournament_results.keys())),
    }
    if simulation_seed is not None:
        params["simulation_seed"] = str(simulation_seed)

    with start_run(
        run_name="inference",
        tags={"stage": "inference"},
    ) as run:
        log_run(params=params)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            pred_path = tmp / "predictions.csv"
            predictions_df.to_csv(pred_path, index=False)
            mlflow.log_artifact(str(pred_path))

            if all_models_predictions_df is not None and not all_models_predictions_df.empty:
                all_path = tmp / "predictions_all_models.csv"
                all_models_predictions_df.to_csv(all_path, index=False)
                mlflow.log_artifact(str(all_path))

            if scoreline_dist is not None and not scoreline_dist.empty:
                sl_path = tmp / "scoreline_distributions.csv"
                scoreline_dist.to_csv(sl_path, index=False)
                mlflow.log_artifact(str(sl_path))

            if combined_advancement is not None:
                tp_path = tmp / "tournament_probabilities.csv"
                combined_advancement.to_csv(tp_path, index=False)
                mlflow.log_artifact(str(tp_path))

            if combined_group_positions is not None:
                gp_path = tmp / "group_positions.csv"
                combined_group_positions.to_csv(gp_path, index=False)
                mlflow.log_artifact(str(gp_path))

            if combined_ko_pairings is not None:
                kp_path = tmp / "ko_pairings.csv"
                combined_ko_pairings.to_csv(kp_path, index=False)
                mlflow.log_artifact(str(kp_path))

        run_id = run.info.run_id

    logger.info(
        "Inference artifacts logged to MLflow run %s (simulated %d models: %s)",
        run_id,
        len(per_model_tournament_results),
        ", ".join(sorted(per_model_tournament_results.keys())),
    )
    return run_id
