"""Strand 2: bracket artifact regeneration and entropy curves."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from src.analysis.replay_common import (
    RECONSTRUCTION_EXPERIMENT,
    compute_advancement_entropy,
    ensure_output_dir,
    log_reconstruction_run,
)
from src.inference.run import _simulate_roster, _seed_from_string
from src.models.config import EXPERIMENT_MODELS
from src.models.mlflow_utils import setup_mlflow
from src.monitoring.monitor import _list_inference_runs

logger = logging.getLogger(__name__)

_PREDICTIONS_FILENAME = "predictions_all_models.csv"
_CHAMPION_PREDICTIONS_FILENAME = "predictions.csv"


def _load_artifact(run_id: str, filename: str) -> pd.DataFrame | None:
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    try:
        local = client.download_artifacts(run_id, filename)
    except Exception:  # noqa: BLE001
        return None
    return pd.read_csv(local)


def _replay_simulation_for_run(
    run_id: str,
    *,
    cadence_mode: str,
    n_sims: int = 10_000,
) -> dict[str, dict]:
    """Re-simulate all roster models from logged predictions."""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params
    seed = int(params["simulation_seed"]) if "simulation_seed" in params else None
    if seed is None and "matchday_label" in params:
        seed = _seed_from_string(params["matchday_label"])

    all_models = _load_artifact(run_id, _PREDICTIONS_FILENAME)
    champion = _load_artifact(run_id, _CHAMPION_PREDICTIONS_FILENAME)
    if all_models is None or champion is None:
        raise ValueError(f"Missing prediction artifacts for run {run_id}")

    champion_model = params.get("champion_model_name", "xgboost")
    from src.inference.features import parse_wc_results

    wc = parse_wc_results()
    return _simulate_roster(
        all_models_predictions_df=all_models,
        champion_predictions_df=champion,
        champion_model_name=champion_model,
        n_sims=n_sims,
        locked_group=wc["group_results"] or None,
        locked_ko=wc["ko_results"] or None,
        seed=seed,
    )


def run_strand2_brackets(*, cadence_modes: tuple[str, ...] = ("frozen", "per_round")) -> str:
    """Replay inference cycles through corrected simulate_tournament; build entropy curves."""
    out_dir = ensure_output_dir("strand2_brackets")
    entropy_rows: list[dict] = []

    for cadence_mode in cadence_modes:
        inference_runs = _list_inference_runs(cadence_mode)
        logger.info(
            "Strand 2: %d inference runs for cadence_mode=%s",
            len(inference_runs),
            cadence_mode,
        )
        for run_info in inference_runs:
            run_id = run_info["run_id"]
            ts = run_info["inference_timestamp"]
            try:
                per_model = _replay_simulation_for_run(run_id, cadence_mode=cadence_mode)
            except Exception:
                logger.exception("Simulation replay failed for run %s", run_id)
                continue

            for model_name in EXPERIMENT_MODELS:
                results = per_model.get(model_name)
                if results is None:
                    continue
                adv = results.get("advancement")
                if adv is None or adv.empty:
                    continue
                h = compute_advancement_entropy(adv)
                entropy_rows.append({
                    "inference_run_id": run_id,
                    "inference_timestamp": ts,
                    "cadence_mode": cadence_mode,
                    "model_name": model_name,
                    "entropy": h,
                })

                model_dir = out_dir / cadence_mode / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                adv.to_csv(model_dir / f"{run_id}_advancement.csv", index=False)
                ko = results.get("ko_pairings")
                if ko is not None and not ko.empty:
                    ko.to_csv(model_dir / f"{run_id}_ko_pairings.csv", index=False)

    entropy_df = pd.DataFrame(entropy_rows)
    entropy_path = out_dir / "entropy_trajectory.csv"
    entropy_df.to_csv(entropy_path, index=False)

    return log_reconstruction_run(
        strand="strand2_brackets",
        params={
            "cadence_modes": ",".join(cadence_modes),
            "inference_runs_replayed": str(entropy_df["inference_run_id"].nunique()),
        },
        metrics={
            "entropy_points": float(len(entropy_df)),
        },
        artifacts={"entropy_trajectory": entropy_path},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_strand2_brackets()
