"""Strand 2: bracket artifact regeneration and entropy curves."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from src.analysis.replay_common import (
    RECONSTRUCTION_EXPERIMENT,
    SETTLE_DELTA,
    check_entropy_trajectory,
    compute_entropy_columns,
    ensure_output_dir,
    log_reconstruction_run,
    parse_wc_results_before_kickoff,
)
from src.inference.run import _simulate_roster, _seed_from_string
from src.models.config import EXPERIMENT_MODELS
from src.models.mlflow_utils import setup_mlflow
from src.monitoring.monitor import _list_inference_runs

logger = logging.getLogger(__name__)

_PREDICTIONS_FILENAME = "predictions_all_models.csv"
_CHAMPION_PREDICTIONS_FILENAME = "predictions.csv"

# Logged run artifacts are immutable, so a replay can reuse them across runs
# instead of re-downloading ~1.8k files from DagsHub each time.  Deliberately
# outside OUTPUT_ROOT, which is DVC-tracked — this is a local scratch cache.
ARTIFACT_CACHE_DIR: Path = Path(".d1_artifact_cache")


def _cache_path(run_id: str, filename: str) -> Path:
    return ARTIFACT_CACHE_DIR / run_id / filename


def _load_artifact(run_id: str, filename: str, *, use_cache: bool = True) -> pd.DataFrame | None:
    """Read a logged artifact, downloading it once and caching it on disk."""
    cached = _cache_path(run_id, filename)
    if use_cache and cached.exists():
        return pd.read_csv(cached)

    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    try:
        local = client.download_artifacts(run_id, filename)
    except Exception:  # noqa: BLE001
        return None

    if use_cache:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            # Write via a temp name so an interrupted copy cannot leave a
            # truncated file that later runs would treat as a cache hit.
            staged = cached.with_suffix(cached.suffix + ".tmp")
            shutil.copyfile(local, staged)
            staged.replace(cached)
        except OSError:
            logger.warning("Could not cache artifact %s/%s", run_id, filename, exc_info=True)

    return pd.read_csv(local)


def _replay_simulation_for_run(
    run_id: str,
    *,
    cadence_mode: str,
    as_of: pd.Timestamp,
    settle_delta: pd.Timedelta = SETTLE_DELTA,
    n_sims: int = 10_000,
) -> dict[str, dict]:
    """Re-simulate all roster models from logged predictions.

    Only results settled by ``as_of`` are locked, so each replayed cycle sees
    the tournament state that cycle actually ran against.
    """
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
    wc = parse_wc_results_before_kickoff(as_of, settle_delta=settle_delta)
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
                per_model = _replay_simulation_for_run(
                    run_id, cadence_mode=cadence_mode, as_of=ts,
                )
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
                entropy_rows.append({
                    "inference_run_id": run_id,
                    "inference_timestamp": ts,
                    "cadence_mode": cadence_mode,
                    "model_name": model_name,
                    **compute_entropy_columns(adv),
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
            "settle_delta": str(SETTLE_DELTA),
        },
        metrics={
            "entropy_points": float(len(entropy_df)),
            **check_entropy_trajectory(entropy_df),
        },
        artifacts={"entropy_trajectory": entropy_path},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_strand2_brackets()
