"""Strand 2: bracket artifact regeneration and entropy curves."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

from src.analysis.replay_common import (
    RECONSTRUCTION_EXPERIMENT,
    SETTLE_DELTA,
    InferenceCycle,
    check_entropy_trajectory,
    compute_entropy_columns,
    ensure_output_dir,
    inference_cycles_for,
    load_snapshot_predictions,
    log_reconstruction_run,
    parse_wc_results_before_kickoff,
    prediction_snapshot_available,
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


def _load_predictions(run_id: str, kind: str) -> pd.DataFrame | None:
    """Logged predictions for one run: snapshot → local cache → MLflow.

    A present snapshot is authoritative.  A run missing from it logged no such
    artifact (two pre-tournament cycles did not), so reaching for MLflow would
    only spend retry backoff on a 404.
    """
    if prediction_snapshot_available(kind):
        return load_snapshot_predictions(run_id, kind)
    filename = (
        _PREDICTIONS_FILENAME if kind == "all_models" else _CHAMPION_PREDICTIONS_FILENAME
    )
    return _load_artifact(run_id, filename)


def _cycle_params_from_mlflow(run_id: str) -> tuple[int | None, str]:
    """Fallback seed / champion lookup when no cycle table row is available."""
    client = mlflow.tracking.MlflowClient()
    params = client.get_run(run_id).data.params
    seed = int(params["simulation_seed"]) if "simulation_seed" in params else None
    if seed is None and "matchday_label" in params:
        seed = _seed_from_string(params["matchday_label"])
    return seed, params.get("champion_model_name", "xgboost")


def _replay_simulation_for_run(
    run_id: str,
    *,
    cadence_mode: str,
    as_of: pd.Timestamp,
    cycle: InferenceCycle | None = None,
    settle_delta: pd.Timedelta = SETTLE_DELTA,
    n_sims: int = 10_000,
) -> dict[str, dict]:
    """Re-simulate all roster models from logged predictions.

    Only results settled by ``as_of`` are locked, so each replayed cycle sees
    the tournament state that cycle actually ran against.  ``cycle`` supplies
    the logged seed / champion / n_sims offline; without it they are read back
    from MLflow.
    """
    if cycle is not None:
        seed = cycle.simulation_seed
        champion_model = cycle.champion_model_name
        n_sims = cycle.n_sims
    else:
        seed, champion_model = _cycle_params_from_mlflow(run_id)

    all_models = _load_predictions(run_id, "all_models")
    champion = _load_predictions(run_id, "champion")
    if all_models is None or champion is None:
        raise ValueError(f"Missing prediction artifacts for run {run_id}")

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


@dataclass(frozen=True)
class _ReplayTarget:
    """One cycle to replay.  ``cycle`` is None when only MLflow knows its params."""

    run_id: str
    inference_timestamp: pd.Timestamp
    cycle: InferenceCycle | None


def _replay_targets(cadence_mode: str) -> list[_ReplayTarget]:
    """Cycles to replay for one cadence, from the frozen table when available."""
    try:
        cycles = inference_cycles_for(cadence_mode)
    except FileNotFoundError:
        logger.warning(
            "No frozen inference-cycle table — listing runs from MLflow instead.",
        )
        return [
            _ReplayTarget(str(r["run_id"]), r["inference_timestamp"], None)
            for r in _list_inference_runs(cadence_mode)
        ]
    return [_ReplayTarget(c.run_id, c.inference_timestamp, c) for c in cycles]


def run_strand2_brackets(
    *,
    cadence_modes: tuple[str, ...] = ("frozen", "per_round"),
    log_mlflow: bool = False,
    subdir: str = "strand2_brackets",
) -> str:
    """Replay inference cycles through corrected simulate_tournament; build entropy curves.

    ``subdir`` writes the brackets somewhere other than the committed output,
    which is how the offline snapshot path is validated against it.
    """
    out_dir = ensure_output_dir(subdir)
    entropy_rows: list[dict] = []

    for cadence_mode in cadence_modes:
        targets = _replay_targets(cadence_mode)
        logger.info(
            "Strand 2: %d inference runs for cadence_mode=%s",
            len(targets),
            cadence_mode,
        )
        for target in targets:
            run_id = target.run_id
            ts = target.inference_timestamp
            try:
                per_model = _replay_simulation_for_run(
                    run_id,
                    cadence_mode=cadence_mode,
                    as_of=ts,
                    cycle=target.cycle,
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
        enabled=log_mlflow,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_strand2_brackets()
