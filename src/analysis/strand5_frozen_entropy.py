"""Strand 5: cadence-correct entropy trajectories from regime-pinned models.

Supersedes Strand 4 and corrects the frozen half of Strand 2.  Two problems are
fixed here:

- Strand 2 replays the *logged* lambdas, so its frozen rows inherit the
  shadow-resolution bug (``wc_shadow`` had no cadence alias, so live frozen
  cycles were served the newest per_round shadow).  Strand 5 re-predicts every
  pairing with the pre-tournament pin the frozen cadence should have used.
- Strand 4 re-ran live inference at reconstruction time, so its per_round gap
  snapshots used terminal models instead of the R32/R16 regime models.  Strand 5
  pins each snapshot to the versions that actually served that round.

Both halves predict all ~1128 pairings in one batch and re-simulate, because
``simulate_tournament`` needs the whole bracket and several features depend on
the full pairing table.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.analysis.replay_common import (
    FROZEN_REGIME,
    SETTLE_DELTA,
    GoldCommit,
    augment_gold_for_inference,
    batch_lambdas,
    build_gold_commit_index,
    check_entropy_trajectory,
    compute_entropy_columns,
    ensure_output_dir,
    inference_cycles_for,
    load_gold_at_commit,
    load_model_manifest,
    load_pinned_shadow_model,
    log_reconstruction_run,
    parse_wc_results_before_kickoff,
    pinned_versions,
    resolve_gold_commit,
    snapshot_key,
)
from src.analysis.rq_datasets.paths import (
    PROVENANCE_FROZEN_SHADOW,
    PROVENANCE_SNAPSHOT,
)
from src.analysis.strand4_entropy import build_entropy_snapshot_specs
from src.inference.run import _seed_from_string, _simulate_roster
from src.models.config import EXPERIMENT_MODELS
from src.monitoring.monitor import parse_wc_settled_matches

logger = logging.getLogger(__name__)

CADENCE_MODES: Final[tuple[str, ...]] = ("frozen", "per_round")

# Lambdas kept for the audit's pre-refit identity check (Strand 5 must equal the
# logged values before the first refit, since until then the buggy resolver
# served the frozen pin anyway).
PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "inference_run_id",
    "cadence_mode",
    "model_name",
    "home_team",
    "away_team",
    "lambda_h",
    "lambda_a",
)

# A pinned pyfunc is reused across every cycle in its regime; loading it once
# per cycle would dominate the runtime.
_MODEL_CACHE: dict[tuple[str, int], Any] = {}


def _predictions_schema() -> pa.Schema:
    return pa.schema([
        pa.field(col, pa.float64() if col.startswith("lambda_") else pa.string())
        for col in PREDICTION_COLUMNS
    ])


def clear_model_cache() -> None:
    """Drop loaded pinned models (tests, memory pressure)."""
    _MODEL_CACHE.clear()


def load_regime_models(
    manifest: pd.DataFrame,
    *,
    cadence_role: str,
    regime: str,
) -> dict[str, tuple[str, int, Any]]:
    """Load the pinned roster for one regime as ``{model: (registry, version, model)}``."""
    pins = pinned_versions(manifest, cadence_role=cadence_role, regime=regime)
    loaded: dict[str, tuple[str, int, Any]] = {}
    for model_name, (registry_name, version) in pins.items():
        cache_key = (registry_name, version)
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = load_pinned_shadow_model(
                model_name, version, registry_name=registry_name,
            )
        loaded[model_name] = (registry_name, version, _MODEL_CACHE[cache_key])
    return loaded


def _predict_and_simulate(
    *,
    models: dict[str, tuple[str, int, Any]],
    gold_commit: GoldCommit,
    wc: dict[str, Any],
    seed: int | None,
    n_sims: int,
    champion_model_name: str,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Re-predict every pairing with the pinned roster and simulate the bracket."""
    gold_df = load_gold_at_commit(gold_commit)
    augmented, reference_date = augment_gold_for_inference(gold_df, wc)
    key = snapshot_key(gold_commit.gold_hash, wc)

    frames: list[pd.DataFrame] = []
    for model_name, (_registry, _version, model) in models.items():
        lambdas = batch_lambdas(model, model_name, augmented, reference_date, key=key)
        frame = lambdas.copy()
        frame.insert(0, "model_name", model_name)
        frames.append(frame)
    all_models = pd.concat(frames, ignore_index=True)

    champion_rows = all_models.loc[all_models["model_name"] == champion_model_name]
    if champion_rows.empty:
        raise ValueError(
            f"Champion {champion_model_name!r} is not in the pinned roster "
            f"{sorted(models)} — the manifest and the cycle table disagree",
        )
    champion = champion_rows.drop(columns=["model_name"]).reset_index(drop=True)

    per_model = _simulate_roster(
        all_models_predictions_df=all_models,
        champion_predictions_df=champion,
        champion_model_name=champion_model_name,
        n_sims=n_sims,
        locked_group=wc["group_results"] or None,
        locked_ko=wc["ko_results"] or None,
        seed=seed,
    )
    return per_model, all_models


def _write_artifacts(
    per_model: dict[str, dict],
    *,
    out_dir: Path,
    cadence_mode: str,
    file_stem: str,
) -> dict[str, pd.DataFrame]:
    """Persist advancement / ko_pairings per model; return the advancement frames."""
    advancement: dict[str, pd.DataFrame] = {}
    for model_name in EXPERIMENT_MODELS:
        results = per_model.get(model_name)
        if results is None:
            continue
        adv = results.get("advancement")
        if adv is None or adv.empty:
            continue
        advancement[model_name] = adv

        model_dir = out_dir / cadence_mode / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        adv.to_csv(model_dir / f"{file_stem}_advancement.csv", index=False)
        ko = results.get("ko_pairings")
        if ko is not None and not ko.empty:
            ko.to_csv(model_dir / f"{file_stem}_ko_pairings.csv", index=False)
    return advancement


def _prediction_rows(
    all_models: pd.DataFrame,
    *,
    run_id: str,
    cadence_mode: str,
) -> pd.DataFrame:
    out = all_models.copy()
    out.insert(0, "cadence_mode", cadence_mode)
    out.insert(0, "inference_run_id", run_id)
    return out[list(PREDICTION_COLUMNS)]


def run_strand5_frozen_entropy(
    *,
    log_mlflow: bool = False,
    subdir: str = "strand5_frozen_entropy",
    scopes: tuple[str, ...] = ("live", "snapshots"),
    max_cycles: int | None = None,
) -> str:
    """Rebuild frozen live-cycle entropy and the four gap snapshots.

    ``scopes`` and ``max_cycles`` exist so a smoke run can time a handful of
    real cycles without producing a partial committed output — point ``subdir``
    somewhere else when using them.
    """
    out_dir = ensure_output_dir(subdir)
    manifest = load_model_manifest()
    gold_index = build_gold_commit_index()

    entropy_rows: list[dict] = []
    started = time.monotonic()
    live_cycles = 0
    skipped_no_gold: list[str] = []

    predictions_path = out_dir / "predictions.parquet"
    staged = predictions_path.with_suffix(predictions_path.suffix + ".tmp")
    writer = pq.ParquetWriter(staged, _predictions_schema(), compression="snappy")

    try:
        if "live" in scopes:
            # Frozen never refits, so one pre-tournament roster serves every cycle.
            models = load_regime_models(
                manifest, cadence_role="frozen", regime=FROZEN_REGIME,
            )
            versions = {name: (reg, ver) for name, (reg, ver, _) in models.items()}
            cycles = inference_cycles_for("frozen")
            if max_cycles is not None:
                cycles = cycles[:max_cycles]
            logger.info(
                "Strand 5: %d frozen cycles with pins %s", len(cycles), versions,
            )

            for cycle in cycles:
                commit = resolve_gold_commit(cycle.inference_timestamp, gold_index)
                if commit is None:
                    skipped_no_gold.append(cycle.run_id)
                    logger.warning(
                        "No pre-cycle Gold commit for %s (%s) — skipping",
                        cycle.run_id,
                        cycle.inference_timestamp,
                    )
                    continue
                if cycle.simulation_seed is None:
                    logger.warning(
                        "Cycle %s has no seed — its simulation is not reproducible",
                        cycle.run_id,
                    )

                wc = parse_wc_results_before_kickoff(
                    cycle.inference_timestamp, settle_delta=SETTLE_DELTA,
                )
                try:
                    per_model, all_models = _predict_and_simulate(
                        models=models,
                        gold_commit=commit,
                        wc=wc,
                        seed=cycle.simulation_seed,
                        n_sims=cycle.n_sims,
                        champion_model_name=cycle.champion_model_name,
                    )
                except Exception:
                    logger.exception("Strand 5 replay failed for cycle %s", cycle.run_id)
                    continue

                advancement = _write_artifacts(
                    per_model,
                    out_dir=out_dir,
                    cadence_mode="frozen",
                    file_stem=cycle.run_id,
                )
                for model_name, adv in advancement.items():
                    registry_name, version, _ = models[model_name]
                    entropy_rows.append({
                        "inference_run_id": cycle.run_id,
                        "inference_timestamp": cycle.inference_timestamp,
                        "cadence_mode": "frozen",
                        "model_name": model_name,
                        **compute_entropy_columns(adv),
                        "synthetic": False,
                        "snapshot_label": pd.NA,
                        "registry_name": registry_name,
                        "model_version": version,
                        "provenance": PROVENANCE_FROZEN_SHADOW,
                    })
                writer.write_table(
                    pa.Table.from_pandas(
                        _prediction_rows(
                            all_models, run_id=cycle.run_id, cadence_mode="frozen",
                        ),
                        schema=_predictions_schema(),
                        preserve_index=False,
                    ),
                )
                live_cycles += 1

        if "snapshots" in scopes:
            settled = parse_wc_settled_matches()
            specs = build_entropy_snapshot_specs(settled)
            logger.info("Strand 5: %d gap snapshots x %d cadences", len(specs), len(CADENCE_MODES))

            for spec in specs:
                # Strand 4 semantics deliberately preserved: these synthetic
                # states lock on raw kickoff (settle delta 0), and Gold is
                # resolved at the synthetic timestamp.
                wc = parse_wc_results_before_kickoff(spec.max_kickoff)
                commit = resolve_gold_commit(spec.synthetic_timestamp, gold_index)
                if commit is None:
                    raise RuntimeError(f"No Gold commit before snapshot {spec.label}")
                seed = _seed_from_string(spec.round_seed)

                for cadence_mode in CADENCE_MODES:
                    regime = FROZEN_REGIME if cadence_mode == "frozen" else spec.round_seed
                    models = load_regime_models(
                        manifest, cadence_role=cadence_mode, regime=regime,
                    )
                    run_id = f"{spec.label}__{cadence_mode}"
                    per_model, all_models = _predict_and_simulate(
                        models=models,
                        gold_commit=commit,
                        wc=wc,
                        seed=seed,
                        n_sims=10_000,
                        champion_model_name="xgboost",
                    )
                    advancement = _write_artifacts(
                        per_model,
                        out_dir=out_dir,
                        cadence_mode=cadence_mode,
                        file_stem=spec.label,
                    )
                    for model_name, adv in advancement.items():
                        registry_name, version, _ = models[model_name]
                        entropy_rows.append({
                            "inference_run_id": run_id,
                            "inference_timestamp": spec.synthetic_timestamp,
                            "cadence_mode": cadence_mode,
                            "model_name": model_name,
                            **compute_entropy_columns(adv),
                            "synthetic": True,
                            "snapshot_label": spec.label,
                            "registry_name": registry_name,
                            "model_version": version,
                            "provenance": PROVENANCE_SNAPSHOT,
                        })
                    writer.write_table(
                        pa.Table.from_pandas(
                            _prediction_rows(
                                all_models, run_id=run_id, cadence_mode=cadence_mode,
                            ),
                            schema=_predictions_schema(),
                            preserve_index=False,
                        ),
                    )
    finally:
        writer.close()
    staged.replace(predictions_path)

    entropy_df = pd.DataFrame(entropy_rows)
    entropy_path = out_dir / "entropy_trajectory.csv"
    entropy_df.to_csv(entropy_path, index=False)

    elapsed = time.monotonic() - started
    summary = {
        "live_cycles": live_cycles,
        "entropy_rows": len(entropy_df),
        "synthetic_rows": int(entropy_df["synthetic"].sum()) if not entropy_df.empty else 0,
        "skipped_no_gold": skipped_no_gold,
        "scopes": list(scopes),
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_live_cycle": round(elapsed / live_cycles, 2) if live_cycles else None,
        "predictions_mb": round(predictions_path.stat().st_size / 1e6, 1),
        "pins": (
            manifest[["registry_name", "version", "model_name", "cadence_role", "regime"]]
            .to_dict(orient="records")
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Strand 5 summary: %s", summary)

    return log_reconstruction_run(
        strand="strand5_frozen_entropy",
        params={
            "live_cycles": str(live_cycles),
            "scopes": ",".join(scopes),
            "settle_delta": str(SETTLE_DELTA),
        },
        metrics={
            "entropy_points": float(len(entropy_df)),
            **check_entropy_trajectory(entropy_df),
        },
        artifacts={"entropy_trajectory": entropy_path, "summary": summary_path},
        enabled=log_mlflow,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_strand5_frozen_entropy()
