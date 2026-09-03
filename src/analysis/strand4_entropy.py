"""Strand 4: reconstruct four missed entropy-trajectory snapshots (SUPERSEDED).

Superseded by :mod:`src.analysis.strand5_frozen_entropy`, which rebuilds the
same four states with regime-pinned model versions.  Only
:func:`build_entropy_snapshot_specs` is still consumed — it remains the source
of truth for *which* states are missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.analysis.replay_common import (
    RECONSTRUCTION_EXPERIMENT,
    check_entropy_trajectory,
    compute_entropy_columns,
    ensure_output_dir,
    log_reconstruction_run,
    parse_wc_results_before_kickoff,
)
from src.inference.run import _seed_from_string, run_inference_and_simulation
from src.models.config import EXPERIMENT_MODELS
from src.monitoring.monitor import _list_inference_runs, parse_wc_settled_matches

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntropySnapshotSpec:
    """One intermediate locked state to reconstruct."""

    label: str
    round_seed: str
    max_kickoff: pd.Timestamp
    synthetic_timestamp: pd.Timestamp


def _fixture_kickoff_by_teams(home: str, away: str, settled: pd.DataFrame) -> pd.Timestamp:
    row = settled[(settled["home"] == home) & (settled["away"] == away)]
    if row.empty:
        row = settled[(settled["home"] == away) & (settled["away"] == home)]
    if row.empty:
        raise KeyError(f"No settled match for {home} vs {away}")
    return pd.Timestamp(row.iloc[0]["kickoff_utc"])


def build_entropy_snapshot_specs(settled: pd.DataFrame) -> list[EntropySnapshotSpec]:
    """Return the four intermediate states documented in wc_live.md."""
    japan_brazil = _fixture_kickoff_by_teams("Japan", "Brazil", settled)
    germany_paraguay = _fixture_kickoff_by_teams("Germany", "Paraguay", settled)
    brazil_norway = _fixture_kickoff_by_teams("Brazil", "Norway", settled)
    mexico_england = _fixture_kickoff_by_teams("Mexico", "England", settled)

    mid_r32_a = japan_brazil - pd.Timedelta(hours=1)
    mid_r32_b = japan_brazil + pd.Timedelta(hours=1)
    mid_r32_c = germany_paraguay + pd.Timedelta(hours=1)
    mid_r16 = brazil_norway + (mexico_england - brazil_norway) / 2

    return [
        EntropySnapshotSpec(
            label="r32_pre_japan_brazil",
            round_seed="R32",
            max_kickoff=japan_brazil - pd.Timedelta(seconds=1),
            synthetic_timestamp=mid_r32_a,
        ),
        EntropySnapshotSpec(
            label="r32_japan_brazil_locked",
            round_seed="R32",
            max_kickoff=japan_brazil,
            synthetic_timestamp=mid_r32_b,
        ),
        EntropySnapshotSpec(
            label="r32_germany_paraguay_locked",
            round_seed="R32",
            max_kickoff=germany_paraguay,
            synthetic_timestamp=mid_r32_c,
        ),
        EntropySnapshotSpec(
            label="r16_brazil_norway_locked",
            round_seed="R16",
            max_kickoff=brazil_norway,
            synthetic_timestamp=mid_r16,
        ),
    ]


def confirm_missing_cycle_counts() -> dict[str, int]:
    """Confirm R32 has 3 and R16 has 1 missing snapshot vs kickoff windows."""
    settled = parse_wc_settled_matches()
    specs = build_entropy_snapshot_specs(settled)
    return {
        "r32_missing": sum(1 for s in specs if s.round_seed == "R32"),
        "r16_missing": sum(1 for s in specs if s.round_seed == "R16"),
        "total": len(specs),
    }


def run_strand4_entropy(
    *,
    cadence_modes: tuple[str, ...] = ("frozen", "per_round"),
    log_mlflow: bool = True,
) -> str:
    """Re-run inference+simulation at four synthetic locked states.

    Superseded by :mod:`src.analysis.strand5_frozen_entropy`: this driver
    re-ran live inference at reconstruction time, so its per_round rows used
    the newest (terminal) models rather than the models that actually served
    the R32/R16 regimes.  Kept only for provenance and for its snapshot specs.
    """
    out_dir = ensure_output_dir("strand4_entropy")
    settled = parse_wc_settled_matches()
    specs = build_entropy_snapshot_specs(settled)
    counts = confirm_missing_cycle_counts()
    logger.info("Missing-cycle confirmation: %s", counts)
    assert counts["r32_missing"] == 3 and counts["r16_missing"] == 1

    entropy_rows: list[dict] = []

    for spec in specs:
        wc_partial = parse_wc_results_before_kickoff(spec.max_kickoff)
        seed = _seed_from_string(spec.round_seed)
        ts_str = spec.synthetic_timestamp.isoformat()

        for cadence_mode in cadence_modes:
            run_id = run_inference_and_simulation(
                cadence_mode=cadence_mode,
                simulation_seed=seed,
                wc_results_override=wc_partial,
                inference_timestamp_override=ts_str,
                reconstruction_experiment=RECONSTRUCTION_EXPERIMENT,
                matchday_label=spec.round_seed,
            )
            import mlflow

            client = mlflow.tracking.MlflowClient()
            local = client.download_artifacts(run_id, "tournament_probabilities.csv")
            adv = pd.read_csv(local)
            for model_name in EXPERIMENT_MODELS:
                model_adv = adv[adv["model_name"] == model_name]
                if model_adv.empty:
                    logger.warning(
                        "No advancement rows for %s (%s, %s) — model was skipped "
                        "upstream; snapshot will be incomplete.",
                        model_name,
                        spec.label,
                        cadence_mode,
                    )
                    continue
                entropy_rows.append({
                    "snapshot_label": spec.label,
                    "inference_run_id": run_id,
                    "inference_timestamp": spec.synthetic_timestamp,
                    "cadence_mode": cadence_mode,
                    "model_name": model_name,
                    **compute_entropy_columns(model_adv),
                    "synthetic": True,
                })

    entropy_df = pd.DataFrame(entropy_rows)
    entropy_path = out_dir / "reconstructed_entropy_snapshots.csv"
    entropy_df.to_csv(entropy_path, index=False)

    expected_rows = len(specs) * len(cadence_modes) * len(EXPERIMENT_MODELS)
    if len(entropy_df) != expected_rows:
        logger.warning(
            "Strand 4 produced %d of %d expected entropy rows — some models "
            "were skipped upstream.",
            len(entropy_df),
            expected_rows,
        )

    return log_reconstruction_run(
        strand="strand4_entropy",
        params={
            "snapshots_reconstructed": str(len(specs)),
            "cadence_modes": ",".join(cadence_modes),
            **{k: str(v) for k, v in counts.items()},
        },
        metrics={
            "entropy_points": float(len(entropy_df)),
            "expected_entropy_points": float(expected_rows),
            **check_entropy_trajectory(entropy_df),
        },
        artifacts={"reconstructed_snapshots": entropy_path},
        enabled=log_mlflow,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_strand4_entropy()
