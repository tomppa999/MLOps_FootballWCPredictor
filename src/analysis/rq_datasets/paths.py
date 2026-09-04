"""Paths and schema constants for RQ-ready analysis datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from src.models.config import LIVE_SHADOW_MODELS

ANALYSIS_ROOT: Final[Path] = Path("data/analysis")
LIVE_ROOT: Final[Path] = ANALYSIS_ROOT / "live"
RECONSTRUCTION_ROOT: Final[Path] = Path("data/reconstruction")

N_MATCHES: Final[int] = 104
EXPECTED_ROWS_PER_CADENCE: Final[int] = len(LIVE_SHADOW_MODELS) * N_MATCHES  # 832
CADENCE_MODES: Final[tuple[str, ...]] = ("frozen", "per_round")

# Start of the analysis window: the last frozen inference cycle before the
# tournament opened (first kickoff Jun 11 19:00 UTC).  34 earlier cycles (20
# frozen, 14 per_round) are dry runs against an empty tournament and answer
# none of the RQs; three of them also logged neither a simulation seed nor a
# matchday label, so they draw from OS entropy and can never be reproduced.
# Keeping this timestamp (and its per_round twin a few minutes later) leaves
# one pre-tournament baseline per cadence and makes the whole replay
# deterministic.
ANALYSIS_START: Final[pd.Timestamp] = pd.Timestamp("2026-06-11 16:27:37", tz="UTC")

MONITORING_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "kickoff_utc",
    "home",
    "away",
    "actual_h",
    "actual_a",
    "actual_outcome",
    "model_name",
    "lambda_h",
    "lambda_a",
    "p_home",
    "p_draw",
    "p_away",
    "rps",
    "nll",
    "rmse_h",
    "rmse_a",
    "inference_run_id",
    "cadence_mode",
)

RQ1_KEY: Final[tuple[str, ...]] = ("match_id", "model_name", "cadence_mode")
RQ2_KEY: Final[tuple[str, ...]] = ("inference_run_id", "cadence_mode", "model_name")

PROVENANCE_LIVE: Final[str] = "live"
PROVENANCE_FROZEN_SHADOW: Final[str] = "reconstructed_frozen_shadow"
PROVENANCE_BACKFILL_REPRED: Final[str] = "backfill_repredicted"
PROVENANCE_BACKFILL_COPIED: Final[str] = "backfill_copied"
# RQ2 only: Strand 2 replays the logged lambdas of a real cycle, Strand 5
# re-predicts with the regime-pinned model at a synthetic locked state.
PROVENANCE_LIVE_REPLAY: Final[str] = "live_replay"
PROVENANCE_SNAPSHOT: Final[str] = "reconstructed_snapshot"

PROVENANCE_VALUES: Final[frozenset[str]] = frozenset({
    PROVENANCE_LIVE,
    PROVENANCE_FROZEN_SHADOW,
    PROVENANCE_BACKFILL_REPRED,
    PROVENANCE_BACKFILL_COPIED,
    PROVENANCE_LIVE_REPLAY,
    PROVENANCE_SNAPSHOT,
})

STRAND1_COMBINED: Final[Path] = (
    RECONSTRUCTION_ROOT / "strand1_frozen_shadow" / "frozen_shadow_combined.csv"
)
STRAND3_BACKFILL: Final[Path] = RECONSTRUCTION_ROOT / "strand3_backfill" / "backfill_rows.csv"
STRAND2_TRAJECTORY: Final[Path] = (
    RECONSTRUCTION_ROOT / "strand2_brackets" / "entropy_trajectory.csv"
)
STRAND4_SNAPSHOTS: Final[Path] = (
    RECONSTRUCTION_ROOT / "strand4_entropy" / "reconstructed_entropy_snapshots.csv"
)
STRAND5_ROOT: Final[Path] = RECONSTRUCTION_ROOT / "strand5_frozen_entropy"
STRAND5_TRAJECTORY: Final[Path] = STRAND5_ROOT / "entropy_trajectory.csv"
STRAND5_PREDICTIONS: Final[Path] = STRAND5_ROOT / "predictions.parquet"

RQ1_PATH: Final[Path] = ANALYSIS_ROOT / "rq1_matches.csv"
RQ2_PATH: Final[Path] = ANALYSIS_ROOT / "rq2_entropy.csv"
RQ3_PATH: Final[Path] = ANALYSIS_ROOT / "rq3_calibration.csv"

METRIC_TOL: Final[float] = 1e-6
N_PROBABILITY_BINS: Final[int] = 10
# Entropy is normalised by column sum (n_teams=48), so H_max = log(48).
ENTROPY_H_MAX: Final[float] = float(__import__("math").log(48))
