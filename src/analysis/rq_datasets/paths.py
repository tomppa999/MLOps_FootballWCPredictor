"""Paths and schema constants for RQ-ready analysis datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from src.models.config import LIVE_SHADOW_MODELS

ANALYSIS_ROOT: Final[Path] = Path("data/analysis")
LIVE_ROOT: Final[Path] = ANALYSIS_ROOT / "live"
RECONSTRUCTION_ROOT: Final[Path] = Path("data/reconstruction")

N_MATCHES: Final[int] = 104
EXPECTED_ROWS_PER_CADENCE: Final[int] = len(LIVE_SHADOW_MODELS) * N_MATCHES  # 832
CADENCE_MODES: Final[tuple[str, ...]] = ("frozen", "per_round")

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

PROVENANCE_VALUES: Final[frozenset[str]] = frozenset({
    PROVENANCE_LIVE,
    PROVENANCE_FROZEN_SHADOW,
    PROVENANCE_BACKFILL_REPRED,
    PROVENANCE_BACKFILL_COPIED,
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

RQ1_PATH: Final[Path] = ANALYSIS_ROOT / "rq1_matches.csv"
RQ2_PATH: Final[Path] = ANALYSIS_ROOT / "rq2_entropy.csv"
RQ3_PATH: Final[Path] = ANALYSIS_ROOT / "rq3_calibration.csv"

METRIC_TOL: Final[float] = 1e-6
N_PROBABILITY_BINS: Final[int] = 10
# Entropy is normalised by column sum (n_teams=48), so H_max = log(48).
ENTROPY_H_MAX: Final[float] = float(__import__("math").log(48))
