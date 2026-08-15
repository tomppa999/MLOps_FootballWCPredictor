"""Export canonical live tournament tables from MLflow + Bronze to disk.

Outputs under ``data/analysis/live/``:

- ``monitoring_{cadence}.csv`` — full cumulative monitoring per cadence
- ``inference_cycles.csv`` — one row per inference run (provenance for RQ2)
- ``fixtures.csv`` — finished WC fixtures with ``round_label`` (needed by RQ1)

Idempotent: skips existing files unless ``refresh=True``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.analysis.replay_common import load_finished_wc_fixtures
from src.analysis.rq_datasets.paths import CADENCE_MODES, LIVE_ROOT
from src.models.mlflow_utils import EXPERIMENT_NAME, setup_mlflow

logger = logging.getLogger(__name__)

_MONITORING_ARTIFACT = "wc2026_monitoring.csv"

INFERENCE_CYCLE_COLUMNS: tuple[str, ...] = (
    "inference_run_id",
    "inference_timestamp",
    "cadence_mode",
    "matchday_label",
    "champion_model_name",
    "champion_run_id",
    "simulation_seed",
    "n_sims",
)


def monitoring_path(cadence_mode: str, live_root: Path = LIVE_ROOT) -> Path:
    return live_root / f"monitoring_{cadence_mode}.csv"


def inference_cycles_path(live_root: Path = LIVE_ROOT) -> Path:
    return live_root / "inference_cycles.csv"


def fixtures_path(live_root: Path = LIVE_ROOT) -> Path:
    return live_root / "fixtures.csv"


def _download_latest_monitoring(cadence_mode: str) -> pd.DataFrame:
    """Download ``wc2026_monitoring.csv`` from the latest monitoring run."""
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(
            f'tags.stage = "monitoring" AND tags.cadence_mode = "{cadence_mode}"'
        ),
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No monitoring runs found for cadence_mode={cadence_mode!r}.")
    local_path = client.download_artifacts(runs[0].info.run_id, _MONITORING_ARTIFACT)
    df = pd.read_csv(local_path)
    if "kickoff_utc" in df.columns:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    return df


def _list_all_inference_cycles() -> pd.DataFrame:
    """Return one row per inference run with cycle provenance params."""
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string='tags.stage = "inference"',
        order_by=["start_time ASC"],
        max_results=5000,
    )
    rows: list[dict[str, Any]] = []
    for r in runs:
        params = r.data.params
        ts = params.get("inference_timestamp")
        if not ts:
            continue
        try:
            ts_parsed = pd.to_datetime(ts, utc=True)
        except (ValueError, TypeError):
            continue
        seed_raw = params.get("simulation_seed")
        n_sims_raw = params.get("n_sims")
        rows.append({
            "inference_run_id": r.info.run_id,
            "inference_timestamp": ts_parsed,
            "cadence_mode": params.get("cadence_mode", "frozen"),
            "matchday_label": params.get("matchday_label"),
            "champion_model_name": params.get("champion_model_name"),
            "champion_run_id": params.get("champion_run_id"),
            "simulation_seed": int(seed_raw) if seed_raw not in (None, "") else pd.NA,
            "n_sims": int(n_sims_raw) if n_sims_raw not in (None, "") else pd.NA,
        })
    df = pd.DataFrame(rows, columns=list(INFERENCE_CYCLE_COLUMNS))
    if df.empty:
        return df
    return df.sort_values(["cadence_mode", "inference_timestamp"]).reset_index(drop=True)


def build_fixtures_frame() -> pd.DataFrame:
    """Finished WC fixtures with ``round_label`` from Bronze.

    The pipeline's ``_round_to_matchday_label`` returns ``None`` for the
    3rd-place final; for RQ analysis we label it ``3rd_place`` so every
    monitoring row joins cleanly.
    """
    fixtures = load_finished_wc_fixtures()
    rows = []
    for fx in fixtures:
        label = fx.round_label
        if label is None and fx.round_str and "3rd" in fx.round_str.lower():
            label = "3rd_place"
        rows.append({
            "match_id": fx.fixture_id,
            "kickoff_utc": fx.kickoff,
            "home": fx.home_team,
            "away": fx.away_team,
            "actual_h": fx.home_goals,
            "actual_a": fx.away_goals,
            "round_label": label,
            "round_str": fx.round_str,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("kickoff_utc").reset_index(drop=True)


def export_live(*, refresh: bool = False, live_root: Path = LIVE_ROOT) -> dict[str, Path]:
    """Freeze live monitoring, inference cycles, and fixtures to ``live_root``.

    Returns a mapping of logical name → written path.
    """
    live_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for cadence in CADENCE_MODES:
        path = monitoring_path(cadence, live_root)
        if path.exists() and not refresh:
            logger.info("Skipping existing %s (pass refresh=True to re-download)", path)
        else:
            logger.info("Downloading monitoring for cadence_mode=%s", cadence)
            df = _download_latest_monitoring(cadence)
            df.to_csv(path, index=False)
            logger.info("Wrote %s (%d rows)", path, len(df))
        written[f"monitoring_{cadence}"] = path

    cycles_path = inference_cycles_path(live_root)
    if cycles_path.exists() and not refresh:
        logger.info("Skipping existing %s", cycles_path)
    else:
        logger.info("Listing inference cycles from MLflow")
        cycles = _list_all_inference_cycles()
        cycles.to_csv(cycles_path, index=False)
        logger.info("Wrote %s (%d rows)", cycles_path, len(cycles))
    written["inference_cycles"] = cycles_path

    fx_path = fixtures_path(live_root)
    if fx_path.exists() and not refresh:
        logger.info("Skipping existing %s", fx_path)
    else:
        fixtures = build_fixtures_frame()
        fixtures.to_csv(fx_path, index=False)
        logger.info("Wrote %s (%d rows)", fx_path, len(fixtures))
    written["fixtures"] = fx_path

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download even when output files already exist",
    )
    parser.add_argument(
        "--live-root",
        type=Path,
        default=LIVE_ROOT,
        help=f"Output directory (default: {LIVE_ROOT})",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    export_live(refresh=args.refresh, live_root=args.live_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
