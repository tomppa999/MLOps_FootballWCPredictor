"""Helpers to load inference artifacts from the latest MLflow run.

This mirrors the pattern used in ``src/models/plot_feature_importance.py`` but
targets the inference runs (tags.stage = "inference") and downloads the
dashboard-relevant CSV artifacts.

Tournament artifacts (tournament_probabilities, group_positions, ko_pairings)
are stored in multi-model long format (one row per model_name × entity) since
Option A (A.9).  ``load_latest_inference_artifacts`` filters them to the
champion automatically, so the Streamlit app requires no changes.  Pre-Option-A
runs that lack a ``model_name`` column are passed through unchanged
(backward-compatible).

Offline fallback: every successful MLflow load is persisted to
``_offline_cache/`` (CSVs + ``_meta.json``) so the dashboard stays functional
when DagsHub is in maintenance.  The ``is_stale`` flag on ``InferenceRunInfo``
signals that the cache was used.  The ``@st.cache_data(ttl=300)`` wrapper means
the live path is retried automatically every 5 minutes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import mlflow
import pandas as pd
import streamlit as st

try:
    from src.models.mlflow_utils import EXPERIMENT_NAME, setup_mlflow
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.models.mlflow_utils import EXPERIMENT_NAME, setup_mlflow

logger = logging.getLogger(__name__)

ARTIFACT_FILENAMES: tuple[str, ...] = (
    "tournament_probabilities.csv",
    "group_positions.csv",
    "predictions.csv",
    "scoreline_distributions.csv",
    "ko_pairings.csv",
    "ko_fixtures.csv",
)

# Artifact keys that are stored in multi-model long format and should be
# filtered to the champion before being handed to the dashboard.
_MULTI_MODEL_ARTIFACTS: frozenset[str] = frozenset({
    "tournament_probabilities",
    "group_positions",
    "ko_pairings",
})

# Disk cache lives next to this file so it is committed with the repo and
# survives Streamlit Cloud container restarts.
_CACHE_DIR = Path(__file__).parent / "_offline_cache"
_META_FILE = _CACHE_DIR / "_meta.json"
_PRETOURNAMENT_DIR = Path(__file__).parent / "_pretournament"

_MONITORING_ARTIFACT_FILENAME = "wc2026_monitoring.csv"
_MONITORING_CACHE_FILE = _CACHE_DIR / "wc2026_monitoring.csv"


@dataclass(frozen=True)
class InferenceRunInfo:
    """Metadata for the latest inference run used by the dashboard."""

    run_id: str
    n_sims: int | None
    champion_run_id: str | None
    champion_model_name: str | None
    inference_timestamp: str | None
    is_stale: bool = False


def _filter_to_champion(df: pd.DataFrame, champion_model_name: str) -> pd.DataFrame:
    """Return rows matching the champion and drop the model_name column.

    Backward-compatible: if the DataFrame has no ``model_name`` column
    (pre-Option-A runs), it is returned unchanged.
    """
    if "model_name" not in df.columns:
        return df
    filtered = df[df["model_name"] == champion_model_name].drop(columns=["model_name"])
    return filtered.reset_index(drop=True)


def _write_offline_cache(
    dfs: dict[str, pd.DataFrame],
    info: InferenceRunInfo,
) -> None:
    """Persist champion-filtered DataFrames + metadata to _offline_cache/."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for key, df in dfs.items():
            df.to_csv(_CACHE_DIR / f"{key}.csv", index=False)
        meta = {
            "run_id": info.run_id,
            "n_sims": info.n_sims,
            "champion_run_id": info.champion_run_id,
            "champion_model_name": info.champion_model_name,
            "inference_timestamp": info.inference_timestamp,
        }
        _META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Offline cache written to %s", _CACHE_DIR)
    except Exception:
        logger.warning("Failed to write offline cache — live data unaffected.", exc_info=True)


def _read_offline_cache() -> tuple[dict[str, pd.DataFrame], InferenceRunInfo] | None:
    """Load cached DataFrames + metadata from _offline_cache/, or None if absent."""
    if not _META_FILE.exists():
        return None
    try:
        meta = json.loads(_META_FILE.read_text(encoding="utf-8"))
        dfs: dict[str, pd.DataFrame] = {}
        for csv_path in _CACHE_DIR.glob("*.csv"):
            dfs[csv_path.stem] = pd.read_csv(csv_path)
        info = InferenceRunInfo(
            run_id=meta.get("run_id", "offline"),
            n_sims=meta.get("n_sims"),
            champion_run_id=meta.get("champion_run_id"),
            champion_model_name=meta.get("champion_model_name"),
            inference_timestamp=meta.get("inference_timestamp"),
            is_stale=True,
        )
        logger.info("Offline cache loaded from %s", _CACHE_DIR)
        return dfs, info
    except Exception:
        logger.warning("Failed to read offline cache.", exc_info=True)
        return None


def _get_latest_inference_run() -> mlflow.entities.Run:
    """Return the most recent frozen-lineage inference run for the dashboard.

    Prefers runs with ``params.cadence_mode = frozen`` (B.2 display lineage).
    Falls back to the latest inference run when no tagged run exists (pre-B.1).
    """
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found")

    frozen_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(
            'tags.stage = "inference" AND params.cadence_mode = "frozen"'
        ),
        order_by=["start_time DESC"],
        max_results=1,
    )
    if frozen_runs:
        return frozen_runs[0]

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string='tags.stage = "inference"',
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No inference runs found in MLflow (tags.stage = 'inference').")
    return runs[0]


def _load_from_mlflow() -> tuple[dict[str, pd.DataFrame], InferenceRunInfo]:
    """Core MLflow fetch — separated so the cache wrapper can call it cleanly."""
    run = _get_latest_inference_run()
    client = mlflow.tracking.MlflowClient()

    artifact_dir = Path(client.download_artifacts(run.info.run_id, ""))

    params: dict[str, Any] = run.data.params
    champion_model_name: str | None = params.get("champion_model_name") or None

    dfs: dict[str, pd.DataFrame] = {}
    for filename in ARTIFACT_FILENAMES:
        path = artifact_dir / filename
        if not path.exists():
            continue
        key = path.stem
        df = pd.read_csv(path)
        if key in _MULTI_MODEL_ARTIFACTS and champion_model_name:
            df = _filter_to_champion(df, champion_model_name)
        dfs[key] = df

    info = InferenceRunInfo(
        run_id=run.info.run_id,
        n_sims=int(params["n_sims"]) if "n_sims" in params else None,
        champion_run_id=params.get("champion_run_id"),
        champion_model_name=champion_model_name,
        inference_timestamp=params.get("inference_timestamp"),
        is_stale=False,
    )
    return dfs, info


@st.cache_data(ttl=300)
def load_latest_inference_artifacts() -> tuple[dict[str, pd.DataFrame], InferenceRunInfo]:
    """Download CSV artifacts from the latest inference run.

    Tournament-related artifacts (tournament_probabilities, group_positions,
    ko_pairings) are filtered to the champion model before being returned,
    so the Streamlit app sees single-model data exactly as before Option A.

    Falls back to the committed offline cache when DagsHub is unreachable.
    The ``is_stale`` flag on the returned ``InferenceRunInfo`` signals that
    the cache was used.  The ``ttl=300`` means the live path is retried every
    5 minutes and recovers automatically once the tracking server is back.

    Returns:
        A tuple of:
          - mapping of base artifact name (without .csv) to DataFrame.
          - ``InferenceRunInfo`` with basic provenance metadata.
    """
    try:
        dfs, info = _load_from_mlflow()
        _write_offline_cache(dfs, info)
        return dfs, info
    except Exception:
        logger.warning("MLflow unreachable — attempting offline cache.", exc_info=True)
        cached = _read_offline_cache()
        if cached is None:
            raise
        return cached


def _load_monitoring_from_mlflow(cadence_mode: str = "frozen") -> pd.DataFrame:
    """Download wc2026_monitoring.csv from the latest monitoring run."""
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
    local_path = client.download_artifacts(
        runs[0].info.run_id, _MONITORING_ARTIFACT_FILENAME,
    )
    return pd.read_csv(local_path)


@st.cache_data(ttl=300)
def load_latest_monitoring_results(cadence_mode: str = "frozen") -> "pd.DataFrame | None":
    """Latest pre-kickoff predictions + actual results for settled WC matches.

    Returns None (never raises) so the dashboard degrades gracefully
    pre-WC or during a DagsHub outage. Falls back to the committed
    offline cache when MLflow is unreachable.
    """
    try:
        df = _load_monitoring_from_mlflow(cadence_mode)
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(_MONITORING_CACHE_FILE, index=False)
        except Exception:
            logger.warning("Failed to cache monitoring artifact.", exc_info=True)
        return df
    except Exception:
        logger.warning("Monitoring artifact unavailable — trying offline cache.", exc_info=True)
        if _MONITORING_CACHE_FILE.exists():
            try:
                return pd.read_csv(_MONITORING_CACHE_FILE)
            except Exception:
                pass
        return None


def load_pretournament_snapshot(name: str) -> pd.DataFrame | None:
    """Load a committed pre-tournament snapshot CSV, or None if absent."""
    path = _PRETOURNAMENT_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def load_group_mapping(config_path: Path | str = Path("data/tournament/wc2026.json")) -> dict[str, str]:
    """Return a mapping of team -> group letter from the tournament config."""
    import json

    path = Path(config_path)
    with path.open() as f:
        config = json.load(f)

    mapping: dict[str, str] = {}
    groups: dict[str, list[str]] = config.get("groups", {})
    for group_letter, teams in groups.items():
        for team in teams:
            mapping[team] = group_letter
    return mapping
