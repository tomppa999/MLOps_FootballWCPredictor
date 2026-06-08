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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import mlflow
import pandas as pd

try:
    from src.models.mlflow_utils import EXPERIMENT_NAME, setup_mlflow
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.models.mlflow_utils import EXPERIMENT_NAME, setup_mlflow


ARTIFACT_FILENAMES: tuple[str, ...] = (
    "tournament_probabilities.csv",
    "group_positions.csv",
    "predictions.csv",
    "scoreline_distributions.csv",
    "ko_pairings.csv",
)

# Artifact keys that are stored in multi-model long format and should be
# filtered to the champion before being handed to the dashboard.
_MULTI_MODEL_ARTIFACTS: frozenset[str] = frozenset({
    "tournament_probabilities",
    "group_positions",
    "ko_pairings",
})


@dataclass(frozen=True)
class InferenceRunInfo:
    """Metadata for the latest inference run used by the dashboard."""

    run_id: str
    n_sims: int | None
    champion_run_id: str | None
    champion_model_name: str | None
    inference_timestamp: str | None


def _filter_to_champion(df: pd.DataFrame, champion_model_name: str) -> pd.DataFrame:
    """Return rows matching the champion and drop the model_name column.

    Backward-compatible: if the DataFrame has no ``model_name`` column
    (pre-Option-A runs), it is returned unchanged.
    """
    if "model_name" not in df.columns:
        return df
    filtered = df[df["model_name"] == champion_model_name].drop(columns=["model_name"])
    return filtered.reset_index(drop=True)


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


def load_latest_inference_artifacts() -> tuple[dict[str, pd.DataFrame], InferenceRunInfo]:
    """Download CSV artifacts from the latest inference run.

    Tournament-related artifacts (tournament_probabilities, group_positions,
    ko_pairings) are filtered to the champion model before being returned,
    so the Streamlit app sees single-model data exactly as before Option A.

    Returns:
        A tuple of:
          - mapping of base artifact name (without .csv) to DataFrame.
          - ``InferenceRunInfo`` with basic provenance metadata.
    """
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
    )
    return dfs, info


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
