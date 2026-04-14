"""Log inference and simulation artifacts to MLflow."""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.models.mlflow_utils import (
    get_latest_production_run_id,
    log_run,
    setup_mlflow,
    start_run,
)

logger = logging.getLogger(__name__)


def log_inference_artifacts(
    predictions_df: pd.DataFrame,
    scoreline_dist: pd.DataFrame | None,
    tournament_results: dict[str, Any],
    *,
    n_sims: int,
    gold_row_count: int,
) -> str:
    """Start an MLflow run tagged stage=inference and log all artifacts.

    Logged artifacts:
      - predictions.csv: per-match lambda and outcome probabilities
      - scoreline_distributions.csv: sampled scoreline probabilities
      - tournament_probabilities.csv: per-team advancement by round

    Returns the inference MLflow run_id.
    """
    setup_mlflow()

    champion_run_id = get_latest_production_run_id() or "unknown"

    with start_run(
        run_name="inference",
        tags={"stage": "inference"},
    ) as run:
        log_run(
            params={
                "n_sims": str(n_sims),
                "champion_run_id": champion_run_id,
                "gold_row_count": str(gold_row_count),
                "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            pred_path = tmp / "predictions.csv"
            predictions_df.to_csv(pred_path, index=False)
            mlflow.log_artifact(str(pred_path))

            if scoreline_dist is not None and not scoreline_dist.empty:
                sl_path = tmp / "scoreline_distributions.csv"
                scoreline_dist.to_csv(sl_path, index=False)
                mlflow.log_artifact(str(sl_path))

            advancement = tournament_results.get("advancement")
            if advancement is not None and not advancement.empty:
                tp_path = tmp / "tournament_probabilities.csv"
                advancement.to_csv(tp_path, index=False)
                mlflow.log_artifact(str(tp_path))

        run_id = run.info.run_id

    logger.info("Inference artifacts logged to MLflow run %s", run_id)
    return run_id
