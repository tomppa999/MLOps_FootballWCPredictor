"""MLflow helpers: experiment setup, run tagging, model registry."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion

logger = logging.getLogger(__name__)

TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME: str = "wc_prediction"
STAGING_MODEL_NAME: str = "wc_staging"
PRODUCTION_MODEL_NAME: str = "wc_production"


def setup_mlflow(tracking_uri: str = TRACKING_URI) -> None:
    """Point MLflow at the local file store."""
    mlflow.set_tracking_uri(tracking_uri)


def get_or_create_experiment(name: str = EXPERIMENT_NAME) -> str:
    """Return the experiment ID, creating the experiment if needed."""
    setup_mlflow()
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    return mlflow.create_experiment(name)


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def start_run(
    *,
    experiment_name: str = EXPERIMENT_NAME,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> mlflow.ActiveRun:
    """Start an MLflow run with standard setup.

    Use as a context manager::

        with start_run(run_name="ridge_trial_0", tags={"stage": "experimental"}):
            mlflow.log_params(...)
            ...
    """
    experiment_id = get_or_create_experiment(experiment_name)
    return mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,
        nested=nested,
    )


def log_run(
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    """Log params, metrics, and tags to the active run."""
    if params:
        mlflow.log_params(params)
    if metrics:
        mlflow.log_metrics(metrics)
    if tags:
        mlflow.set_tags(tags)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def register_model(
    run_id: str,
    artifact_path: str = "model",
    *,
    model_name: str,
) -> ModelVersion:
    """Register a logged model artifact in the MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri, model_name)
    logger.info("Registered %s version %s", model_name, mv.version)
    return mv


def promote_to_production(
    model_name: str = PRODUCTION_MODEL_NAME,
    version: int | str = 1,
) -> None:
    """Set the 'champion' alias on a registered model version."""
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(model_name, "champion", str(version))
    logger.info("Promoted %s v%s to champion", model_name, version)


def set_challenger_alias(
    model_name: str = STAGING_MODEL_NAME,
    version: int | str = 1,
) -> None:
    """Set the 'challenger' alias on a registered model version."""
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(model_name, "challenger", str(version))
    logger.info("Set challenger alias on %s v%s", model_name, version)


def load_champion(model_name: str = PRODUCTION_MODEL_NAME) -> Any:
    """Load the current champion model from the registry."""
    return mlflow.pyfunc.load_model(f"models:/{model_name}@champion")


def get_latest_production_run_id(model_name: str = PRODUCTION_MODEL_NAME) -> str | None:
    """Return the run_id of the current champion model version, or None.

    Returns None when no champion alias exists or the model is not yet
    registered (MLflow raises MlflowException in that case).
    """
    client = mlflow.tracking.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
    except mlflow.exceptions.MlflowException:
        return None
    return mv.run_id


def get_champion_rps(model_name: str = PRODUCTION_MODEL_NAME) -> float | None:
    """Return the qa_holdout_rps of the current champion, or None.

    Returns None when no champion exists (first run — always promote).
    """
    run_id = get_latest_production_run_id(model_name)
    if run_id is None:
        return None
    client = mlflow.tracking.MlflowClient()
    metrics = client.get_run(run_id).data.metrics
    return metrics.get("qa_holdout_rps")
