"""MLflow helpers: experiment setup, run tagging, model registry."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion

logger = logging.getLogger(__name__)

TRACKING_URI: str = "file:./mlruns"
EXPERIMENT_NAME: str = "wc_prediction"
REGISTRY_MODEL_NAME: str = "wc_champion"


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
    model_name: str = REGISTRY_MODEL_NAME,
) -> ModelVersion:
    """Register a logged model artifact in the MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri, model_name)
    logger.info("Registered %s version %s", model_name, mv.version)
    return mv


def promote_to_production(
    model_name: str = REGISTRY_MODEL_NAME,
    version: int | str = 1,
) -> None:
    """Transition a model version to the Production stage."""
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info("Promoted %s v%s to Production", model_name, version)


def load_champion(model_name: str = REGISTRY_MODEL_NAME) -> Any:
    """Load the current Production model from the registry."""
    return mlflow.pyfunc.load_model(f"models:/{model_name}/Production")


def get_latest_production_run_id(model_name: str = REGISTRY_MODEL_NAME) -> str | None:
    """Return the run_id of the current Production model version, or None.

    Returns None when no production version exists or the model is not yet
    registered (MLflow 3.x raises MlflowException in that case).
    """
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except mlflow.exceptions.MlflowException:
        return None
    if versions:
        return versions[0].run_id
    return None


def get_champion_rps(model_name: str = REGISTRY_MODEL_NAME) -> float | None:
    """Return the qa_holdout_rps of the current Production champion, or None.

    Returns None when no production model exists (first run — always promote).
    """
    run_id = get_latest_production_run_id(model_name)
    if run_id is None:
        return None
    client = mlflow.tracking.MlflowClient()
    metrics = client.get_run(run_id).data.metrics
    return metrics.get("qa_holdout_rps")
