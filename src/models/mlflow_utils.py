"""MLflow helpers: experiment setup, run tagging, model registry."""

from __future__ import annotations

import logging
import os
from typing import Any, NamedTuple

import mlflow
from mlflow.entities.model_registry import ModelVersion

logger = logging.getLogger(__name__)

TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME: str = "wc_prediction"
STAGING_MODEL_NAME: str = "wc_staging"
PRODUCTION_MODEL_NAME: str = "wc_production"
SHADOW_MODEL_NAME: str = "wc_shadow"


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
    model_uri: str,
    *,
    model_name: str,
) -> ModelVersion:
    """Register a logged model artifact in the MLflow Model Registry.

    Args:
        model_uri: URI returned by ``mlflow.pyfunc.log_model`` (typically
            ``models:/<model_id>`` in MLflow 3.x).
        model_name: Registered model name in the registry.
    """
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


# ---------------------------------------------------------------------------
# Champion metadata
# ---------------------------------------------------------------------------

_DEPLOY_INTERNAL_PARAMS: frozenset[str] = frozenset({"evaluation_run_id", "gold_row_count"})


class ChampionMeta(NamedTuple):
    """Champion model identity, hyperparameters, and holdout metrics."""

    model_name: str
    best_params: dict[str, Any]
    holdout_metrics: dict[str, float]


def _cast_params(model_name: str, raw_params: dict[str, str]) -> dict[str, Any]:
    """Cast MLflow string params back to typed values using SEARCH_SPACES."""
    from src.models.config import SEARCH_SPACES  # local import avoids circular dep

    space = SEARCH_SPACES.get(model_name, {})
    result: dict[str, Any] = {}
    for k, v in raw_params.items():
        spec = space.get(k)
        if spec is None:
            result[k] = v
            continue
        t = spec["type"]
        if t == "int":
            result[k] = int(v)
        elif t == "float":
            result[k] = float(v)
        elif t == "categorical":
            for choice in spec.get("choices", []):
                if str(choice) == v:
                    result[k] = choice
                    break
            else:
                result[k] = v
        else:
            result[k] = v
    return result


def get_champion_metadata(
    model_name: str = PRODUCTION_MODEL_NAME,
) -> ChampionMeta:
    """Return identity, hyperparameters, and holdout metrics of the champion.

    Raises:
        ValueError: if no champion exists or required metadata is missing.
    """
    run_id = get_latest_production_run_id(model_name)
    if run_id is None:
        raise ValueError(f"No champion found for registered model '{model_name}'")
    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run_id).data
    champion_model_name = run_data.tags.get("model_name")
    if not champion_model_name:
        raise ValueError(f"Champion run {run_id} has no 'model_name' tag")
    raw_params = {
        k: v
        for k, v in run_data.params.items()
        if k not in _DEPLOY_INTERNAL_PARAMS
    }
    best_params = _cast_params(champion_model_name, raw_params)
    holdout_metrics = {
        k: v
        for k, v in run_data.metrics.items()
        if k.startswith("qa_holdout_")
    }
    return ChampionMeta(
        model_name=champion_model_name,
        best_params=best_params,
        holdout_metrics=holdout_metrics,
    )


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


# ---------------------------------------------------------------------------
# Shadow metadata (wc_shadow with cold-start fallback to wc_staging)
# ---------------------------------------------------------------------------


def _latest_version_with_model_tag(
    registered_model: str,
    model_name: str,
) -> ModelVersion | None:
    """Return the most recently created version of ``registered_model`` whose
    backing run has tag ``model_name=<x>``.

    Walks model versions newest-first and inspects each underlying run.
    Returns ``None`` if no matching version exists.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{registered_model}'")
    except mlflow.exceptions.MlflowException:
        return None

    versions = sorted(
        versions,
        key=lambda mv: int(mv.version),
        reverse=True,
    )
    for mv in versions:
        try:
            run = client.get_run(mv.run_id)
        except mlflow.exceptions.MlflowException:
            continue
        if run.data.tags.get("model_name") == model_name:
            return mv
    return None


def get_shadow_metadata(
    model_name: str,
    *,
    shadow_model: str = SHADOW_MODEL_NAME,
    staging_model: str = STAGING_MODEL_NAME,
) -> ChampionMeta:
    """Return identity, best_params, and holdout metrics for a shadow candidate.

    Looks up the latest ``wc_shadow`` version with tag ``model_name=<x>``.
    Falls back to the latest ``wc_staging`` version with the same tag if no
    shadow version exists yet (cold-start, before the first backfill).

    Mirrors :func:`get_champion_metadata` so integer hyperparameters survive
    the MLflow string round-trip via :func:`_cast_params`.

    Raises:
        ValueError: if neither registry surfaces a version with that tag.
    """
    mv = _latest_version_with_model_tag(shadow_model, model_name)
    source = shadow_model
    if mv is None:
        mv = _latest_version_with_model_tag(staging_model, model_name)
        source = staging_model
    if mv is None:
        raise ValueError(
            f"No registered version found for model_name='{model_name}' in "
            f"either '{shadow_model}' or '{staging_model}'."
        )

    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(mv.run_id).data
    raw_params = {
        k: v
        for k, v in run_data.params.items()
        if k not in _DEPLOY_INTERNAL_PARAMS
    }
    best_params = _cast_params(model_name, raw_params)
    holdout_metrics = {
        k: v
        for k, v in run_data.metrics.items()
        if k.startswith("qa_holdout_") or k.startswith("holdout_")
    }
    logger.info(
        "Loaded shadow metadata for %s from %s v%s (run=%s)",
        model_name,
        source,
        mv.version,
        mv.run_id,
    )
    return ChampionMeta(
        model_name=model_name,
        best_params=best_params,
        holdout_metrics=holdout_metrics,
    )


def get_all_shadow_metadata(
    candidate_names: list[str],
    *,
    exclude_model_name: str | None = None,
) -> list[ChampionMeta]:
    """Return one ChampionMeta per non-excluded candidate.

    Args:
        candidate_names: All registered candidate model names (e.g. the
            keys of ``CANDIDATE_MODELS``).
        exclude_model_name: Optional name to skip — typically the current
            champion, which is already refit by ``run_champion_refit``.
    """
    out: list[ChampionMeta] = []
    for name in candidate_names:
        if exclude_model_name is not None and name == exclude_model_name:
            continue
        out.append(get_shadow_metadata(name))
    return out


def load_shadow_model(
    model_name: str,
    *,
    shadow_model: str = SHADOW_MODEL_NAME,
    staging_model: str = STAGING_MODEL_NAME,
) -> Any:
    """Load the latest shadow pyfunc artifact for the given candidate.

    Falls back to the latest ``wc_staging`` version with the same tag if no
    shadow version exists yet (cold-start). Mirrors the resolution order
    used by :func:`get_shadow_metadata`.
    """
    mv = _latest_version_with_model_tag(shadow_model, model_name)
    if mv is None:
        mv = _latest_version_with_model_tag(staging_model, model_name)
    if mv is None:
        raise ValueError(
            f"No registered version found for model_name='{model_name}' in "
            f"either '{shadow_model}' or '{staging_model}'."
        )
    return mlflow.pyfunc.load_model(f"runs:/{mv.run_id}/model")
