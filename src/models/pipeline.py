"""Three-phase training pipeline: experimental → QA → deploy.

Phase 1 (Experimental): Tune every candidate via walk-forward CV.  Top-K advance.
Phase 2 (QA):           Retrain top-K on pre-WC data, evaluate on WC 2022 holdout.
Phase 3 (Deploy):       Refit the winner on *all* Gold data, register, promote.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.models.candidates.bayesian_poisson import BayesianPoissonModel
from src.models.candidates.cnn import CNNModel
from src.models.candidates.lstm import LSTMModel
from src.models.candidates.negbin_glm import NegativeBinomialGLM
from src.models.candidates.poisson_glm import BivariatePoisson
from src.models.candidates.random_forest import RandomForestModel
from src.models.candidates.ridge import RidgeModel
from src.models.candidates.sarimax import SARIMAXModel
from src.models.candidates.xgboost_model import XGBoostModel
from src.models.config import DEFAULT_N_TRIALS, MODEL_FEATURE_SETS, SEARCH_SPACES
from src.models.data_split import DataSplits, load_gold, make_splits, walk_forward_cv
from src.models.evaluation import (
    compute_mean_nll,
    compute_mean_rps,
    compute_permutation_importance,
    compute_rmse,
)
from src.models.mlflow_utils import (
    PRODUCTION_MODEL_NAME,
    STAGING_MODEL_NAME,
    get_champion_metadata,
    get_champion_rps,
    log_run,
    promote_to_production,
    register_model,
    set_challenger_alias,
    setup_mlflow,
    start_run,
)
from src.models.tuning import run_tuning

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Candidate registry (9 models)
# ---------------------------------------------------------------------------

CANDIDATE_MODELS: dict[str, type[BaseModel]] = {
    "poisson_glm": BivariatePoisson,
    "negbin_glm": NegativeBinomialGLM,
    "ridge": RidgeModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "bayesian_poisson": BayesianPoissonModel,
    "sarimax": SARIMAXModel,
    "lstm": LSTMModel,
    "cnn": CNNModel,
}

RETRAIN_THRESHOLD: int = 10

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ExperimentalResult:
    """Carries tuning output for one candidate into the QA phase."""

    model_name: str
    model_cls: type[BaseModel]
    best_params: dict[str, Any]
    cv_nll: float
    importance: pd.DataFrame
    splits: DataSplits
    feature_cols: list[str]


@dataclass
class QAResult:
    """QA-phase output for one candidate, with holdout metrics."""

    model_name: str
    model_cls: type[BaseModel]
    best_params: dict[str, Any]
    cv_nll: float
    holdout_rps: float
    holdout_nll: float
    holdout_rmse_home: float
    holdout_rmse_away: float
    qa_run_id: str
    splits: DataSplits
    feature_cols: list[str]


# ---------------------------------------------------------------------------
# Model artifact helper (pyfunc wrapper for uniform MLflow serialisation)
# ---------------------------------------------------------------------------


class _ModelWrapper(mlflow.pyfunc.PythonModel):
    """Thin pyfunc wrapper so any BaseModel can be registered in MLflow."""

    def __init__(self, model: BaseModel) -> None:
        self.model = model

    def predict(  # noqa: ANN201
        self,
        context: Any,  # noqa: ARG002
        model_input: Any,
        params: dict | None = None,  # noqa: ARG002
    ):
        X = (
            model_input.values
            if hasattr(model_input, "values")
            else np.asarray(model_input)
        )
        lam_h, lam_a = self.model.predict(X)
        return np.column_stack([lam_h, lam_a])


def _log_model_artifact(model: BaseModel) -> str:
    """Log a fitted BaseModel as an MLflow pyfunc model artifact.

    Returns the model_uri needed for registration (MLflow 3.x stores
    model artifacts under a ``models:/`` namespace, not under the run).
    """
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=_ModelWrapper(model),
    )
    return model_info.model_uri


# ---------------------------------------------------------------------------
# Phase 1 — Experimental (tuning + top-K selection)
# ---------------------------------------------------------------------------


def run_experimental_phase(
    df: pd.DataFrame,
    *,
    n_trials_override: int | None = None,
    models: dict[str, type[BaseModel]] | None = None,
    pipeline_run_id: str | None = None,
) -> list[ExperimentalResult]:
    """Tune each candidate via walk-forward CV; return all sorted by CV NLL."""
    setup_mlflow()
    candidates = models or CANDIDATE_MODELS
    results: list[ExperimentalResult] = []

    for model_name, model_cls in candidates.items():
        feature_cols = MODEL_FEATURE_SETS[model_name]
        dropna = model_name != "xgboost"
        splits = make_splits(df, feature_cols, dropna=dropna)
        cv_folds = walk_forward_cv(len(splits.X_train))
        search_space = SEARCH_SPACES[model_name]
        n_trials = n_trials_override or DEFAULT_N_TRIALS[model_name]

        logger.info("Tuning %s (%d trials)…", model_name, n_trials)
        best_params, study = run_tuning(
            model_cls,
            search_space,
            splits.X_train,
            splits.y_train,
            cv_folds,
            n_trials=n_trials,
            pipeline_run_id=pipeline_run_id,
        )

        model = model_cls(**best_params)
        model.fit(splits.X_train, splits.y_train)
        importance = compute_permutation_importance(
            model, splits.X_train, splits.y_train, feature_cols, n_repeats=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / f"importance_{model_name}.csv"
            importance.to_csv(csv_path, index=False)
            best_tags: dict[str, str] = {
                "stage": "experimental",
                "model_name": model_name,
                "role": "best",
            }
            if pipeline_run_id:
                best_tags["pipeline_run_id"] = pipeline_run_id
            with start_run(
                run_name=f"best_{model_name}",
                tags=best_tags,
            ):
                log_run(
                    params=best_params,
                    metrics={"cv_nll": study.best_value},
                )
                mlflow.log_artifact(str(csv_path), artifact_path="importance")

        results.append(
            ExperimentalResult(
                model_name=model_name,
                model_cls=model_cls,
                best_params=best_params,
                cv_nll=study.best_value,
                importance=importance,
                splits=splits,
                feature_cols=feature_cols,
            )
        )
        logger.info("%s — best CV NLL: %.4f", model_name, study.best_value)

    results.sort(key=lambda r: r.cv_nll)
    _log_importance_summary(results)
    logger.info(
        "Experimental done — all %d models advance: %s",
        len(results),
        [(r.model_name, f"{r.cv_nll:.4f}") for r in results],
    )
    return results


def _log_importance_summary(results: list[ExperimentalResult]) -> None:
    """Print a top-5 feature importance summary for every model."""
    logger.info("=== Feature Importance Summary (top 5 per model) ===")
    for r in results:
        top5 = r.importance.head(5)
        lines = [
            f"  {row.feature:>30s}  {row.importance_mean:+.6f} ± {row.importance_std:.6f}"
            for row in top5.itertuples()
        ]
        logger.info(
            "%s (CV NLL: %.4f):\n%s", r.model_name, r.cv_nll, "\n".join(lines),
        )


# ---------------------------------------------------------------------------
# Phase 2 — QA (holdout evaluation on WC 2022)
# ---------------------------------------------------------------------------


def run_qa_phase(
    top_models: list[ExperimentalResult],
    *,
    pipeline_run_id: str | None = None,
) -> QAResult:
    """Retrain top models on pre-WC data; pick best by WC 2022 holdout RPS.

    Each finalist is serialized and registered as a version of wc_staging
    in the MLflow model registry.  The winner receives the ``challenger``
    alias.
    """
    setup_mlflow()
    qa_results: list[QAResult] = []
    run_id_to_version: dict[str, str] = {}

    for entry in top_models:
        model = entry.model_cls(**entry.best_params)
        model.fit(entry.splits.X_train, entry.splits.y_train)

        lam_h, lam_a = model.predict(entry.splits.X_holdout)
        holdout_rps = compute_mean_rps(
            lam_h,
            lam_a,
            entry.splits.y_holdout[:, 0],
            entry.splits.y_holdout[:, 1],
        )
        holdout_nll = compute_mean_nll(
            lam_h,
            lam_a,
            entry.splits.y_holdout[:, 0],
            entry.splits.y_holdout[:, 1],
        )
        rmse_h = compute_rmse(lam_h, entry.splits.y_holdout[:, 0])
        rmse_a = compute_rmse(lam_a, entry.splits.y_holdout[:, 1])

        qa_tags: dict[str, str] = {"stage": "qa", "model_name": entry.model_name}
        if pipeline_run_id:
            qa_tags["pipeline_run_id"] = pipeline_run_id
        with start_run(
            run_name=f"qa_{entry.model_name}",
            tags=qa_tags,
        ) as run:
            log_run(
                params=entry.best_params,
                metrics={
                    "holdout_rps": holdout_rps,
                    "holdout_nll": holdout_nll,
                    "holdout_rmse_home": rmse_h,
                    "holdout_rmse_away": rmse_a,
                    "cv_nll": entry.cv_nll,
                },
            )
            model_uri = _log_model_artifact(model)
            qa_run_id = run.info.run_id

        qa_mv = register_model(model_uri=model_uri, model_name=STAGING_MODEL_NAME)
        run_id_to_version[qa_run_id] = qa_mv.version

        qa_results.append(
            QAResult(
                model_name=entry.model_name,
                model_cls=entry.model_cls,
                best_params=entry.best_params,
                cv_nll=entry.cv_nll,
                holdout_rps=holdout_rps,
                holdout_nll=holdout_nll,
                holdout_rmse_home=rmse_h,
                holdout_rmse_away=rmse_a,
                qa_run_id=qa_run_id,
                splits=entry.splits,
                feature_cols=entry.feature_cols,
            )
        )
        logger.info("%s — holdout RPS: %.4f, NLL: %.4f", entry.model_name, holdout_rps, holdout_nll)

    qa_results.sort(key=lambda r: r.holdout_rps)
    winner = qa_results[0]
    set_challenger_alias(version=run_id_to_version[winner.qa_run_id])
    logger.info(
        "QA winner: %s (holdout RPS: %.4f)",
        winner.model_name,
        winner.holdout_rps,
    )
    return winner


# ---------------------------------------------------------------------------
# Phase 3 — Deploy (production refit + registry promotion)
# ---------------------------------------------------------------------------


class ChallengeFailed(Exception):
    """Raised when the challenger does not beat the current production champion."""


def run_deploy_phase(
    winner: QAResult,
    *,
    pipeline_run_id: str | None = None,
) -> str:
    """Refit winner on all Gold data, serialize, register, and promote.

    Raises:
        ChallengeFailed: if a production champion already exists and the
            challenger's holdout RPS does not improve on it.

    Returns:
        The MLflow run_id of the production-refit run.
    """
    setup_mlflow()

    champion_rps = get_champion_rps()
    if champion_rps is not None and winner.holdout_rps > champion_rps:
        raise ChallengeFailed(
            f"Challenger {winner.model_name} holdout RPS {winner.holdout_rps:.4f} "
            f"does not beat champion RPS {champion_rps:.4f} — promotion skipped."
        )
    if champion_rps is None:
        logger.info("No existing champion — first-run promotion.")
    else:
        logger.info(
            "Challenger %s (%.4f) beats champion (%.4f) — promoting.",
            winner.model_name,
            winner.holdout_rps,
            champion_rps,
        )

    model = winner.model_cls(**winner.best_params)
    model.fit(winner.splits.X_full, winner.splits.y_full)

    deploy_tags: dict[str, str] = {
        "stage": "production-refit",
        "model_name": winner.model_name,
    }
    if pipeline_run_id:
        deploy_tags["pipeline_run_id"] = pipeline_run_id
    with start_run(
        run_name=f"deploy_{winner.model_name}",
        tags=deploy_tags,
    ) as run:
        log_run(
            params={
                **winner.best_params,
                "evaluation_run_id": winner.qa_run_id,
                "gold_row_count": str(len(winner.splits.df_full)),
            },
            metrics={
                "qa_holdout_rps": winner.holdout_rps,
                "qa_holdout_nll": winner.holdout_nll,
                "qa_holdout_rmse_home": winner.holdout_rmse_home,
                "qa_holdout_rmse_away": winner.holdout_rmse_away,
            },
        )
        model_uri = _log_model_artifact(model)
        run_id = run.info.run_id

    mv = register_model(model_uri=model_uri, model_name=PRODUCTION_MODEL_NAME)
    promote_to_production(version=mv.version)

    logger.info(
        "Deployed %s (run_id=%s, version=%s)",
        winner.model_name,
        run_id,
        mv.version,
    )
    return run_id


# ---------------------------------------------------------------------------
# Champion refit (skip experimental + QA; just refit the winner on fresh data)
# ---------------------------------------------------------------------------


def run_champion_refit(df: pd.DataFrame) -> str:
    """Refit the current production champion on all available Gold data.

    Reads the champion's model class and hyperparameters from MLflow, refits
    on the full Gold dataset, then registers and promotes the new version.
    The WC 2022 holdout metrics from the original champion run are forwarded
    unchanged — re-evaluating after fitting on all data would be misleading
    because the model has then seen the holdout rows.

    Returns:
        The MLflow run_id of the refit run.
    """
    setup_mlflow()
    meta = get_champion_metadata()
    model_cls = CANDIDATE_MODELS[meta.model_name]
    feature_cols = MODEL_FEATURE_SETS[meta.model_name]
    dropna = meta.model_name != "xgboost"
    splits = make_splits(df, feature_cols, dropna=dropna)

    logger.info(
        "Champion refit: fitting %s on %d Gold rows.",
        meta.model_name,
        len(splits.df_full),
    )

    model = model_cls(**meta.best_params)
    model.fit(splits.X_full, splits.y_full)

    with start_run(
        run_name=f"refit_{meta.model_name}",
        tags={"stage": "champion-refit", "model_name": meta.model_name},
    ) as run:
        log_run(
            params={**meta.best_params, "gold_row_count": str(len(splits.df_full))},
            metrics=meta.holdout_metrics,
        )
        model_uri = _log_model_artifact(model)
        run_id = run.info.run_id

    mv = register_model(model_uri=model_uri, model_name=PRODUCTION_MODEL_NAME)
    promote_to_production(version=mv.version)

    logger.info(
        "Champion refit complete: %s (run_id=%s, version=%s, gold_rows=%d)",
        meta.model_name,
        run_id,
        mv.version,
        len(splits.df_full),
    )
    return run_id


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


def run_full_pipeline(
    df: pd.DataFrame | None = None,
    *,
    n_trials_override: int | None = None,
    models: dict[str, type[BaseModel]] | None = None,
) -> str:
    """Run experimental → QA → deploy.  Returns the production run_id."""
    if df is None:
        df = load_gold()

    pipeline_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info("Pipeline run ID: %s", pipeline_run_id)

    logger.info("=== Phase 1: Experimental ===")
    top_k = run_experimental_phase(
        df,
        n_trials_override=n_trials_override,
        models=models,
        pipeline_run_id=pipeline_run_id,
    )

    logger.info("=== Phase 2: QA ===")
    winner = run_qa_phase(top_k, pipeline_run_id=pipeline_run_id)

    logger.info("=== Phase 3: Deploy ===")
    try:
        run_id = run_deploy_phase(winner, pipeline_run_id=pipeline_run_id)
    except ChallengeFailed as exc:
        logger.info(str(exc))
        return winner.qa_run_id

    logger.info("=== Pipeline complete (run_id=%s) ===", run_id)
    return run_id
