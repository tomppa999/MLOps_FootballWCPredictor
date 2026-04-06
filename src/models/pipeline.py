"""Three-phase training pipeline: experimental → QA → deploy.

Phase 1 (Experimental): Tune every candidate via walk-forward CV.  Top-K advance.
Phase 2 (QA):           Retrain top-K on pre-WC data, evaluate on WC 2022 holdout.
Phase 3 (Deploy):       Refit the winner on *all* Gold data, register, promote.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
    compute_mean_rps,
    compute_permutation_importance,
    compute_rmse,
)
from src.models.mlflow_utils import (
    get_champion_rps,
    log_run,
    promote_to_production,
    register_model,
    setup_mlflow,
    start_run,
)
from src.models.tuning import run_tuning

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Candidate registry (all 9 models)
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

TOP_K_FOR_QA: int = 4
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
    cv_rps: float
    importance: pd.DataFrame
    splits: DataSplits
    feature_cols: list[str]


@dataclass
class QAResult:
    """QA-phase output for one candidate, with holdout metrics."""

    model_name: str
    model_cls: type[BaseModel]
    best_params: dict[str, Any]
    cv_rps: float
    holdout_rps: float
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


def _log_model_artifact(model: BaseModel) -> None:
    """Log a fitted BaseModel as an MLflow pyfunc model artifact."""
    mlflow.pyfunc.log_model(
        name="model",
        python_model=_ModelWrapper(model),
    )


# ---------------------------------------------------------------------------
# Phase 1 — Experimental (tuning + top-K selection)
# ---------------------------------------------------------------------------


def run_experimental_phase(
    df: pd.DataFrame,
    *,
    n_trials_override: int | None = None,
    models: dict[str, type[BaseModel]] | None = None,
) -> list[ExperimentalResult]:
    """Tune each candidate via walk-forward CV; return top-K by CV RPS."""
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
        )

        model = model_cls(**best_params)
        model.fit(splits.X_train, splits.y_train)
        importance = compute_permutation_importance(
            model, splits.X_train, splits.y_train, feature_cols, n_repeats=5,
        )

        results.append(
            ExperimentalResult(
                model_name=model_name,
                model_cls=model_cls,
                best_params=best_params,
                cv_rps=study.best_value,
                importance=importance,
                splits=splits,
                feature_cols=feature_cols,
            )
        )
        logger.info("%s — best CV RPS: %.4f", model_name, study.best_value)

    results.sort(key=lambda r: r.cv_rps)
    top_k = results[:TOP_K_FOR_QA]
    logger.info(
        "Experimental done — top %d: %s",
        TOP_K_FOR_QA,
        [(r.model_name, f"{r.cv_rps:.4f}") for r in top_k],
    )
    return top_k


# ---------------------------------------------------------------------------
# Phase 2 — QA (holdout evaluation on WC 2022)
# ---------------------------------------------------------------------------


def run_qa_phase(top_models: list[ExperimentalResult]) -> QAResult:
    """Retrain top models on pre-WC data; pick best by WC 2022 holdout RPS."""
    setup_mlflow()
    qa_results: list[QAResult] = []

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
        rmse_h = compute_rmse(lam_h, entry.splits.y_holdout[:, 0])
        rmse_a = compute_rmse(lam_a, entry.splits.y_holdout[:, 1])

        with start_run(
            run_name=f"qa_{entry.model_name}",
            tags={"stage": "qa", "model_name": entry.model_name},
        ) as run:
            log_run(
                params=entry.best_params,
                metrics={
                    "holdout_rps": holdout_rps,
                    "holdout_rmse_home": rmse_h,
                    "holdout_rmse_away": rmse_a,
                    "cv_rps": entry.cv_rps,
                },
            )
            qa_run_id = run.info.run_id

        qa_results.append(
            QAResult(
                model_name=entry.model_name,
                model_cls=entry.model_cls,
                best_params=entry.best_params,
                cv_rps=entry.cv_rps,
                holdout_rps=holdout_rps,
                holdout_rmse_home=rmse_h,
                holdout_rmse_away=rmse_a,
                qa_run_id=qa_run_id,
                splits=entry.splits,
                feature_cols=entry.feature_cols,
            )
        )
        logger.info("%s — holdout RPS: %.4f", entry.model_name, holdout_rps)

    qa_results.sort(key=lambda r: r.holdout_rps)
    winner = qa_results[0]
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


def run_deploy_phase(winner: QAResult) -> str:
    """Refit winner on all Gold data, serialize, register, and promote.

    Raises:
        ChallengeFailed: if a production champion already exists and the
            challenger's holdout RPS does not improve on it.

    Returns:
        The MLflow run_id of the production-refit run.
    """
    setup_mlflow()

    champion_rps = get_champion_rps()
    if champion_rps is not None and winner.holdout_rps >= champion_rps:
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

    with start_run(
        run_name=f"deploy_{winner.model_name}",
        tags={"stage": "production-refit", "model_name": winner.model_name},
    ) as run:
        log_run(
            params={
                **winner.best_params,
                "evaluation_run_id": winner.qa_run_id,
                "gold_row_count": str(len(winner.splits.df_full)),
            },
            metrics={
                "qa_holdout_rps": winner.holdout_rps,
                "qa_holdout_rmse_home": winner.holdout_rmse_home,
                "qa_holdout_rmse_away": winner.holdout_rmse_away,
            },
        )
        _log_model_artifact(model)
        run_id = run.info.run_id

    mv = register_model(run_id, artifact_path="model")
    promote_to_production(version=mv.version)

    logger.info(
        "Deployed %s (run_id=%s, version=%s)",
        winner.model_name,
        run_id,
        mv.version,
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

    logger.info("=== Phase 1: Experimental ===")
    top_k = run_experimental_phase(
        df, n_trials_override=n_trials_override, models=models,
    )

    logger.info("=== Phase 2: QA ===")
    winner = run_qa_phase(top_k)

    logger.info("=== Phase 3: Deploy ===")
    run_id = run_deploy_phase(winner)

    logger.info("=== Pipeline complete (run_id=%s) ===", run_id)
    return run_id
