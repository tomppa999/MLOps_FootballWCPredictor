"""Optuna study factory with walk-forward CV and MLflow trial logging."""

from __future__ import annotations

import logging
import time
from typing import Any

import mlflow
import numpy as np
import optuna

from src.models.base import BaseModel
from src.models.evaluation import compute_mean_nll, compute_mean_rps, compute_rmse
from src.models.mlflow_utils import get_or_create_experiment, setup_mlflow

logger = logging.getLogger(__name__)

# Silence Optuna's per-trial INFO logging (results are logged to MLflow)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Hyperparameter sampling from config search-space dicts
# ---------------------------------------------------------------------------


def sample_params(trial: optuna.Trial, search_space: dict[str, dict]) -> dict[str, Any]:
    """Sample hyperparameters from an Optuna trial according to *search_space*."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        stype = spec["type"]
        if stype == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif stype == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif stype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search-space type '{stype}' for param '{name}'")
    return params


# ---------------------------------------------------------------------------
# Objective factory
# ---------------------------------------------------------------------------


def _make_objective(
    model_cls: type[BaseModel],
    search_space: dict[str, dict],
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: list[tuple[np.ndarray, np.ndarray]],
    fixed_params: dict[str, Any] | None = None,
    pipeline_run_id: str | None = None,
):
    """Return an Optuna objective that evaluates *model_cls* via walk-forward CV."""

    _fixed = fixed_params or {}

    def objective(trial: optuna.Trial) -> float:
        sampled = sample_params(trial, search_space)
        all_params = {**_fixed, **sampled}

        fold_nll: list[float] = []
        fold_rps: list[float] = []
        fold_rmse_h: list[float] = []
        fold_rmse_a: list[float] = []
        for train_idx, val_idx in cv_folds:
            model = model_cls(**all_params)
            model.fit(X[train_idx], y[train_idx])
            lam_h, lam_a = model.predict(X[val_idx])
            nll = compute_mean_nll(lam_h, lam_a, y[val_idx, 0], y[val_idx, 1])
            rps = compute_mean_rps(lam_h, lam_a, y[val_idx, 0], y[val_idx, 1])
            fold_nll.append(nll)
            fold_rps.append(rps)
            fold_rmse_h.append(compute_rmse(lam_h, y[val_idx, 0]))
            fold_rmse_a.append(compute_rmse(lam_a, y[val_idx, 1]))

        mean_nll = float(np.mean(fold_nll))
        mean_rps = float(np.mean(fold_rps))
        mean_rmse_h = float(np.mean(fold_rmse_h))
        mean_rmse_a = float(np.mean(fold_rmse_a))

        trial_tags: dict[str, str] = {
            "stage": "experimental",
            "model_name": model_cls.__name__,
            "trial_number": str(trial.number),
        }
        if pipeline_run_id:
            trial_tags["pipeline_run_id"] = pipeline_run_id

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(all_params)
            mlflow.log_metric("cv_nll_mean", mean_nll)
            mlflow.log_metric("cv_rps_mean", mean_rps)
            mlflow.log_metric("cv_rmse_home_mean", mean_rmse_h)
            mlflow.log_metric("cv_rmse_away_mean", mean_rmse_a)
            for i, nll_val in enumerate(fold_nll):
                mlflow.log_metric(f"cv_nll_fold_{i}", nll_val)
            mlflow.set_tags(trial_tags)

        return mean_nll

    return objective


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_tuning(
    model_cls: type[BaseModel],
    search_space: dict[str, dict],
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    n_trials: int = 50,
    fixed_params: dict[str, Any] | None = None,
    experiment_name: str = "wc_prediction",
    random_state: int = 42,
    pipeline_run_id: str | None = None,
) -> tuple[dict[str, Any], optuna.Study]:
    """Run an Optuna TPE study with walk-forward CV, logging every trial to MLflow.

    Args:
        model_cls: A concrete ``BaseModel`` subclass.
        search_space: Optuna search-space dict (from ``config.SEARCH_SPACES``).
        X: Feature matrix for the training period.
        y: Target matrix (n, 2) for the training period.
        cv_folds: Output of ``walk_forward_cv``.
        n_trials: Number of Optuna trials.
        fixed_params: Non-tuned params passed to every model instantiation.
        experiment_name: MLflow experiment to log into.
        random_state: Seed for the TPE sampler.

    Returns:
        (best_params, study) — best_params includes both fixed and sampled params.
    """
    setup_mlflow()
    experiment_id = get_or_create_experiment(experiment_name)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        study_name=f"tune_{model_cls.__name__}",
    )

    objective = _make_objective(
        model_cls=model_cls,
        search_space=search_space,
        X=X,
        y=y,
        cv_folds=cv_folds,
        fixed_params=fixed_params,
        pipeline_run_id=pipeline_run_id,
    )

    start_time = time.monotonic()

    def _log_trial(
        study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        elapsed = time.monotonic() - start_time
        logger.info(
            "%s trial %d/%d — NLL: %.4f (best: %.4f) | params: %s | elapsed: %.0fs",
            model_cls.__name__,
            trial.number + 1,
            n_trials,
            trial.value,
            study.best_value,
            trial.params,
            elapsed,
        )

    tuning_tags: dict[str, str] = {
        "stage": "experimental",
        "model_name": model_cls.__name__,
    }
    if pipeline_run_id:
        tuning_tags["pipeline_run_id"] = pipeline_run_id
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"tuning_{model_cls.__name__}",
        tags=tuning_tags,
    ):
        study.optimize(objective, n_trials=n_trials, callbacks=[_log_trial])

        mlflow.log_params(
            {f"best_{k}": v for k, v in study.best_params.items()}
        )
        mlflow.log_metric("best_cv_nll", study.best_value)

    best_params = {**(fixed_params or {}), **study.best_params}
    logger.info(
        "%s tuning done — best CV NLL: %.4f, params: %s",
        model_cls.__name__,
        study.best_value,
        best_params,
    )
    return best_params, study
