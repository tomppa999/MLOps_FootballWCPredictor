"""Three-phase training pipeline: experimental → QA → deploy.

Phase 1 (Experimental): Tune every candidate via walk-forward CV.  Top-K advance.
Phase 2 (QA):           Retrain top-K on pre-WC data, evaluate on WC 2022 holdout.
Phase 3 (Deploy):       Refit the winner on *all* Gold data, register, promote.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cloudpickle
import mlflow
import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.models.candidates.bayesian_poisson import BayesianPoissonModel
from src.models.candidates.cnn import CNNModel
from src.models.candidates.lstm import LSTMModel
from src.models.candidates.mean_rate_poisson import MeanRatePoisson
from src.models.candidates.negbin_glm import NegativeBinomialGLM
from src.models.candidates.poisson_glm import BivariatePoisson
from src.models.candidates.random_forest import RandomForestModel
from src.models.candidates.ridge import RidgeModel
from src.models.candidates.sarimax import SARIMAXModel
from src.models.candidates.xgboost_model import XGBoostModel
from src.models.config import (
    DEFAULT_N_TRIALS,
    EXPERIMENT_MODELS,
    LIVE_SHADOW_MODELS,
    MODEL_FEATURE_SETS,
    SEARCH_SPACES,
    TUNED_HALF_PERIODS,
)
from src.models.data_split import (
    DEFAULT_HALF_PERIOD_YEARS,
    DataSplits,
    load_gold,
    make_splits,
    walk_forward_cv,
)
from src.models.evaluation import (
    compute_mean_nll,
    compute_mean_rps,
    compute_permutation_importance,
    compute_rmse,
)
from src.models.mlflow_utils import (
    CHAMPION_ALIAS_PER_ROUND,
    PRODUCTION_MODEL_NAME,
    SHADOW_MODEL_NAME,
    STAGING_MODEL_NAME,
    get_all_shadow_metadata,
    get_champion_metadata,
    get_champion_rps,
    get_shadow_metadata,
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
    "mean_rate_poisson": MeanRatePoisson,
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

# Shadow-refit hang guards (B.3). A native sampler (PyMC NUTS) can wedge
# indefinitely; a thread-based timeout cannot interrupt a C-extension call, so
# each fit runs in a child process that is killed on timeout.
# No per-model cap: each model gets the remaining overall budget. bayesian_poisson
# is the only model that ever approaches the limit and it runs last, so the fast
# models are never blocked by it.
SHADOW_REFIT_TOTAL_TIMEOUT_S: float = 1800.0  # overall cap across candidates (30 min)
# bayesian_poisson is the slow/risky model — refit it last within the
# experiment roster so the other selected models always complete first.
_SLOW_EXPERIMENT_MODEL: str = "bayesian_poisson"

# Refit order for the 4-model EXPERIMENT_MODELS roster: fast models first,
# MCMC (bayesian_poisson) last so the other 3 always complete even on timeout.
_PER_ROUND_REFIT_ORDER: list[str] = [
    m for m in EXPERIMENT_MODELS if m != _SLOW_EXPERIMENT_MODEL
] + ([_SLOW_EXPERIMENT_MODEL] if _SLOW_EXPERIMENT_MODEL in EXPERIMENT_MODELS else [])

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
    # A.6: per-model tuned time-decay half-life; defaults to 3yr if not yet tuned.
    half_period_years: float = DEFAULT_HALF_PERIOD_YEARS


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
    # A.6: per-model tuned time-decay half-life; defaults to 3yr if not yet tuned.
    half_period_years: float = DEFAULT_HALF_PERIOD_YEARS


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
        half_period_years = TUNED_HALF_PERIODS.get(model_name, DEFAULT_HALF_PERIOD_YEARS)
        splits = make_splits(df, feature_cols, dropna=dropna, half_period_years=half_period_years)
        cv_folds = walk_forward_cv(len(splits.X_train))
        search_space = SEARCH_SPACES[model_name]
        n_trials = n_trials_override or DEFAULT_N_TRIALS[model_name]

        logger.info(
            "Tuning %s (%d trials, half_period=%.2fyr)…",
            model_name, n_trials, half_period_years,
        )
        t0 = time.time()
        best_params, study = run_tuning(
            model_cls,
            search_space,
            splits.X_train,
            splits.y_train,
            cv_folds,
            n_trials=n_trials,
            pipeline_run_id=pipeline_run_id,
            sample_weight=splits.w_train,
        )

        model = model_cls(**best_params)
        model.fit(splits.X_train, splits.y_train, sample_weight=splits.w_train)
        importance = compute_permutation_importance(
            model, splits.X_holdout, splits.y_holdout, feature_cols, n_repeats=5,
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
                    params={**best_params, "half_period_years": half_period_years},
                    metrics={
                        "cv_nll": study.best_value,
                        "experimental_wall_sec": time.time() - t0,
                    },
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
                half_period_years=half_period_years,
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
        t0 = time.time()
        model = entry.model_cls(**entry.best_params)
        model.fit(
            entry.splits.X_train, entry.splits.y_train, sample_weight=entry.splits.w_train
        )

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
                params={**entry.best_params, "half_period_years": entry.half_period_years},
                metrics={
                    "holdout_rps": holdout_rps,
                    "holdout_nll": holdout_nll,
                    "holdout_rmse_home": rmse_h,
                    "holdout_rmse_away": rmse_a,
                    "cv_nll": entry.cv_nll,
                    "qa_wall_sec": time.time() - t0,
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
                half_period_years=entry.half_period_years,
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
    if champion_rps is not None and winner.holdout_rps >= champion_rps:
        raise ChallengeFailed(
            f"Challenger {winner.model_name} holdout RPS {winner.holdout_rps:.4f} "
            f"does not strictly improve champion RPS {champion_rps:.4f} — promotion skipped."
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

    t0 = time.time()
    model = winner.model_cls(**winner.best_params)
    model.fit(winner.splits.X_full, winner.splits.y_full, sample_weight=winner.splits.w_full)

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
                "half_period_years": winner.half_period_years,
                "evaluation_run_id": winner.qa_run_id,
                "gold_row_count": str(len(winner.splits.df_full)),
            },
            metrics={
                "qa_holdout_rps": winner.holdout_rps,
                "qa_holdout_nll": winner.holdout_nll,
                "qa_holdout_rmse_home": winner.holdout_rmse_home,
                "qa_holdout_rmse_away": winner.holdout_rmse_away,
                "deploy_wall_sec": time.time() - t0,
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
    splits = make_splits(df, feature_cols, dropna=dropna, half_period_years=meta.half_period_years)

    logger.info(
        "Champion refit: fitting %s on %d Gold rows (half_period=%.2fyr).",
        meta.model_name,
        len(splits.df_full),
        meta.half_period_years,
    )

    t0 = time.time()
    model = model_cls(**meta.best_params)
    model.fit(splits.X_full, splits.y_full, sample_weight=splits.w_full)

    with start_run(
        run_name=f"refit_{meta.model_name}",
        tags={"stage": "champion-refit", "model_name": meta.model_name},
    ) as run:
        log_run(
            params={
                **meta.best_params,
                "half_period_years": meta.half_period_years,
                "gold_row_count": str(len(splits.df_full)),
            },
            metrics={**meta.holdout_metrics, "refit_wall_sec": time.time() - t0},
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
# Per-round refit (B.3) — 4-model EXPERIMENT_MODELS roster, per-round cadence
# ---------------------------------------------------------------------------


def run_per_round_refit(
    df: pd.DataFrame,
    *,
    matchday: str,
    completed_matchday: str,
) -> dict[str, str]:
    """Refit the 4 EXPERIMENT_MODELS roster entries for the per-round cadence.

    Reads each model's hyperparameters from the registry (champion from
    wc_production, non-champion roster models from wc_staging), refits on
    the full Gold dataset, and registers results tagged cadence_mode=per_round.

    The display champion receives the ``champion_per_round`` alias on
    wc_production.  The other 3 roster models register to wc_shadow with
    ``cadence_mode=per_round`` + ``model_name`` tags so
    ``load_shadow_model(name, cadence_mode="per_round")`` picks them up.

    ``bayesian_poisson`` is always refitted last (slow MCMC); per-model and
    overall timeouts match the shadow-refit guards so a hung NUTS sampler
    cannot stall the pipeline.

    Args:
        df: Full Gold DataFrame (already loaded by the caller).
        matchday: Tournament stage label for the round being predicted
            (e.g. ``"2"``, ``"3"``, ``"R32"``, …).  Stored as an MLflow run
            tag so downstream consumers know what round this model serves.
        completed_matchday: The round that just fully completed and triggered
            this refit (e.g. ``"1"`` after all MD1 games finished).  Stored
            as ``per_round_last_completed_matchday`` so the trigger gate can
            detect the next boundary without re-firing on the same round.

    Returns:
        Dict mapping model_name → MLflow run_id for each model that was
        successfully refitted.  Timed-out or failed models are skipped and
        logged as warnings.
    """
    setup_mlflow()
    champion_meta = get_champion_metadata()
    run_ids: dict[str, str] = {}
    overall_deadline = time.time() + SHADOW_REFIT_TOTAL_TIMEOUT_S

    for model_name in _PER_ROUND_REFIT_ORDER:
        if model_name not in CANDIDATE_MODELS:
            logger.warning(
                "Per-round refit: %s not in CANDIDATE_MODELS — skipping.", model_name,
            )
            continue

        remaining = overall_deadline - time.time()
        if remaining <= 0:
            logger.warning(
                "Per-round refit: overall timeout (%.0fs) reached — "
                "skipping %s and remainder.",
                SHADOW_REFIT_TOTAL_TIMEOUT_S,
                model_name,
            )
            break

        if model_name == champion_meta.model_name:
            meta = champion_meta
        else:
            try:
                meta = get_shadow_metadata(model_name)
            except ValueError:
                logger.warning(
                    "Per-round refit: no wc_staging metadata for %s — skipping.",
                    model_name,
                )
                continue

        feature_cols = MODEL_FEATURE_SETS[model_name]
        dropna = model_name != "xgboost"
        splits = make_splits(
            df, feature_cols, dropna=dropna, half_period_years=meta.half_period_years,
        )
        logger.info(
            "Per-round refit: fitting %s on %d Gold rows "
            "(matchday=%s, half_period=%.2fyr, timeout=%.0fs).",
            model_name,
            len(splits.df_full),
            matchday,
            meta.half_period_years,
            remaining,
        )

        t0 = time.time()
        model_obj = CANDIDATE_MODELS[model_name](**meta.best_params)
        per_model_timeout = remaining
        try:
            model_obj = _fit_with_timeout(
                model_obj, splits.X_full, splits.y_full, splits.w_full, per_model_timeout,
            )
        except TimeoutError:
            logger.warning(
                "Per-round refit: %s timed out (%.0fs) — skipping.",
                model_name, per_model_timeout,
            )
            continue
        except Exception:
            logger.exception("Per-round refit: %s failed during fit — skipping.", model_name)
            continue

        run_tags: dict[str, str] = {
            "stage": "per-round-refit",
            "model_name": model_name,
            "cadence_mode": "per_round",
            "per_round_refit_matchday": matchday,
            "per_round_last_completed_matchday": completed_matchday,
        }
        with start_run(
            run_name=f"per_round_refit_{model_name}_md{matchday}",
            tags=run_tags,
        ) as run:
            log_run(
                params={
                    **meta.best_params,
                    "half_period_years": meta.half_period_years,
                    "gold_row_count": str(len(splits.df_full)),
                },
                metrics={**meta.holdout_metrics, "refit_wall_sec": time.time() - t0},
            )
            model_uri = _log_model_artifact(model_obj)
            run_id = run.info.run_id

        if model_name == champion_meta.model_name:
            mv = register_model(model_uri=model_uri, model_name=PRODUCTION_MODEL_NAME)
            promote_to_production(version=mv.version, alias=CHAMPION_ALIAS_PER_ROUND)
            logger.info(
                "Per-round refit: %s → wc_production v%s (champion_per_round, matchday=%s)",
                model_name, mv.version, matchday,
            )
        else:
            mv = register_model(model_uri=model_uri, model_name=SHADOW_MODEL_NAME)
            logger.info(
                "Per-round refit: %s → wc_shadow v%s (cadence_mode=per_round, matchday=%s)",
                model_name, mv.version, matchday,
            )

        run_ids[model_name] = run_id

    logger.info(
        "Per-round refit complete — %d/%d models fitted (matchday=%s).",
        len(run_ids), len(EXPERIMENT_MODELS), matchday,
    )
    return run_ids


# ---------------------------------------------------------------------------
# Shadow refit (8 non-champion candidates on full Gold, frozen hyperparameters)
# ---------------------------------------------------------------------------


def _shadow_fit_worker(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None,
    out_path: str,
) -> None:
    """Child-process target: fit *model* and cloudpickle it to *out_path*.

    Runs in a separate process so a hung native sampler (e.g. PyMC NUTS) can be
    killed via termination — a thread-based timeout cannot interrupt a
    C-extension call. cloudpickle matches MLflow's own model serialisation, so
    keras/xgboost/PyMC models round-trip cleanly.
    """
    model.fit(X, y, sample_weight=w)
    with open(out_path, "wb") as fh:
        cloudpickle.dump(model, fh)


def _fit_with_timeout(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None,
    timeout_s: float,
) -> BaseModel:
    """Fit *model* in a child process, enforcing a hard wall-clock timeout.

    Returns the fitted model (read back from the child via cloudpickle).

    Raises:
        TimeoutError: if fitting exceeds *timeout_s* (the child is terminated).
        RuntimeError: if the child exits non-zero or produces no artifact.
    """
    ctx = mp.get_context()  # fork on Linux (cheap), spawn on macOS
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "fitted_model.pkl")
        proc = ctx.Process(
            target=_shadow_fit_worker, args=(model, X, y, w, out_path)
        )
        proc.start()
        proc.join(timeout_s)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            raise TimeoutError(f"fit exceeded {timeout_s:.0f}s")
        if proc.exitcode != 0:
            raise RuntimeError(f"fit subprocess exited with code {proc.exitcode}")
        artifact = Path(out_path)
        if not artifact.exists():
            raise RuntimeError("fit subprocess produced no model artifact")
        with artifact.open("rb") as fh:
            return cloudpickle.load(fh)


def _ordered_shadow_candidates(champion_name: str) -> list[str]:
    """Order non-champion candidates for refit.

    Only models in LIVE_SHADOW_MODELS are considered (lstm and cnn are
    excluded from the live pipeline).  Experiment-roster models come first
    so the simulated models always refit before the RPS-only shadows; the
    slow/risky ``bayesian_poisson`` is placed last within that group.
    """
    live = set(LIVE_SHADOW_MODELS)
    non_champion = [n for n in CANDIDATE_MODELS if n != champion_name and n in live]
    experiment = [
        n for n in EXPERIMENT_MODELS if n != champion_name and n in CANDIDATE_MODELS and n in live
    ]
    # Stable sort: keeps roster order but pushes the slow model to the end.
    experiment.sort(key=lambda n: n == _SLOW_EXPERIMENT_MODEL)
    experiment_set = set(experiment)
    non_experiment = [n for n in non_champion if n not in experiment_set]
    return experiment + non_experiment


def run_shadow_refit(df: pd.DataFrame) -> list[str]:
    """Refit all non-champion candidates on full Gold using stored best_params.

    Reads each candidate's Optuna-tuned ``best_params`` from the shadow
    registry (``wc_shadow``) with cold-start fallback to ``wc_staging``,
    refits on the full Gold dataset, and registers the result as a new
    version of ``wc_shadow`` tagged with the candidate name. NEVER re-tunes.

    The current champion is excluded — it is refit by ``run_champion_refit``
    on the same data, so a duplicate refit would waste compute.

    Returns:
        List of MLflow run_ids, one per refit shadow candidate.
    """
    setup_mlflow()
    champion = get_champion_metadata()
    candidate_names = _ordered_shadow_candidates(champion.model_name)
    shadow_metas = get_all_shadow_metadata(
        candidate_names,
        exclude_model_name=champion.model_name,
    )

    overall_deadline = time.time() + SHADOW_REFIT_TOTAL_TIMEOUT_S
    run_ids: list[str] = []
    for meta in shadow_metas:
        if meta.model_name == champion.model_name:
            continue  # defensive guard
        if meta.model_name not in CANDIDATE_MODELS:
            logger.warning(
                "Shadow candidate %s not in CANDIDATE_MODELS — skipping.",
                meta.model_name,
            )
            continue

        remaining = overall_deadline - time.time()
        if remaining <= 0:
            logger.warning(
                "Shadow refit overall timeout (%.0fs) reached — skipping "
                "remaining candidates from %s onward.",
                SHADOW_REFIT_TOTAL_TIMEOUT_S,
                meta.model_name,
            )
            break
        per_model_timeout = remaining

        model_cls = CANDIDATE_MODELS[meta.model_name]
        feature_cols = MODEL_FEATURE_SETS[meta.model_name]
        dropna = meta.model_name != "xgboost"
        splits = make_splits(
            df, feature_cols, dropna=dropna, half_period_years=meta.half_period_years
        )

        logger.info(
            "Shadow refit: fitting %s on %d Gold rows (half_period=%.2fyr, timeout=%.0fs).",
            meta.model_name,
            len(splits.df_full),
            meta.half_period_years,
            per_model_timeout,
        )
        t0 = time.time()
        model = model_cls(**meta.best_params)
        try:
            model = _fit_with_timeout(
                model,
                splits.X_full,
                splits.y_full,
                splits.w_full,
                per_model_timeout,
            )
        except TimeoutError:
            logger.warning(
                "Shadow refit: %s exceeded %.0fs — skipping (previous version kept).",
                meta.model_name,
                per_model_timeout,
            )
            continue
        except Exception:
            logger.exception(
                "Shadow refit: %s failed during fit — skipping.", meta.model_name
            )
            continue

        with start_run(
            run_name=f"shadow_refit_{meta.model_name}",
            tags={"stage": "shadow-refit", "model_name": meta.model_name},
        ) as run:
            log_run(
                params={
                    **meta.best_params,
                    "half_period_years": meta.half_period_years,
                    "gold_row_count": str(len(splits.df_full)),
                },
                metrics={**meta.holdout_metrics, "shadow_wall_sec": time.time() - t0},
            )
            model_uri = _log_model_artifact(model)
            run_id = run.info.run_id

        mv = register_model(model_uri=model_uri, model_name=SHADOW_MODEL_NAME)
        logger.info(
            "Shadow refit complete: %s (run_id=%s, version=%s, gold_rows=%d)",
            meta.model_name,
            run_id,
            mv.version,
            len(splits.df_full),
        )
        run_ids.append(run_id)

    logger.info("Shadow refit done — %d candidates registered.", len(run_ids))
    return run_ids


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
