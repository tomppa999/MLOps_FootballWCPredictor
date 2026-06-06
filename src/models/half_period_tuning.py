"""A.6 — Marginal half-period tuning orchestrator.

Loads each weighted model's A.5 best_params from the MLflow registry
(wc_staging, with cold-start fallback from wc_shadow), then runs a cheap
1-D Optuna search over ``half_period_years ∈ [1.0, 5.0]`` to find the
per-model optimum while keeping all other hyperparameters frozen.

Usage::

    conda activate modelops
    python -m src.models.half_period_tuning [--n-trials N]

After the run, paste the printed ``TUNED_HALF_PERIODS`` block into
``src/models/config.py`` to freeze the values for A.7 and B.3.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import pandas as pd

from src.models.config import (
    HALF_PERIOD_N_TRIALS,
    HALF_PERIOD_SEARCH,
    MODEL_FEATURE_SETS,
    WEIGHTED_MODELS,
)
from src.models.data_split import compute_weight_components, load_gold, make_splits, walk_forward_cv
from src.models.mlflow_utils import get_shadow_metadata, setup_mlflow
from src.models.tuning import tune_half_period

logger = logging.getLogger(__name__)

# Candidate class registry (mirrors pipeline.CANDIDATE_MODELS; imported lazily
# to avoid heavy optional deps (keras, pymc) at module import time).
def _candidate_models() -> dict[str, Any]:
    from src.models.candidates.bayesian_poisson import BayesianPoissonModel  # noqa: PLC0415
    from src.models.candidates.cnn import CNNModel  # noqa: PLC0415
    from src.models.candidates.lstm import LSTMModel  # noqa: PLC0415
    from src.models.candidates.negbin_glm import NegativeBinomialGLM  # noqa: PLC0415
    from src.models.candidates.poisson_glm import BivariatePoisson  # noqa: PLC0415
    from src.models.candidates.random_forest import RandomForestModel  # noqa: PLC0415
    from src.models.candidates.ridge import RidgeModel  # noqa: PLC0415
    from src.models.candidates.xgboost_model import XGBoostModel  # noqa: PLC0415

    return {
        "poisson_glm": BivariatePoisson,
        "negbin_glm": NegativeBinomialGLM,
        "ridge": RidgeModel,
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "bayesian_poisson": BayesianPoissonModel,
        "lstm": LSTMModel,
        "cnn": CNNModel,
    }


def run_half_period_tuning(
    df: pd.DataFrame | None = None,
    *,
    n_trials: int | None = None,
    models: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Tune ``half_period_years`` for every weighted model (A.6).

    For each weighted model:
    1. Load A.5 ``best_params`` from the MLflow registry (wc_staging / wc_shadow).
    2. Build the training split and walk-forward CV folds.
    3. Pre-compute ``(days_ago, w_importance)`` — stable across trials.
    4. Run a 1-D Optuna search over ``half_period_years`` minimising CV NLL.

    Args:
        df: Gold DataFrame.  Loaded from disk when ``None``.
        n_trials: Number of Optuna trials per model.  Defaults to
            ``HALF_PERIOD_N_TRIALS`` (25).
        models: ``{model_name: model_cls}`` override for testing.

    Returns:
        ``{model_name: best_half_period_years}`` for all weighted models that
        completed successfully.  Failed models are logged and omitted.
    """
    setup_mlflow()
    if df is None:
        df = load_gold()
    if n_trials is None:
        n_trials = HALF_PERIOD_N_TRIALS

    candidates = models if models is not None else _candidate_models()
    # Only tune weighted models; skip any candidate not in WEIGHTED_MODELS.
    candidates = {k: v for k, v in candidates.items() if k in WEIGHTED_MODELS}

    results: dict[str, float] = {}
    low = HALF_PERIOD_SEARCH["low"]
    high = HALF_PERIOD_SEARCH["high"]

    for model_name, model_cls in candidates.items():
        logger.info("=== A.6: tuning half_period for %s (%d trials) ===", model_name, n_trials)
        try:
            meta = get_shadow_metadata(model_name)
        except ValueError:
            logger.warning(
                "%s: no wc_staging entry — skipping. "
                "Run the full pipeline (QA phase) first.",
                model_name,
            )
            continue

        fixed_params = {
            k: v for k, v in meta.best_params.items()
            # Strip any infrastructure keys that are not model constructor args.
            if k not in {"half_period_years", "gold_row_count", "evaluation_run_id"}
        }

        feature_cols = MODEL_FEATURE_SETS[model_name]
        dropna = model_name != "xgboost"
        splits = make_splits(df, feature_cols, dropna=dropna)
        cv_folds = walk_forward_cv(len(splits.X_train))
        days_ago, w_importance = compute_weight_components(splits.df_train)

        try:
            best_hp, _ = tune_half_period(
                model_cls,
                fixed_params,
                splits.X_train,
                splits.y_train,
                cv_folds,
                days_ago,
                w_importance,
                n_trials=n_trials,
                low=low,
                high=high,
            )
        except Exception:
            logger.exception("half_period tuning failed for %s — skipping.", model_name)
            continue

        results[model_name] = best_hp
        logger.info("%s → best half_period_years = %.3f", model_name, best_hp)

    _print_paste_block(results)
    return results


def _print_paste_block(results: dict[str, float]) -> None:
    """Print a copy-paste ready TUNED_HALF_PERIODS block for config.py."""
    lines = ["", "Paste the following into TUNED_HALF_PERIODS in src/models/config.py:", ""]
    lines.append("TUNED_HALF_PERIODS: Final[dict[str, float]] = {")
    for model_name in sorted(results):
        lines.append(f'    "{model_name}": {results[model_name]:.4f},')
    lines.append("}")
    print("\n".join(lines))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A.6: marginal half-period tuning for weighted models."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=f"Optuna trials per model (default: {HALF_PERIOD_N_TRIALS}).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    args = _parse_args()
    results = run_half_period_tuning(n_trials=args.n_trials)
    if not results:
        logger.error("No models tuned — check MLflow registry and logs above.")
        sys.exit(1)
    logger.info("A.6 complete: %s", results)
