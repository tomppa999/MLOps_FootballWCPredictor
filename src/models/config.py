"""Feature sets, model→feature-set mapping, and Optuna search spaces."""

from __future__ import annotations

from typing import Final

from src.gold.schema import (
    CONTEXT_COLUMNS,
    ROLLING_ELO_CHANGE_COLUMNS,
    ROLLING_GOALS_COLUMNS,
    TARGET_COLUMNS,
    TEMPORAL_COLUMNS,
)

# Re-export for convenience
TARGET_COLS: Final[list[str]] = TARGET_COLUMNS

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

# Core: nearly complete across all rows (<2% NaN after dropna).
# Rolling goals are included here (~108–109 NaN / 6,663 rows).
# Gold v2 expansion: ``elo_sum`` (strength) and the three
# TEMPORAL_COLUMNS join Core.  ``is_cross_confederation`` was removed
# (invisible in all model importances, confounded with competition_tier).
# Thesis A.2: rolling Elo-change (full coverage) joins Core.
CORE_FEATURE_COLUMNS: Final[list[str]] = (
    ["elo_diff", "elo_sum"]
    + CONTEXT_COLUMNS
    + TEMPORAL_COLUMNS
    + ROLLING_GOALS_COLUMNS
    + ROLLING_ELO_CHANGE_COLUMNS
)  # 14 features

# Thesis A.1: rolling shot and tactical columns are dropped from the model
# feature set entirely.  No model trains on in-game statistics, so the Full set
# now equals the Core set (kept as a distinct name for backward-compatible
# imports and the per-model mapping below).
FULL_FEATURE_COLUMNS: Final[list[str]] = list(CORE_FEATURE_COLUMNS)

# Which feature set each model uses.
# mean_rate_poisson ignores all features but needs a consistent key here.
MODEL_FEATURE_SETS: Final[dict[str, list[str]]] = {
    "mean_rate_poisson": CORE_FEATURE_COLUMNS,
    "poisson_glm": CORE_FEATURE_COLUMNS,
    "negbin_glm": CORE_FEATURE_COLUMNS,
    "ridge": CORE_FEATURE_COLUMNS,
    "random_forest": CORE_FEATURE_COLUMNS,
    "xgboost": FULL_FEATURE_COLUMNS,
    "bayesian_poisson": CORE_FEATURE_COLUMNS,
    "sarimax": CORE_FEATURE_COLUMNS,
    "lstm": CORE_FEATURE_COLUMNS,
    "cnn": CORE_FEATURE_COLUMNS,
}

# ---------------------------------------------------------------------------
# Optuna search spaces
# ---------------------------------------------------------------------------
# Each value is a dict whose keys are hyperparameter names.
# Spec per param: {"type": "float"|"int"|"categorical", ...}

SEARCH_SPACES: Final[dict[str, dict]] = {
    "mean_rate_poisson": {},
    "poisson_glm": {
        "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
    },
    "negbin_glm": {
        "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
    },
    "ridge": {
        "alpha": {"type": "float", "low": 1e-3, "high": 1e5, "log": True},
    },
    "random_forest": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 50},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.5, 0.8]},
    },
    "xgboost": {
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    },
    # Milestone 4 models — search spaces defined upfront
    "bayesian_poisson": {
        "prior_sigma": {"type": "float", "low": 0.1, "high": 5.0, "log": False},
        "draws": {"type": "int", "low": 500, "high": 2000},
        "tune_steps": {"type": "int", "low": 500, "high": 2000},
    },
    "sarimax": {
        "p": {"type": "int", "low": 0, "high": 3},
        "d": {"type": "int", "low": 0, "high": 2},
        "q": {"type": "int", "low": 0, "high": 3},
    },
    "lstm": {
        "units": {"type": "int", "low": 16, "high": 128},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    },
    "cnn": {
        "filters": {"type": "int", "low": 16, "high": 128},
        "kernel_size": {"type": "int", "low": 2, "high": 7},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    },
}

# Default Optuna trial counts per model
DEFAULT_N_TRIALS: Final[dict[str, int]] = {
    "mean_rate_poisson": 1,
    "poisson_glm": 30,
    "negbin_glm": 30,
    "ridge": 30,
    "random_forest": 80,
    "xgboost": 100,
    "bayesian_poisson": 40,
    "sarimax": 48,
    "lstm": 50,
    "cnn": 50,
}

# ---------------------------------------------------------------------------
# A.6 — Half-period tuning
# ---------------------------------------------------------------------------

# Models that receive time-decay weights (mean_rate_poisson ignores weights;
# sarimax has no per-observation weight hook in statsmodels SARIMAX).
WEIGHTED_MODELS: Final[frozenset[str]] = frozenset({
    "poisson_glm",
    "negbin_glm",
    "ridge",
    "random_forest",
    "xgboost",
    "bayesian_poisson",
    "lstm",
    "cnn",
})

# Optuna search bounds and trial budget for the marginal half-period search.
HALF_PERIOD_SEARCH: Final[dict[str, float]] = {"low": 1.0, "high": 5.0}
HALF_PERIOD_N_TRIALS: Final[int] = 25

# Per-model tuned half-period (years), filled manually after running
# `python -m src.models.half_period_tuning` (A.6).  Empty dict → all weighted
# models fall back to DEFAULT_HALF_PERIOD_YEARS (3.0) until populated.
# Pattern mirrors HOLDOUT_RPS_BASELINES in src/monitoring/baselines.py.
# Update all 8 weighted-model values after A.6; leave unweighted models absent.
# A.6 final decisions (2026-06-04):
# ridge is the only model with a clean, deterministic preference for a longer
# half-life (~4.8yr, Δ NLL = −0.0045).  All others show a flat objective
# across [1,5]; 3yr (Ley et al. optimum) is at/near optimal for every one.
# xgboost "best" was marginally *worse* than 3yr (within RNG); poisson_glm
# result was unreliable (optimizer blow-ups).  No rerun needed.
TUNED_HALF_PERIODS: Final[dict[str, float]] = {
    "poisson_glm": 3.0,    # unreliable first-run result; flat → 3yr
    "negbin_glm": 3.0,     # Δ NLL = +0.0003 (noise)
    "ridge": 4.803,        # Δ NLL = −0.0045 (deterministic gain)
    "random_forest": 3.0,  # Δ NLL = +0.0001 (noise)
    "xgboost": 3.0,        # best at 4.92 was +0.001 vs 3yr → 3yr preferred
    "bayesian_poisson": 3.0,  # Δ NLL = +0.0001 (flat; divergences expected)
    "lstm": 3.0,           # landscape flat; A.6 offset = Keras nondeterminism
    "cnn": 3.0,            # Δ NLL = −0.0017 (within noise; TPE artefact)
}
