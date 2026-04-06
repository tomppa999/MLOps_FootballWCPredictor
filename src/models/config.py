"""Feature sets, model→feature-set mapping, and Optuna search spaces."""

from __future__ import annotations

from typing import Final

from src.gold.schema import (
    CONTEXT_COLUMNS,
    ROLLING_GOALS_COLUMNS,
    ROLLING_SHOT_COLUMNS,
    ROLLING_TACTICAL_COLUMNS,
    TARGET_COLUMNS,
    TACTICAL_CLUSTER_COLUMNS,
)

# Re-export for convenience
TARGET_COLS: Final[list[str]] = TARGET_COLUMNS

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

# Core: nearly complete across all rows (<2% NaN after dropna).
# Rolling goals are included here (~108–109 NaN / 6,663 rows).
# Rolling shots and tactical columns have substantial NaN (~1,800–2,100)
# and are Full-only; only XGBoost handles NaN natively.
CORE_FEATURE_COLUMNS: Final[list[str]] = (
    ["elo_diff"]
    + CONTEXT_COLUMNS
    + ROLLING_GOALS_COLUMNS
)  # 8 features

FULL_FEATURE_COLUMNS: Final[list[str]] = (
    CORE_FEATURE_COLUMNS
    + ROLLING_SHOT_COLUMNS
    + ROLLING_TACTICAL_COLUMNS
    + TACTICAL_CLUSTER_COLUMNS
)  # 28 features

# Which feature set each model uses
MODEL_FEATURE_SETS: Final[dict[str, list[str]]] = {
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
    "poisson_glm": {
        "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
    },
    "negbin_glm": {
        "alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
    },
    "ridge": {
        "alpha": {"type": "float", "low": 1e-3, "high": 1e3, "log": True},
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
        "prior_sigma": {"type": "float", "low": 0.1, "high": 5.0, "log": True},
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
    "poisson_glm": 30,
    "negbin_glm": 20,
    "ridge": 30,
    "random_forest": 50,
    "xgboost": 60,
    "bayesian_poisson": 20,
    "sarimax": 30,
    "lstm": 25,
    "cnn": 25,
}
