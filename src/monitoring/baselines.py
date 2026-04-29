"""Frozen WC 2022 holdout RPS baselines and alert thresholds.

Values are sourced from each candidate's ``qa_holdout_rps`` metric on its
``wc_staging`` MLflow run (filterable by ``model_name`` tag) and are
identical to the Results table in the project report. They are committed
in code so the alert logic remains offline-checkable and deterministic.

A model is flagged when its rolling-mean RPS over the last
``ALERT_WINDOW`` scored WC 2026 matches exceeds
``WC2022_RPS_BASELINES[model] * ALERT_FACTOR``.
"""

from __future__ import annotations

from typing import Final

WC2022_RPS_BASELINES: Final[dict[str, float]] = {
    "xgboost": 0.2118,
    "lstm": 0.2133,
    "ridge": 0.2136,
    "poisson_glm": 0.2148,
    "random_forest": 0.2148,
    "bayesian_poisson": 0.2149,
    "negbin_glm": 0.2155,
    "sarimax": 0.2178,
    "cnn": 0.2296,
}

# Multiplicative factor applied to the per-model baseline to derive the
# alert threshold. 1.3 follows the project plan: a 30% degradation over
# the WC 2022 audit value is a meaningful, manually investigable signal.
ALERT_FACTOR: Final[float] = 1.3

# Number of most recent scored WC 2026 matches included in the rolling
# mean. ~1.5 group-stage matchdays — long enough to suppress early-
# tournament noise, short enough to react within the 5-week tournament.
ALERT_WINDOW: Final[int] = 16
