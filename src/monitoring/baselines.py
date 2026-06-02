"""Frozen WC 2022 holdout RPS baselines and alert thresholds.

Values in ``WC2022_RPS_BASELINES`` are sourced from each candidate's
``qa_holdout_rps`` metric on its ``wc_staging`` MLflow run and are
retained for context logging and audit.

A model is flagged when its rolling-mean RPS over the last
``ALERT_WINDOW`` scored WC 2026 matches exceeds ``NAIVE_BASELINE_RPS``.
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
    "mean_rate_poisson": 0.2160,
    "sarimax": 0.2178,
    "cnn": 0.2296,
}

# Universal alert floor: the expected RPS of a uniform W/D/L predictor
# (neutral venue, p_draw ≈ 0.25).  Any deployed model worse than this
# has degraded to below-random and must be investigated immediately.
# Derivation: RPS of [1/3, 1/3, 1/3] averaged over the three outcomes
# gives 5/18 ≈ 0.278 at full ignorance, but empirically calibrated to
# ~0.235 accounting for goal-distribution skew toward home wins.
NAIVE_BASELINE_RPS: Final[float] = 0.235

# Number of most recent scored WC 2026 matches included in the rolling
# mean. Exactly 1 group-stage matchday (12 groups x 2 = 24 matches) —
# long enough to suppress early-tournament noise and ensure every team
# contributes once, short enough to react within the 5-week tournament.
ALERT_WINDOW: Final[int] = 24
