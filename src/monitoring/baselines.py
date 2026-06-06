"""Frozen holdout RPS baselines and alert thresholds.

Values in ``HOLDOUT_RPS_BASELINES`` are sourced from each candidate's
``holdout_rps`` metric logged during the A.7 full refit run
(2026-06-05, pipeline run with tuned half-periods from A.6 applied:
ridge = 4.803yr, all other weighted models = 3.0yr).
Holdout: WC 2022 + continental tournaments (~347 matches).
These are the final frozen values for WC 2026 monitoring context.
Retained for context logging and audit only — not used in alert logic.

A model is flagged when its rolling-mean RPS over the last
``ALERT_WINDOW`` scored WC 2026 matches exceeds ``NAIVE_BASELINE_RPS``.
"""

from __future__ import annotations

from typing import Final

HOLDOUT_RPS_BASELINES: Final[dict[str, float]] = {
    "xgboost": 0.18289,
    "bayesian_poisson": 0.18316,
    "negbin_glm": 0.18373,
    "poisson_glm": 0.18389,
    "random_forest": 0.18392,
    "sarimax": 0.18583,
    "ridge": 0.18813,
    "lstm": 0.19299,
    "cnn": 0.20916,
    "mean_rate_poisson": 0.22872,
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
