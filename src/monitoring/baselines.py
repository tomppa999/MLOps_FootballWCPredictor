"""Frozen holdout RPS baselines and alert thresholds.

Values in ``HOLDOUT_RPS_BASELINES`` are sourced from each candidate's
``qa_holdout_rps`` metric logged during the most recent full refit run
(A.5 thesis feature set: WC 2022 + continental tournaments holdout,
fixed 3yr time-decay half-period).  Refresh all values after each major
refit (A.5 → interim values below; A.7 → final frozen values).
Retained for context logging and audit only — not used in alert logic.

A model is flagged when its rolling-mean RPS over the last
``ALERT_WINDOW`` scored WC 2026 matches exceeds ``NAIVE_BASELINE_RPS``.
"""

from __future__ import annotations

from typing import Final

HOLDOUT_RPS_BASELINES: Final[dict[str, float]] = {
    "xgboost": 0.18242,
    "poisson_glm": 0.18316,
    "bayesian_poisson": 0.18320,
    "random_forest": 0.18331,
    "negbin_glm": 0.18377,
    "sarimax": 0.18595,
    "ridge": 0.18819,
    "lstm": 0.18898,
    "cnn": 0.20181,
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
