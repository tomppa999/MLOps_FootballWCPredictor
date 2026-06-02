"""Symmetric mean-rate Poisson baseline (intercept-only case of Maher 1982).

Predicts the grand-mean goal rate λ = (mean home goals + mean away goals) / 2
for every match, regardless of features.  This is the no-features reference
threshold: any candidate that cannot beat it on holdout RPS has no demonstrated
feature value.

Reference:
    Maher, M. J. (1982). Modelling association football scores.
    Statistica Neerlandica, 36(3), 109–118.

Note on Dixon & Coles (1997): their low-score correction factor ρ and
exponential time-decay improve exact-scoreline prediction but are mathematically
irrelevant for RPS (goal-difference distribution is identical regardless of
ρ — the common terms cancel).  For national-team RPS evaluation, independent
Poisson at the grand mean is the correct intercept-only baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models.base import BaseModel


class MeanRatePoisson(BaseModel):
    """Symmetric mean-rate Poisson baseline.

    Predicts λ = grand mean of (home_goals + away_goals) / 2 for every match.
    No hyperparameters; Optuna tuning is not needed or run for this model.
    """

    def __init__(self) -> None:
        self._lambda: float | None = None

    @property
    def name(self) -> str:
        return "mean_rate_poisson"

    @property
    def distribution_family(self) -> str:
        return "poisson"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,  # noqa: ARG002
    ) -> MeanRatePoisson:
        """Compute and store the (unweighted) grand mean goal rate.

        Args:
            X: Feature matrix (ignored — this model uses no features).
            y: Target matrix (n_samples, 2) — columns [home_goals, away_goals].
            sample_weight: Accepted for interface parity but **ignored**.  This
                is a no-information floor; its prediction is a single constant
                per match, so time-weighting would only nudge the scalar and
                muddy its role as the entropy floor (see docs/notes/decisions.md).
        """
        self._lambda = float((y[:, 0].mean() + y[:, 1].mean()) / 2)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._lambda is None:
            raise RuntimeError("Model has not been fitted yet.")
        n = len(X)
        lam = np.full(n, self._lambda)
        return lam, lam.copy()

    def get_params(self) -> dict[str, Any]:
        if self._lambda is not None:
            return {"lambda": self._lambda}
        return {}
