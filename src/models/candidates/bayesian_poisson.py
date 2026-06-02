"""Bayesian Poisson regression via PyMC.

Two independent Bayesian Poisson regressions (home goals, away goals) with
Normal(0, prior_sigma) priors on all coefficients.  MCMC posterior means serve
as the λ estimates for the standard ``predict()`` method.  Full posterior
samples are retained for distribution-aware scoring via ``predict_samples()``.

Note: models the goal rates as independent marginals rather than the full
bivariate Poisson joint distribution.  The key distinction from poisson_glm is
full Bayesian inference (MCMC) with prior-based regularisation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pymc as pm
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel

logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class BayesianPoissonModel(BaseModel):
    """Bayesian Poisson regression (independent marginals) via PyMC NUTS.

    Args:
        prior_sigma: Std of the Normal prior on all coefficients.
        draws: Posterior samples to draw per chain (1 chain used).
        tune_steps: Burn-in / tuning steps.
        random_seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        prior_sigma: float = 1.0,
        draws: int = 500,
        tune_steps: int = 500,
        random_seed: int = 42,
    ) -> None:
        self.prior_sigma = prior_sigma
        self.draws = draws
        self.tune_steps = tune_steps
        self.random_seed = random_seed
        self._scaler: StandardScaler | None = None
        # Posterior means (used by standard predict())
        self._beta_h: np.ndarray | None = None
        self._beta_a: np.ndarray | None = None
        self._intercept_h: float = 0.0
        self._intercept_a: float = 0.0
        # Full posterior samples — shape (draws, n_features) / (draws,)
        self._beta_h_samples: np.ndarray | None = None
        self._beta_a_samples: np.ndarray | None = None
        self._intercept_h_samples: np.ndarray | None = None
        self._intercept_a_samples: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "bayesian_poisson"

    @property
    def distribution_family(self) -> str:
        return "bayesian_poisson"

    def fit(self, X: np.ndarray, y: np.ndarray) -> BayesianPoissonModel:
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X).astype(np.float64)
        n, p = Xs.shape

        h_obs = np.maximum(np.round(y[:, 0]).astype(int), 0)
        a_obs = np.maximum(np.round(y[:, 1]).astype(int), 0)

        with pm.Model():
            intercept_h = pm.Normal("intercept_h", 0.0, sigma=self.prior_sigma)
            beta_h = pm.Normal("beta_h", 0.0, sigma=self.prior_sigma, shape=p)
            intercept_a = pm.Normal("intercept_a", 0.0, sigma=self.prior_sigma)
            beta_a = pm.Normal("beta_a", 0.0, sigma=self.prior_sigma, shape=p)

            lam_h = pm.math.exp(intercept_h + pm.math.dot(Xs, beta_h))
            lam_a = pm.math.exp(intercept_a + pm.math.dot(Xs, beta_a))

            pm.Poisson("home_goals", mu=lam_h, observed=h_obs)
            pm.Poisson("away_goals", mu=lam_a, observed=a_obs)

            trace = pm.sample(
                draws=self.draws,
                tune=self.tune_steps,
                chains=1,
                progressbar=False,
                random_seed=self.random_seed,
            )

        post = trace.posterior

        # Retain full posterior sample arrays (chain squeezed: shape (draws, p))
        self._beta_h_samples = post["beta_h"].values.squeeze(0)  # (draws, p)
        self._beta_a_samples = post["beta_a"].values.squeeze(0)
        self._intercept_h_samples = post["intercept_h"].values.squeeze(0)  # (draws,)
        self._intercept_a_samples = post["intercept_a"].values.squeeze(0)

        # Posterior means for fast predict()
        self._beta_h = self._beta_h_samples.mean(axis=0)
        self._intercept_h = float(self._intercept_h_samples.mean())
        self._beta_a = self._beta_a_samples.mean(axis=0)
        self._intercept_a = float(self._intercept_a_samples.mean())

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._scaler is None or self._beta_h is None:
            raise RuntimeError("Model has not been fitted yet.")
        Xs = self._scaler.transform(X).astype(np.float64)
        lam_h = np.exp(self._intercept_h + Xs @ self._beta_h).clip(1e-6)
        lam_a = np.exp(self._intercept_a + Xs @ self._beta_a).clip(1e-6)
        return lam_h, lam_a

    def predict_samples(
        self, X: np.ndarray, n_samples: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict lambda distributions using posterior samples.

        Args:
            X: Feature matrix, shape (n_obs, n_features).
            n_samples: Number of posterior draws to use.  Defaults to all.

        Returns:
            (lambda_h_samples, lambda_a_samples) each shape (n_samples, n_obs).
        """
        if self._scaler is None or self._beta_h_samples is None:
            raise RuntimeError("Model has not been fitted yet.")
        Xs = self._scaler.transform(X).astype(np.float64)  # (n_obs, p)

        bh = self._beta_h_samples  # (draws, p)
        ba = self._beta_a_samples
        ih = self._intercept_h_samples  # (draws,)
        ia = self._intercept_a_samples

        if n_samples is not None:
            bh = bh[:n_samples]
            ba = ba[:n_samples]
            ih = ih[:n_samples]
            ia = ia[:n_samples]

        # (draws, n_obs) via vectorised matmul
        lam_h_samples = np.exp(ih[:, None] + bh @ Xs.T).clip(1e-6)
        lam_a_samples = np.exp(ia[:, None] + ba @ Xs.T).clip(1e-6)
        return lam_h_samples, lam_a_samples

    def get_params(self) -> dict[str, Any]:
        return {
            "prior_sigma": self.prior_sigma,
            "draws": self.draws,
            "tune_steps": self.tune_steps,
        }
