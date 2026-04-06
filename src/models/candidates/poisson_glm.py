"""Bivariate Poisson GLM via custom MLE.

Reference:
    Karlis, D. & Ntzoufras, I. (2003). Analysis of sports data by using bivariate
    Poisson models. Journal of the Royal Statistical Society: Series D
    (The Statistician), 52(3), 381–393.

The bivariate Poisson distribution models a match as three independent Poisson
variates: X1 ~ Pois(λ1), X2 ~ Pois(λ2), X3 ~ Pois(λ3), where

    home_goals = X1 + X3
    away_goals = X2 + X3

and λ3 captures positive score dependence between sides.  Each rate is
parameterised as exp(β^T x) with an optional L2 penalty α on the coefficients.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel


def _bvp_log_pmf(
    h: np.ndarray,
    a: np.ndarray,
    lam1: np.ndarray,
    lam2: np.ndarray,
    lam3: np.ndarray,
) -> np.ndarray:
    """Log-PMF of the bivariate Poisson distribution (fully vectorized).

    P(H=h, A=a) = exp(-(λ1+λ2+λ3)) * (λ1^h / h!) * (λ2^a / a!)
                  * Σ_{k=0}^{min(h,a)} C(h,k)*C(a,k)*k! * (λ3/(λ1*λ2))^k

    Computed in log-space for numerical stability.  The inner sum is broadcast
    over a (n, K_max+1) grid — no Python loop over samples.
    """
    h = np.asarray(h, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)

    log_base = (
        -(lam1 + lam2 + lam3)
        + h * np.log(lam1)
        - gammaln(h + 1)
        + a * np.log(lam2)
        - gammaln(a + 1)
    )

    min_ha = np.minimum(h, a).astype(int)
    K_max = int(min_ha.max()) if len(min_ha) > 0 else 0

    # (1, K+1) grid of k values broadcast against (n, 1) sample vectors
    k = np.arange(K_max + 1, dtype=np.float64)[None, :]
    h_col = h[:, None]
    a_col = a[:, None]

    # Clamp differences to >= 0 so gammaln doesn't produce NaN on invalid k;
    # those entries are masked to -inf below.
    h_minus_k = np.maximum(h_col - k, 0.0)
    a_minus_k = np.maximum(a_col - k, 0.0)

    log_terms = (
        gammaln(h_col + 1) - gammaln(k + 1) - gammaln(h_minus_k + 1)
        + gammaln(a_col + 1) - gammaln(k + 1) - gammaln(a_minus_k + 1)
        + gammaln(k + 1)
        + k * (np.log(lam3[:, None]) - np.log(lam1[:, None]) - np.log(lam2[:, None]))
    )

    valid = k <= min_ha[:, None]
    log_terms = np.where(valid, log_terms, -np.inf)
    log_sum = np.logaddexp.reduce(log_terms, axis=1)

    return log_base + log_sum


class BivariatePoisson(BaseModel):
    """Bivariate Poisson GLM with L2-regularised MLE (Karlis & Ntzoufras, 2003).

    Fits three log-linear models:
        log λ1 = X β1   (home goals, unique component)
        log λ2 = X β2   (away goals, unique component)
        log λ3 = X β3   (shared dependence component)

    Coefficients are initialised at zero (i.e. λ ≈ 1) and optimised via
    L-BFGS-B with an optional L2 penalty (``alpha``).
    """

    def __init__(self, alpha: float = 0.1, maxiter: int = 500) -> None:
        """
        Args:
            alpha: L2 penalty strength on all coefficients (excluding intercepts).
            maxiter: Maximum L-BFGS-B iterations (lower for fast unit tests).
        """
        self.alpha = alpha
        self.maxiter = maxiter
        self._scaler: StandardScaler | None = None
        self._coef1: np.ndarray | None = None
        self._coef2: np.ndarray | None = None
        self._coef3: np.ndarray | None = None
        self._n_features: int = 0

    @property
    def name(self) -> str:
        return "poisson_glm"

    def _build_design(self, X: np.ndarray) -> np.ndarray:
        """Prepend intercept column."""
        return np.column_stack([np.ones(len(X)), X])

    def _unpack(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self._n_features + 1  # +1 for intercept
        return params[:p], params[p : 2 * p], params[2 * p :]

    def _neg_log_likelihood(
        self, params: np.ndarray, Xd: np.ndarray, h: np.ndarray, a: np.ndarray
    ) -> float:
        b1, b2, b3 = self._unpack(params)
        lam1 = np.exp(Xd @ b1).clip(1e-6)
        lam2 = np.exp(Xd @ b2).clip(1e-6)
        lam3 = np.exp(Xd @ b3).clip(1e-6)

        ll = _bvp_log_pmf(h, a, lam1, lam2, lam3).sum()

        # L2 penalty on non-intercept coefficients
        penalty = 0.5 * self.alpha * (
            (b1[1:] ** 2).sum() + (b2[1:] ** 2).sum() + (b3[1:] ** 2).sum()
        )
        return -ll + penalty

    def fit(self, X: np.ndarray, y: np.ndarray) -> BivariatePoisson:
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        self._n_features = Xs.shape[1]

        h = y[:, 0].astype(np.float64)
        a = y[:, 1].astype(np.float64)

        Xd = self._build_design(Xs)
        p = self._n_features + 1
        x0 = np.zeros(3 * p)

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(Xd, h, a),
            method="L-BFGS-B",
            options={"maxiter": self.maxiter, "ftol": 1e-9},
        )

        b1, b2, b3 = self._unpack(result.x)
        self._coef1, self._coef2, self._coef3 = b1, b2, b3
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._scaler is None or self._coef1 is None:
            raise RuntimeError("Model has not been fitted yet.")
        Xs = self._scaler.transform(X)
        Xd = self._build_design(Xs)
        lam1 = np.exp(Xd @ self._coef1).clip(1e-6)
        lam2 = np.exp(Xd @ self._coef2).clip(1e-6)
        lam3 = np.exp(Xd @ self._coef3).clip(1e-6)
        # Expected goals per side = λ1 + λ3 (home) and λ2 + λ3 (away)
        return lam1 + lam3, lam2 + lam3

    def get_params(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "maxiter": self.maxiter}
