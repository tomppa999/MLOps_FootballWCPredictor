"""Bayesian Bivariate Poisson regression via PyMC.

Implements the Karlis & Ntzoufras (2003) bivariate Poisson distribution in a
fully Bayesian framework using PyMC's NUTS sampler.  A match is modelled as
three latent Poisson variates:

    X1 ~ Pois(λ1),  X2 ~ Pois(λ2),  X3 ~ Pois(λ3)
    home_goals = X1 + X3,  away_goals = X2 + X3

where λ3 captures positive score dependence.  Each rate has its own coefficient
vector with Normal(0, prior_sigma) priors.  The bivariate Poisson log-PMF is
implemented entirely in PyTensor ops for compatibility with pm.CustomDist.

Reference:
    Karlis, D. & Ntzoufras, I. (2003). Analysis of sports data by using
    bivariate Poisson models. JRSS-D, 52(3), 381–393.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel

logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

K_MAX = 10


def _bvp_logp_pytensor(
    h: pt.TensorVariable,
    a: pt.TensorVariable,
    lam1: pt.TensorVariable,
    lam2: pt.TensorVariable,
    lam3: pt.TensorVariable,
) -> pt.TensorVariable:
    """Bivariate Poisson log-PMF in PyTensor ops.

    P(H=h, A=a) = exp(-(λ1+λ2+λ3)) * (λ1^h / h!) * (λ2^a / a!)
                  * Σ_{k=0}^{min(h,a)} C(h,k)*C(a,k)*k! * (λ3/(λ1*λ2))^k

    The inner sum is truncated at K_MAX for tractability. Uses gammaln and
    logsumexp for numerical stability.
    """
    # #region agent log
    import json, time; open("/Users/tomfarnschlaeder/Projects/MLOps_FootballWCPredictor/.cursor/debug-837e79.log", "a").write(json.dumps({"sessionId":"837e79","hypothesisId":"A,B,C","location":"bayesian_bivariate_poisson.py:_bvp_logp_pytensor:entry","message":"logp_fn called","data":{"lam1_type":str(type(lam1)),"lam2_type":str(type(lam2)),"lam3_type":str(type(lam3)),"lam1_is_none":lam1 is None,"lam2_is_none":lam2 is None,"lam3_is_none":lam3 is None,"h_type":str(type(h)),"a_type":str(type(a))},"timestamp":int(time.time()*1000)}) + "\n")
    # #endregion
    h = pt.flatten(pt.cast(h, "float64"))
    a = pt.flatten(pt.cast(a, "float64"))
    lam1 = pt.flatten(lam1)
    lam2 = pt.flatten(lam2)
    lam3 = pt.flatten(lam3)

    # #region agent log
    open("/Users/tomfarnschlaeder/Projects/MLOps_FootballWCPredictor/.cursor/debug-837e79.log", "a").write(json.dumps({"sessionId":"837e79","hypothesisId":"D,E","runId":"post-fix","location":"bayesian_bivariate_poisson.py:shapes","message":"tensor ndims after flatten","data":{"h_ndim":h.ndim,"lam1_ndim":lam1.ndim},"timestamp":int(time.time()*1000)}) + "\n")
    # #endregion

    log_lam1 = pt.log(pt.maximum(lam1, 1e-12))
    log_lam2 = pt.log(pt.maximum(lam2, 1e-12))
    log_lam3 = pt.log(pt.maximum(lam3, 1e-12))

    log_base = (
        -(lam1 + lam2 + lam3)
        + h * log_lam1
        - pt.gammaln(h + 1)
        + a * log_lam2
        - pt.gammaln(a + 1)
    )

    k = pt.arange(K_MAX + 1, dtype="float64")

    h_col = pt.shape_padright(h)
    a_col = pt.shape_padright(a)
    k_row = pt.shape_padleft(k)
    log_lam1_col = pt.shape_padright(log_lam1)
    log_lam2_col = pt.shape_padright(log_lam2)
    log_lam3_col = pt.shape_padright(log_lam3)

    h_minus_k = pt.maximum(h_col - k_row, 0.0)
    a_minus_k = pt.maximum(a_col - k_row, 0.0)

    log_terms = (
        pt.gammaln(h_col + 1) - pt.gammaln(k_row + 1) - pt.gammaln(h_minus_k + 1)
        + pt.gammaln(a_col + 1) - pt.gammaln(k_row + 1) - pt.gammaln(a_minus_k + 1)
        + pt.gammaln(k_row + 1)
        + k_row * (log_lam3_col - log_lam1_col - log_lam2_col)
    )

    valid = k_row <= pt.minimum(h_col, a_col)
    log_terms = pt.switch(valid, log_terms, -1e30)
    log_sum = pt.logsumexp(log_terms, axis=-1)

    return log_base + log_sum


class BayesianBivariatePoisson(BaseModel):
    """Bayesian Bivariate Poisson (Karlis-Ntzoufras) via PyMC NUTS.

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
        self._beta_1: np.ndarray | None = None
        self._beta_2: np.ndarray | None = None
        self._beta_3: np.ndarray | None = None
        self._intercept_1: float = 0.0
        self._intercept_2: float = 0.0
        self._intercept_3: float = 0.0

    @property
    def name(self) -> str:
        return "bayesian_bvp"

    def fit(self, X: np.ndarray, y: np.ndarray) -> BayesianBivariatePoisson:
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X).astype(np.float64)
        n, p = Xs.shape

        h_obs = np.maximum(np.round(y[:, 0]).astype(int), 0)
        a_obs = np.maximum(np.round(y[:, 1]).astype(int), 0)

        obs = np.column_stack([h_obs, a_obs]).astype(np.float64)

        with pm.Model():
            Xs_shared = pm.Data("Xs", Xs)
            obs_shared = pm.Data("obs", obs)

            intercept_1 = pm.Normal("intercept_1", 0.0, sigma=self.prior_sigma)
            beta_1 = pm.Normal("beta_1", 0.0, sigma=self.prior_sigma, shape=p)
            intercept_2 = pm.Normal("intercept_2", 0.0, sigma=self.prior_sigma)
            beta_2 = pm.Normal("beta_2", 0.0, sigma=self.prior_sigma, shape=p)
            intercept_3 = pm.Normal("intercept_3", -1.0, sigma=self.prior_sigma)
            beta_3 = pm.Normal("beta_3", 0.0, sigma=self.prior_sigma, shape=p)

            lam1 = pm.math.exp(intercept_1 + pm.math.dot(Xs_shared, beta_1))
            lam2 = pm.math.exp(intercept_2 + pm.math.dot(Xs_shared, beta_2))
            lam3 = pm.math.exp(intercept_3 + pm.math.dot(Xs_shared, beta_3))

            def logp_fn(value, lam1, lam2, lam3):
                # #region agent log
                import json, time; open("/Users/tomfarnschlaeder/Projects/MLOps_FootballWCPredictor/.cursor/debug-837e79.log", "a").write(json.dumps({"sessionId":"837e79","hypothesisId":"C","location":"bayesian_bivariate_poisson.py:logp_fn:entry","message":"logp_fn closure called","data":{"value_type":str(type(value)),"value_is_none":value is None,"lam1_type":str(type(lam1)),"lam1_is_none":lam1 is None,"lam2_type":str(type(lam2)),"lam2_is_none":lam2 is None,"lam3_type":str(type(lam3)),"lam3_is_none":lam3 is None},"timestamp":int(time.time()*1000)}) + "\n")
                # #endregion
                h = value[:, 0]
                a = value[:, 1]
                return _bvp_logp_pytensor(h, a, lam1, lam2, lam3)

            pm.CustomDist(
                "score",
                lam1,
                lam2,
                lam3,
                logp=logp_fn,
                observed=obs_shared,
            )

            trace = pm.sample(
                draws=self.draws,
                tune=self.tune_steps,
                chains=1,
                progressbar=False,
                random_seed=self.random_seed,
            )

        post = trace.posterior
        self._beta_1 = post["beta_1"].mean(dim=["chain", "draw"]).values
        self._intercept_1 = float(post["intercept_1"].mean(dim=["chain", "draw"]).values)
        self._beta_2 = post["beta_2"].mean(dim=["chain", "draw"]).values
        self._intercept_2 = float(post["intercept_2"].mean(dim=["chain", "draw"]).values)
        self._beta_3 = post["beta_3"].mean(dim=["chain", "draw"]).values
        self._intercept_3 = float(post["intercept_3"].mean(dim=["chain", "draw"]).values)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._scaler is None or self._beta_1 is None:
            raise RuntimeError("Model has not been fitted yet.")
        Xs = self._scaler.transform(X).astype(np.float64)
        lam1 = np.exp(self._intercept_1 + Xs @ self._beta_1).clip(1e-6)
        lam2 = np.exp(self._intercept_2 + Xs @ self._beta_2).clip(1e-6)
        lam3 = np.exp(self._intercept_3 + Xs @ self._beta_3).clip(1e-6)
        return lam1 + lam3, lam2 + lam3

    def get_params(self) -> dict[str, Any]:
        return {
            "prior_sigma": self.prior_sigma,
            "draws": self.draws,
            "tune_steps": self.tune_steps,
        }
