"""SARIMAX model — two independent instances for home and away goals.

Each model is a SARIMAX(p, d, q) fitted on the time-ordered training sequence
with standardised Core features as exogenous regressors.

Prediction uses ``forecast(steps=n, exog=X_test)``, which produces n out-of-
sample predictions.  For long horizons the AR/MA contribution decays toward the
unconditional mean; the exogenous feature contribution dominates for large
feature effects, making this pragmatically acceptable as a candidate model.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.base import BaseModel


class SARIMAXModel(BaseModel):
    """Two independent SARIMAX(p, d, q) models with Core exogenous features.

    Args:
        p: AR order.
        d: Differencing order.
        q: MA order.
        maxiter: Maximum MLE optimisation iterations.
    """

    def __init__(
        self,
        p: int = 1,
        d: int = 0,
        q: int = 0,
        maxiter: int = 100,
    ) -> None:
        self.p = p
        self.d = d
        self.q = q
        self.maxiter = maxiter
        self._scaler: StandardScaler | None = None
        self._result_home: Any = None
        self._result_away: Any = None

    @property
    def name(self) -> str:
        return "sarimax"

    def fit(self, X: np.ndarray, y: np.ndarray) -> SARIMAXModel:
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)

        h = y[:, 0].astype(np.float64)
        a = y[:, 1].astype(np.float64)

        order = (self.p, self.d, self.q)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result_home = SARIMAX(
                h, exog=Xs, order=order, trend="c"
            ).fit(disp=False, maxiter=self.maxiter)
            self._result_away = SARIMAX(
                a, exog=Xs, order=order, trend="c"
            ).fit(disp=False, maxiter=self.maxiter)

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._scaler is None or self._result_home is None:
            raise RuntimeError("Model has not been fitted yet.")
        Xs = self._scaler.transform(X)
        n = len(Xs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lam_h = np.asarray(
                self._result_home.forecast(steps=n, exog=Xs)
            ).clip(0)
            lam_a = np.asarray(
                self._result_away.forecast(steps=n, exog=Xs)
            ).clip(0)

        return lam_h, lam_a

    def get_params(self) -> dict[str, Any]:
        return {"p": self.p, "d": self.d, "q": self.q, "maxiter": self.maxiter}
