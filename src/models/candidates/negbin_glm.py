"""Negative Binomial GLM (independent marginals).

Two separate statsmodels NegativeBinomial GLMs — one for home goals,
one for away goals.  Features are standardised inside the model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel


class NegativeBinomialGLM(BaseModel):
    """Independent NegBin GLMs for home and away expected goals.

    Args:
        alpha: Initial dispersion parameter passed to statsmodels as the
            starting value for the ancillary parameter.  statsmodels
            estimates the final value via MLE.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._scaler: StandardScaler | None = None
        self._model_home: Any = None
        self._model_away: Any = None

    @property
    def name(self) -> str:
        return "negbin_glm"

    def fit(self, X: np.ndarray, y: np.ndarray) -> NegativeBinomialGLM:
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        Xd = sm.add_constant(Xs)

        h = y[:, 0].astype(np.float64)
        a = y[:, 1].astype(np.float64)

        self._model_home = sm.GLM(
            h, Xd, family=sm.families.NegativeBinomial(alpha=self.alpha)
        ).fit(disp=False)

        self._model_away = sm.GLM(
            a, Xd, family=sm.families.NegativeBinomial(alpha=self.alpha)
        ).fit(disp=False)

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._scaler is None or self._model_home is None:
            raise RuntimeError("Model has not been fitted yet.")
        Xs = self._scaler.transform(X)
        Xd = sm.add_constant(Xs, has_constant="add")
        lam_h = self._model_home.predict(Xd)
        lam_a = self._model_away.predict(Xd)
        return np.asarray(lam_h), np.asarray(lam_a)

    def get_params(self) -> dict[str, Any]:
        return {"alpha": self.alpha}
