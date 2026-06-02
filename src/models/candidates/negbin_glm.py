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
        self._fitted_alpha_home: float | None = None
        self._fitted_alpha_away: float | None = None

    @property
    def name(self) -> str:
        return "negbin_glm"

    @property
    def distribution_family(self) -> str:
        return "negbin"

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

        # Store the MLE-estimated dispersion from statsmodels.
        # NegativeBinomial family exposes the estimated ancillary
        # parameter via .scale on the fitted result.
        self._fitted_alpha_home = float(self._model_home.scale)
        self._fitted_alpha_away = float(self._model_away.scale)

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._scaler is None or self._model_home is None:
            raise RuntimeError("Model has not been fitted yet.")
        Xs = self._scaler.transform(X)
        Xd = sm.add_constant(Xs, has_constant="add")
        lam_h = self._model_home.predict(Xd)
        lam_a = self._model_away.predict(Xd)
        return np.asarray(lam_h), np.asarray(lam_a)

    def predict_with_dispersion(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Predict expected goals together with fitted NegBin dispersion.

        Returns:
            (lambda_h, lambda_a, alpha_h, alpha_a) where alpha_* are the
            MLE-estimated dispersion parameters from the fitted GLMs.
        """
        if self._fitted_alpha_home is None or self._fitted_alpha_away is None:
            raise RuntimeError("Model has not been fitted yet.")
        lam_h, lam_a = self.predict(X)
        return lam_h, lam_a, self._fitted_alpha_home, self._fitted_alpha_away

    def get_params(self) -> dict[str, Any]:
        return {"alpha": self.alpha}
