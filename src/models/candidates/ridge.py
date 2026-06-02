"""Ridge regression baseline.

StandardScaler applied explicitly, then a MultiOutputRegressor(Ridge) so both
home and away goals share the same interface and per-sample weights forward
cleanly to the estimators.  Negative predictions are clipped to 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel


class RidgeModel(BaseModel):
    """Ridge regression with standard scaling, predicting home and away goals.

    Args:
        alpha: L2 regularisation strength (sklearn Ridge convention).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._scaler: StandardScaler | None = None
        self._model: MultiOutputRegressor | None = None

    @property
    def name(self) -> str:
        return "ridge"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> RidgeModel:
        # Scale explicitly (rather than via a Pipeline) so sample_weight can be
        # forwarded straight to the Ridge estimators by MultiOutputRegressor.
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        self._model = MultiOutputRegressor(Ridge(alpha=self.alpha))
        fit_kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
        self._model.fit(Xs, y, **fit_kwargs)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model has not been fitted yet.")
        preds = self._model.predict(self._scaler.transform(X)).clip(0)
        return preds[:, 0], preds[:, 1]

    def get_params(self) -> dict[str, Any]:
        return {"alpha": self.alpha}
