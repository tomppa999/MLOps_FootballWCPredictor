"""Ridge regression baseline.

Two sklearn Pipelines (StandardScaler → Ridge), wrapped in a
MultiOutputRegressor so both home and away goals share the same interface.
Negative predictions are clipped to 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel


class RidgeModel(BaseModel):
    """Ridge regression with standard scaling, predicting home and away goals.

    Args:
        alpha: L2 regularisation strength (sklearn Ridge convention).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._pipeline: MultiOutputRegressor | None = None

    @property
    def name(self) -> str:
        return "ridge"

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeModel:
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=self.alpha))])
        self._pipeline = MultiOutputRegressor(pipe)
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._pipeline is None:
            raise RuntimeError("Model has not been fitted yet.")
        preds = self._pipeline.predict(X).clip(0)
        return preds[:, 0], preds[:, 1]

    def get_params(self) -> dict[str, Any]:
        return {"alpha": self.alpha}
