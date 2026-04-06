"""Random Forest regressor (native multi-output).

No feature scaling needed.  Uses sklearn's native multi-output support so
a single forest predicts both home and away goals simultaneously.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.models.base import BaseModel


class RandomForestModel(BaseModel):
    """Multi-output Random Forest for expected goals prediction.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth (None = unlimited).
        min_samples_leaf: Minimum samples at a leaf node.
        max_features: Number of features to consider at each split.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: str | float = "sqrt",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._model: RandomForestRegressor | None = None

    @property
    def name(self) -> str:
        return "random_forest"

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestModel:
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        preds = self._model.predict(X).clip(0)
        return preds[:, 0], preds[:, 1]

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": self.random_state,
        }
