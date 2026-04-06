"""XGBoost regressor (two independent models, Full feature set).

Two separate XGBRegressor instances — one for home goals, one for away goals.
XGBoost handles NaN natively, so this is the only model that uses the Full
feature set (Core + rolling shots + tactical + clusters).
No feature scaling needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBRegressor

from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    """Two XGBRegressors predicting home and away expected goals independently.

    Args:
        learning_rate: Step size shrinkage (eta).
        max_depth: Maximum tree depth.
        n_estimators: Number of boosting rounds.
        subsample: Row sub-sampling ratio per tree.
        colsample_bytree: Column sub-sampling ratio per tree.
        reg_lambda: L2 regularisation term on leaf weights.
        random_state: Random seed.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self._model_home: XGBRegressor | None = None
        self._model_away: XGBRegressor | None = None

    @property
    def name(self) -> str:
        return "xgboost"

    def _make_estimator(self) -> XGBRegressor:
        return XGBRegressor(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbosity=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBoostModel:
        self._model_home = self._make_estimator()
        self._model_away = self._make_estimator()
        self._model_home.fit(X, y[:, 0])
        self._model_away.fit(X, y[:, 1])
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model_home is None or self._model_away is None:
            raise RuntimeError("Model has not been fitted yet.")
        lam_h = self._model_home.predict(X).clip(0)
        lam_a = self._model_away.predict(X).clip(0)
        return lam_h, lam_a

    def get_params(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
        }
