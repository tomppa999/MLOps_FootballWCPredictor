"""Abstract base class for all candidate models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    """All 10 candidate models implement this interface.

    ``predict`` must return Poisson-rate estimates (λ_home, λ_away).
    Non-distributional models (Ridge, RF, XGBoost) treat their point
    predictions as Poisson means for downstream outcome-probability
    computation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique identifier, e.g. 'poisson_glm'."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseModel:
        """Fit on training data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target matrix (n_samples, 2) — columns [home_goals, away_goals].

        Returns:
            self (for chaining).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict expected goals.

        Returns:
            (lambda_home, lambda_away) — each shape (n_samples,).
        """

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return all hyperparameters (for MLflow logging)."""
