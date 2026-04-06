"""Tests for BayesianPoissonModel.

Uses minimal draws/tune_steps to keep the test suite fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.candidates.bayesian_poisson import BayesianPoissonModel

_FAST = {"draws": 20, "tune_steps": 20}


def _make_data(
    n: int = 60,
    n_features: int = 8,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 5, size=(n, 2)).astype(np.float64)
    return X, y


class TestBayesianPoissonModel:
    def test_name(self):
        assert BayesianPoissonModel().name == "bayesian_poisson"

    def test_get_params_keys(self):
        m = BayesianPoissonModel(prior_sigma=2.0, draws=100, tune_steps=200)
        p = m.get_params()
        assert p["prior_sigma"] == 2.0
        assert p["draws"] == 100
        assert p["tune_steps"] == 200

    def test_fit_returns_self(self):
        X, y = _make_data()
        m = BayesianPoissonModel(**_FAST)
        assert m.fit(X, y) is m

    def test_predict_shape(self):
        X, y = _make_data()
        m = BayesianPoissonModel(**_FAST).fit(X, y)
        lh, la = m.predict(X)
        assert lh.shape == (len(X),)
        assert la.shape == (len(X),)

    def test_predictions_positive(self):
        X, y = _make_data()
        m = BayesianPoissonModel(**_FAST).fit(X, y)
        lh, la = m.predict(X)
        assert (lh > 0).all()
        assert (la > 0).all()

    def test_predict_new_samples(self):
        X, y = _make_data(n=60)
        m = BayesianPoissonModel(**_FAST).fit(X[:50], y[:50])
        lh, la = m.predict(X[50:])
        assert lh.shape == (10,)
        assert la.shape == (10,)

    def test_predict_before_fit_raises(self):
        m = BayesianPoissonModel()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 8)))

    def test_predictions_finite(self):
        X, y = _make_data()
        m = BayesianPoissonModel(**_FAST).fit(X, y)
        lh, la = m.predict(X)
        assert np.isfinite(lh).all()
        assert np.isfinite(la).all()
