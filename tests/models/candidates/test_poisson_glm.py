"""Tests for BivariatePoisson."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.candidates.poisson_glm import BivariatePoisson


def _make_data(n: int = 80, n_features: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 5, size=(n, 2)).astype(np.float64)
    return X, y


_FAST = dict(maxiter=50)


class TestBivariatePoisson:
    def test_name(self):
        assert BivariatePoisson().name == "poisson_glm"

    def test_get_params_roundtrip(self):
        m = BivariatePoisson(alpha=0.5, maxiter=100)
        assert m.get_params() == {"alpha": 0.5, "maxiter": 100}

    def test_fit_returns_self(self):
        X, y = _make_data()
        m = BivariatePoisson(**_FAST)
        assert m.fit(X, y) is m

    def test_predict_shape(self):
        X, y = _make_data()
        m = BivariatePoisson(**_FAST).fit(X, y)
        lh, la = m.predict(X)
        assert lh.shape == (len(X),)
        assert la.shape == (len(X),)

    def test_predictions_positive(self):
        X, y = _make_data()
        m = BivariatePoisson(**_FAST).fit(X, y)
        lh, la = m.predict(X)
        assert (lh > 0).all()
        assert (la > 0).all()

    def test_predict_before_fit_raises(self):
        m = BivariatePoisson()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 3)))

    def test_single_feature(self):
        X, y = _make_data(n=60, n_features=1)
        m = BivariatePoisson(**_FAST).fit(X, y)
        lh, la = m.predict(X[:10])
        assert lh.shape == (10,)

    def test_alpha_zero_still_converges(self):
        X, y = _make_data()
        m = BivariatePoisson(alpha=0.0, **_FAST).fit(X, y)
        lh, la = m.predict(X)
        assert np.isfinite(lh).all()
        assert np.isfinite(la).all()
