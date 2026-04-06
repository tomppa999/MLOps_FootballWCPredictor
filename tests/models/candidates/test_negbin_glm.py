"""Tests for NegativeBinomialGLM."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.candidates.negbin_glm import NegativeBinomialGLM


def _make_data(n: int = 100, n_features: int = 3, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 6, size=(n, 2)).astype(np.float64)
    return X, y


class TestNegativeBinomialGLM:
    def test_name(self):
        assert NegativeBinomialGLM().name == "negbin_glm"

    def test_get_params_roundtrip(self):
        m = NegativeBinomialGLM(alpha=2.0)
        assert m.get_params() == {"alpha": 2.0}

    def test_fit_returns_self(self):
        X, y = _make_data()
        assert NegativeBinomialGLM().fit(X, y) is NegativeBinomialGLM().fit(X, y).__class__.__mro__[0] or True
        m = NegativeBinomialGLM()
        assert m.fit(X, y) is m

    def test_predict_shape(self):
        X, y = _make_data()
        m = NegativeBinomialGLM().fit(X, y)
        lh, la = m.predict(X)
        assert lh.shape == (len(X),)
        assert la.shape == (len(X),)

    def test_predictions_positive(self):
        X, y = _make_data()
        m = NegativeBinomialGLM().fit(X, y)
        lh, la = m.predict(X)
        assert (lh > 0).all()
        assert (la > 0).all()

    def test_predict_before_fit_raises(self):
        m = NegativeBinomialGLM()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 3)))

    def test_predictions_finite(self):
        X, y = _make_data()
        m = NegativeBinomialGLM().fit(X, y)
        lh, la = m.predict(X)
        assert np.isfinite(lh).all()
        assert np.isfinite(la).all()
