"""Tests for SARIMAXModel."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.candidates.sarimax import SARIMAXModel


def _make_data(
    n: int = 60,
    n_features: int = 8,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 5, size=(n, 2)).astype(np.float64)
    return X, y


class TestSARIMAXModel:
    def test_name(self):
        assert SARIMAXModel().name == "sarimax"

    def test_get_params_keys(self):
        m = SARIMAXModel(p=2, d=1, q=1)
        p = m.get_params()
        assert p["p"] == 2
        assert p["d"] == 1
        assert p["q"] == 1

    def test_fit_returns_self(self):
        X, y = _make_data()
        m = SARIMAXModel(p=1, d=0, q=0)
        assert m.fit(X, y) is m

    def test_predict_shape(self):
        X, y = _make_data()
        m = SARIMAXModel(p=1, d=0, q=0).fit(X[:50], y[:50])
        lh, la = m.predict(X[50:])
        assert lh.shape == (10,)
        assert la.shape == (10,)

    def test_predictions_non_negative(self):
        X, y = _make_data()
        m = SARIMAXModel(p=1, d=0, q=0).fit(X[:50], y[:50])
        lh, la = m.predict(X[50:])
        assert (lh >= 0).all()
        assert (la >= 0).all()

    def test_predict_before_fit_raises(self):
        m = SARIMAXModel()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 8)))

    def test_predict_multiple_steps(self):
        X, y = _make_data(n=80)
        m = SARIMAXModel(p=1, d=0, q=0).fit(X[:60], y[:60])
        lh, la = m.predict(X[60:])
        assert lh.shape == (20,)
        assert la.shape == (20,)

    def test_predictions_finite(self):
        X, y = _make_data()
        m = SARIMAXModel(p=0, d=0, q=0).fit(X[:50], y[:50])
        lh, la = m.predict(X[50:])
        assert np.isfinite(lh).all()
        assert np.isfinite(la).all()
