"""Tests for RidgeModel."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.candidates.ridge import RidgeModel


def _make_data(n: int = 100, n_features: int = 4, seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 6, size=(n, 2)).astype(np.float64)
    return X, y


class TestRidgeModel:
    def test_name(self):
        assert RidgeModel().name == "ridge"

    def test_get_params_roundtrip(self):
        m = RidgeModel(alpha=10.0)
        assert m.get_params() == {"alpha": 10.0}

    def test_fit_returns_self(self):
        X, y = _make_data()
        m = RidgeModel()
        assert m.fit(X, y) is m

    def test_predict_shape(self):
        X, y = _make_data()
        m = RidgeModel().fit(X, y)
        lh, la = m.predict(X)
        assert lh.shape == (len(X),)
        assert la.shape == (len(X),)

    def test_predictions_non_negative(self):
        X, y = _make_data()
        m = RidgeModel().fit(X, y)
        lh, la = m.predict(X)
        assert (lh >= 0).all()
        assert (la >= 0).all()

    def test_predict_before_fit_raises(self):
        m = RidgeModel()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 4)))

    def test_predict_on_new_samples(self):
        X, y = _make_data()
        m = RidgeModel().fit(X[:80], y[:80])
        lh, la = m.predict(X[80:])
        assert lh.shape == (20,)
        assert la.shape == (20,)

    def test_high_alpha_shrinks_predictions(self):
        """Very high regularisation should produce predictions closer to the mean."""
        X, y = _make_data(n=200)
        m_low = RidgeModel(alpha=0.001).fit(X, y)
        m_high = RidgeModel(alpha=1e6).fit(X, y)
        lh_low, _ = m_low.predict(X)
        lh_high, _ = m_high.predict(X)
        assert lh_high.std() <= lh_low.std()
