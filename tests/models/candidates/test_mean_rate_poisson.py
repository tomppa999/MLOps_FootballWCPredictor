"""Tests for MeanRatePoisson baseline model."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.candidates.mean_rate_poisson import MeanRatePoisson


def _make_data(n: int = 50, n_features: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 5, size=(n, 2)).astype(np.float64)
    return X, y


class TestMeanRatePoisson:
    def test_name(self):
        assert MeanRatePoisson().name == "mean_rate_poisson"

    def test_distribution_family(self):
        assert MeanRatePoisson().distribution_family == "poisson"

    def test_get_params_empty_before_fit(self):
        assert MeanRatePoisson().get_params() == {}

    def test_get_params_after_fit(self):
        X, y = _make_data()
        m = MeanRatePoisson().fit(X, y)
        params = m.get_params()
        assert "lambda" in params
        assert params["lambda"] > 0

    def test_fit_returns_self(self):
        X, y = _make_data()
        m = MeanRatePoisson()
        assert m.fit(X, y) is m

    def test_predict_shape(self):
        X, y = _make_data()
        m = MeanRatePoisson().fit(X, y)
        lh, la = m.predict(X)
        assert lh.shape == (len(X),)
        assert la.shape == (len(X),)

    def test_predict_symmetric(self):
        """Both sides must receive the same lambda."""
        X, y = _make_data()
        m = MeanRatePoisson().fit(X, y)
        lh, la = m.predict(X)
        np.testing.assert_array_equal(lh, la)

    def test_predict_constant(self):
        """All rows receive the same value regardless of features."""
        X, y = _make_data()
        m = MeanRatePoisson().fit(X, y)
        lh, _ = m.predict(X)
        assert np.all(lh == lh[0])

    def test_lambda_equals_grand_mean(self):
        X, y = _make_data()
        expected = (y[:, 0].mean() + y[:, 1].mean()) / 2
        m = MeanRatePoisson().fit(X, y)
        lh, la = m.predict(X)
        np.testing.assert_allclose(lh[0], expected, atol=1e-10)

    def test_predict_ignores_features(self):
        """Predictions must not change when features differ."""
        X, y = _make_data()
        m = MeanRatePoisson().fit(X, y)
        lh1, _ = m.predict(X)
        rng = np.random.default_rng(42)
        X_random = rng.standard_normal(X.shape)
        lh2, _ = m.predict(X_random)
        np.testing.assert_array_equal(lh1, lh2)

    def test_predict_before_fit_raises(self):
        m = MeanRatePoisson()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 3)))
