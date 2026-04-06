"""Tests for RandomForestModel."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.candidates.random_forest import RandomForestModel


def _make_data(n: int = 120, n_features: int = 4, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 6, size=(n, 2)).astype(np.float64)
    return X, y


class TestRandomForestModel:
    def test_name(self):
        assert RandomForestModel().name == "random_forest"

    def test_get_params_keys(self):
        m = RandomForestModel(n_estimators=50, max_depth=5)
        params = m.get_params()
        assert "n_estimators" in params
        assert "max_depth" in params
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 5

    def test_fit_returns_self(self):
        X, y = _make_data()
        m = RandomForestModel(n_estimators=10)
        assert m.fit(X, y) is m

    def test_predict_shape(self):
        X, y = _make_data()
        m = RandomForestModel(n_estimators=10).fit(X, y)
        lh, la = m.predict(X)
        assert lh.shape == (len(X),)
        assert la.shape == (len(X),)

    def test_predictions_non_negative(self):
        X, y = _make_data()
        m = RandomForestModel(n_estimators=10).fit(X, y)
        lh, la = m.predict(X)
        assert (lh >= 0).all()
        assert (la >= 0).all()

    def test_predict_before_fit_raises(self):
        m = RandomForestModel()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 4)))

    def test_predict_on_new_samples(self):
        X, y = _make_data()
        m = RandomForestModel(n_estimators=10).fit(X[:100], y[:100])
        lh, la = m.predict(X[100:])
        assert lh.shape == (20,)

    def test_deterministic_with_seed(self):
        X, y = _make_data()
        m1 = RandomForestModel(n_estimators=10, random_state=42).fit(X, y)
        m2 = RandomForestModel(n_estimators=10, random_state=42).fit(X, y)
        lh1, _ = m1.predict(X)
        lh2, _ = m2.predict(X)
        np.testing.assert_array_equal(lh1, lh2)
