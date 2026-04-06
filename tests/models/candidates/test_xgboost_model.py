"""Tests for XGBoostModel."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.candidates.xgboost_model import XGBoostModel


def _make_data(
    n: int = 120,
    n_features: int = 8,
    nan_frac: float = 0.0,
    seed: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    if nan_frac > 0:
        mask = rng.random((n, n_features)) < nan_frac
        X[mask] = np.nan
    y = rng.integers(0, 6, size=(n, 2)).astype(np.float64)
    return X, y


class TestXGBoostModel:
    def test_name(self):
        assert XGBoostModel().name == "xgboost"

    def test_get_params_keys(self):
        m = XGBoostModel(learning_rate=0.05, max_depth=4)
        params = m.get_params()
        assert params["learning_rate"] == 0.05
        assert params["max_depth"] == 4

    def test_fit_returns_self(self):
        X, y = _make_data()
        m = XGBoostModel(n_estimators=10)
        assert m.fit(X, y) is m

    def test_predict_shape(self):
        X, y = _make_data()
        m = XGBoostModel(n_estimators=10).fit(X, y)
        lh, la = m.predict(X)
        assert lh.shape == (len(X),)
        assert la.shape == (len(X),)

    def test_predictions_non_negative(self):
        X, y = _make_data()
        m = XGBoostModel(n_estimators=10).fit(X, y)
        lh, la = m.predict(X)
        assert (lh >= 0).all()
        assert (la >= 0).all()

    def test_predict_before_fit_raises(self):
        m = XGBoostModel()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 8)))

    def test_handles_nan_features(self):
        """XGBoost must train and predict without error when features contain NaN."""
        X, y = _make_data(nan_frac=0.2)
        m = XGBoostModel(n_estimators=10).fit(X, y)
        lh, la = m.predict(X)
        assert np.isfinite(lh).all()
        assert np.isfinite(la).all()

    def test_predict_on_new_samples(self):
        X, y = _make_data()
        m = XGBoostModel(n_estimators=10).fit(X[:100], y[:100])
        lh, la = m.predict(X[100:])
        assert lh.shape == (20,)

    def test_deterministic_with_seed(self):
        X, y = _make_data()
        m1 = XGBoostModel(n_estimators=10, random_state=0).fit(X, y)
        m2 = XGBoostModel(n_estimators=10, random_state=0).fit(X, y)
        lh1, _ = m1.predict(X)
        lh2, _ = m2.predict(X)
        np.testing.assert_array_almost_equal(lh1, lh2)
