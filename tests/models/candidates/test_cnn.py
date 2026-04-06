"""Tests for CNNModel.

Uses minimal filters and epochs to keep the suite fast.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

from src.models.candidates.cnn import CNNModel

_FAST = {"filters": 8, "kernel_size": 2, "epochs": 2, "batch_size": 10, "seq_len": 3}


def _make_data(
    n: int = 60,
    n_features: int = 8,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 5, size=(n, 2)).astype(np.float64)
    return X, y


class TestCNNModel:
    def test_name(self):
        assert CNNModel().name == "cnn"

    def test_get_params_keys(self):
        m = CNNModel(filters=64, kernel_size=5, dropout=0.2)
        p = m.get_params()
        assert p["filters"] == 64
        assert p["kernel_size"] == 5
        assert p["dropout"] == 0.2

    def test_fit_returns_self(self):
        X, y = _make_data()
        m = CNNModel(**_FAST)
        assert m.fit(X, y) is m

    def test_predict_shape(self):
        X, y = _make_data()
        m = CNNModel(**_FAST).fit(X, y)
        lh, la = m.predict(X)
        assert lh.shape == (len(X),)
        assert la.shape == (len(X),)

    def test_predictions_non_negative(self):
        X, y = _make_data()
        m = CNNModel(**_FAST).fit(X, y)
        lh, la = m.predict(X)
        assert (lh >= 0).all()
        assert (la >= 0).all()

    def test_predict_new_samples_uses_training_tail(self):
        X, y = _make_data(n=60)
        m = CNNModel(**_FAST).fit(X[:50], y[:50])
        lh, la = m.predict(X[50:])
        assert lh.shape == (10,)
        assert la.shape == (10,)

    def test_predict_before_fit_raises(self):
        m = CNNModel()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 8)))

    def test_predictions_finite(self):
        X, y = _make_data()
        m = CNNModel(**_FAST).fit(X, y)
        lh, la = m.predict(X)
        assert np.isfinite(lh).all()
        assert np.isfinite(la).all()

    def test_kernel_clamped_to_seq_len(self):
        """kernel_size > seq_len should not crash — gets clamped internally."""
        X, y = _make_data()
        m = CNNModel(filters=4, kernel_size=10, epochs=2, batch_size=10, seq_len=3)
        m.fit(X, y)
        lh, la = m.predict(X[:5])
        assert lh.shape == (5,)
