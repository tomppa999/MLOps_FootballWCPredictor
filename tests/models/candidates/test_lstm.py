"""Tests for LSTMModel.

Uses minimal units and epochs to keep the suite fast.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

from src.models.candidates.lstm import LSTMModel

_FAST = {"units": 8, "epochs": 2, "batch_size": 10, "seq_len": 3}


def _make_data(
    n: int = 60,
    n_features: int = 8,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = rng.integers(0, 5, size=(n, 2)).astype(np.float64)
    return X, y


class TestLSTMModel:
    def test_name(self):
        assert LSTMModel().name == "lstm"

    def test_get_params_keys(self):
        m = LSTMModel(units=64, num_layers=2, dropout=0.1)
        p = m.get_params()
        assert p["units"] == 64
        assert p["num_layers"] == 2
        assert p["dropout"] == 0.1

    def test_fit_returns_self(self):
        X, y = _make_data()
        m = LSTMModel(**_FAST)
        assert m.fit(X, y) is m

    def test_predict_shape(self):
        X, y = _make_data()
        m = LSTMModel(**_FAST).fit(X, y)
        lh, la = m.predict(X)
        assert lh.shape == (len(X),)
        assert la.shape == (len(X),)

    def test_predictions_non_negative(self):
        X, y = _make_data()
        m = LSTMModel(**_FAST).fit(X, y)
        lh, la = m.predict(X)
        assert (lh >= 0).all()
        assert (la >= 0).all()

    def test_predict_new_samples_uses_training_tail(self):
        X, y = _make_data(n=60)
        m = LSTMModel(**_FAST).fit(X[:50], y[:50])
        lh, la = m.predict(X[50:])
        assert lh.shape == (10,)
        assert la.shape == (10,)

    def test_predict_before_fit_raises(self):
        m = LSTMModel()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((5, 8)))

    def test_predictions_finite(self):
        X, y = _make_data()
        m = LSTMModel(**_FAST).fit(X, y)
        lh, la = m.predict(X)
        assert np.isfinite(lh).all()
        assert np.isfinite(la).all()

    def test_multilayer_lstm(self):
        X, y = _make_data()
        m = LSTMModel(units=4, num_layers=2, epochs=2, batch_size=10, seq_len=3)
        m.fit(X, y)
        lh, la = m.predict(X[:5])
        assert lh.shape == (5,)
