"""LSTM sequence model for goal prediction.

Builds sliding-window sequences from the flat feature matrix internally so the
BaseModel interface (fit/predict on 2-D arrays) is preserved.  The last
``seq_len - 1`` training rows are stored and prepended during inference to
provide proper temporal context for test sequences.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

import keras
from keras import layers

from src.models.base import BaseModel

_DEFAULT_SEQ_LEN = 5


class LSTMModel(BaseModel):
    """Stacked LSTM predicting home and away expected goals.

    Sequences are built via a sliding window of length ``seq_len`` over the
    (standardised) feature matrix.  Rows without enough history are zero-padded
    at the start.

    Args:
        units: LSTM hidden units per layer.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate applied after each LSTM layer.
        learning_rate: Adam learning rate.
        batch_size: Mini-batch size.
        epochs: Maximum epochs (EarlyStopping may stop earlier).
        seq_len: Look-back window (number of past matches per sequence).
        random_seed: Reproducibility seed.
    """

    def __init__(
        self,
        units: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 50,
        seq_len: int = _DEFAULT_SEQ_LEN,
        random_seed: int = 42,
    ) -> None:
        self.units = units
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seq_len = seq_len
        self.random_seed = random_seed
        self._scaler: StandardScaler | None = None
        self._model: keras.Model | None = None
        self._train_tail: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "lstm"

    # ------------------------------------------------------------------
    # Sequence helpers
    # ------------------------------------------------------------------

    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        """Slide a window of length ``seq_len`` over X → (n, seq_len, n_feats)."""
        n, n_feats = X.shape
        seqs = np.zeros((n, self.seq_len, n_feats), dtype=np.float32)
        for i in range(n):
            start = max(0, i - self.seq_len + 1)
            actual = i - start + 1
            seqs[i, self.seq_len - actual :] = X[start : i + 1]
        return seqs

    def _build_model(self, n_features: int) -> keras.Model:
        keras.utils.set_random_seed(self.random_seed)
        inp = keras.Input(shape=(self.seq_len, n_features))
        x = inp
        for _ in range(self.num_layers - 1):
            x = layers.LSTM(self.units, return_sequences=True)(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
        x = layers.LSTM(self.units)(x)
        if self.dropout > 0:
            x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(2)(x)
        model = keras.Model(inp, out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
        return model

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> LSTMModel:
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        # Store trailing context for inference-time sequence building
        self._train_tail = Xs[-(self.seq_len - 1) :] if len(Xs) >= self.seq_len else Xs.copy()

        seqs = self._build_sequences(Xs)
        y_f = y.astype(np.float32)

        self._model = self._build_model(Xs.shape[1])
        self._model.fit(
            seqs,
            y_f,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True, verbose=0
                )
            ],
            validation_split=0.1,
        )
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._scaler is None or self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        Xs = self._scaler.transform(X).astype(np.float32)

        if self._train_tail is not None and len(self._train_tail) > 0:
            X_ctx = np.vstack([self._train_tail, Xs])
            seqs_ctx = self._build_sequences(X_ctx)
            seqs = seqs_ctx[len(self._train_tail) :]
        else:
            seqs = self._build_sequences(Xs)

        preds = np.asarray(self._model.predict(seqs, verbose=0)).clip(0)
        return preds[:, 0], preds[:, 1]

    def get_params(self) -> dict[str, Any]:
        return {
            "units": self.units,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "seq_len": self.seq_len,
        }
