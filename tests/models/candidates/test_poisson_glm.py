"""Tests for BivariatePoisson."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.candidates.poisson_glm import BivariatePoisson
from src.models.config import MODEL_FEATURE_SETS
from src.models.data_split import (
    GOLD_PATH,
    compute_sample_weights,
    load_gold,
    make_splits,
    walk_forward_cv,
)
from src.models.evaluation import compute_nll_dispatch


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

    def test_weight_scale_invariant_after_normalization(self):
        """Global weight scale cancels out; only relative weights matter."""
        X, y = _make_data(n=100)
        rng = np.random.default_rng(7)
        w = rng.uniform(0.2, 4.0, size=len(X))
        m1 = BivariatePoisson(alpha=1.0, **_FAST).fit(X, y, sample_weight=w)
        m2 = BivariatePoisson(alpha=1.0, **_FAST).fit(X, y, sample_weight=w * 10.0)
        lh1, la1 = m1.predict(X)
        lh2, la2 = m2.predict(X)
        np.testing.assert_allclose(lh1, lh2, rtol=1e-3)
        np.testing.assert_allclose(la1, la2, rtol=1e-3)


# A.5-tuned alpha from MLflow registry (A.6 first-run blow-up reproduced at ~9.9).
_A5_ALPHA = 9.9


@pytest.mark.skipif(not GOLD_PATH.exists(), reason="Gold parquet not available")
class TestWeightedHalfPeriodStability:
    """Regression: weighted MLE must stay stable across half-periods (A.6 blocker)."""

    def _cv_mean_nll(self, half_period_years: float) -> float:
        feature_cols = MODEL_FEATURE_SETS["poisson_glm"]
        splits = make_splits(load_gold(), feature_cols)
        cv_folds = walk_forward_cv(len(splits.X_train))
        fold_nll: list[float] = []

        for train_idx, val_idx in cv_folds:
            df_fold = splits.df_train.iloc[train_idx]
            w = compute_sample_weights(df_fold, half_period_years=half_period_years)
            model = BivariatePoisson(alpha=_A5_ALPHA)
            model.fit(
                splits.X_train[train_idx],
                splits.y_train[train_idx],
                sample_weight=w,
            )
            nll = compute_nll_dispatch(
                model,
                splits.X_train[val_idx],
                splits.y_train[val_idx, 0],
                splits.y_train[val_idx, 1],
            )
            fold_nll.append(nll)

        return float(np.mean(fold_nll))

    def test_stable_cv_nll_across_half_periods(self) -> None:
        """NLL must be finite and similar at hp=1, 3, 5 (pre-fix blew up to 21+)."""
        nll_by_hp = {hp: self._cv_mean_nll(hp) for hp in (1.0, 3.0, 5.0)}

        for hp, nll in nll_by_hp.items():
            assert np.isfinite(nll), f"NLL not finite at half_period_years={hp}"
            assert nll < 4.0, f"NLL blow-up at half_period_years={hp}: {nll}"

        spread = max(nll_by_hp.values()) - min(nll_by_hp.values())
        assert spread < 0.5, (
            f"CV NLL varies too much across half-periods: {nll_by_hp}"
        )
