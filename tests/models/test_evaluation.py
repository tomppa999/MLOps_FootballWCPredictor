"""Tests for src.models.evaluation — Poisson NLL, RPS, outcome probs, RMSE."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluation import (
    compute_mean_nll,
    compute_mean_nll_bayes,
    compute_mean_nll_nb,
    compute_mean_rps,
    compute_nll_dispatch,
    compute_outcome_probs,
    compute_outcome_probs_bayes,
    compute_outcome_probs_dispatch,
    compute_outcome_probs_nb,
    compute_rmse,
    compute_rps,
    goals_to_outcome,
)


# ---------------------------------------------------------------------------
# compute_outcome_probs
# ---------------------------------------------------------------------------


class TestComputeOutcomeProbs:
    def test_probs_sum_to_one(self):
        probs = compute_outcome_probs(np.array([1.5]), np.array([1.0]))
        assert probs.shape == (1, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_probs_sum_to_one_vectorised(self):
        lam_h = np.array([0.5, 1.5, 2.5, 3.0])
        lam_a = np.array([1.0, 1.0, 0.5, 2.0])
        probs = compute_outcome_probs(lam_h, lam_a)
        assert probs.shape == (4, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_symmetric_rates_give_higher_draw(self):
        probs = compute_outcome_probs(np.array([1.0]), np.array([1.0]))
        p_home, p_draw, p_away = probs[0]
        assert abs(p_home - p_away) < 1e-6
        assert p_draw > 0.15  # meaningful draw probability at low rates

    def test_dominant_home_rate(self):
        probs = compute_outcome_probs(np.array([4.0]), np.array([0.3]))
        assert probs[0, 0] > 0.85  # strong home favourite

    def test_scalar_input(self):
        probs = compute_outcome_probs(1.5, 1.0)
        assert probs.shape == (1, 3)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)

    def test_all_probs_non_negative(self):
        probs = compute_outcome_probs(np.array([0.1, 5.0]), np.array([5.0, 0.1]))
        assert (probs >= 0).all()

    def test_extreme_lambda_no_nan(self):
        """Very large lambda must not produce NaN (underflow guard)."""
        probs = compute_outcome_probs(np.array([50.0]), np.array([50.0]))
        assert not np.any(np.isnan(probs))


# ---------------------------------------------------------------------------
# goals_to_outcome
# ---------------------------------------------------------------------------


class TestGoalsToOutcome:
    def test_home_win(self):
        assert goals_to_outcome(np.array([3]), np.array([1]))[0] == 0

    def test_draw(self):
        assert goals_to_outcome(np.array([2]), np.array([2]))[0] == 1

    def test_away_win(self):
        assert goals_to_outcome(np.array([0]), np.array([1]))[0] == 2

    def test_vectorised(self):
        outcomes = goals_to_outcome(np.array([3, 1, 1]), np.array([1, 1, 3]))
        np.testing.assert_array_equal(outcomes, [0, 1, 2])


# ---------------------------------------------------------------------------
# compute_rps
# ---------------------------------------------------------------------------


class TestComputeRPS:
    def test_perfect_prediction_rps_zero(self):
        """Putting all probability mass on the correct outcome → RPS = 0."""
        probs = np.array([[1.0, 0.0, 0.0]])
        actual = np.array([0])  # home win
        rps = compute_rps(probs, actual)
        np.testing.assert_allclose(rps, 0.0, atol=1e-10)

    def test_worst_prediction_rps_one(self):
        """All mass on the opposite outcome → RPS = 1."""
        probs = np.array([[0.0, 0.0, 1.0]])
        actual = np.array([0])  # home win, predicted away
        rps = compute_rps(probs, actual)
        np.testing.assert_allclose(rps, 1.0, atol=1e-10)

    def test_uniform_prediction(self):
        """Uniform [1/3, 1/3, 1/3] → known RPS for each outcome."""
        probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
        # actual = home win (0): cum_f = [1/3, 2/3], cum_o = [1, 1]
        # RPS = 0.5 * ((1/3-1)^2 + (2/3-1)^2) = 0.5*(4/9 + 1/9) = 5/18
        rps = compute_rps(probs, np.array([0]))
        np.testing.assert_allclose(rps, 5 / 18, atol=1e-10)

    def test_rps_is_non_negative(self):
        rng = np.random.default_rng(42)
        n = 50
        raw = rng.random((n, 3))
        probs = raw / raw.sum(axis=1, keepdims=True)
        actual = rng.integers(0, 3, n)
        rps = compute_rps(probs, actual)
        assert (rps >= 0).all()

    def test_rps_bounded_by_one(self):
        rng = np.random.default_rng(42)
        n = 50
        raw = rng.random((n, 3))
        probs = raw / raw.sum(axis=1, keepdims=True)
        actual = rng.integers(0, 3, n)
        rps = compute_rps(probs, actual)
        assert (rps <= 1.0 + 1e-10).all()


# ---------------------------------------------------------------------------
# compute_mean_rps
# ---------------------------------------------------------------------------


class TestComputeMeanRPS:
    def test_end_to_end(self):
        rps = compute_mean_rps(
            lambda_h=np.array([2.0, 0.5]),
            lambda_a=np.array([0.5, 2.0]),
            home_goals=np.array([3, 0]),
            away_goals=np.array([1, 2]),
        )
        assert isinstance(rps, float)
        assert 0 <= rps <= 1


# ---------------------------------------------------------------------------
# compute_mean_nll
# ---------------------------------------------------------------------------


class TestComputeMeanNLL:
    def test_good_prediction_beats_bad(self):
        """Rates close to actual goals should score lower NLL than distant rates."""
        actual_h = np.array([2, 1, 0])
        actual_a = np.array([1, 0, 3])
        good_nll = compute_mean_nll(
            np.array([2.0, 1.0, 0.5]), np.array([1.0, 0.5, 2.5]),
            actual_h, actual_a,
        )
        bad_nll = compute_mean_nll(
            np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0]),
            actual_h, actual_a,
        )
        assert good_nll < bad_nll

    def test_nll_non_negative(self):
        nll = compute_mean_nll(
            np.array([1.5, 2.0]), np.array([1.0, 0.8]),
            np.array([1, 3]), np.array([0, 1]),
        )
        assert nll >= 0.0

    def test_known_value(self):
        """Single match: lambda_h=1, lambda_a=1, actual=(1,1).
        NLL = -(logpmf(1,1) + logpmf(1,1)) = -2*log(e^{-1}) = 2."""
        from scipy.stats import poisson

        expected = -(poisson.logpmf(1, 1.0) + poisson.logpmf(1, 1.0))
        nll = compute_mean_nll(np.array([1.0]), np.array([1.0]),
                               np.array([1]), np.array([1]))
        np.testing.assert_allclose(nll, expected, atol=1e-10)

    def test_returns_float(self):
        nll = compute_mean_nll(np.array([1.0]), np.array([1.0]),
                               np.array([1]), np.array([1]))
        assert isinstance(nll, float)

    def test_zero_lambda_returns_finite(self):
        """Lambda=0 with non-zero goals must produce a large but finite NLL."""
        nll = compute_mean_nll(
            np.array([0.0, 0.0]), np.array([0.0, 0.0]),
            np.array([2, 1]), np.array([1, 3]),
        )
        assert np.isfinite(nll)
        assert nll > 0


# ---------------------------------------------------------------------------
# compute_rmse
# ---------------------------------------------------------------------------


class TestComputeRMSE:
    def test_perfect(self):
        assert compute_rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0

    def test_known_value(self):
        rmse = compute_rmse(np.array([3.0, 4.0]), np.array([1.0, 2.0]))
        np.testing.assert_allclose(rmse, 2.0)

    def test_scalar(self):
        rmse = compute_rmse(1.0, 2.0)
        np.testing.assert_allclose(rmse, 1.0)


# ---------------------------------------------------------------------------
# compute_outcome_probs_nb
# ---------------------------------------------------------------------------


class TestComputeOutcomeProbsNB:
    def test_shape(self):
        probs = compute_outcome_probs_nb(
            np.array([1.5, 1.0]), np.array([1.0, 1.5]), 0.5, 0.5
        )
        assert probs.shape == (2, 3)

    def test_probs_sum_to_one(self):
        probs = compute_outcome_probs_nb(
            np.array([1.5, 1.0]), np.array([1.0, 1.5]), 0.5, 0.5
        )
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_all_probs_non_negative(self):
        probs = compute_outcome_probs_nb(
            np.array([0.5, 3.0]), np.array([3.0, 0.5]), 1.0, 1.0
        )
        assert (probs >= 0).all()

    def test_small_alpha_approaches_poisson(self):
        """Very small dispersion (alpha → 0) should give results close to Poisson."""
        lh = np.array([1.5])
        la = np.array([1.0])
        nb_probs = compute_outcome_probs_nb(lh, la, 1e-4, 1e-4)
        poisson_probs = compute_outcome_probs(lh, la)
        np.testing.assert_allclose(nb_probs, poisson_probs, atol=0.01)


# ---------------------------------------------------------------------------
# compute_mean_nll_nb
# ---------------------------------------------------------------------------


class TestComputeMeanNLLNB:
    def test_returns_finite(self):
        nll = compute_mean_nll_nb(
            np.array([1.5, 1.0]), np.array([1.0, 1.5]),
            0.5, 0.5,
            np.array([1, 2]), np.array([2, 1]),
        )
        assert np.isfinite(nll)
        assert nll > 0

    def test_good_beats_bad(self):
        actual_h = np.array([2, 1])
        actual_a = np.array([1, 2])
        good = compute_mean_nll_nb(
            np.array([2.0, 1.0]), np.array([1.0, 2.0]), 0.5, 0.5,
            actual_h, actual_a,
        )
        bad = compute_mean_nll_nb(
            np.array([0.1, 5.0]), np.array([5.0, 0.1]), 0.5, 0.5,
            actual_h, actual_a,
        )
        assert good < bad


# ---------------------------------------------------------------------------
# compute_outcome_probs_bayes
# ---------------------------------------------------------------------------


class TestComputeOutcomeProbsBayes:
    def _make_samples(self, n_draws: int = 10, n_obs: int = 4) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        lh = rng.uniform(0.5, 2.5, size=(n_draws, n_obs))
        la = rng.uniform(0.5, 2.5, size=(n_draws, n_obs))
        return lh, la

    def test_shape(self):
        lh, la = self._make_samples()
        probs = compute_outcome_probs_bayes(lh, la)
        assert probs.shape == (4, 3)

    def test_probs_sum_to_one(self):
        lh, la = self._make_samples()
        probs = compute_outcome_probs_bayes(lh, la)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_all_probs_non_negative(self):
        lh, la = self._make_samples()
        probs = compute_outcome_probs_bayes(lh, la)
        assert (probs >= 0).all()

    def test_single_draw_matches_poisson(self):
        """With one posterior draw, result should match compute_outcome_probs."""
        lh = np.array([[1.5, 1.0]])
        la = np.array([[1.0, 1.5]])
        bayes_probs = compute_outcome_probs_bayes(lh, la)
        poisson_probs = compute_outcome_probs(lh[0], la[0])
        np.testing.assert_allclose(bayes_probs, poisson_probs, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_mean_nll_bayes
# ---------------------------------------------------------------------------


class TestComputeMeanNLLBayes:
    def test_returns_finite(self):
        rng = np.random.default_rng(1)
        lh_s = rng.uniform(0.5, 2.5, size=(10, 5))
        la_s = rng.uniform(0.5, 2.5, size=(10, 5))
        nll = compute_mean_nll_bayes(
            lh_s, la_s, np.array([1, 2, 1, 0, 3]), np.array([0, 1, 2, 1, 1])
        )
        assert np.isfinite(nll)
        assert nll > 0

    def test_single_draw_matches_poisson(self):
        """Single draw: Bayesian NLL should match standard Poisson NLL."""
        lh = np.array([[1.5, 2.0]])
        la = np.array([[1.0, 0.8]])
        hg = np.array([1, 2])
        ag = np.array([0, 1])
        bayes_nll = compute_mean_nll_bayes(lh, la, hg, ag)
        poisson_nll = compute_mean_nll(lh[0], la[0], hg, ag)
        np.testing.assert_allclose(bayes_nll, poisson_nll, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_outcome_probs_dispatch / compute_nll_dispatch
# ---------------------------------------------------------------------------


class TestDispatchers:
    def _make_X_y(self, n: int = 30, p: int = 3) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(99)
        X = rng.standard_normal((n, p))
        y = rng.integers(0, 5, size=(n, 2)).astype(np.float64)
        return X, y

    def test_dispatch_poisson_family(self):
        from src.models.candidates.ridge import RidgeModel
        X, y = self._make_X_y()
        m = RidgeModel().fit(X, y)
        probs = compute_outcome_probs_dispatch(m, X)
        assert probs.shape == (len(X), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_dispatch_negbin_family(self):
        from src.models.candidates.negbin_glm import NegativeBinomialGLM
        X, y = self._make_X_y()
        m = NegativeBinomialGLM().fit(X, y)
        probs = compute_outcome_probs_dispatch(m, X)
        assert probs.shape == (len(X), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_dispatch_bayesian_family(self):
        from src.models.candidates.bayesian_poisson import BayesianPoissonModel
        X, y = self._make_X_y()
        m = BayesianPoissonModel(draws=20, tune_steps=20).fit(X, y)
        probs = compute_outcome_probs_dispatch(m, X)
        assert probs.shape == (len(X), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_nll_dispatch_poisson_matches_direct(self):
        from src.models.candidates.ridge import RidgeModel
        X, y = self._make_X_y()
        m = RidgeModel().fit(X, y)
        lh, la = m.predict(X)
        direct = compute_mean_nll(lh, la, y[:, 0], y[:, 1])
        dispatched = compute_nll_dispatch(m, X, y[:, 0], y[:, 1])
        np.testing.assert_allclose(dispatched, direct, atol=1e-10)

    def test_nll_dispatch_negbin_returns_finite(self):
        from src.models.candidates.negbin_glm import NegativeBinomialGLM
        X, y = self._make_X_y()
        m = NegativeBinomialGLM().fit(X, y)
        nll = compute_nll_dispatch(m, X, y[:, 0], y[:, 1])
        assert np.isfinite(nll)
        assert nll > 0

    def test_nll_dispatch_bayesian_returns_finite(self):
        from src.models.candidates.bayesian_poisson import BayesianPoissonModel
        X, y = self._make_X_y()
        m = BayesianPoissonModel(draws=20, tune_steps=20).fit(X, y)
        nll = compute_nll_dispatch(m, X, y[:, 0], y[:, 1])
        assert np.isfinite(nll)
        assert nll > 0
