"""Tests for src.models.evaluation — RPS, outcome probs, RMSE."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluation import (
    compute_mean_rps,
    compute_outcome_probs,
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
