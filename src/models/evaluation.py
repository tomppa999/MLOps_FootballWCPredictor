"""Evaluation metrics: RPS, RMSE, outcome probabilities, permutation importance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import poisson

if TYPE_CHECKING:
    from src.models.base import BaseModel

# ---------------------------------------------------------------------------
# Outcome probabilities (analytical Poisson grid)
# ---------------------------------------------------------------------------


def compute_outcome_probs(
    lambda_h: np.ndarray,
    lambda_a: np.ndarray,
    max_goals: int = 10,
) -> np.ndarray:
    """Compute P(home_win), P(draw), P(away_win) from Poisson rate parameters.

    Builds the joint (h, a) Poisson grid truncated at ``max_goals`` and
    sums probabilities into the three outcome buckets.

    Args:
        lambda_h: Expected home goals, shape (n,) or scalar.
        lambda_a: Expected away goals, shape (n,) or scalar.

    Returns:
        Array of shape (n, 3) — columns [P(home), P(draw), P(away)].
    """
    lambda_h = np.atleast_1d(np.asarray(lambda_h, dtype=np.float64))
    lambda_a = np.atleast_1d(np.asarray(lambda_a, dtype=np.float64))

    goals = np.arange(max_goals + 1)

    # pmf tables: (n, max_goals+1)
    pmf_h = poisson.pmf(goals[None, :], lambda_h[:, None])
    pmf_a = poisson.pmf(goals[None, :], lambda_a[:, None])

    # Joint probability grid: (n, max_goals+1, max_goals+1)
    joint = pmf_h[:, :, None] * pmf_a[:, None, :]

    h_idx, a_idx = np.meshgrid(goals, goals, indexing="ij")

    p_home = (joint * (h_idx > a_idx)[None]).sum(axis=(1, 2))
    p_draw = (joint * (h_idx == a_idx)[None]).sum(axis=(1, 2))
    p_away = (joint * (h_idx < a_idx)[None]).sum(axis=(1, 2))

    result = np.column_stack([p_home, p_draw, p_away])
    # Normalize to account for truncation-induced probability loss at high rates
    result /= result.sum(axis=1, keepdims=True)
    return result


# ---------------------------------------------------------------------------
# Ranked Probability Score
# ---------------------------------------------------------------------------


def goals_to_outcome(home_goals: np.ndarray, away_goals: np.ndarray) -> np.ndarray:
    """Convert goal counts to ordinal outcome: 0=home_win, 1=draw, 2=away_win."""
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    outcome = np.ones(len(home_goals), dtype=int)  # default: draw
    outcome[home_goals > away_goals] = 0
    outcome[home_goals < away_goals] = 2
    return outcome


def compute_rps(probs: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Compute per-match Ranked Probability Score.

    RPS = 1/(R-1) * Σ_{r=1}^{R-1} (F_r − O_r)²
    where F_r and O_r are cumulative forecast and observation.

    Args:
        probs: (n, 3) predicted probabilities [P(home), P(draw), P(away)].
        actual: (n,) integer outcomes — 0=home_win, 1=draw, 2=away_win.

    Returns:
        (n,) RPS values.  Lower is better; 0 = perfect.
    """
    probs = np.asarray(probs)
    actual = np.asarray(actual, dtype=int)

    n = len(actual)
    actual_onehot = np.zeros((n, 3))
    actual_onehot[np.arange(n), actual] = 1.0

    cum_f = np.cumsum(probs, axis=1)[:, :2]
    cum_o = np.cumsum(actual_onehot, axis=1)[:, :2]

    return 0.5 * np.sum((cum_f - cum_o) ** 2, axis=1)


def compute_mean_rps(
    lambda_h: np.ndarray,
    lambda_a: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> float:
    """End-to-end mean RPS from predicted rates and actual goals."""
    probs = compute_outcome_probs(lambda_h, lambda_a)
    outcomes = goals_to_outcome(home_goals, away_goals)
    return float(compute_rps(probs, outcomes).mean())


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------


def compute_rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Root mean squared error between two arrays."""
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(actual)) ** 2)))


# ---------------------------------------------------------------------------
# Permutation importance (RPS-based)
# ---------------------------------------------------------------------------


def compute_permutation_importance(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Feature importance via permutation using mean RPS as the scoring metric.

    Positive importance means shuffling the feature *worsens* RPS (higher = more
    important).

    Returns:
        DataFrame with columns [feature, importance_mean, importance_std].
    """
    rng = np.random.default_rng(random_state)

    lam_h, lam_a = model.predict(X)
    baseline = compute_mean_rps(lam_h, lam_a, y[:, 0], y[:, 1])

    records: list[dict] = []
    for j in range(X.shape[1]):
        deltas: list[float] = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            lh, la = model.predict(X_perm)
            shuffled_rps = compute_mean_rps(lh, la, y[:, 0], y[:, 1])
            deltas.append(shuffled_rps - baseline)
        records.append(
            {
                "feature": feature_names[j],
                "importance_mean": float(np.mean(deltas)),
                "importance_std": float(np.std(deltas)),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
