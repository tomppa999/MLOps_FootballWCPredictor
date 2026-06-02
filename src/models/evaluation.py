"""Evaluation metrics: Poisson NLL, RPS, RMSE, outcome probabilities, permutation importance.

Distribution-aware scoring:
- ``"poisson"``        — standard independent Poisson grid (all models by default)
- ``"negbin"``         — Negative Binomial PMF grid using fitted dispersion
- ``"bayesian_poisson"`` — MC average over posterior lambda samples
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

if TYPE_CHECKING:
    from src.models.base import BaseModel

logger = logging.getLogger(__name__)

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
    # Clip to a safe ceiling: poisson.pmf underflows to 0 for lambda >> max_goals,
    # which would produce a zero-sum row and a NaN after normalization.
    lambda_h = lambda_h.clip(1e-6, 15.0)
    lambda_a = lambda_a.clip(1e-6, 15.0)

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
    # Normalize to account for truncation-induced probability loss at high rates.
    # Guard against zero-sum rows (should not occur after ceiling clip, but kept
    # as a second line of defense).
    row_sum = result.sum(axis=1, keepdims=True)
    if np.any(row_sum == 0):
        logger.warning(
            "compute_outcome_probs: %d row(s) have zero probability mass — "
            "lambda ceiling may be too low",
            int((row_sum == 0).sum()),
        )
    result /= np.where(row_sum == 0, 1.0, row_sum)
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
# Poisson Negative Log-Likelihood
# ---------------------------------------------------------------------------


def compute_mean_nll(
    lambda_h: np.ndarray,
    lambda_a: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> float:
    """Mean Poisson negative log-likelihood over both sides.

    Scores how well the predicted rates explain the observed scoreline.
    Lower is better; used as the Optuna tuning objective.

    Lambdas are clipped to 1e-6 to prevent -inf from poisson.logpmf when
    a model predicts zero or negative rates (Ridge, SARIMAX, some XGBoost
    trials).  The clip is applied here rather than in individual model
    predict() methods so that model outputs are not altered.
    """
    lambda_h = np.asarray(lambda_h, dtype=np.float64).clip(1e-6)
    lambda_a = np.asarray(lambda_a, dtype=np.float64).clip(1e-6)
    nll = -(
        poisson.logpmf(home_goals, lambda_h) + poisson.logpmf(away_goals, lambda_a)
    )
    return float(nll.mean())


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


# ---------------------------------------------------------------------------
# Negative Binomial outcome probabilities and NLL
# ---------------------------------------------------------------------------


def compute_outcome_probs_nb(
    lambda_h: np.ndarray,
    lambda_a: np.ndarray,
    alpha_h: float,
    alpha_a: float,
    max_goals: int = 10,
) -> np.ndarray:
    """Compute outcome probabilities using the NegBin PMF grid.

    Parameterisation follows statsmodels NB-2: Var(Y) = mu + alpha * mu^2.
    scipy.stats.nbinom uses (n, p) where n = 1/alpha and p = n/(n+mu).

    Args:
        lambda_h: Expected home goals, shape (n,).
        lambda_a: Expected away goals, shape (n,).
        alpha_h: Home dispersion parameter (MLE from NegBin GLM).
        alpha_a: Away dispersion parameter.
        max_goals: Truncation point for the goal grid.

    Returns:
        Array of shape (n, 3) — columns [P(home), P(draw), P(away)].
    """
    lambda_h = np.atleast_1d(np.asarray(lambda_h, dtype=np.float64)).clip(1e-6, 15.0)
    lambda_a = np.atleast_1d(np.asarray(lambda_a, dtype=np.float64)).clip(1e-6, 15.0)
    alpha_h = max(float(alpha_h), 1e-6)
    alpha_a = max(float(alpha_a), 1e-6)

    goals = np.arange(max_goals + 1)

    n_h = 1.0 / alpha_h
    p_h = n_h / (n_h + lambda_h)  # shape (n,)
    pmf_h = nbinom.pmf(goals[None, :], n_h, p_h[:, None])  # (n, max_goals+1)

    n_a = 1.0 / alpha_a
    p_a = n_a / (n_a + lambda_a)
    pmf_a = nbinom.pmf(goals[None, :], n_a, p_a[:, None])

    joint = pmf_h[:, :, None] * pmf_a[:, None, :]
    h_idx, a_idx = np.meshgrid(goals, goals, indexing="ij")

    p_home = (joint * (h_idx > a_idx)[None]).sum(axis=(1, 2))
    p_draw = (joint * (h_idx == a_idx)[None]).sum(axis=(1, 2))
    p_away = (joint * (h_idx < a_idx)[None]).sum(axis=(1, 2))

    result = np.column_stack([p_home, p_draw, p_away])
    row_sum = result.sum(axis=1, keepdims=True)
    result /= np.where(row_sum == 0, 1.0, row_sum)
    return result


def compute_mean_nll_nb(
    lambda_h: np.ndarray,
    lambda_a: np.ndarray,
    alpha_h: float,
    alpha_a: float,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> float:
    """Mean NegBin negative log-likelihood.

    Uses scipy.stats.nbinom with the NB-2 parameterisation (alpha = dispersion).
    """
    lambda_h = np.asarray(lambda_h, dtype=np.float64).clip(1e-6)
    lambda_a = np.asarray(lambda_a, dtype=np.float64).clip(1e-6)
    alpha_h = max(float(alpha_h), 1e-6)
    alpha_a = max(float(alpha_a), 1e-6)

    n_h, n_a = 1.0 / alpha_h, 1.0 / alpha_a
    p_h = n_h / (n_h + lambda_h)
    p_a = n_a / (n_a + lambda_a)

    nll = -(
        nbinom.logpmf(home_goals, n_h, p_h)
        + nbinom.logpmf(away_goals, n_a, p_a)
    )
    return float(nll.mean())


# ---------------------------------------------------------------------------
# Bayesian Poisson outcome probabilities and NLL
# ---------------------------------------------------------------------------


def compute_outcome_probs_bayes(
    lambda_h_samples: np.ndarray,
    lambda_a_samples: np.ndarray,
    max_goals: int = 10,
) -> np.ndarray:
    """Compute outcome probabilities by averaging Poisson PMFs over posterior draws.

    Args:
        lambda_h_samples: Shape (n_samples, n_obs) — posterior home rate draws.
        lambda_a_samples: Shape (n_samples, n_obs) — posterior away rate draws.
        max_goals: Truncation point.

    Returns:
        Array of shape (n_obs, 3) — columns [P(home), P(draw), P(away)].
    """
    lambda_h_samples = np.asarray(lambda_h_samples, dtype=np.float64).clip(1e-6, 15.0)
    lambda_a_samples = np.asarray(lambda_a_samples, dtype=np.float64).clip(1e-6, 15.0)

    n_draws, n_obs = lambda_h_samples.shape
    goals = np.arange(max_goals + 1)
    h_idx, a_idx = np.meshgrid(goals, goals, indexing="ij")

    p_home_acc = np.zeros(n_obs)
    p_draw_acc = np.zeros(n_obs)
    p_away_acc = np.zeros(n_obs)

    for s in range(n_draws):
        lh = lambda_h_samples[s]  # (n_obs,)
        la = lambda_a_samples[s]
        pmf_h = poisson.pmf(goals[None, :], lh[:, None])  # (n_obs, G+1)
        pmf_a = poisson.pmf(goals[None, :], la[:, None])
        joint = pmf_h[:, :, None] * pmf_a[:, None, :]  # (n_obs, G+1, G+1)
        p_home_acc += (joint * (h_idx > a_idx)[None]).sum(axis=(1, 2))
        p_draw_acc += (joint * (h_idx == a_idx)[None]).sum(axis=(1, 2))
        p_away_acc += (joint * (h_idx < a_idx)[None]).sum(axis=(1, 2))

    result = np.column_stack([p_home_acc, p_draw_acc, p_away_acc]) / n_draws
    row_sum = result.sum(axis=1, keepdims=True)
    result /= np.where(row_sum == 0, 1.0, row_sum)
    return result


def compute_mean_nll_bayes(
    lambda_h_samples: np.ndarray,
    lambda_a_samples: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> float:
    """Mean NLL averaged over posterior samples via log-sum-exp.

    Computes log(1/S * sum_s P(goals | lambda_s)) for each observation,
    then negates and averages.  Numerically stable via log-sum-exp.
    """
    lambda_h_samples = np.asarray(lambda_h_samples, dtype=np.float64).clip(1e-6)
    lambda_a_samples = np.asarray(lambda_a_samples, dtype=np.float64).clip(1e-6)
    home_goals = np.asarray(home_goals, dtype=int)
    away_goals = np.asarray(away_goals, dtype=int)

    n_draws, n_obs = lambda_h_samples.shape
    log_n = np.log(n_draws)

    # log p(h, a | lambda_s) for each draw s and observation i
    # shape (n_draws, n_obs)
    log_lik = (
        poisson.logpmf(home_goals[None, :], lambda_h_samples)
        + poisson.logpmf(away_goals[None, :], lambda_a_samples)
    )

    # log-sum-exp over draws, then subtract log(n_draws)
    max_ll = log_lik.max(axis=0)  # (n_obs,)
    log_mean_lik = max_ll + np.log(np.exp(log_lik - max_ll).sum(axis=0)) - log_n

    return float(-log_mean_lik.mean())


# ---------------------------------------------------------------------------
# Distribution-aware dispatcher
# ---------------------------------------------------------------------------


def compute_outcome_probs_dispatch(
    model: "BaseModel",
    X: np.ndarray,
    max_goals: int = 10,
) -> np.ndarray:
    """Select outcome probability computation based on model.distribution_family.

    For ``"negbin"`` models: calls ``predict_with_dispersion`` and uses the
    NegBin PMF grid.  For ``"bayesian_poisson"`` models: draws posterior samples
    and averages PMFs.  All other models fall back to the standard Poisson grid.

    Returns:
        Array of shape (n, 3) — columns [P(home), P(draw), P(away)].
    """
    family = model.distribution_family

    if family == "negbin":
        from src.models.candidates.negbin_glm import NegativeBinomialGLM  # noqa: PLC0415
        assert isinstance(model, NegativeBinomialGLM)
        lam_h, lam_a, alpha_h, alpha_a = model.predict_with_dispersion(X)
        return compute_outcome_probs_nb(lam_h, lam_a, alpha_h, alpha_a, max_goals)

    if family == "bayesian_poisson":
        from src.models.candidates.bayesian_poisson import BayesianPoissonModel  # noqa: PLC0415
        assert isinstance(model, BayesianPoissonModel)
        lam_h_s, lam_a_s = model.predict_samples(X)
        return compute_outcome_probs_bayes(lam_h_s, lam_a_s, max_goals)

    # Default: treat predict() output as Poisson rates
    lam_h, lam_a = model.predict(X)
    return compute_outcome_probs(lam_h, lam_a, max_goals)


def compute_nll_dispatch(
    model: "BaseModel",
    X: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> float:
    """Select NLL computation based on model.distribution_family."""
    family = model.distribution_family

    if family == "negbin":
        from src.models.candidates.negbin_glm import NegativeBinomialGLM  # noqa: PLC0415
        assert isinstance(model, NegativeBinomialGLM)
        lam_h, lam_a, alpha_h, alpha_a = model.predict_with_dispersion(X)
        return compute_mean_nll_nb(lam_h, lam_a, alpha_h, alpha_a, home_goals, away_goals)

    if family == "bayesian_poisson":
        from src.models.candidates.bayesian_poisson import BayesianPoissonModel  # noqa: PLC0415
        assert isinstance(model, BayesianPoissonModel)
        lam_h_s, lam_a_s = model.predict_samples(X)
        return compute_mean_nll_bayes(lam_h_s, lam_a_s, home_goals, away_goals)

    lam_h, lam_a = model.predict(X)
    return compute_mean_nll(lam_h, lam_a, home_goals, away_goals)
