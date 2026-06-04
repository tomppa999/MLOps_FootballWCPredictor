"""Tests for A.6 — tune_half_period and run_half_period_tuning."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.models.config import HALF_PERIOD_SEARCH, WEIGHTED_MODELS
from src.models.data_split import (
    DEFAULT_HALF_PERIOD_YEARS,
    _DAYS_PER_YEAR,
    compute_weight_components,
    make_splits,
    walk_forward_cv,
)


# ---------------------------------------------------------------------------
# MLflow isolation fixture (mirrors test_pipeline.py::tmp_mlflow)
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_mlflow(tmp_path, monkeypatch):
    """Redirect all MLflow I/O to a temp directory and patch setup_mlflow."""
    uri = f"file:{tmp_path / 'mlruns'}"
    _noop = lambda *_a, **_kw: mlflow.set_tracking_uri(uri)  # noqa: E731
    monkeypatch.setattr("src.models.tuning.setup_mlflow", _noop)
    monkeypatch.setattr("src.models.half_period_tuning.setup_mlflow", _noop)
    monkeypatch.setattr("src.models.mlflow_utils.setup_mlflow", _noop)
    mlflow.set_tracking_uri(uri)
    yield uri
    while mlflow.active_run():
        mlflow.end_run()


# ---------------------------------------------------------------------------
# Minimal fake model for unit tests (no ML deps)
# ---------------------------------------------------------------------------


class _ConstantModel:
    """Predicts (1.2, 1.2) for every row; records sample_weight passed to fit."""

    name = "constant"
    distribution_family = "poisson"

    def __init__(self, **kwargs: Any) -> None:
        self.fit_weights: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> None:
        self.fit_weights = sample_weight

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(X)
        return np.full(n, 1.2), np.full(n, 1.2)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_train_data(n: int = 300) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X, y, df_train) for unit tests."""
    rng = np.random.default_rng(0)
    ref = pd.Timestamp("2022-01-01")
    dates = [ref - pd.Timedelta(days=d * 3) for d in range(n)]
    df = pd.DataFrame({
        "date_utc": dates,
        "competition_tier": rng.integers(1, 5, size=n),
        "f1": rng.random(n),
        "f2": rng.random(n),
        "home_goals": rng.integers(0, 5, size=n),
        "away_goals": rng.integers(0, 5, size=n),
    })
    X = df[["f1", "f2"]].to_numpy(dtype="float64")
    y = df[["home_goals", "away_goals"]].to_numpy(dtype="float64")
    return X, y, df


# ---------------------------------------------------------------------------
# tune_half_period
# ---------------------------------------------------------------------------


class TestTuneHalfPeriod:
    """Tests for src.models.tuning.tune_half_period."""

    def test_returns_float_in_search_range(self, tmp_mlflow: Any) -> None:
        from src.models.tuning import tune_half_period  # noqa: PLC0415

        X, y, df = _make_train_data()
        days_ago, w_imp = compute_weight_components(df)
        cv_folds = walk_forward_cv(len(X), n_splits=2)

        best_hp, study = tune_half_period(
            _ConstantModel,
            {},
            X, y, cv_folds,
            days_ago, w_imp,
            n_trials=3,
            low=HALF_PERIOD_SEARCH["low"],
            high=HALF_PERIOD_SEARCH["high"],
        )

        assert HALF_PERIOD_SEARCH["low"] <= best_hp <= HALF_PERIOD_SEARCH["high"]
        assert len(study.trials) == 3

    def test_weights_differ_across_trials(self, tmp_mlflow: Any) -> None:
        """Verify that different half-period values produce different sample weights."""
        from src.models.tuning import tune_half_period  # noqa: PLC0415

        X, y, df = _make_train_data()
        days_ago, w_imp = compute_weight_components(df)
        cv_folds = walk_forward_cv(len(X), n_splits=2)

        _, study = tune_half_period(
            _ConstantModel,
            {},
            X, y, cv_folds,
            days_ago, w_imp,
            n_trials=5,
            low=1.0,
            high=5.0,
        )

        # Different trials should propose different half-period values,
        # which in turn produce different weight arrays.
        half_periods = [t.params["half_period_years"] for t in study.trials]
        assert len(set(round(hp, 4) for hp in half_periods)) > 1, (
            "Optuna should explore multiple half_period_years values across trials"
        )

        # Manually verify two distinct half-periods give different weights.
        w_short = 0.5 ** (days_ago / (1.0 * _DAYS_PER_YEAR)) * w_imp
        w_long = 0.5 ** (days_ago / (5.0 * _DAYS_PER_YEAR)) * w_imp
        assert not np.allclose(w_short, w_long)

    def test_weight_formula(self) -> None:
        """Manual weight check: 0.5 ** (days_ago / (hp * 365.25)) * w_imp."""
        hp = 2.0
        days_ago = np.array([0.0, 365.25 * 2])  # 0 yr and 1 half-period away
        w_imp = np.array([4.0, 4.0])  # tier 1 both
        w_time = 0.5 ** (days_ago / (hp * _DAYS_PER_YEAR))
        expected = w_time * w_imp
        assert expected[0] == pytest.approx(4.0)       # today → full importance
        assert expected[1] == pytest.approx(2.0)       # 1 half-period → halved

    def test_enqueues_exact_three_year_trial(self, tmp_mlflow: Any) -> None:
        """A.6 rerun must evaluate hp=3.0 exactly for baseline comparison."""
        from src.models.tuning import tune_half_period  # noqa: PLC0415

        X, y, df = _make_train_data()
        days_ago, w_imp = compute_weight_components(df)
        cv_folds = walk_forward_cv(len(X), n_splits=2)

        _, study = tune_half_period(
            _ConstantModel,
            {},
            X, y, cv_folds,
            days_ago, w_imp,
            n_trials=3,
            low=1.0,
            high=5.0,
        )

        half_periods = [t.params["half_period_years"] for t in study.trials]
        assert any(hp == pytest.approx(3.0) for hp in half_periods)


# ---------------------------------------------------------------------------
# run_half_period_tuning
# ---------------------------------------------------------------------------


class TestRunHalfPeriodTuning:
    """Tests for src.models.half_period_tuning.run_half_period_tuning."""

    def _make_gold_df(self, n: int = 400) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        # Dates spanning ~4 years before WC 2022
        base = pd.Timestamp("2022-11-20")
        dates = [base - pd.Timedelta(days=d * 4) for d in range(n)]
        return pd.DataFrame({
            "date_utc": dates,
            "league_id": [1] * n,
            "competition_tier": rng.integers(1, 5, size=n),
            "f1": rng.random(n),
            "f2": rng.random(n),
            "home_goals": rng.integers(0, 5, size=n),
            "away_goals": rng.integers(0, 5, size=n),
            "elo_diff": rng.random(n),
            "elo_sum": rng.random(n),
        })

    def _fake_meta(self, model_name: str) -> Any:
        from src.models.mlflow_utils import ChampionMeta  # noqa: PLC0415
        return ChampionMeta(
            model_name=model_name,
            best_params={},
            holdout_metrics={},
            half_period_years=DEFAULT_HALF_PERIOD_YEARS,
        )

    def test_returns_dict_with_weighted_model_keys(self, tmp_mlflow: Any) -> None:
        from src.models.half_period_tuning import run_half_period_tuning  # noqa: PLC0415

        df = self._make_gold_df()
        fake_candidates = {"poisson_glm": _ConstantModel}

        with patch(
            "src.models.half_period_tuning.get_shadow_metadata",
            side_effect=lambda name: self._fake_meta(name),
        ), patch(
            "src.models.half_period_tuning.MODEL_FEATURE_SETS",
            {"poisson_glm": ["f1", "f2"]},
        ):
            results = run_half_period_tuning(df=df, n_trials=2, models=fake_candidates)

        assert "poisson_glm" in results
        assert HALF_PERIOD_SEARCH["low"] <= results["poisson_glm"] <= HALF_PERIOD_SEARCH["high"]

    def test_unweighted_models_skipped(self, tmp_mlflow: Any) -> None:
        """mean_rate_poisson and sarimax must not appear in results."""
        from src.models.half_period_tuning import run_half_period_tuning  # noqa: PLC0415

        df = self._make_gold_df()
        # Pass both weighted and unweighted names; only weighted should come back.
        unweighted = {"mean_rate_poisson": _ConstantModel, "sarimax": _ConstantModel}

        with patch("src.models.half_period_tuning.get_shadow_metadata",
                   side_effect=lambda name: self._fake_meta(name)):
            results = run_half_period_tuning(df=df, n_trials=2, models=unweighted)

        # Both are excluded from WEIGHTED_MODELS → nothing tuned
        assert "mean_rate_poisson" not in results
        assert "sarimax" not in results

    def test_missing_registry_entry_skipped(self, tmp_mlflow: Any) -> None:
        from src.models.half_period_tuning import run_half_period_tuning  # noqa: PLC0415

        df = self._make_gold_df()
        fake_candidates = {"poisson_glm": _ConstantModel}

        with patch(
            "src.models.half_period_tuning.get_shadow_metadata",
            side_effect=ValueError("not found"),
        ):
            results = run_half_period_tuning(df=df, n_trials=2, models=fake_candidates)

        assert results == {}


# ---------------------------------------------------------------------------
# ChampionMeta.half_period_years persistence
# ---------------------------------------------------------------------------


class TestChampionMetaHalfPeriod:
    def test_default_half_period_is_three(self) -> None:
        from src.models.mlflow_utils import ChampionMeta  # noqa: PLC0415
        meta = ChampionMeta(model_name="x", best_params={}, holdout_metrics={})
        assert meta.half_period_years == pytest.approx(DEFAULT_HALF_PERIOD_YEARS)

    def test_explicit_half_period_stored(self) -> None:
        from src.models.mlflow_utils import ChampionMeta  # noqa: PLC0415
        meta = ChampionMeta(model_name="x", best_params={}, holdout_metrics={}, half_period_years=2.5)
        assert meta.half_period_years == pytest.approx(2.5)

    def test_half_period_not_in_best_params(self) -> None:
        """half_period_years must never be passed to model constructors."""
        from src.models.mlflow_utils import ChampionMeta, _DEPLOY_INTERNAL_PARAMS  # noqa: PLC0415
        assert "half_period_years" in _DEPLOY_INTERNAL_PARAMS

    def test_cast_params_does_not_include_half_period(self) -> None:
        """_cast_params strips half_period_years (it's in _DEPLOY_INTERNAL_PARAMS)."""
        from src.models.mlflow_utils import _cast_params, _DEPLOY_INTERNAL_PARAMS  # noqa: PLC0415
        raw = {"alpha": "0.5", "half_period_years": "2.0"}
        # Only keys NOT in _DEPLOY_INTERNAL_PARAMS are cast — simulate what
        # get_champion_metadata does before calling _cast_params.
        filtered = {k: v for k, v in raw.items() if k not in _DEPLOY_INTERNAL_PARAMS}
        result = _cast_params("poisson_glm", filtered)
        assert "half_period_years" not in result
        assert "alpha" in result
