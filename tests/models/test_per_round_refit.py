"""Tests for B.3 per-round roster refit (run_per_round_refit in pipeline.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from src.models.mlflow_utils import (
    CHAMPION_ALIAS_PER_ROUND,
    PRODUCTION_MODEL_NAME,
    SHADOW_MODEL_NAME,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_df() -> pd.DataFrame:
    return pd.DataFrame({"a": range(10)})


def _mock_splits():
    """Return a MagicMock DataSplits with numpy-typed arrays."""
    splits = MagicMock()
    splits.X_full = np.zeros((10, 5))
    splits.y_full = np.zeros((10, 2))
    splits.w_full = np.ones(10)
    splits.df_full = pd.DataFrame({"a": range(10)})
    return splits


def _champ_meta(model_name: str = "xgboost") -> MagicMock:
    m = MagicMock()
    m.model_name = model_name
    m.best_params = {}
    m.holdout_metrics = {"qa_holdout_rps": 0.2}
    m.half_period_years = 3.0
    return m


def _shadow_meta(model_name: str) -> MagicMock:
    m = MagicMock()
    m.model_name = model_name
    m.best_params = {}
    m.holdout_metrics = {"holdout_rps": 0.25}
    m.half_period_years = 3.0
    return m


def _setup_start_run(mock_start_run, run_id: str = "run-xyz") -> MagicMock:
    run_ctx = MagicMock()
    run_ctx.info = MagicMock(run_id=run_id)
    mock_start_run.return_value.__enter__ = MagicMock(return_value=run_ctx)
    mock_start_run.return_value.__exit__ = MagicMock(return_value=False)
    return run_ctx


# ---------------------------------------------------------------------------
# Patch decorator helper (all the pipeline internals we need to stub)
# ---------------------------------------------------------------------------

_PATCHES = [
    patch("src.models.pipeline.setup_mlflow"),
    patch("src.models.pipeline.get_champion_metadata"),
    patch("src.models.pipeline.get_shadow_metadata"),
    patch("src.models.pipeline.make_splits"),
    patch("src.models.pipeline._fit_with_timeout"),
    patch("src.models.pipeline.start_run"),
    patch("src.models.pipeline.log_run"),
    patch("src.models.pipeline._log_model_artifact", return_value="models:/fake"),
    patch("src.models.pipeline.register_model"),
    patch("src.models.pipeline.promote_to_production"),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunPerRoundRefit:
    def _apply_patches(self, mocker):
        """Return a dict of name→mock for all pipeline internals."""
        mocks = {}
        names = [
            "setup_mlflow", "get_champion_metadata", "get_shadow_metadata",
            "make_splits", "_fit_with_timeout", "start_run", "log_run",
            "_log_model_artifact", "register_model", "promote_to_production",
        ]
        for name in names:
            mocks[name] = mocker.patch(f"src.models.pipeline.{name}")
        mocks["_log_model_artifact"].return_value = "models:/fake"
        mocks["register_model"].return_value = MagicMock(version="42", name=PRODUCTION_MODEL_NAME)
        return mocks

    def _standard_setup(self, mocks):
        """Apply default return values to the shared mocks."""
        champ = _champ_meta("xgboost")
        mocks["get_champion_metadata"].return_value = champ
        mocks["get_shadow_metadata"].side_effect = lambda name: _shadow_meta(name)
        mocks["make_splits"].return_value = _mock_splits()
        mocks["_fit_with_timeout"].return_value = MagicMock()
        _setup_start_run(mocks["start_run"])
        return champ

    def test_champion_gets_per_round_alias(self, mocker):
        from src.models.pipeline import run_per_round_refit

        mocks = self._apply_patches(mocker)
        self._standard_setup(mocks)

        run_per_round_refit(_make_df(), matchday="2", completed_matchday="1")

        # promote_to_production must be called exactly once with champion_per_round
        mocks["promote_to_production"].assert_called_once_with(
            version="42", alias=CHAMPION_ALIAS_PER_ROUND,
        )

    def test_non_champion_roster_models_go_to_shadow(self, mocker):
        from src.models.config import EXPERIMENT_MODELS
        from src.models.pipeline import run_per_round_refit

        mocks = self._apply_patches(mocker)
        self._standard_setup(mocks)
        # Make register_model return different names per call so we can inspect them
        def _register(*, model_uri, model_name):
            mv = MagicMock(version="1", name=model_name)
            return mv
        mocks["register_model"].side_effect = _register

        run_per_round_refit(_make_df(), matchday="R32", completed_matchday="3")

        register_calls = mocks["register_model"].call_args_list
        shadow_calls = [c for c in register_calls if c.kwargs.get("model_name") == SHADOW_MODEL_NAME]
        prod_calls = [c for c in register_calls if c.kwargs.get("model_name") == PRODUCTION_MODEL_NAME]
        # Exactly one model (xgboost champion) goes to production
        assert len(prod_calls) == 1
        # The remaining EXPERIMENT_MODELS go to shadow
        assert len(shadow_calls) == len(EXPERIMENT_MODELS) - 1

    def test_runs_tagged_with_cadence_mode_and_matchday(self, mocker):
        from src.models.pipeline import run_per_round_refit

        mocks = self._apply_patches(mocker)
        self._standard_setup(mocks)

        run_per_round_refit(_make_df(), matchday="QF", completed_matchday="R32")

        for c in mocks["start_run"].call_args_list:
            tags = c.kwargs.get("tags", {})
            assert tags.get("cadence_mode") == "per_round"
            assert tags.get("per_round_refit_matchday") == "QF"
            assert tags.get("per_round_last_completed_matchday") == "R32"

    def test_all_models_produce_run_ids(self, mocker):
        from src.models.config import EXPERIMENT_MODELS
        from src.models.pipeline import run_per_round_refit

        mocks = self._apply_patches(mocker)
        self._standard_setup(mocks)
        # Each call to start_run context manager returns a unique run_id
        run_counter = {"n": 0}
        def _start_run_factory(**kwargs):
            run_counter["n"] += 1
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(
                return_value=MagicMock(info=MagicMock(run_id=f"run-{run_counter['n']}"))
            )
            ctx.__exit__ = MagicMock(return_value=False)
            return ctx
        mocks["start_run"].side_effect = _start_run_factory

        result = run_per_round_refit(_make_df(), matchday="2", completed_matchday="1")

        assert len(result) == len(EXPERIMENT_MODELS)

    def test_timeout_skips_model_without_raising(self, mocker):
        from src.models.pipeline import run_per_round_refit

        mocks = self._apply_patches(mocker)
        champ = _champ_meta("xgboost")
        mocks["get_champion_metadata"].return_value = champ
        mocks["get_shadow_metadata"].side_effect = lambda name: _shadow_meta(name)
        mocks["make_splits"].return_value = _mock_splits()
        mocks["_fit_with_timeout"].side_effect = TimeoutError("timed out")

        result = run_per_round_refit(_make_df(), matchday="2", completed_matchday="1")

        assert result == {}
        mocks["promote_to_production"].assert_not_called()
        mocks["register_model"].assert_not_called()

    def test_fit_exception_skips_model_without_raising(self, mocker):
        from src.models.pipeline import run_per_round_refit

        mocks = self._apply_patches(mocker)
        champ = _champ_meta("xgboost")
        mocks["get_champion_metadata"].return_value = champ
        mocks["get_shadow_metadata"].side_effect = lambda name: _shadow_meta(name)
        mocks["make_splits"].return_value = _mock_splits()
        mocks["_fit_with_timeout"].side_effect = RuntimeError("kaboom")

        result = run_per_round_refit(_make_df(), matchday="SF", completed_matchday="QF")

        assert result == {}

    def test_missing_shadow_metadata_skips_model(self, mocker):
        from src.models.config import EXPERIMENT_MODELS
        from src.models.pipeline import run_per_round_refit

        mocks = self._apply_patches(mocker)
        champ = _champ_meta("xgboost")
        mocks["get_champion_metadata"].return_value = champ
        # Only xgboost (champion) has metadata; others raise
        mocks["get_shadow_metadata"].side_effect = ValueError("no staging version")
        mocks["make_splits"].return_value = _mock_splits()
        mocks["_fit_with_timeout"].return_value = MagicMock()
        _setup_start_run(mocks["start_run"])

        result = run_per_round_refit(_make_df(), matchday="3", completed_matchday="2")

        # Only champion fitted (no shadow metadata for others)
        assert "xgboost" in result
        assert len(result) == 1

    def test_bayesian_poisson_ordered_last(self, mocker):
        """bayesian_poisson should be the last model fitted."""
        from src.models.pipeline import _PER_ROUND_REFIT_ORDER

        assert _PER_ROUND_REFIT_ORDER[-1] == "bayesian_poisson"
