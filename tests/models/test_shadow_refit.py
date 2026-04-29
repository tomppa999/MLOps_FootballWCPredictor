"""Tests for ``run_shadow_refit``: registry-driven, no re-tuning, champion skipped."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.models.base import BaseModel
from src.models.data_split import DataSplits
from src.models.mlflow_utils import ChampionMeta


class _FakeShadowModel(BaseModel):
    """Returns a fixed pair of rates; only used to assert fit was called."""

    fit_calls: int = 0

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return "fake_shadow"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_FakeShadowModel":
        type(self).fit_calls += 1
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        return np.full(n, 1.2), np.full(n, 1.0)

    def get_params(self) -> dict[str, Any]:
        return self.kwargs


@pytest.fixture()
def fake_splits() -> DataSplits:
    rng = np.random.default_rng(0)
    n_train, n_holdout, n_feat = 60, 10, 8
    X_train = rng.standard_normal((n_train, n_feat))
    y_train = np.column_stack([
        rng.poisson(1.5, n_train), rng.poisson(1.0, n_train),
    ]).astype(float)
    X_holdout = rng.standard_normal((n_holdout, n_feat))
    y_holdout = np.column_stack([
        rng.poisson(1.5, n_holdout), rng.poisson(1.0, n_holdout),
    ]).astype(float)
    X_full = np.vstack([X_train, X_holdout])
    y_full = np.vstack([y_train, y_holdout])
    cols = [f"f{i}" for i in range(n_feat)]
    df_train = pd.DataFrame(X_train, columns=cols)
    df_holdout = pd.DataFrame(X_holdout, columns=cols)
    df_full = pd.concat([df_train, df_holdout], ignore_index=True)
    df_full["date_utc"] = pd.date_range("2020-01-01", periods=len(df_full))
    return DataSplits(
        X_train=X_train, y_train=y_train,
        X_holdout=X_holdout, y_holdout=y_holdout,
        X_full=X_full, y_full=y_full,
        df_train=df_train, df_holdout=df_holdout, df_full=df_full,
    )


@pytest.fixture()
def tmp_mlflow(tmp_path, monkeypatch):
    uri = f"file:{tmp_path / 'mlruns'}"
    _noop_setup = lambda *_a, **_kw: mlflow.set_tracking_uri(uri)
    monkeypatch.setattr("src.models.pipeline.setup_mlflow", _noop_setup)
    monkeypatch.setattr("src.models.mlflow_utils.setup_mlflow", _noop_setup)
    mlflow.set_tracking_uri(uri)
    yield uri
    while mlflow.active_run():
        mlflow.end_run()


# ---------------------------------------------------------------------------
# Champion-skip + 8-candidate refit
# ---------------------------------------------------------------------------


@patch("src.models.pipeline._log_model_artifact", return_value="models:/fake/1")
@patch("src.models.pipeline.register_model")
@patch("src.models.pipeline.make_splits")
@patch("src.models.pipeline.get_all_shadow_metadata")
@patch("src.models.pipeline.get_champion_metadata")
def test_shadow_refit_skips_champion_and_fits_eight(
    mock_get_champion,
    mock_get_all_shadows,
    mock_make_splits,
    mock_register,
    mock_log_artifact,
    tmp_mlflow,
    fake_splits,
):
    """Champion is excluded; remaining 8 candidates are fit + registered."""
    from src.models import pipeline as pipeline_module

    candidate_names = list(pipeline_module.CANDIDATE_MODELS.keys())
    champion_name = candidate_names[0]
    expected_shadow_names = candidate_names[1:]

    mock_get_champion.return_value = ChampionMeta(
        model_name=champion_name,
        best_params={},
        holdout_metrics={"qa_holdout_rps": 0.21},
    )
    mock_get_all_shadows.return_value = [
        ChampionMeta(
            model_name=name,
            best_params={},
            holdout_metrics={"qa_holdout_rps": 0.22},
        )
        for name in expected_shadow_names
    ]
    mock_make_splits.return_value = fake_splits
    mock_register.side_effect = [
        MagicMock(version=str(i + 1)) for i in range(len(expected_shadow_names))
    ]

    fake_candidates = {name: _FakeShadowModel for name in candidate_names}
    _FakeShadowModel.fit_calls = 0

    with patch.dict(pipeline_module.CANDIDATE_MODELS, fake_candidates, clear=True):
        run_ids = pipeline_module.run_shadow_refit(pd.DataFrame({"x": range(50)}))

    assert len(run_ids) == len(expected_shadow_names)
    for call in mock_register.call_args_list:
        assert call.kwargs["model_name"] == "wc_shadow"
    refit_names = [meta.model_name for meta in mock_get_all_shadows.return_value]
    assert champion_name not in refit_names
    assert _FakeShadowModel.fit_calls == len(expected_shadow_names)


@patch("src.models.pipeline._log_model_artifact", return_value="models:/fake/1")
@patch("src.models.pipeline.register_model")
@patch("src.models.pipeline.make_splits")
@patch("src.models.pipeline.get_all_shadow_metadata")
@patch("src.models.pipeline.get_champion_metadata")
def test_shadow_refit_does_not_invoke_optuna(
    mock_get_champion,
    mock_get_all_shadows,
    mock_make_splits,
    mock_register,
    mock_log_artifact,
    tmp_mlflow,
    fake_splits,
):
    """Shadow refit must reuse stored best_params — no Optuna call."""
    from src.models import pipeline as pipeline_module

    mock_get_champion.return_value = ChampionMeta(
        model_name="xgboost",
        best_params={},
        holdout_metrics={},
    )
    mock_get_all_shadows.return_value = [
        ChampionMeta(model_name="ridge", best_params={"alpha": 0.5}, holdout_metrics={}),
    ]
    mock_make_splits.return_value = fake_splits
    mock_register.return_value = MagicMock(version="1")

    with patch.dict(
        pipeline_module.CANDIDATE_MODELS,
        {"xgboost": _FakeShadowModel, "ridge": _FakeShadowModel},
        clear=True,
    ):
        with patch("src.models.tuning.run_tuning") as mock_tuning:
            pipeline_module.run_shadow_refit(pd.DataFrame({"x": range(20)}))
        mock_tuning.assert_not_called()


# ---------------------------------------------------------------------------
# get_shadow_metadata casts integer hyperparameters
# ---------------------------------------------------------------------------


def test_get_shadow_metadata_casts_int_params(monkeypatch, tmp_path):
    """``get_shadow_metadata`` must call ``_cast_params`` like the champion path."""
    from src.models import mlflow_utils

    fake_mv = MagicMock(version="3", run_id="run-xyz")
    fake_run = MagicMock()
    # raw params are always strings in MLflow; cast must turn n_estimators into int.
    fake_run.data.params = {
        "n_estimators": "100",
        "max_depth": "5",
        "min_samples_leaf": "10",
        "max_features": "sqrt",
    }
    fake_run.data.tags = {"model_name": "random_forest"}
    fake_run.data.metrics = {"qa_holdout_rps": 0.21}

    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = [fake_mv]
    fake_client.get_run.return_value = fake_run
    monkeypatch.setattr(
        mlflow_utils.mlflow.tracking, "MlflowClient", lambda: fake_client,
    )

    meta = mlflow_utils.get_shadow_metadata("random_forest")
    assert meta.model_name == "random_forest"
    assert isinstance(meta.best_params["n_estimators"], int)
    assert meta.best_params["n_estimators"] == 100
    assert isinstance(meta.best_params["max_depth"], int)


def test_get_shadow_metadata_falls_back_to_staging(monkeypatch):
    """Cold start: no ``wc_shadow`` version → fall back to ``wc_staging``."""
    from src.models import mlflow_utils

    fake_run = MagicMock()
    fake_run.data.params = {"alpha": "1.0"}
    fake_run.data.tags = {"model_name": "ridge"}
    fake_run.data.metrics = {"holdout_rps": 0.214}
    fake_mv = MagicMock(version="2", run_id="staging-run")

    def search_versions(filter_str: str):
        if "wc_shadow" in filter_str:
            return []  # cold start
        return [fake_mv]

    fake_client = MagicMock()
    fake_client.search_model_versions.side_effect = search_versions
    fake_client.get_run.return_value = fake_run
    monkeypatch.setattr(
        mlflow_utils.mlflow.tracking, "MlflowClient", lambda: fake_client,
    )

    meta = mlflow_utils.get_shadow_metadata("ridge")
    assert meta.model_name == "ridge"
    assert meta.best_params["alpha"] == pytest.approx(1.0)
    assert "holdout_rps" in meta.holdout_metrics
