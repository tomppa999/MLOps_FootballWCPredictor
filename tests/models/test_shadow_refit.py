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

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "_FakeShadowModel":
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


def _inproc_fit(model, X, y, w, timeout_s):  # noqa: ANN001, ANN202, ARG001
    """In-process stand-in for ``_fit_with_timeout`` so tests avoid spawning a
    child process (which would hide the fake model's class-level fit counter)."""
    model.fit(X, y, sample_weight=w)
    return model


# ---------------------------------------------------------------------------
# Champion-skip + 8-candidate refit
# ---------------------------------------------------------------------------


@patch("src.models.pipeline._fit_with_timeout", side_effect=_inproc_fit)
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
    mock_fit_timeout,
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


@patch("src.models.pipeline._fit_with_timeout", side_effect=_inproc_fit)
@patch("src.models.pipeline._log_model_artifact", return_value="models:/fake/1")
@patch("src.models.pipeline.register_model")
@patch("src.models.pipeline.make_splits")
@patch("src.models.pipeline.get_all_shadow_metadata")
@patch("src.models.pipeline.get_champion_metadata")
def test_shadow_refit_tags_cadence_mode_frozen(
    mock_get_champion,
    mock_get_all_shadows,
    mock_make_splits,
    mock_register,
    mock_log_artifact,
    mock_fit_timeout,
    tmp_mlflow,
    fake_splits,
):
    """Frozen shadow refit runs must carry cadence_mode=frozen."""
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
        with patch("src.models.pipeline.start_run") as mock_start_run:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=MagicMock(info=MagicMock(run_id="r1")))
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_start_run.return_value = mock_ctx
            pipeline_module.run_shadow_refit(pd.DataFrame({"x": range(20)}))

    tags = mock_start_run.call_args.kwargs["tags"]
    assert tags["cadence_mode"] == "frozen"
    assert tags["stage"] == "shadow-refit"


@patch("src.models.pipeline._fit_with_timeout", side_effect=_inproc_fit)
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
    mock_fit_timeout,
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
# Candidate ordering (experiment roster first; bayesian_poisson last in group)
# ---------------------------------------------------------------------------


def test_ordered_shadow_candidates_experiment_first_bayesian_last():
    """Experiment models lead, bayesian_poisson is last of them, champion absent."""
    from src.models import pipeline as pipeline_module
    from src.models.config import EXPERIMENT_MODELS, LIVE_SHADOW_MODELS

    ordered = pipeline_module._ordered_shadow_candidates("xgboost")

    # Champion excluded; only LIVE_SHADOW_MODELS are considered (lstm/cnn excluded).
    assert "xgboost" not in ordered
    assert set(ordered) == set(LIVE_SHADOW_MODELS) - {"xgboost"}
    assert len(ordered) == len(set(ordered))

    # Experiment roster (minus champion) forms the prefix.
    experiment_minus_champion = [m for m in EXPERIMENT_MODELS if m != "xgboost"]
    assert set(ordered[: len(experiment_minus_champion)]) == set(experiment_minus_champion)

    # bayesian_poisson is last within the experiment prefix, before any shadow.
    bayes_idx = ordered.index("bayesian_poisson")
    assert bayes_idx == len(experiment_minus_champion) - 1
    non_experiment = [m for m in ordered if m not in set(EXPERIMENT_MODELS)]
    assert all(ordered.index(m) > bayes_idx for m in non_experiment)


# ---------------------------------------------------------------------------
# Hang guard — per-model fit timeout
# ---------------------------------------------------------------------------


class _SlowModel(BaseModel):
    """Fit blocks longer than any test timeout, to exercise the hang guard."""

    def __init__(self, sleep_s: float = 30.0) -> None:
        self.sleep_s = sleep_s

    @property
    def name(self) -> str:
        return "slow"

    def fit(self, X, y, sample_weight=None):  # noqa: ANN001, ANN201, ARG002
        import time as _time

        _time.sleep(self.sleep_s)
        return self

    def predict(self, X):  # noqa: ANN001, ANN201
        n = X.shape[0]
        return np.ones(n), np.ones(n)

    def get_params(self) -> dict[str, Any]:
        return {"sleep_s": self.sleep_s}


class _FastModel(BaseModel):
    """Records its own fitted state so the cloudpickle round-trip is verifiable."""

    def __init__(self) -> None:
        self.fitted = False

    @property
    def name(self) -> str:
        return "fast"

    def fit(self, X, y, sample_weight=None):  # noqa: ANN001, ANN201, ARG002
        self.fitted = True
        return self

    def predict(self, X):  # noqa: ANN001, ANN201
        n = X.shape[0]
        return np.ones(n), np.ones(n)

    def get_params(self) -> dict[str, Any]:
        return {}


def test_fit_with_timeout_raises_on_hang():
    """A fit exceeding the timeout is killed and surfaces TimeoutError."""
    from src.models import pipeline as pipeline_module

    X = np.zeros((4, 3))
    y = np.zeros((4, 2))
    with pytest.raises(TimeoutError):
        pipeline_module._fit_with_timeout(_SlowModel(sleep_s=30.0), X, y, None, 0.5)


def test_fit_with_timeout_returns_fitted_model():
    """A fast fit returns the fitted model (state survives the subprocess)."""
    from src.models import pipeline as pipeline_module

    X = np.zeros((4, 3))
    y = np.zeros((4, 2))
    fitted = pipeline_module._fit_with_timeout(_FastModel(), X, y, None, 60.0)
    assert isinstance(fitted, _FastModel)
    assert fitted.fitted is True


# ---------------------------------------------------------------------------
# get_shadow_metadata casts integer hyperparameters
# ---------------------------------------------------------------------------


def test_get_shadow_metadata_casts_int_params(monkeypatch, tmp_path):
    """``get_shadow_metadata`` reads from wc_staging and casts int hyperparameters."""
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
    # QA runs log metrics with "holdout_" prefix (not "qa_holdout_").
    fake_run.data.metrics = {"holdout_rps": 0.21, "holdout_nll": 2.78}

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
    assert "holdout_rps" in meta.holdout_metrics


def test_get_shadow_metadata_reads_from_staging_only(monkeypatch):
    """``get_shadow_metadata`` must read from wc_staging, never wc_shadow."""
    from src.models import mlflow_utils

    fake_run = MagicMock()
    fake_run.data.params = {"alpha": "1.0"}
    fake_run.data.tags = {"model_name": "ridge"}
    fake_run.data.metrics = {"holdout_rps": 0.188}
    fake_mv = MagicMock(version="5", run_id="staging-run-latest")

    searched: list[str] = []

    def search_versions(filter_str: str):
        searched.append(filter_str)
        if "wc_staging" in filter_str:
            return [fake_mv]
        return []

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
    # wc_shadow must never be queried as a metadata source.
    assert all("wc_shadow" not in q for q in searched)


def test_get_shadow_metadata_raises_if_staging_missing(monkeypatch):
    """Missing wc_staging entry raises ValueError — no silent fallback."""
    from src.models import mlflow_utils

    fake_client = MagicMock()
    fake_client.search_model_versions.return_value = []
    monkeypatch.setattr(
        mlflow_utils.mlflow.tracking, "MlflowClient", lambda: fake_client,
    )

    with pytest.raises(ValueError, match="wc_staging"):
        mlflow_utils.get_shadow_metadata("cnn")
