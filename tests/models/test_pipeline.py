"""Tests for the three-phase training pipeline and trigger threshold gating."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.models.base import BaseModel
from src.models.data_split import DataSplits
from src.models.pipeline import (
    ChallengeFailed,
    ExperimentalResult,
    QAResult,
    run_deploy_phase,
    run_qa_phase,
)


# ---------------------------------------------------------------------------
# Deterministic fake model
# ---------------------------------------------------------------------------


class _FakeModel(BaseModel):
    """Predicts fixed rates; useful for verifying pipeline selection logic."""

    def __init__(self, lam_h: float = 1.5, lam_a: float = 1.0, **kwargs: Any) -> None:
        self._lam_h = lam_h
        self._lam_a = lam_a

    @property
    def name(self) -> str:
        return "fake"

    def fit(self, X: np.ndarray, y: np.ndarray) -> _FakeModel:
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        return np.full(n, self._lam_h), np.full(n, self._lam_a)

    def get_params(self) -> dict[str, Any]:
        return {"lam_h": self._lam_h, "lam_a": self._lam_a}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_mlflow(tmp_path, monkeypatch):
    """Redirect all MLflow I/O to a temp directory for test isolation."""
    uri = f"file:{tmp_path / 'mlruns'}"

    _noop_setup = lambda *_a, **_kw: mlflow.set_tracking_uri(uri)  # noqa: E731
    monkeypatch.setattr("src.models.pipeline.setup_mlflow", _noop_setup)
    monkeypatch.setattr("src.models.mlflow_utils.setup_mlflow", _noop_setup)

    mlflow.set_tracking_uri(uri)
    yield uri
    while mlflow.active_run():
        mlflow.end_run()


@pytest.fixture()
def fake_splits():
    """Minimal DataSplits with Poisson(1.5) / Poisson(1.0) targets."""
    rng = np.random.default_rng(42)
    n_train, n_holdout = 100, 20
    n_feat = 8

    X_train = rng.standard_normal((n_train, n_feat))
    y_train = np.column_stack([
        rng.poisson(1.5, n_train),
        rng.poisson(1.0, n_train),
    ]).astype(float)

    X_holdout = rng.standard_normal((n_holdout, n_feat))
    y_holdout = np.column_stack([
        rng.poisson(1.5, n_holdout),
        rng.poisson(1.0, n_holdout),
    ]).astype(float)

    X_full = np.vstack([X_train, X_holdout])
    y_full = np.vstack([y_train, y_holdout])

    cols = [f"f{i}" for i in range(n_feat)]
    tcols = ["home_goals", "away_goals"]

    df_train = pd.DataFrame(X_train, columns=cols)
    for i, tc in enumerate(tcols):
        df_train[tc] = y_train[:, i]
    df_train["date_utc"] = pd.date_range("2020-01-01", periods=n_train)

    df_holdout = pd.DataFrame(X_holdout, columns=cols)
    for i, tc in enumerate(tcols):
        df_holdout[tc] = y_holdout[:, i]
    df_holdout["date_utc"] = pd.date_range("2022-11-20", periods=n_holdout)

    df_full = pd.concat([df_train, df_holdout], ignore_index=True)

    return DataSplits(
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        X_full=X_full,
        y_full=y_full,
        df_train=df_train,
        df_holdout=df_holdout,
        df_full=df_full,
    )


def _make_exp_result(
    model_name: str,
    cv_nll: float,
    splits: DataSplits,
    *,
    model_cls: type[BaseModel] = _FakeModel,
    best_params: dict[str, Any] | None = None,
) -> ExperimentalResult:
    return ExperimentalResult(
        model_name=model_name,
        model_cls=model_cls,
        best_params=best_params or {},
        cv_nll=cv_nll,
        importance=pd.DataFrame(
            {"feature": ["f0"], "importance_mean": [0.1], "importance_std": [0.01]}
        ),
        splits=splits,
        feature_cols=[f"f{i}" for i in range(8)],
    )


# ---------------------------------------------------------------------------
# Top-K selection (pure logic, no MLflow needed)
# ---------------------------------------------------------------------------


class TestAllModelsAdvance:
    """All models now advance to QA (no TOP_K cutoff)."""

    def test_sorted_by_cv_nll_ascending(self, fake_splits):
        results = [
            _make_exp_result("worst", 0.50, fake_splits),
            _make_exp_result("best", 0.15, fake_splits),
            _make_exp_result("mid", 0.25, fake_splits),
            _make_exp_result("good", 0.20, fake_splits),
            _make_exp_result("ok", 0.30, fake_splits),
        ]
        results.sort(key=lambda r: r.cv_nll)

        assert len(results) == 5
        assert results[0].model_name == "best"
        assert results[1].model_name == "good"
        assert results[2].model_name == "mid"
        assert results[3].model_name == "ok"
        assert results[4].model_name == "worst"

    def test_all_models_advance(self, fake_splits):
        results = [
            _make_exp_result(f"model_{i}", i * 0.1, fake_splits)
            for i in range(5)
        ]
        results.sort(key=lambda r: r.cv_nll)
        assert len(results) == 5
        assert results[0].model_name == "model_0"
        assert results[-1].model_name == "model_4"


# ---------------------------------------------------------------------------
# Phase 2: QA — best holdout RPS wins
# ---------------------------------------------------------------------------


class TestQAPhase:
    def test_best_holdout_rps_wins(self, tmp_mlflow, fake_splits):
        """Model predicting closer to actual means should win."""
        good = _make_exp_result(
            "good", 0.20, fake_splits,
            best_params={"lam_h": 1.5, "lam_a": 1.0},
        )
        bad = _make_exp_result(
            "bad", 0.25, fake_splits,
            best_params={"lam_h": 5.0, "lam_a": 5.0},
        )

        winner = run_qa_phase([bad, good])

        assert winner.model_name == "good"
        assert winner.holdout_rps < 1.0

    def test_qa_produces_valid_metrics(self, tmp_mlflow, fake_splits):
        entry = _make_exp_result(
            "only", 0.20, fake_splits, best_params={"lam_h": 1.5, "lam_a": 1.0}
        )
        winner = run_qa_phase([entry])

        assert winner.qa_run_id
        assert winner.holdout_rps >= 0.0
        assert np.isfinite(winner.holdout_nll)
        assert winner.holdout_rmse_home >= 0.0
        assert winner.holdout_rmse_away >= 0.0

    def test_qa_logs_to_mlflow(self, tmp_mlflow, fake_splits):
        entry = _make_exp_result(
            "logged", 0.20, fake_splits, best_params={"lam_h": 1.5, "lam_a": 1.0}
        )
        winner = run_qa_phase([entry])

        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(winner.qa_run_id).data
        assert "holdout_rps" in run_data.metrics
        assert "holdout_nll" in run_data.metrics
        assert run_data.tags["stage"] == "qa"
        assert run_data.tags["model_name"] == "logged"


# ---------------------------------------------------------------------------
# Phase 3: Deploy — register + promote
# ---------------------------------------------------------------------------


class TestDeployPhase:
    @patch("src.models.pipeline.promote_to_production")
    @patch("src.models.pipeline.register_model")
    def test_returns_valid_run_id(
        self, mock_register, mock_promote, tmp_mlflow, fake_splits
    ):
        mock_register.return_value = MagicMock(version="1")

        qa = QAResult(
            model_name="fake",
            model_cls=_FakeModel,
            best_params={"lam_h": 1.5, "lam_a": 1.0},
            cv_nll=0.20,
            holdout_rps=0.18,
            holdout_nll=2.75,
            holdout_rmse_home=1.0,
            holdout_rmse_away=0.8,
            qa_run_id="qa_abc",
            splits=fake_splits,
            feature_cols=[f"f{i}" for i in range(8)],
        )
        run_id = run_deploy_phase(qa)

        assert isinstance(run_id, str) and len(run_id) > 0
        mock_register.assert_called_once()
        mock_promote.assert_called_once_with(version="1")

    @patch("src.models.pipeline.promote_to_production")
    @patch("src.models.pipeline.register_model")
    def test_logs_gold_row_count_and_tags(
        self, mock_register, mock_promote, tmp_mlflow, fake_splits
    ):
        mock_register.return_value = MagicMock(version="1")

        qa = QAResult(
            model_name="fake",
            model_cls=_FakeModel,
            best_params={"lam_h": 1.5, "lam_a": 1.0},
            cv_nll=0.20,
            holdout_rps=0.18,
            holdout_nll=2.75,
            holdout_rmse_home=1.0,
            holdout_rmse_away=0.8,
            qa_run_id="qa_xyz",
            splits=fake_splits,
            feature_cols=[f"f{i}" for i in range(8)],
        )
        run_id = run_deploy_phase(qa)

        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run_id).data

        assert run_data.params["gold_row_count"] == str(len(fake_splits.df_full))
        assert run_data.params["evaluation_run_id"] == "qa_xyz"
        assert run_data.tags["stage"] == "production-refit"
        assert run_data.tags["model_name"] == "fake"
        assert "qa_holdout_rps" in run_data.metrics
        assert "qa_holdout_nll" in run_data.metrics

    @patch("src.models.pipeline.promote_to_production")
    @patch("src.models.pipeline.register_model")
    def test_model_artifact_logged(
        self, mock_register, mock_promote, tmp_mlflow, fake_splits
    ):
        """Verify that an MLflow model artifact is created by the deploy phase."""
        mock_register.return_value = MagicMock(version="1")

        qa = QAResult(
            model_name="fake",
            model_cls=_FakeModel,
            best_params={},
            cv_nll=0.20,
            holdout_rps=0.18,
            holdout_nll=2.75,
            holdout_rmse_home=1.0,
            holdout_rmse_away=0.8,
            qa_run_id="qa_art",
            splits=fake_splits,
            feature_cols=[f"f{i}" for i in range(8)],
        )
        run_id = run_deploy_phase(qa)

        client = mlflow.tracking.MlflowClient()
        model_artifacts = client.list_artifacts(run_id, "model")
        artifact_names = [a.path for a in model_artifacts]
        assert any("MLmodel" in p for p in artifact_names)


# ---------------------------------------------------------------------------
# Trigger threshold gating
# ---------------------------------------------------------------------------


class TestDispatchThreshold:
    @patch("src.models.pipeline.run_champion_refit")
    @patch("src.models.pipeline.run_full_pipeline")
    @patch("src.models.data_split.load_gold")
    @patch("src.models.mlflow_utils.get_latest_production_run_id")
    @patch("src.models.mlflow_utils.setup_mlflow")
    def test_skip_retrain_when_delta_small(
        self, mock_setup, mock_prod_id, mock_gold, mock_pipeline, mock_refit
    ):
        """Delta < RETRAIN_THRESHOLD — neither full pipeline nor refit is called."""
        from src.pipeline.trigger import dispatch_training_or_inference

        mock_gold.return_value = pd.DataFrame({"x": range(100)})
        mock_prod_id.return_value = "run_existing"

        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.data.params = {"gold_row_count": "95"}
        mock_client.get_run.return_value = mock_run

        with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            dispatch_training_or_inference(mode="auto")

        mock_pipeline.assert_not_called()
        mock_refit.assert_not_called()

    @patch("src.models.pipeline.run_champion_refit")
    @patch("src.models.pipeline.run_full_pipeline")
    @patch("src.models.data_split.load_gold")
    @patch("src.models.mlflow_utils.get_latest_production_run_id")
    @patch("src.models.mlflow_utils.setup_mlflow")
    def test_refit_when_delta_sufficient(
        self, mock_setup, mock_prod_id, mock_gold, mock_pipeline, mock_refit
    ):
        """Delta >= RETRAIN_THRESHOLD with existing champion → refit champion."""
        from src.pipeline.trigger import dispatch_training_or_inference

        mock_df = pd.DataFrame({"x": range(110)})
        mock_gold.return_value = mock_df
        mock_prod_id.return_value = "run_existing"

        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.data.params = {"gold_row_count": "95"}
        mock_client.get_run.return_value = mock_run

        with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            dispatch_training_or_inference(mode="auto")

        mock_refit.assert_called_once_with(mock_df)
        mock_pipeline.assert_not_called()

    @patch("src.models.pipeline.run_champion_refit")
    @patch("src.models.pipeline.run_full_pipeline")
    @patch("src.models.data_split.load_gold")
    @patch("src.models.mlflow_utils.get_latest_production_run_id")
    @patch("src.models.mlflow_utils.setup_mlflow")
    def test_first_run_always_trains(
        self, mock_setup, mock_prod_id, mock_gold, mock_pipeline, mock_refit
    ):
        """No production champion → full pipeline (Experimental + QA + Deploy)."""
        from src.pipeline.trigger import dispatch_training_or_inference

        mock_df = pd.DataFrame({"x": range(50)})
        mock_gold.return_value = mock_df
        mock_prod_id.return_value = None

        dispatch_training_or_inference(mode="auto")

        mock_pipeline.assert_called_once_with(mock_df)
        mock_refit.assert_not_called()

    @patch("src.models.pipeline.run_champion_refit")
    @patch("src.models.pipeline.run_full_pipeline")
    @patch("src.models.data_split.load_gold")
    @patch("src.models.mlflow_utils.setup_mlflow")
    def test_inference_only_skips_retrain(
        self, mock_setup, mock_gold, mock_pipeline, mock_refit
    ):
        from src.pipeline.trigger import dispatch_training_or_inference

        mock_gold.return_value = pd.DataFrame({"x": range(100)})
        dispatch_training_or_inference(mode="inference_only")

        mock_pipeline.assert_not_called()
        mock_refit.assert_not_called()

    @patch("src.models.pipeline.run_champion_refit")
    @patch("src.models.pipeline.run_full_pipeline")
    @patch("src.models.data_split.load_gold")
    @patch("src.models.mlflow_utils.get_latest_production_run_id")
    @patch("src.models.mlflow_utils.setup_mlflow")
    def test_exact_threshold_triggers_refit(
        self, mock_setup, mock_prod_id, mock_gold, mock_pipeline, mock_refit
    ):
        """delta == RETRAIN_THRESHOLD (10) with champion → champion refit, not full pipeline."""
        from src.pipeline.trigger import dispatch_training_or_inference

        mock_df = pd.DataFrame({"x": range(110)})
        mock_gold.return_value = mock_df
        mock_prod_id.return_value = "run_existing"

        mock_client = MagicMock()
        mock_run = MagicMock()
        mock_run.data.params = {"gold_row_count": "100"}
        mock_client.get_run.return_value = mock_run

        with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            dispatch_training_or_inference(mode="auto")

        mock_refit.assert_called_once_with(mock_df)
        mock_pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# Phase ordering (end-to-end with mocked internals)
# ---------------------------------------------------------------------------


class TestPhaseOrdering:
    @patch("src.models.pipeline.promote_to_production")
    @patch("src.models.pipeline.set_challenger_alias")
    @patch("src.models.pipeline.register_model")
    def test_qa_winner_flows_to_deploy(
        self, mock_register, mock_challenger, mock_promote, tmp_mlflow, fake_splits
    ):
        """Verify that the QA winner's params appear in the deploy run."""
        mock_register.return_value = MagicMock(version="1")

        good = _make_exp_result(
            "good", 0.15, fake_splits,
            best_params={"lam_h": 1.5, "lam_a": 1.0},
        )
        bad = _make_exp_result(
            "bad", 0.25, fake_splits,
            best_params={"lam_h": 5.0, "lam_a": 5.0},
        )

        winner = run_qa_phase([bad, good])
        assert winner.model_name == "good"

        run_id = run_deploy_phase(winner)

        client = mlflow.tracking.MlflowClient()
        run_data = client.get_run(run_id).data
        assert run_data.params["evaluation_run_id"] == winner.qa_run_id
        assert run_data.tags["stage"] == "production-refit"
        assert run_data.tags["model_name"] == "good"


# ---------------------------------------------------------------------------
# Champion-vs-challenger gate
# ---------------------------------------------------------------------------


class TestChampionGate:
    @patch("src.models.pipeline.promote_to_production")
    @patch("src.models.pipeline.register_model")
    @patch("src.models.pipeline.get_champion_rps")
    def test_no_champion_always_promotes(
        self, mock_champion_rps, mock_register, mock_promote, tmp_mlflow, fake_splits
    ):
        """First run: no existing champion → promotion always happens."""
        mock_champion_rps.return_value = None
        mock_register.return_value = MagicMock(version="1")

        qa = QAResult(
            model_name="fake",
            model_cls=_FakeModel,
            best_params={},
            cv_nll=0.20,
            holdout_rps=0.25,
            holdout_nll=2.80,
            holdout_rmse_home=1.0,
            holdout_rmse_away=0.8,
            qa_run_id="qa_first",
            splits=fake_splits,
            feature_cols=[f"f{i}" for i in range(8)],
        )
        run_id = run_deploy_phase(qa)

        assert isinstance(run_id, str) and len(run_id) > 0
        mock_promote.assert_called_once()

    @patch("src.models.pipeline.promote_to_production")
    @patch("src.models.pipeline.register_model")
    @patch("src.models.pipeline.get_champion_rps")
    def test_challenger_beats_champion_promotes(
        self, mock_champion_rps, mock_register, mock_promote, tmp_mlflow, fake_splits
    ):
        """Challenger RPS < champion RPS → improvement → promotion happens."""
        mock_champion_rps.return_value = 0.30  # existing champion
        mock_register.return_value = MagicMock(version="2")

        qa = QAResult(
            model_name="fake",
            model_cls=_FakeModel,
            best_params={},
            cv_nll=0.20,
            holdout_rps=0.25,  # better than champion 0.30
            holdout_nll=2.75,
            holdout_rmse_home=1.0,
            holdout_rmse_away=0.8,
            qa_run_id="qa_better",
            splits=fake_splits,
            feature_cols=[f"f{i}" for i in range(8)],
        )
        run_id = run_deploy_phase(qa)

        assert isinstance(run_id, str) and len(run_id) > 0
        mock_promote.assert_called_once_with(version="2")

    @patch("src.models.pipeline.promote_to_production")
    @patch("src.models.pipeline.register_model")
    @patch("src.models.pipeline.get_champion_rps")
    def test_challenger_loses_raises(
        self, mock_champion_rps, mock_register, mock_promote, tmp_mlflow, fake_splits
    ):
        """Challenger RPS > champion RPS → no improvement → ChallengeFailed raised."""
        mock_champion_rps.return_value = 0.20  # existing champion is very good

        qa = QAResult(
            model_name="fake",
            model_cls=_FakeModel,
            best_params={},
            cv_nll=0.25,
            holdout_rps=0.22,  # worse than champion 0.20
            holdout_nll=2.80,
            holdout_rmse_home=1.0,
            holdout_rmse_away=0.8,
            qa_run_id="qa_worse",
            splits=fake_splits,
            feature_cols=[f"f{i}" for i in range(8)],
        )
        with pytest.raises(ChallengeFailed, match="does not beat champion"):
            run_deploy_phase(qa)

        mock_register.assert_not_called()
        mock_promote.assert_not_called()

    @patch("src.models.pipeline.promote_to_production")
    @patch("src.models.pipeline.register_model")
    @patch("src.models.pipeline.get_champion_rps")
    def test_challenger_ties_promotes(
        self, mock_champion_rps, mock_register, mock_promote, tmp_mlflow, fake_splits
    ):
        """Challenger RPS == champion RPS → tie → promotion happens (newer model)."""
        mock_champion_rps.return_value = 0.25
        mock_register.return_value = MagicMock(version="3")

        qa = QAResult(
            model_name="fake",
            model_cls=_FakeModel,
            best_params={},
            cv_nll=0.20,
            holdout_rps=0.25,  # equal to champion
            holdout_nll=2.75,
            holdout_rmse_home=1.0,
            holdout_rmse_away=0.8,
            qa_run_id="qa_tie",
            splits=fake_splits,
            feature_cols=[f"f{i}" for i in range(8)],
        )
        run_id = run_deploy_phase(qa)

        assert isinstance(run_id, str) and len(run_id) > 0
        mock_register.assert_called_once()
        mock_promote.assert_called_once_with(version="3")


# ---------------------------------------------------------------------------
# Champion refit (bypass Experimental + QA on subsequent runs)
# ---------------------------------------------------------------------------


class TestChampionRefit:
    @patch("src.models.pipeline.promote_to_production")
    @patch("src.models.pipeline.register_model")
    @patch("src.models.pipeline.get_champion_metadata")
    @patch("src.models.pipeline.make_splits")
    def test_refit_promotes_new_version(
        self,
        mock_splits,
        mock_meta,
        mock_register,
        mock_promote,
        tmp_mlflow,
        fake_splits,
    ):
        """run_champion_refit loads champion metadata, refits on X_full, and promotes."""
        from src.models.mlflow_utils import ChampionMeta
        from src.models.pipeline import run_champion_refit

        mock_splits.return_value = fake_splits
        mock_meta.return_value = ChampionMeta(
            model_name="ridge",
            best_params={"alpha": 1.0},
            holdout_metrics={"qa_holdout_rps": 0.21, "qa_holdout_nll": 2.95},
        )
        mock_register.return_value = MagicMock(version="5")

        df = pd.DataFrame({"x": range(50)})
        run_id = run_champion_refit(df)

        assert isinstance(run_id, str) and len(run_id) > 0
        mock_meta.assert_called_once()
        mock_promote.assert_called_once_with(version="5")

    @patch("src.models.pipeline.promote_to_production")
    @patch("src.models.pipeline.register_model")
    @patch("src.models.pipeline.get_champion_metadata")
    @patch("src.models.pipeline.make_splits")
    def test_refit_forwards_holdout_metrics(
        self,
        mock_splits,
        mock_meta,
        mock_register,
        mock_promote,
        tmp_mlflow,
        fake_splits,
    ):
        """Holdout metrics from the champion run are logged unchanged to the new run."""
        from src.models.mlflow_utils import ChampionMeta
        from src.models.pipeline import run_champion_refit

        mock_splits.return_value = fake_splits
        original_metrics = {"qa_holdout_rps": 0.19, "qa_holdout_nll": 2.88}
        mock_meta.return_value = ChampionMeta(
            model_name="ridge",
            best_params={"alpha": 0.5},
            holdout_metrics=original_metrics,
        )
        mock_register.return_value = MagicMock(version="6")

        run_id = run_champion_refit(pd.DataFrame({"x": range(50)}))

        # Verify the metrics were forwarded by inspecting the MLflow run
        import mlflow as _mlflow

        run_data = _mlflow.tracking.MlflowClient().get_run(run_id).data
        assert run_data.metrics["qa_holdout_rps"] == pytest.approx(0.19)
        assert run_data.metrics["qa_holdout_nll"] == pytest.approx(2.88)
