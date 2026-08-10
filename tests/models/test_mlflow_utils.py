"""Tests for B.2 cadence-mode alias and tag resolution in mlflow_utils,
and B.3 alias-writer (promote_to_production)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlflow
import pytest

from src.models.mlflow_utils import (
    CHAMPION_ALIAS_DEFAULT,
    CHAMPION_ALIAS_FROZEN,
    CHAMPION_ALIAS_PER_ROUND,
    PRODUCTION_MODEL_NAME,
    _alias_for_mode,
    _latest_version_with_tags,
    _resolve_champion_alias,
    get_production_run_id,
    load_champion,
    promote_to_production,
)


class TestAliasForMode:
    def test_frozen_maps_to_champion_frozen(self):
        assert _alias_for_mode("frozen") == CHAMPION_ALIAS_FROZEN

    def test_per_round_maps_to_champion_per_round(self):
        assert _alias_for_mode("per_round") == CHAMPION_ALIAS_PER_ROUND

    def test_unknown_mode_falls_back_to_champion(self):
        assert _alias_for_mode("other") == CHAMPION_ALIAS_DEFAULT


class TestResolveChampionAlias:
    @patch("src.models.mlflow_utils.mlflow.tracking.MlflowClient")
    def test_mode_alias_used_when_present(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        assert _resolve_champion_alias(CHAMPION_ALIAS_FROZEN) == CHAMPION_ALIAS_FROZEN
        mock_client.get_model_version_by_alias.assert_called_once_with(
            PRODUCTION_MODEL_NAME, CHAMPION_ALIAS_FROZEN,
        )

    @patch("src.models.mlflow_utils.mlflow.tracking.MlflowClient")
    def test_missing_mode_alias_falls_back_to_champion(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_model_version_by_alias.side_effect = mlflow.exceptions.MlflowException(
            "not found",
        )
        mock_client_cls.return_value = mock_client
        assert _resolve_champion_alias(CHAMPION_ALIAS_PER_ROUND) == CHAMPION_ALIAS_DEFAULT


class TestGetProductionRunId:
    @patch("src.models.mlflow_utils._resolve_champion_alias", return_value=CHAMPION_ALIAS_FROZEN)
    @patch("src.models.mlflow_utils.setup_mlflow")
    @patch("src.models.mlflow_utils.mlflow.tracking.MlflowClient")
    def test_returns_run_id_for_resolved_alias(
        self, mock_client_cls, mock_setup, mock_resolve,
    ):
        mock_mv = MagicMock(run_id="run-abc")
        mock_client = MagicMock()
        mock_client.get_model_version_by_alias.return_value = mock_mv
        mock_client_cls.return_value = mock_client

        assert get_production_run_id(CHAMPION_ALIAS_FROZEN) == "run-abc"
        mock_resolve.assert_called_once_with(CHAMPION_ALIAS_FROZEN, PRODUCTION_MODEL_NAME)


class TestLoadChampion:
    @patch("mlflow.pyfunc.load_model")
    @patch("src.models.mlflow_utils._resolve_champion_alias", return_value=CHAMPION_ALIAS_FROZEN)
    @patch("src.models.mlflow_utils.setup_mlflow")
    def test_loads_resolved_alias_uri(self, mock_setup, mock_resolve, mock_load):
        load_champion(alias=CHAMPION_ALIAS_FROZEN)
        mock_load.assert_called_once_with(
            f"models:/{PRODUCTION_MODEL_NAME}@{CHAMPION_ALIAS_FROZEN}",
        )


class TestLatestVersionWithTags:
    def _make_mv(self, version: int, run_id: str) -> MagicMock:
        mv = MagicMock()
        mv.version = str(version)
        mv.run_id = run_id
        mv.name = "wc_shadow"
        return mv

    @patch("src.models.mlflow_utils.mlflow.tracking.MlflowClient")
    def test_prefers_cadence_mode_match(self, mock_client_cls):
        mv_old = self._make_mv(1, "run-old")
        mv_new = self._make_mv(2, "run-new")
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mv_old, mv_new]
        mock_client.get_run.side_effect = lambda run_id: MagicMock(
            data=MagicMock(tags={
                "model_name": "poisson_glm",
                "cadence_mode": "per_round" if run_id == "run-new" else "frozen",
            }),
        )
        mock_client_cls.return_value = mock_client

        result = _latest_version_with_tags(
            "wc_shadow", "poisson_glm", cadence_mode="per_round",
        )
        assert result is mv_new

    @patch("src.models.mlflow_utils.mlflow.tracking.MlflowClient")
    def test_falls_back_to_model_name_only(self, mock_client_cls):
        mv = self._make_mv(1, "run-1")
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mv]
        mock_client.get_run.return_value = MagicMock(
            data=MagicMock(tags={"model_name": "ridge"}),
        )
        mock_client_cls.return_value = mock_client

        result = _latest_version_with_tags(
            "wc_shadow", "ridge", cadence_mode="per_round",
        )
        assert result is mv

    @patch("src.models.mlflow_utils.mlflow.tracking.MlflowClient")
    def test_frozen_skips_newer_per_round_version(self, mock_client_cls):
        """Untagged frozen shadow must not resolve to a newer per_round version."""
        mv_frozen = self._make_mv(88, "run-frozen")
        mv_per_round = self._make_mv(116, "run-per-round")
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mv_frozen, mv_per_round]
        mock_client.get_run.side_effect = lambda run_id: MagicMock(
            data=MagicMock(tags={
                "model_name": "poisson_glm",
                **(
                    {"cadence_mode": "per_round"}
                    if run_id == "run-per-round"
                    else {}
                ),
            }),
        )
        mock_client_cls.return_value = mock_client

        result = _latest_version_with_tags(
            "wc_shadow", "poisson_glm", cadence_mode="frozen",
        )
        assert result is mv_frozen

    @patch("src.models.mlflow_utils.mlflow.tracking.MlflowClient")
    def test_none_cadence_returns_newest_model_name_match(self, mock_client_cls):
        mv_old = self._make_mv(1, "run-old")
        mv_new = self._make_mv(2, "run-new")
        mock_client = MagicMock()
        mock_client.search_model_versions.return_value = [mv_old, mv_new]
        mock_client.get_run.side_effect = lambda run_id: MagicMock(
            data=MagicMock(tags={
                "model_name": "poisson_glm",
                "cadence_mode": "per_round" if run_id == "run-new" else "frozen",
            }),
        )
        mock_client_cls.return_value = mock_client

        result = _latest_version_with_tags("wc_shadow", "poisson_glm")
        assert result is mv_new


class TestPromoteToProduction:
    @patch("src.models.mlflow_utils.mlflow.tracking.MlflowClient")
    def test_default_sets_champion_alias(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        promote_to_production(version="3")
        mock_client.set_registered_model_alias.assert_called_once_with(
            PRODUCTION_MODEL_NAME, CHAMPION_ALIAS_DEFAULT, "3",
        )

    @patch("src.models.mlflow_utils.mlflow.tracking.MlflowClient")
    def test_per_round_alias_written_when_specified(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        promote_to_production(version="7", alias=CHAMPION_ALIAS_PER_ROUND)
        mock_client.set_registered_model_alias.assert_called_once_with(
            PRODUCTION_MODEL_NAME, CHAMPION_ALIAS_PER_ROUND, "7",
        )

    @patch("src.models.mlflow_utils.mlflow.tracking.MlflowClient")
    def test_frozen_alias_written_when_specified(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        promote_to_production(version="2", alias=CHAMPION_ALIAS_FROZEN)
        mock_client.set_registered_model_alias.assert_called_once_with(
            PRODUCTION_MODEL_NAME, CHAMPION_ALIAS_FROZEN, "2",
        )
