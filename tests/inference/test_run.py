"""Tests for src.inference.run (orchestrator)."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from src.inference.run import (
    _build_ko_fixtures,
    _seed_from_timestamp,
    run_inference_and_simulation,
)
from src.models.config import EXPERIMENT_MODELS


def _make_gold_df() -> pd.DataFrame:
    """Minimal Gold DataFrame for testing."""
    rows = []
    for i in range(5):
        rows.append({
            "fixture_id": 1000 + i,
            "date_utc": f"2025-06-{10 + i}",
            "season": 2025,
            "league_id": 10,
            "league_name": "Friendly",
            "home_team": "France",
            "away_team": "Germany",
            "home_goals": 2,
            "away_goals": 1,
            "home_elo_pre": 2000.0,
            "away_elo_pre": 1980.0,
            "elo_diff": 20.0,
            "competition_tier": 4,
            "is_knockout": False,
            "is_neutral": False,
            "stats_tier": "none",
            "home_shots_on_goal": 4,
            "home_total_shots": 10,
            "home_fouls": 12,
            "home_corner_kicks": 5,
            "home_possession_pct": 55.0,
            "away_shots_on_goal": 3,
            "away_total_shots": 8,
            "away_fouls": 14,
            "away_corner_kicks": 3,
            "away_possession_pct": 45.0,
        })
    return pd.DataFrame(rows)


def _make_pairings() -> pd.DataFrame:
    return pd.DataFrame([{
        "fixture_id": "wc2026_pair_France_Germany",
        "date_utc": "2026-06-15",
        "home_team": "France",
        "away_team": "Germany",
        "competition_tier": 1,
        "is_knockout": False,
        "is_neutral": True,
    }])


def _make_champion_predictions() -> pd.DataFrame:
    """Champion-only prediction (no model_name column)."""
    return pd.DataFrame([{
        "fixture_id": "wc2026_pair_France_Germany",
        "home_team": "France",
        "away_team": "Germany",
        "date_utc": "2026-06-15",
        "lambda_h": 1.5,
        "lambda_a": 1.2,
        "p_home": 0.45,
        "p_draw": 0.25,
        "p_away": 0.30,
    }])


def _make_all_models_predictions() -> pd.DataFrame:
    """Long-format predictions for all EXPERIMENT_MODELS."""
    rows = []
    for model_name in EXPERIMENT_MODELS:
        rows.append({
            "fixture_id": "wc2026_pair_France_Germany",
            "home_team": "France",
            "away_team": "Germany",
            "date_utc": "2026-06-15",
            "model_name": model_name,
            "lambda_h": 1.5,
            "lambda_a": 1.2,
            "p_home": 0.45,
            "p_draw": 0.25,
            "p_away": 0.30,
        })
    return pd.DataFrame(rows)


def _make_sim_results() -> dict:
    return {
        "advancement": pd.DataFrame([{"team": "France", "p_winner": 0.15}]),
        "group_positions": pd.DataFrame([{"team": "France", "p_1st": 0.6}]),
        "ko_pairings": pd.DataFrame([{"team_a": "France", "team_b": "Germany", "count": 500}]),
        "ko_slot_pairings": pd.DataFrame([{
            "stage": "R32",
            "match_num": 49,
            "home_team": "France",
            "away_team": "Germany",
            "count": 100,
            "frequency": 1.0,
        }]),
        "n_sims": 100,
    }


class TestBuildKoFixtures:
    """_build_ko_fixtures consumes team-set-keyed locked_ko (frozenset keys)."""

    def test_no_sim_uses_team_set_values_and_stage(self):
        """With no simulation pairings, locked fixtures are emitted from the
        team-set values, reading stage from the stored field (match_num=None)."""
        locked_ko = {
            frozenset({"France", "Germany"}): {
                "home": "France", "away": "Germany",
                "home_goals": 2, "away_goals": 1,
                "decided_by": "FT", "stage": "R32",
            }
        }
        df = _build_ko_fixtures(locked_ko, None)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["status"] == "locked"
        assert row["stage"] == "R32"
        assert row["home_team"] == "France"
        assert row["away_team"] == "Germany"
        assert row["home_goals"] == 2
        assert row["away_goals"] == 1
        assert row["match_num"] is None
        assert row["pairing_frequency"] == 1.0

    def test_modal_slot_locked_by_team_set(self):
        """A modal slot whose (home, away) team-set is locked is emitted as
        status='locked' with the actual score and pairing_frequency=1.0."""
        locked_ko = {
            frozenset({"France", "Germany"}): {
                "home": "France", "away": "Germany",
                "home_goals": 3, "away_goals": 1,
                "decided_by": "FT", "stage": "R32",
            }
        }
        ko_slot_pairings = pd.DataFrame([{
            "stage": "R32", "match_num": 73,
            "home_team": "France", "away_team": "Germany",
            "count": 100, "frequency": 1.0,
        }])
        df = _build_ko_fixtures(locked_ko, ko_slot_pairings)
        row = df[df["match_num"] == 73].iloc[0]
        assert row["status"] == "locked"
        assert row["home_goals"] == 3
        assert row["away_goals"] == 1
        assert row["decided_by"] == "FT"
        assert row["pairing_frequency"] == 1.0

    def test_modal_slot_predicted_when_team_set_not_locked(self):
        """An unlocked modal slot stays predicted with its observed frequency."""
        ko_slot_pairings = pd.DataFrame([{
            "stage": "R32", "match_num": 73,
            "home_team": "Brazil", "away_team": "Spain",
            "count": 80, "frequency": 0.8,
        }])
        df = _build_ko_fixtures({}, ko_slot_pairings)
        row = df[df["match_num"] == 73].iloc[0]
        assert row["status"] == "predicted"
        assert row["home_goals"] is None
        assert abs(row["pairing_frequency"] - 0.8) < 1e-9


class TestSeedFromTimestamp:
    def test_deterministic(self):
        ts = "2026-06-11T08:00:00+00:00"
        assert _seed_from_timestamp(ts) == _seed_from_timestamp(ts)

    def test_different_timestamps_give_different_seeds(self):
        s1 = _seed_from_timestamp("2026-06-11T08:00:00+00:00")
        s2 = _seed_from_timestamp("2026-06-11T08:30:00+00:00")
        assert s1 != s2

    def test_result_is_valid_32bit_uint(self):
        seed = _seed_from_timestamp("2026-06-11T08:00:00+00:00")
        assert isinstance(seed, int)
        assert 0 <= seed < 2**32


class TestRunInferenceAndSimulation:
    @patch("src.inference.run.log_inference_artifacts", return_value="run_123")
    @patch("src.inference.run.simulate_tournament")
    @patch("src.inference.run.run_prediction_all_models")
    @patch("src.inference.run.run_prediction")
    @patch("src.inference.run._alias_for_mode", side_effect=lambda m: f"alias_{m}")
    @patch("src.inference.run.get_champion_metadata")
    @patch("src.inference.run.build_inference_features")
    @patch("src.inference.run.generate_wc_group_fixtures")
    @patch("src.inference.run.generate_all_wc_pairings")
    @patch("src.inference.run.parse_wc_results")
    @patch("src.inference.run.load_gold")
    def test_happy_path(
        self,
        mock_load_gold,
        mock_parse_results,
        mock_all_pairings,
        mock_group_fixtures,
        mock_features,
        mock_champion_meta,
        mock_alias_for_mode,
        mock_predict,
        mock_predict_all,
        mock_simulate,
        mock_log,
    ):
        mock_load_gold.return_value = _make_gold_df()
        mock_parse_results.return_value = {
            "group_results": {},
            "ko_results": {},
            "next_matchday": 1,
            "finished_fixtures": [],
        }

        mock_all_pairings.return_value = _make_pairings()
        mock_group_fixtures.return_value = _make_pairings()

        features = _make_pairings().copy()
        features["elo_diff"] = 20.0
        mock_features.return_value = features

        mock_champion_meta.return_value = MagicMock(model_name="xgboost")
        mock_predict.return_value = _make_champion_predictions()
        mock_predict_all.return_value = _make_all_models_predictions()
        mock_simulate.return_value = _make_sim_results()

        run_id = run_inference_and_simulation(n_sims=100, cadence_mode="per_round")

        assert run_id == "run_123"
        mock_all_pairings.assert_called_once()
        mock_features.assert_called_once()
        mock_predict.assert_called_once()
        mock_predict.assert_called_with(mock_features.return_value, cadence_mode="per_round")
        mock_predict_all.assert_called_once_with(
            mock_features.return_value, cadence_mode="per_round",
        )
        mock_champion_meta.assert_called_once_with(alias="alias_per_round")
        mock_log.assert_called_once()

        # Option A: one simulate_tournament call per EXPERIMENT_MODELS entry.
        assert mock_simulate.call_count == len(EXPERIMENT_MODELS)

        # All simulate_tournament calls must share the same seed.
        seeds_used = [c.kwargs.get("seed") for c in mock_simulate.call_args_list]
        assert len(set(seeds_used)) == 1, "All models in a cycle must share the same seed"
        assert seeds_used[0] is not None

        # log_inference_artifacts must receive per_model_tournament_results + seed.
        log_kwargs = mock_log.call_args.kwargs
        assert "per_model_tournament_results" in log_kwargs
        assert "xgboost" in log_kwargs["per_model_tournament_results"]
        assert "simulation_seed" in log_kwargs
        assert log_kwargs["simulation_seed"] == seeds_used[0]
        assert "inference_timestamp" in log_kwargs
        assert log_kwargs["cadence_mode"] == "per_round"
        assert log_kwargs["matchday_label"] == "1"
        assert log_kwargs["matches_completed_in_matchday"] == 0
        assert log_kwargs["total_matches_completed"] == 0

        # ko_fixtures is built from champion sim and passed to log_inference_artifacts.
        assert "ko_fixtures" in log_kwargs
        kf = log_kwargs["ko_fixtures"]
        # mock sim result has one predicted R32 slot; ko_results is empty → status=predicted
        assert kf is not None
        assert isinstance(kf, pd.DataFrame)
        assert set(kf.columns) >= {"match_num", "stage", "home_team", "away_team", "status"}

    @patch("src.inference.run.generate_all_wc_pairings")
    @patch("src.inference.run.parse_wc_results")
    @patch("src.inference.run.load_gold")
    def test_returns_empty_when_no_pairings(
        self, mock_load_gold, mock_parse_results, mock_all_pairings,
    ):
        mock_load_gold.return_value = _make_gold_df()
        mock_parse_results.return_value = {
            "group_results": {},
            "ko_results": {},
            "next_matchday": 1,
            "finished_fixtures": [],
        }
        mock_all_pairings.return_value = pd.DataFrame()

        run_id = run_inference_and_simulation()
        assert run_id == ""

    @patch("src.inference.run.log_inference_artifacts", return_value="run_456")
    @patch("src.inference.run.simulate_tournament")
    @patch("src.inference.run.run_prediction_all_models", side_effect=RuntimeError("registry down"))
    @patch("src.inference.run.run_prediction")
    @patch("src.inference.run.get_champion_metadata")
    @patch("src.inference.run.build_inference_features")
    @patch("src.inference.run.generate_wc_group_fixtures")
    @patch("src.inference.run.generate_all_wc_pairings")
    @patch("src.inference.run.parse_wc_results")
    @patch("src.inference.run.load_gold")
    def test_fallback_to_champion_only_when_all_models_fails(
        self,
        mock_load_gold,
        mock_parse_results,
        mock_all_pairings,
        mock_group_fixtures,
        mock_features,
        mock_champion_meta,
        mock_predict,
        mock_predict_all,
        mock_simulate,
        mock_log,
    ):
        """When run_prediction_all_models raises, champion sim still runs."""
        mock_load_gold.return_value = _make_gold_df()
        mock_parse_results.return_value = {
            "group_results": {},
            "ko_results": {},
            "next_matchday": 1,
            "finished_fixtures": [],
        }
        mock_all_pairings.return_value = _make_pairings()
        mock_group_fixtures.return_value = _make_pairings()
        features = _make_pairings().copy()
        features["elo_diff"] = 20.0
        mock_features.return_value = features
        mock_champion_meta.return_value = MagicMock(model_name="xgboost")
        mock_predict.return_value = _make_champion_predictions()
        mock_simulate.return_value = _make_sim_results()

        run_id = run_inference_and_simulation(
            n_sims=100,
            cadence_mode="per_round",
            matchday_label="R32",
            matches_completed_in_matchday=5,
            total_matches_completed=72,
        )

        assert run_id == "run_456"
        # Champion-only fallback: exactly one sim call, seed still set.
        assert mock_simulate.call_count == 1
        assert mock_simulate.call_args.kwargs.get("seed") is not None
        mock_log.assert_called_once()
        log_kwargs = mock_log.call_args.kwargs
        assert "xgboost" in log_kwargs["per_model_tournament_results"]
        assert "simulation_seed" in log_kwargs
        assert log_kwargs["cadence_mode"] == "per_round"
        assert log_kwargs["matchday_label"] == "R32"
        assert log_kwargs["matches_completed_in_matchday"] == 5
        assert log_kwargs["total_matches_completed"] == 72
