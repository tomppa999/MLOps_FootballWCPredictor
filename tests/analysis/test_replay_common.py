"""Unit tests for D.1 replay_common utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.replay_common import (
    GoldCommit,
    compute_advancement_entropy,
    leaderboard_summary,
    resolve_gold_commit,
    score_prediction_row,
)


class TestResolveGoldCommit:
    def test_returns_last_commit_before_kickoff(self):
        index = [
            GoldCommit("a", pd.Timestamp("2026-06-10T10:00:00Z"), "hash1"),
            GoldCommit("b", pd.Timestamp("2026-06-11T10:00:00Z"), "hash2"),
            GoldCommit("c", pd.Timestamp("2026-06-12T10:00:00Z"), "hash3"),
        ]
        kickoff = pd.Timestamp("2026-06-11T20:00:00Z")
        result = resolve_gold_commit(kickoff, index)
        assert result is not None
        assert result.commit_sha == "b"

    def test_returns_none_when_no_prior_commit(self):
        index = [GoldCommit("a", pd.Timestamp("2026-06-12T10:00:00Z"), "hash1")]
        kickoff = pd.Timestamp("2026-06-11T10:00:00Z")
        assert resolve_gold_commit(kickoff, index) is None


class TestAdvancementEntropy:
    def test_uniform_distribution_has_max_entropy(self):
        teams = [f"T{i}" for i in range(48)]
        p = np.full(48, 32.0 / 48.0)
        df = pd.DataFrame({"team": teams, "p_r32": p})
        h = compute_advancement_entropy(df)
        assert h == pytest.approx(np.log(48), rel=1e-6)

    def test_empty_returns_nan(self):
        assert np.isnan(compute_advancement_entropy(pd.DataFrame()))


class TestLeaderboardSummary:
    def test_rmse_is_average_of_sides(self):
        df = pd.DataFrame({
            "match_id": [1, 2],
            "model_name": ["ridge", "ridge"],
            "cadence_mode": ["frozen", "frozen"],
            "rps": [0.2, 0.3],
            "nll": [2.0, 2.5],
            "rmse_h": [0.4, 0.6],
            "rmse_a": [0.2, 0.4],
        })
        summary = leaderboard_summary(df)
        assert summary.iloc[0]["mean_rmse"] == pytest.approx(0.4)


class TestScorePredictionRow:
    def test_produces_monitoring_columns(self):
        match = pd.Series({
            "match_id": 1,
            "kickoff_utc": pd.Timestamp("2026-06-11T20:00:00Z"),
            "home": "A",
            "away": "B",
            "actual_h": 2,
            "actual_a": 1,
            "actual_outcome": 0,
        })
        pred = {
            "lambda_h": 1.5,
            "lambda_a": 1.0,
            "p_home": 0.4,
            "p_draw": 0.3,
            "p_away": 0.3,
        }
        row = score_prediction_row(match, "ridge", pred, cadence_mode="frozen")
        assert row["model_name"] == "ridge"
        assert row["cadence_mode"] == "frozen"
        assert "rps" in row
        assert row["rmse_h"] == pytest.approx(0.5)
