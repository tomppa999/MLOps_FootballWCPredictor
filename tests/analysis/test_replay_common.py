"""Unit tests for D.1 replay_common utilities."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.replay_common import (
    GoldCommit,
    check_entropy_trajectory,
    compute_advancement_entropy,
    compute_entropy_columns,
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

    def test_normalises_by_column_sum_not_hardcoded_32(self):
        """Regression: the helper divided every column by 32, which is only
        correct for p_r32 and made other rounds unreadable."""
        teams = [f"T{i}" for i in range(48)]
        df = pd.DataFrame({"team": teams, "p_qf": np.full(48, 8.0 / 48.0)})
        h = compute_advancement_entropy(df, prob_col="p_qf")
        assert h == pytest.approx(np.log(48), rel=1e-6)

    def test_resolved_round_has_zero_entropy(self):
        p = np.zeros(48)
        p[0] = 1.0
        df = pd.DataFrame({"team": [f"T{i}" for i in range(48)], "p_winner": p})
        assert compute_advancement_entropy(df, prob_col="p_winner") == pytest.approx(0.0)

    def test_all_zero_column_returns_nan(self):
        df = pd.DataFrame({"team": ["A", "B"], "p_winner": [0.0, 0.0]})
        assert np.isnan(compute_advancement_entropy(df, prob_col="p_winner"))


class TestEntropyColumns:
    def _advancement(self) -> pd.DataFrame:
        teams = [f"T{i}" for i in range(48)]
        data = {"team": teams, "p_group": np.ones(48)}
        for col, slots in [
            ("p_r32", 32), ("p_r16", 16), ("p_qf", 8),
            ("p_sf", 4), ("p_final", 2), ("p_winner", 1),
        ]:
            data[col] = np.full(48, slots / 48.0)
        return pd.DataFrame(data)

    def test_returns_all_rounds_and_omits_group(self):
        result = compute_entropy_columns(self._advancement())
        assert set(result) == {
            "entropy_r32", "entropy_r16", "entropy_qf",
            "entropy_sf", "entropy_final", "entropy_winner",
        }
        assert "entropy_group" not in result

    def test_rounds_share_one_scale_when_uniform(self):
        result = compute_entropy_columns(self._advancement())
        for value in result.values():
            assert value == pytest.approx(np.log(48), rel=1e-6)


class TestCheckEntropyTrajectory:
    def _df(self, values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({
            "inference_timestamp": pd.date_range("2026-06-06", periods=len(values), tz="UTC"),
            "entropy_winner": values,
        })

    def test_reports_first_and_last(self):
        metrics = check_entropy_trajectory(self._df([3.8, 2.0, 0.0]))
        assert metrics["entropy_winner_first"] == pytest.approx(3.8)
        assert metrics["entropy_winner_last"] == pytest.approx(0.0)

    def test_warns_on_constant_curve(self, caplog):
        with caplog.at_level(logging.WARNING):
            check_entropy_trajectory(self._df([1.5, 1.5, 1.5]))
        assert "constant" in caplog.text

    def test_no_warning_on_varying_curve(self, caplog):
        with caplog.at_level(logging.WARNING):
            check_entropy_trajectory(self._df([3.8, 2.0, 0.0]))
        assert "constant" not in caplog.text

    def test_orders_by_timestamp(self):
        df = self._df([3.8, 2.0, 0.0]).iloc[::-1].reset_index(drop=True)
        metrics = check_entropy_trajectory(df)
        assert metrics["entropy_winner_first"] == pytest.approx(3.8)


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
