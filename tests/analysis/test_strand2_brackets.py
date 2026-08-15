"""Regression tests for the strand 2 as-of cutoff.

The D.1 leak: every replayed cycle locked the *final* tournament result, so
all ~880 replays came back byte-identical and fully determined.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.analysis import strand2_brackets as strand2
from src.analysis.replay_common import SETTLE_DELTA, FinishedFixture, parse_wc_results_before_kickoff
from src.analysis.strand2_brackets import _replay_simulation_for_run


def _run_with_params(params: dict[str, str]) -> MagicMock:
    run = MagicMock()
    run.data.params = params
    return run


def _predictions() -> pd.DataFrame:
    return pd.DataFrame({"model_name": ["xgboost"], "lambda_h": [1.2], "lambda_a": [1.0]})


class TestReplayCutoff:
    def _replay(self, as_of: pd.Timestamp) -> MagicMock:
        """Run one replay with everything stubbed; return the cutoff spy."""
        cutoff_spy = MagicMock(
            return_value={"group_results": {}, "ko_results": {}},
        )
        client = MagicMock()
        client.get_run.return_value = _run_with_params(
            {"simulation_seed": "42", "champion_model_name": "xgboost"},
        )
        with (
            patch("src.analysis.strand2_brackets.mlflow.tracking.MlflowClient", return_value=client),
            patch("src.analysis.strand2_brackets._load_artifact", return_value=_predictions()),
            patch("src.analysis.strand2_brackets.parse_wc_results_before_kickoff", cutoff_spy),
            patch("src.analysis.strand2_brackets._simulate_roster", return_value={}),
        ):
            _replay_simulation_for_run("run1", cadence_mode="frozen", as_of=as_of)
        return cutoff_spy

    def test_cutoff_derives_from_the_runs_own_timestamp(self):
        as_of = pd.Timestamp("2026-06-20T12:00:00Z")
        spy = self._replay(as_of)
        spy.assert_called_once()
        assert spy.call_args.args[0] == as_of

    def test_different_runs_get_different_cutoffs(self):
        early = self._replay(pd.Timestamp("2026-06-10T12:00:00Z"))
        late = self._replay(pd.Timestamp("2026-07-01T12:00:00Z"))
        assert early.call_args.args[0] != late.call_args.args[0]

    def test_settle_delta_is_passed_through(self):
        spy = self._replay(pd.Timestamp("2026-06-20T12:00:00Z"))
        assert spy.call_args.kwargs["settle_delta"] == SETTLE_DELTA


def _fixture(kickoff: str) -> FinishedFixture:
    return FinishedFixture(
        fixture_id=1,
        kickoff=pd.Timestamp(kickoff),
        kickoff_raw=kickoff,
        round_str="Group A - 1",
        round_label="MD1",
        home_team="A",
        away_team="B",
        home_goals=2,
        away_goals=1,
        status="FT",
        teams={},
        score={},
    )


class TestSettleDelta:
    """A match still being played at the cutoff must not leak its final score."""

    def _group_results(self, kickoff: str, as_of: str) -> dict:
        with patch(
            "src.analysis.replay_common.load_finished_wc_fixtures",
            return_value=(_fixture(kickoff),),
        ):
            wc = parse_wc_results_before_kickoff(
                pd.Timestamp(as_of), settle_delta=SETTLE_DELTA,
            )
        return wc["group_results"]

    def test_match_still_in_play_is_excluded(self):
        # Kicked off 30 minutes ago — cannot be known yet.
        assert self._group_results("2026-06-20T11:30:00Z", "2026-06-20T12:00:00Z") == {}

    def test_match_finished_well_before_cutoff_is_included(self):
        results = self._group_results("2026-06-20T09:00:00Z", "2026-06-20T12:00:00Z")
        assert results == {("A", "B"): (2, 1)}

    def test_zero_delta_preserves_kickoff_semantics(self):
        with patch(
            "src.analysis.replay_common.load_finished_wc_fixtures",
            return_value=(_fixture("2026-06-20T11:30:00Z"),),
        ):
            wc = parse_wc_results_before_kickoff(pd.Timestamp("2026-06-20T12:00:00Z"))
        assert wc["group_results"] == {("A", "B"): (2, 1)}


class TestArtifactCache:
    """Logged artifacts are immutable, so each one is downloaded at most once."""

    def _client(self, tmp_path):
        source = tmp_path / "downloaded" / "predictions.csv"
        source.parent.mkdir(parents=True, exist_ok=True)
        _predictions().to_csv(source, index=False)
        client = MagicMock()
        client.download_artifacts.return_value = str(source)
        return client

    def test_second_read_does_not_download_again(self, tmp_path, monkeypatch):
        monkeypatch.setattr(strand2, "ARTIFACT_CACHE_DIR", tmp_path / "cache")
        client = self._client(tmp_path)
        with (
            patch("src.analysis.strand2_brackets.mlflow.tracking.MlflowClient", return_value=client),
            patch("src.analysis.strand2_brackets.setup_mlflow"),
        ):
            first = strand2._load_artifact("run1", "predictions.csv")
            second = strand2._load_artifact("run1", "predictions.csv")

        assert client.download_artifacts.call_count == 1
        pd.testing.assert_frame_equal(first, second)

    def test_distinct_runs_are_cached_separately(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        monkeypatch.setattr(strand2, "ARTIFACT_CACHE_DIR", cache)
        client = self._client(tmp_path)
        with (
            patch("src.analysis.strand2_brackets.mlflow.tracking.MlflowClient", return_value=client),
            patch("src.analysis.strand2_brackets.setup_mlflow"),
        ):
            strand2._load_artifact("run1", "predictions.csv")
            strand2._load_artifact("run2", "predictions.csv")

        assert client.download_artifacts.call_count == 2
        assert (cache / "run1" / "predictions.csv").exists()
        assert (cache / "run2" / "predictions.csv").exists()

    def test_failed_download_is_not_cached(self, tmp_path, monkeypatch):
        monkeypatch.setattr(strand2, "ARTIFACT_CACHE_DIR", tmp_path / "cache")
        client = MagicMock()
        client.download_artifacts.side_effect = RuntimeError("missing")
        with (
            patch("src.analysis.strand2_brackets.mlflow.tracking.MlflowClient", return_value=client),
            patch("src.analysis.strand2_brackets.setup_mlflow"),
        ):
            assert strand2._load_artifact("run1", "predictions.csv") is None
            assert strand2._load_artifact("run1", "predictions.csv") is None
        assert client.download_artifacts.call_count == 2

    def test_use_cache_false_always_downloads(self, tmp_path, monkeypatch):
        monkeypatch.setattr(strand2, "ARTIFACT_CACHE_DIR", tmp_path / "cache")
        client = self._client(tmp_path)
        with (
            patch("src.analysis.strand2_brackets.mlflow.tracking.MlflowClient", return_value=client),
            patch("src.analysis.strand2_brackets.setup_mlflow"),
        ):
            strand2._load_artifact("run1", "predictions.csv", use_cache=False)
            strand2._load_artifact("run1", "predictions.csv", use_cache=False)
        assert client.download_artifacts.call_count == 2
        assert not (tmp_path / "cache").exists()


class TestFixtureCache:
    def test_parsed_once_across_many_cutoffs(self):
        from src.analysis import replay_common

        replay_common.load_finished_wc_fixtures.cache_clear()
        with patch.object(
            replay_common,
            "_load_api_id_to_canonical",
            return_value={},
        ) as loader, patch.object(replay_common.Path, "glob", return_value=[]):
            for day in range(5):
                replay_common.load_finished_wc_fixtures()
        assert loader.call_count == 1
        replay_common.load_finished_wc_fixtures.cache_clear()
