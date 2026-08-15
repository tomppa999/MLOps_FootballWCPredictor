"""Regression tests for the strand 1 frozen-shadow replay.

Two D.1 defects are pinned here:

1. The replay called ``parse_wc_results_before_kickoff`` without a settle
   delta, so every match had its own final score appended to Gold before its
   features were built — the frozen arm scored ~0.002 RPS better than it should
   have, the same magnitude as the RQ1 effect under study.
2. The replay predicted one fixture at a time while live inference predicts all
   WC pairings in a single batch.  ``days_since_last_match`` is derived from the
   upcoming rows collectively, so a single-row replay silently disagrees with
   the live value it is meant to reproduce.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.analysis import strand1_frozen_shadow as strand1
from src.analysis.replay_common import (
    SETTLE_DELTA,
    FinishedFixture,
    clear_batch_caches,
    parse_wc_results_before_kickoff,
    predict_fixture_from_batch,
)
from src.models.config import MODEL_FEATURE_SETS

_KICKOFF = "2026-06-20T12:00:00Z"


def _settled_match(match_id: int = 42) -> pd.DataFrame:
    return pd.DataFrame([{
        "match_id": match_id,
        "kickoff_utc": pd.Timestamp(_KICKOFF),
        "home": "A",
        "away": "B",
        "actual_h": 1,
        "actual_a": 0,
        "actual_outcome": 0,
    }])


def _fixture(kickoff: str, fixture_id: int = 42) -> FinishedFixture:
    return FinishedFixture(
        fixture_id=fixture_id,
        kickoff=pd.Timestamp(kickoff),
        kickoff_raw=kickoff,
        round_str="Group A - 1",
        round_label="MD1",
        home_team="A",
        away_team="B",
        home_goals=1,
        away_goals=0,
        status="FT",
        teams={},
        score={},
    )


class TestStrand1Cutoff:
    """The replay must ask only for results settled before each kickoff."""

    def _run(self) -> MagicMock:
        cutoff_spy = MagicMock(
            return_value={
                "group_results": {},
                "ko_results": {},
                "finished_fixtures": [],
            },
        )
        commit = MagicMock(commit_sha="abc123")
        with (
            patch.object(strand1, "parse_wc_settled_matches", return_value=_settled_match()),
            patch.object(strand1, "build_gold_commit_index", return_value=[commit]),
            patch.object(strand1, "resolve_gold_commit", return_value=commit),
            patch.object(strand1, "load_gold_at_commit", return_value=pd.DataFrame()),
            patch.object(strand1, "load_pinned_shadow_model", return_value=MagicMock()),
            patch.object(strand1, "parse_wc_results_before_kickoff", cutoff_spy),
            patch.object(
                strand1, "augment_gold_for_inference", return_value=(pd.DataFrame(), None),
            ),
            patch.object(strand1, "snapshot_key", return_value=("abc123", "deadbeef")),
            patch.object(
                strand1,
                "predict_fixture_from_batch",
                return_value={
                    "lambda_h": 1.2,
                    "lambda_a": 1.0,
                    "p_home": 0.4,
                    "p_draw": 0.3,
                    "p_away": 0.3,
                },
            ),
            patch.object(strand1, "ensure_output_dir"),
            patch.object(strand1, "leaderboard_summary", return_value=pd.DataFrame()),
            patch.object(strand1, "log_reconstruction_run", return_value="run"),
            patch.object(pd.DataFrame, "to_csv"),
        ):
            strand1.run_strand1_frozen_shadow()
        return cutoff_spy

    def test_settle_delta_is_passed_through(self):
        spy = self._run()
        assert spy.call_args.kwargs["settle_delta"] == SETTLE_DELTA

    def test_cutoff_is_the_matchs_own_kickoff(self):
        spy = self._run()
        assert spy.call_args.args[0] == pd.Timestamp(_KICKOFF)


class TestSelfLeakage:
    """A match must never appear in the results known at its own kickoff."""

    def _known_fixture_ids(self, fixture_kickoff: str) -> set[int]:
        with patch(
            "src.analysis.replay_common.load_finished_wc_fixtures",
            return_value=(_fixture(fixture_kickoff),),
        ):
            wc = parse_wc_results_before_kickoff(
                pd.Timestamp(_KICKOFF), settle_delta=SETTLE_DELTA,
            )
        return {int(fx["fixture_id"]) for fx in wc["finished_fixtures"]}

    def test_match_does_not_see_its_own_result(self):
        assert self._known_fixture_ids(_KICKOFF) == set()

    def test_simultaneous_kickoff_is_excluded(self):
        assert self._known_fixture_ids("2026-06-20T12:00:00Z") == set()

    def test_earlier_settled_match_is_still_visible(self):
        assert self._known_fixture_ids("2026-06-20T08:00:00Z") == {42}


class TestBatchPredictionShape:
    """Replays must predict the full pairing set, exactly as live inference does."""

    def _predict(self) -> MagicMock:
        clear_batch_caches()
        feature_cols = list(MODEL_FEATURE_SETS["poisson_glm"])
        pairings = pd.DataFrame({
            "home_team": ["A", "A", "C"],
            "away_team": ["B", "C", "B"],
        })
        features = pairings.copy()
        for col in feature_cols:
            features[col] = 1.0
        features_spy = MagicMock(return_value=features)

        model = MagicMock()
        model.predict.return_value = np.array([[1.5, 0.9], [1.1, 1.0], [0.8, 1.4]])

        with (
            patch(
                "src.analysis.replay_common.generate_all_wc_pairings",
                return_value=pairings,
            ),
            patch(
                "src.analysis.replay_common.build_inference_features", features_spy,
            ),
        ):
            pred = predict_fixture_from_batch(
                model,
                "poisson_glm",
                "A",
                "B",
                pd.DataFrame(),
                None,
                key=("sha", "digest"),
            )
        clear_batch_caches()
        return features_spy, model, pred

    def test_features_are_built_over_every_pairing(self):
        features_spy, _, _ = self._predict()
        passed = features_spy.call_args.args[0]
        assert len(passed) == 3, "replay must not slice down to a single fixture"

    def test_model_scores_the_whole_batch(self):
        _, model, _ = self._predict()
        assert len(model.predict.call_args.args[0]) == 3

    def test_requested_fixture_is_selected_from_the_batch(self):
        _, _, pred = self._predict()
        assert pred["lambda_h"] == 1.5
        assert pred["lambda_a"] == 0.9
