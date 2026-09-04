"""Unit tests for Strand 5: pinned-model entropy reconstruction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.analysis.replay_common import SETTLE_DELTA, InferenceCycle
from src.analysis.rq_datasets.paths import (
    PROVENANCE_FROZEN_SHADOW,
    PROVENANCE_SNAPSHOT,
)
from src.analysis.strand4_entropy import EntropySnapshotSpec
from src.analysis.strand5_frozen_entropy import run_strand5_frozen_entropy
from src.models.config import EXPERIMENT_MODELS


def _advancement() -> pd.DataFrame:
    teams = [f"T{i}" for i in range(48)]
    data = {"team": teams, "p_group": np.ones(48)}
    for col, slots in [
        ("p_r32", 32), ("p_r16", 16), ("p_qf", 8),
        ("p_sf", 4), ("p_final", 2), ("p_winner", 1),
    ]:
        data[col] = np.full(48, slots / 48.0)
    return pd.DataFrame(data)


def _per_model() -> dict[str, dict]:
    adv = _advancement()
    ko = pd.DataFrame({"slot": ["R32_1"], "home": ["T0"], "away": ["T1"]})
    return {
        name: {"advancement": adv.copy(), "ko_pairings": ko.copy()}
        for name in EXPERIMENT_MODELS
    }


def _all_models() -> pd.DataFrame:
    rows = []
    for name in EXPERIMENT_MODELS:
        rows.append({
            "model_name": name,
            "home_team": "T0",
            "away_team": "T1",
            "lambda_h": 1.2,
            "lambda_a": 0.9,
        })
    return pd.DataFrame(rows)


def _manifest() -> pd.DataFrame:
    rows = []
    for model_name, (registry, version) in (
        ("xgboost", ("wc_production", 15)),
        ("poisson_glm", ("wc_shadow", 88)),
        ("mean_rate_poisson", ("wc_shadow", 89)),
        ("bayesian_poisson", ("wc_shadow", 90)),
    ):
        rows.append({
            "registry_name": registry,
            "version": version,
            "model_name": model_name,
            "cadence_role": "frozen",
            "regime": "pre_tournament",
            "source_run_id": f"run-{model_name}",
        })
    return pd.DataFrame(rows)


def _models() -> dict[str, tuple[str, int, object]]:
    dummy = object()
    return {
        "xgboost": ("wc_production", 15, dummy),
        "poisson_glm": ("wc_shadow", 88, dummy),
        "mean_rate_poisson": ("wc_shadow", 89, dummy),
        "bayesian_poisson": ("wc_shadow", 90, dummy),
    }


def _cycle() -> InferenceCycle:
    return InferenceCycle(
        run_id="run_live",
        inference_timestamp=pd.Timestamp("2026-06-12 06:07:07+00:00"),
        cadence_mode="frozen",
        matchday_label="2",
        champion_model_name="xgboost",
        simulation_seed=12345,
        n_sims=10_000,
    )


def _spec() -> EntropySnapshotSpec:
    return EntropySnapshotSpec(
        label="r32_pre_japan_brazil",
        round_seed="R32",
        max_kickoff=pd.Timestamp("2026-06-29T19:00:00Z"),
        synthetic_timestamp=pd.Timestamp("2026-06-29T18:00:00Z"),
    )


class TestStrand5Driver:
    def _run(self, tmp_path, *, scopes=("live", "snapshots")):
        predict = MagicMock(return_value=(_per_model(), _all_models()))
        commit = MagicMock(commit_sha="abc", gold_hash="hash1")
        with (
            patch(
                "src.analysis.strand5_frozen_entropy.load_model_manifest",
                return_value=_manifest(),
            ),
            patch(
                "src.analysis.strand5_frozen_entropy.build_gold_commit_index",
                return_value=[commit],
            ),
            patch(
                "src.analysis.strand5_frozen_entropy.resolve_gold_commit",
                return_value=commit,
            ),
            patch(
                "src.analysis.strand5_frozen_entropy.load_regime_models",
                return_value=_models(),
            ),
            patch(
                "src.analysis.strand5_frozen_entropy.inference_cycles_for",
                return_value=[_cycle()],
            ),
            patch(
                "src.analysis.strand5_frozen_entropy.parse_wc_results_before_kickoff",
                return_value={"group_results": {}, "ko_results": {}, "finished_fixtures": []},
            ) as cutoff,
            patch(
                "src.analysis.strand5_frozen_entropy._predict_and_simulate",
                predict,
            ),
            patch(
                "src.analysis.strand5_frozen_entropy.parse_wc_settled_matches",
                return_value=pd.DataFrame(),
            ),
            patch(
                "src.analysis.strand5_frozen_entropy.build_entropy_snapshot_specs",
                return_value=[_spec()],
            ),
            patch(
                "src.analysis.strand5_frozen_entropy.ensure_output_dir",
                return_value=tmp_path,
            ),
            patch(
                "src.analysis.strand5_frozen_entropy.log_reconstruction_run",
                return_value="",
            ),
        ):
            run_strand5_frozen_entropy(subdir="unused", scopes=scopes)
        return predict, cutoff, tmp_path

    def test_live_rows_carry_frozen_shadow_provenance_and_schema(self, tmp_path):
        _, _, out = self._run(tmp_path, scopes=("live",))
        traj = pd.read_csv(out / "entropy_trajectory.csv")
        assert set(traj["model_name"]) == set(EXPERIMENT_MODELS)
        assert (traj["provenance"] == PROVENANCE_FROZEN_SHADOW).all()
        assert (traj["synthetic"] == False).all()  # noqa: E712
        assert traj["inference_run_id"].eq("run_live").all()
        for col in ("entropy_r32", "entropy_winner", "model_version", "registry_name"):
            assert col in traj.columns
        preds = pd.read_parquet(out / "predictions.parquet")
        assert set(preds.columns) == {
            "inference_run_id", "cadence_mode", "model_name",
            "home_team", "away_team", "lambda_h", "lambda_a",
        }

    def test_live_simulation_uses_the_logged_seed(self, tmp_path):
        predict, cutoff, _ = self._run(tmp_path, scopes=("live",))
        predict.assert_called_once()
        assert predict.call_args.kwargs["seed"] == 12345
        assert cutoff.call_args.kwargs["settle_delta"] == SETTLE_DELTA

    def test_gap_snapshots_cover_both_cadences_with_snapshot_provenance(self, tmp_path):
        _, cutoff, out = self._run(tmp_path, scopes=("snapshots",))
        traj = pd.read_csv(out / "entropy_trajectory.csv")
        assert len(traj) == 2 * len(EXPERIMENT_MODELS)
        assert set(traj["cadence_mode"]) == {"frozen", "per_round"}
        assert (traj["provenance"] == PROVENANCE_SNAPSHOT).all()
        assert (traj["synthetic"] == True).all()  # noqa: E712
        assert (traj["snapshot_label"] == "r32_pre_japan_brazil").all()
        # Gap snapshots keep Strand 4 semantics: lock on raw kickoff, settle 0.
        assert "settle_delta" not in cutoff.call_args.kwargs
