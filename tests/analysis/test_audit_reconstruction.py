"""Unit and integration tests for reconstruction audit checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.audit_reconstruction import (
    MONITORING_COLUMNS,
    _check_one_advancement,
    _check_one_ko_pairings,
    _entropy_row_ordering,
    _expected_outcome,
    _prob_and_outcome_ok,
    _recompute_metrics_ok,
    audit_strand1,
    audit_strand3,
    run_audit,
)
from src.analysis.replay_common import OUTPUT_ROOT, score_prediction_row


def _monitoring_row(**overrides) -> dict:
    base = {
        "match_id": 1,
        "kickoff_utc": "2026-06-11 19:00:00+00:00",
        "home": "Mexico",
        "away": "South Africa",
        "actual_h": 2,
        "actual_a": 0,
        "actual_outcome": 0,
        "model_name": "poisson_glm",
        "lambda_h": 1.5,
        "lambda_a": 0.8,
        "p_home": 0.5,
        "p_draw": 0.25,
        "p_away": 0.25,
        "rps": 0.1,
        "nll": 2.0,
        "rmse_h": 0.5,
        "rmse_a": 0.8,
        "inference_run_id": "test",
        "cadence_mode": "frozen",
    }
    base.update(overrides)
    # Recompute consistent metrics from lambdas when not overridden for rps/nll/rmse
    if "rps" not in overrides:
        match = pd.Series(base)
        scored = score_prediction_row(
            match,
            base["model_name"],
            {
                "lambda_h": base["lambda_h"],
                "lambda_a": base["lambda_a"],
                "p_home": base["p_home"],
                "p_draw": base["p_draw"],
                "p_away": base["p_away"],
            },
            inference_run_id=base["inference_run_id"],
            cadence_mode=base["cadence_mode"],
        )
        for k in ("p_home", "p_draw", "p_away", "rps", "nll", "rmse_h", "rmse_a"):
            base[k] = scored[k]
    return base


class TestHelpers:
    def test_expected_outcome_encoding(self):
        assert _expected_outcome(2, 0) == 0
        assert _expected_outcome(1, 1) == 1
        assert _expected_outcome(0, 3) == 2

    def test_prob_sum_pass_and_fail(self):
        good = pd.DataFrame([_monitoring_row()])
        assert all(r.status == "pass" for r in _prob_and_outcome_ok(good, name="t"))

        bad = pd.DataFrame([_monitoring_row(p_home=0.9, p_draw=0.9, p_away=0.9, rps=0.1)])
        statuses = {r.name: r.status for r in _prob_and_outcome_ok(bad, name="t")}
        assert statuses["t.prob_sum"] == "fail"

    def test_recompute_metrics_detects_tamper(self):
        row = _monitoring_row()
        df = pd.DataFrame([row])
        assert _recompute_metrics_ok(df, name="ok").status == "pass"

        tampered = df.copy()
        tampered.loc[0, "rps"] = 0.999
        assert _recompute_metrics_ok(tampered, name="bad").status == "fail"

    def test_entropy_ordering_warns_on_increase(self):
        good = pd.DataFrame(
            [
                {
                    "entropy_r32": 3.5,
                    "entropy_r16": 3.2,
                    "entropy_qf": 2.8,
                    "entropy_sf": 2.4,
                    "entropy_final": 2.0,
                    "entropy_winner": 1.5,
                },
            ],
        )
        assert _entropy_row_ordering(good, name="e").status == "pass"

        bad = good.copy()
        bad.loc[0, "entropy_winner"] = 3.9
        assert _entropy_row_ordering(bad, name="e").status == "warn"


class TestAdvancementAndKo:
    def _advancement(self, n_teams: int = 48) -> pd.DataFrame:
        teams = [f"T{i}" for i in range(n_teams)]
        data = {"team": teams, "p_group": np.ones(n_teams)}
        # Put all mass on first `slots` teams equally for each round.
        for col, slots in [
            ("p_r32", 32),
            ("p_r16", 16),
            ("p_qf", 8),
            ("p_sf", 4),
            ("p_final", 2),
            ("p_winner", 1),
        ]:
            p = np.zeros(n_teams)
            p[:slots] = 1.0
            data[col] = p
        return pd.DataFrame(data)

    def test_advancement_good(self, tmp_path: Path):
        path = tmp_path / "x_advancement.csv"
        self._advancement().to_csv(path, index=False)
        assert _check_one_advancement(path) == []

    def test_advancement_bad_team_count(self, tmp_path: Path):
        path = tmp_path / "x_advancement.csv"
        self._advancement(n_teams=32).to_csv(path, index=False)
        problems = _check_one_advancement(path)
        assert any("teams=" in p for p in problems)

    def test_advancement_zero_revived(self, tmp_path: Path):
        df = self._advancement()
        # Team with p_r32=0 but p_winner>0
        df.loc[40, "p_winner"] = 0.5
        df.loc[0, "p_winner"] = 0.5
        path = tmp_path / "x_advancement.csv"
        df.to_csv(path, index=False)
        problems = _check_one_advancement(path)
        assert any("revived" in p or "sum=" in p for p in problems)

    def test_ko_pairings_good(self, tmp_path: Path):
        rows = []
        # 16 R32 ties each with frequency 1.0 → sum 16 (degenerate but valid sums)
        for i in range(16):
            rows.append(
                {
                    "stage": "R32",
                    "team_a": f"A{i}",
                    "team_b": f"B{i}",
                    "count": 10000,
                    "frequency": 1.0,
                },
            )
        path = tmp_path / "x_ko_pairings.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        assert _check_one_ko_pairings(path) == []

    def test_ko_pairings_bad_freq(self, tmp_path: Path):
        path = tmp_path / "x_ko_pairings.csv"
        pd.DataFrame(
            [
                {
                    "stage": "R32",
                    "team_a": "A",
                    "team_b": "B",
                    "count": 5000,
                    "frequency": 2.5,
                },
            ],
        ).to_csv(path, index=False)
        problems = _check_one_ko_pairings(path)
        assert problems


class TestStrand3Synthetic:
    def test_wrong_row_count_fails(self, tmp_path: Path):
        d = tmp_path / "strand3_backfill"
        d.mkdir(parents=True)
        rows = [_monitoring_row(match_id=i) for i in range(2)]
        pd.DataFrame(rows)[list(MONITORING_COLUMNS)].to_csv(d / "backfill_rows.csv", index=False)
        results = {r.name: r for r in audit_strand3(tmp_path)}
        assert results["s3.row_count"].status == "fail"


@pytest.mark.skipif(
    not (OUTPUT_ROOT / "strand1_frozen_shadow").is_dir(),
    reason="data/reconstruction/ not present",
)
class TestIntegrationOffline:
    def test_strand1_has_no_hard_fails_on_core_checks(self):
        results = audit_strand1(OUTPUT_ROOT)
        # Allow warns; core schema/row checks should pass on real artifacts.
        core = [r for r in results if r.name.endswith((".rows", ".schema", ".metrics", ".lambda_distinct"))]
        fails = [r for r in core if r.status == "fail"]
        assert not fails, [f"{r.name}: {r.detail}" for r in fails]

    def test_full_offline_audit_runs(self):
        results = run_audit(strands=("1", "3", "2", "4"), root=OUTPUT_ROOT)
        assert results
        # Integration documents current state — must not crash; fails are reported.
        assert all(r.status in {"pass", "warn", "fail"} for r in results)
        assert any(r.strand == "1" for r in results)
        assert any(r.strand == "2" for r in results)
