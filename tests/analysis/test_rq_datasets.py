"""Unit tests for RQ-ready dataset merge / loaders (no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.rq_datasets.build_rq_datasets import (
    assert_unique_keys,
    build_rq1,
    build_rq2,
    build_rq3,
)
from src.analysis.rq_datasets.loaders import load_rq1, load_rq2, load_rq3, validate_rq1
from src.analysis.rq_datasets.paths import (
    EXPECTED_ROWS_PER_CADENCE,
    MONITORING_COLUMNS,
    PROVENANCE_BACKFILL_COPIED,
    PROVENANCE_BACKFILL_REPRED,
    PROVENANCE_FROZEN_SHADOW,
    PROVENANCE_LIVE,
    PROVENANCE_LIVE_REPLAY,
    PROVENANCE_SNAPSHOT,
    RQ1_KEY,
    RQ2_KEY,
)


def _monitoring_row(
    match_id: int,
    model_name: str,
    cadence_mode: str,
    *,
    inference_run_id: str = "live_run",
    kickoff: str = "2026-06-12T19:00:00Z",
    actual_outcome: int = 0,
    p_home: float = 0.5,
    p_draw: float = 0.25,
    p_away: float = 0.25,
    rps: float = 0.2,
    nll: float = 2.0,
) -> dict:
    return {
        "match_id": match_id,
        "kickoff_utc": kickoff,
        "home": "Home",
        "away": "Away",
        "actual_h": 2 if actual_outcome == 0 else (1 if actual_outcome == 1 else 0),
        "actual_a": 0 if actual_outcome == 0 else (1 if actual_outcome == 1 else 2),
        "actual_outcome": actual_outcome,
        "model_name": model_name,
        "lambda_h": 1.5,
        "lambda_a": 1.0,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "rps": rps,
        "nll": nll,
        "rmse_h": 0.5,
        "rmse_a": 1.0,
        "inference_run_id": inference_run_id,
        "cadence_mode": cadence_mode,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def tiny_live(tmp_path: Path) -> Path:
    """Minimal live tree: 2 matches × 2 models × 2 cadences = 8 rows.

    Strand1 will replace one frozen poisson_glm row. Strand3 appends one
    missing frozen ridge row. After merge: 9 rows (not 832 — tests call
    build helpers with monkeypatched expectations where needed).
    """
    live = tmp_path / "live"
    live.mkdir()
    models = ["poisson_glm", "xgboost"]
    matches = [1001, 1002]
    for cadence in ("frozen", "per_round"):
        rows = [
            _monitoring_row(
                mid,
                model,
                cadence,
                inference_run_id=f"live_{cadence}_{mid}_{model}",
                kickoff=f"2026-06-{12 + i:02d}T19:00:00Z",
            )
            for i, mid in enumerate(matches)
            for model in models
        ]
        _write_csv(live / f"monitoring_{cadence}.csv", rows)

    fixtures = [
        {
            "match_id": 1001,
            "kickoff_utc": "2026-06-12T19:00:00Z",
            "home": "Home",
            "away": "Away",
            "actual_h": 2,
            "actual_a": 0,
            "round_label": "1",
            "round_str": "Group Stage - 1",
        },
        {
            "match_id": 1002,
            "kickoff_utc": "2026-06-13T19:00:00Z",
            "home": "Home",
            "away": "Away",
            "actual_h": 2,
            "actual_a": 0,
            "round_label": "2",
            "round_str": "Group Stage - 2",
        },
        {
            "match_id": 1003,
            "kickoff_utc": "2026-06-14T19:00:00Z",
            "home": "Home",
            "away": "Away",
            "actual_h": 1,
            "actual_a": 1,
            "round_label": "3",
            "round_str": "Group Stage - 3",
        },
    ]
    _write_csv(live / "fixtures.csv", fixtures)

    cycles = [
        {
            "inference_run_id": "run_a",
            "inference_timestamp": "2026-06-10T10:00:00Z",
            "cadence_mode": "frozen",
            "matchday_label": "1",
            "champion_model_name": "xgboost",
            "champion_run_id": "champ1",
            "simulation_seed": 1,
            "n_sims": 100,
        },
        {
            "inference_run_id": "run_b",
            "inference_timestamp": "2026-06-11T10:00:00Z",
            "cadence_mode": "frozen",
            "matchday_label": "2",
            "champion_model_name": "xgboost",
            "champion_run_id": "champ1",
            "simulation_seed": 2,
            "n_sims": 100,
        },
    ]
    _write_csv(live / "inference_cycles.csv", cycles)
    return live


class TestBuildRq1:
    def test_strand1_replaces_and_preserves_source_run_id(
        self, tiny_live: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        strand1 = [
            _monitoring_row(
                1001,
                "poisson_glm",
                "frozen",
                inference_run_id="reconstruction",
                rps=0.11,  # distinct from live 0.2
            ),
        ]
        s1_path = tmp_path / "s1.csv"
        _write_csv(s1_path, strand1)

        strand3 = [
            _monitoring_row(
                1003,
                "ridge",
                "frozen",
                inference_run_id="backfill_ridge_md3",
                actual_outcome=1,
            ),
        ]
        s3_path = tmp_path / "s3.csv"
        _write_csv(s3_path, strand3)

        # Tiny fixture is not 832 rows — disable the cadence count assert via patch.
        import src.analysis.rq_datasets.build_rq_datasets as mod

        monkeypatch.setattr(mod, "EXPECTED_ROWS_PER_CADENCE", 5)  # frozen: 4 kept/replaced + 1 backfill? 
        # frozen live: 4 rows; replace 1 → still 4; append 1 → 5. per_round: 4 unchanged.
        # But we assert both cadences == EXPECTED. So set different... the code asserts
        # same expected for both. Adjust fixture instead: append a per_round backfill too.
        strand3.append(
            _monitoring_row(
                1003,
                "ridge",
                "per_round",
                inference_run_id="backfill_ridge_md3",
                actual_outcome=1,
            ),
        )
        _write_csv(s3_path, strand3)
        monkeypatch.setattr(mod, "EXPECTED_ROWS_PER_CADENCE", 5)
        monkeypatch.setattr(mod, "assert_rps_nll_recomputable", lambda df, **kw: None)

        rq1 = build_rq1(
            live_root=tiny_live,
            strand1_path=s1_path,
            strand3_path=s3_path,
        )

        replaced = rq1[
            (rq1["match_id"] == 1001)
            & (rq1["model_name"] == "poisson_glm")
            & (rq1["cadence_mode"] == "frozen")
        ]
        assert len(replaced) == 1
        assert replaced.iloc[0]["provenance"] == PROVENANCE_FROZEN_SHADOW
        assert replaced.iloc[0]["inference_run_id"] == "reconstruction"
        assert replaced.iloc[0]["source_run_id"] == "live_frozen_1001_poisson_glm"
        assert replaced.iloc[0]["rps"] == pytest.approx(0.11)

        # Other models untouched.
        xgb = rq1[
            (rq1["match_id"] == 1001)
            & (rq1["model_name"] == "xgboost")
            & (rq1["cadence_mode"] == "frozen")
        ]
        assert xgb.iloc[0]["provenance"] == PROVENANCE_LIVE
        assert xgb.iloc[0]["rps"] == pytest.approx(0.2)

        backfills = rq1[rq1["provenance"] == PROVENANCE_BACKFILL_REPRED]
        assert len(backfills) == 2
        assert backfills["source_run_id"].isna().all()

        assert set(rq1["provenance"]) <= {
            PROVENANCE_LIVE,
            PROVENANCE_FROZEN_SHADOW,
            PROVENANCE_BACKFILL_REPRED,
            PROVENANCE_BACKFILL_COPIED,
        }
        assert rq1["round_label"].notna().all()
        assert_unique_keys(rq1, RQ1_KEY, label="test")

    def test_duplicate_keys_raise(self, tiny_live: Path, tmp_path: Path, monkeypatch):
        import src.analysis.rq_datasets.build_rq_datasets as mod

        monkeypatch.setattr(mod, "assert_rps_nll_recomputable", lambda df, **kw: None)
        # Strand3 key already in live → ValueError before count assert.
        s1_path = tmp_path / "s1_empty.csv"
        _write_csv(s1_path, [])  # empty strand1 — but schema needed
        # Empty CSV has no columns; write header-only via DataFrame.
        pd.DataFrame(columns=list(MONITORING_COLUMNS)).to_csv(s1_path, index=False)

        s3_path = tmp_path / "s3_dup.csv"
        _write_csv(
            s3_path,
            [
                _monitoring_row(
                    1001,
                    "xgboost",
                    "frozen",
                    inference_run_id="backfill_copy_from_per_round",
                ),
            ],
        )
        with pytest.raises(ValueError, match="already present"):
            build_rq1(
                live_root=tiny_live,
                strand1_path=s1_path,
                strand3_path=s3_path,
            )


def _entropy_row(
    run_id: str,
    ts: str,
    model: str,
    *,
    cadence: str = "frozen",
    h: float = 3.5,
    **extra: object,
) -> dict:
    """One entropy trajectory row, stages descending from ``h``.

    ``mean_rate_poisson`` must stay the entropy floor (highest H) for
    ``build_rq2``'s sanity check, so give it a larger ``h`` than any other
    model in the same cadence.
    """
    row: dict = {
        "inference_run_id": run_id,
        "inference_timestamp": ts,
        "cadence_mode": cadence,
        "model_name": model,
        "entropy_r32": h,
        "entropy_r16": h - 0.1,
        "entropy_qf": h - 0.2,
        "entropy_sf": h - 0.3,
        "entropy_final": h - 0.4,
        "entropy_winner": h - 0.5,
    }
    row.update(extra)
    return row


def _write(rows: list[dict], path: Path) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestBuildRq2:
    """Strand 5 outranks Strand 2 on shared keys; Strand 4 is never read."""

    def _strand2(self, tmp_path: Path, *, drop_frozen_mean_rate: bool = False) -> Path:
        rows = [
            _entropy_row("run_f", "2026-06-10T10:00:00.5Z", "xgboost", h=3.5),
            _entropy_row("run_p", "2026-06-10T10:05:00Z", "xgboost",
                         cadence="per_round", h=3.5),
            _entropy_row("run_p", "2026-06-10T10:05:00Z", "mean_rate_poisson",
                         cadence="per_round", h=3.8),
        ]
        if not drop_frozen_mean_rate:
            rows.append(
                _entropy_row("run_f", "2026-06-10T10:00:00.5Z", "mean_rate_poisson", h=3.8)
            )
        return _write(rows, tmp_path / "strand2.csv")

    def _strand5(self, tmp_path: Path, **overrides: object) -> Path:
        common = {"synthetic": False, "snapshot_label": "",
                  "provenance": PROVENANCE_FROZEN_SHADOW}
        common.update(overrides)
        rows = [
            _entropy_row("run_f", "2026-06-10T10:00:00.5Z", "xgboost", h=3.4, **common),
            _entropy_row("run_f", "2026-06-10T10:00:00.5Z", "mean_rate_poisson",
                         h=3.75, **common),
        ]
        return _write(rows, tmp_path / "strand5.csv")

    def test_strand5_supersedes_strand2_on_frozen_keys(
        self, tiny_live: Path, tmp_path: Path,
    ):
        rq2 = build_rq2(
            live_root=tiny_live,
            trajectory_path=self._strand2(tmp_path),
            strand5_path=self._strand5(tmp_path),
        )
        assert len(rq2) == 4
        assert_unique_keys(rq2, RQ2_KEY, label="rq2")

        frozen_xgb = rq2[(rq2["cadence_mode"] == "frozen") & (rq2["model_name"] == "xgboost")]
        assert frozen_xgb["entropy_r32"].iloc[0] == pytest.approx(3.4)  # strand5 value
        assert frozen_xgb["provenance"].iloc[0] == PROVENANCE_FROZEN_SHADOW

        per_round = rq2[rq2["cadence_mode"] == "per_round"]
        assert (per_round["provenance"] == PROVENANCE_LIVE_REPLAY).all()

    def test_cycle_missing_a_model_in_strand2_is_tolerated(
        self, tiny_live: Path, tmp_path: Path,
    ):
        rq2 = build_rq2(
            live_root=tiny_live,
            trajectory_path=self._strand2(tmp_path, drop_frozen_mean_rate=True),
            strand5_path=self._strand5(tmp_path),
        )
        frozen = rq2[rq2["cadence_mode"] == "frozen"]
        assert set(frozen["model_name"]) == {"xgboost", "mean_rate_poisson"}
        assert (frozen["provenance"] == PROVENANCE_FROZEN_SHADOW).all()

    def test_synthetic_rows_get_matchday_and_interleaved_cycle_index(
        self, tiny_live: Path, tmp_path: Path,
    ):
        strand2 = _write(
            [
                _entropy_row("run_a", "2026-06-10T10:00:00Z", "xgboost", h=3.5),
                _entropy_row("run_a", "2026-06-10T10:00:00Z", "mean_rate_poisson", h=3.8),
                _entropy_row("run_b", "2026-06-11T10:00:00Z", "xgboost", h=3.4),
                _entropy_row("run_b", "2026-06-11T10:00:00Z", "mean_rate_poisson", h=3.8),
            ],
            tmp_path / "strand2.csv",
        )
        strand5 = _write(
            [
                _entropy_row(
                    "synth_1", "2026-06-10T16:00:00Z", "xgboost", h=3.4,
                    synthetic=True, snapshot_label="r32_pre_japan_brazil",
                    provenance=PROVENANCE_SNAPSHOT,
                ),
                _entropy_row(
                    "synth_1", "2026-06-10T16:00:00Z", "mean_rate_poisson", h=3.8,
                    synthetic=True, snapshot_label="r32_pre_japan_brazil",
                    provenance=PROVENANCE_SNAPSHOT,
                ),
            ],
            tmp_path / "strand5.csv",
        )

        rq2 = build_rq2(
            live_root=tiny_live, trajectory_path=strand2, strand5_path=strand5,
        )
        assert len(rq2) == 6
        synth = rq2[rq2["synthetic"]]
        assert len(synth) == 2
        assert (synth["matchday_label"] == "R32").all()
        assert (synth["provenance"] == PROVENANCE_SNAPSHOT).all()

        # Synthetic timestamp sits between run_a and run_b.
        run_a_idx = rq2.loc[rq2["inference_run_id"] == "run_a", "cycle_index"].iloc[0]
        run_b_idx = rq2.loc[rq2["inference_run_id"] == "run_b", "cycle_index"].iloc[0]
        assert run_a_idx < synth["cycle_index"].iloc[0] < run_b_idx
        assert_unique_keys(rq2, RQ2_KEY, label="rq2")

    def test_unexpected_strand5_provenance_raises(self, tiny_live: Path, tmp_path: Path):
        with pytest.raises(ValueError, match="unexpected provenance"):
            build_rq2(
                live_root=tiny_live,
                trajectory_path=self._strand2(tmp_path),
                strand5_path=self._strand5(tmp_path, provenance=PROVENANCE_LIVE),
            )

    def test_strand4_is_no_longer_an_input(self):
        import inspect

        from src.analysis.rq_datasets import build_rq_datasets as mod

        assert "snapshots_path" not in inspect.signature(mod.build_rq2).parameters
        assert not hasattr(mod, "STRAND4_SNAPSHOTS")


class TestBuildRq3:
    def test_bin_counts_sum_to_input(self):
        rows = []
        for mid in range(10):
            rows.append(
                _monitoring_row(
                    mid,
                    "xgboost",
                    "frozen",
                    actual_outcome=mid % 3,
                    p_home=0.1 + 0.08 * mid,
                    p_draw=0.3,
                    p_away=0.6 - 0.08 * mid,
                    kickoff=f"2026-06-{11 + mid:02d}T12:00:00Z",
                ),
            )
        rq1 = pd.DataFrame(rows)
        rq1["round_label"] = "1"
        rq1["provenance"] = PROVENANCE_LIVE
        rq1["source_run_id"] = rq1["inference_run_id"]
        rq1["kickoff_utc"] = pd.to_datetime(rq1["kickoff_utc"], utc=True)

        rq3 = build_rq3(rq1, n_bins=5)
        rel = rq3[rq3["record_type"] == "reliability"]
        assert int(rel["n"].sum()) == len(rq1) * 3
        cum = rq3[rq3["record_type"] == "cum_rps"]
        assert len(cum) == len(rq1)
        assert cum["cum_rps"].notna().all()
        assert cum["match_index"].iloc[-1] == len(rq1) - 1


class TestLoaders:
    def test_load_rq1_validates(self, tmp_path: Path, monkeypatch):
        # Build a fake 832-row file is heavy; write minimal and patch expectation.
        import src.analysis.rq_datasets.loaders as loaders_mod
        import src.analysis.rq_datasets.build_rq_datasets as build_mod

        rows = [
            {
                **_monitoring_row(1, "xgboost", "frozen"),
                "provenance": PROVENANCE_LIVE,
                "source_run_id": "live_run",
                "round_label": "1",
            },
            {
                **_monitoring_row(1, "xgboost", "per_round"),
                "provenance": PROVENANCE_LIVE,
                "source_run_id": "live_run",
                "round_label": "1",
            },
        ]
        path = tmp_path / "rq1_matches.csv"
        pd.DataFrame(rows).to_csv(path, index=False)

        monkeypatch.setattr(loaders_mod, "EXPECTED_ROWS_PER_CADENCE", 1)
        monkeypatch.setattr(build_mod, "EXPECTED_ROWS_PER_CADENCE", 1)
        df = load_rq1(path, validate=True)
        assert len(df) == 2
        validate_rq1(df)
