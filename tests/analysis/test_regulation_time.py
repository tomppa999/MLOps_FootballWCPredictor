"""Regulation-time rescoring must change labels, never the saved forecasts."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from src.analysis import rq1_paired, rq3_calibration_drift
from src.analysis.regulation_time import load_regulation_scores, main, rescore_matches
from src.analysis.regulation_time_comparison import compare_tables, write_comparison
from src.models.evaluation import compute_outcome_probs, compute_rps


def _fixture(mid=1, status="AET", fulltime=(1, 1), goals=(2, 1)):
    return {
        "fixture": {"id": mid, "status": {"short": status}},
        "teams": {"home": {"name": "Home"}, "away": {"name": "Away"}},
        "goals": dict(zip(("home", "away"), goals)),
        "score": {
            "fulltime": dict(zip(("home", "away"), fulltime)),
            "penalty": {"home": 5, "away": 4},
        },
    }


def _write(root, entries, window="window"):
    path = root / window / "fixtures.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"response": entries}))


@pytest.mark.parametrize(
    "status,fulltime,goals",
    [
        ("FT", (2, 1), (2, 1)),
        ("AET", (1, 1), (2, 1)),
        ("PEN", (1, 1), (2, 2)),
    ],
)
def test_reads_regulation_score_not_extra_time_or_penalties(tmp_path, status, fulltime, goals):
    _write(tmp_path, [_fixture(status=status, fulltime=fulltime, goals=goals)])
    result = load_regulation_scores(tmp_path, {1}).iloc[0]
    assert (result.regulation_h, result.regulation_a) == fulltime
    assert (result.final_h, result.final_a) == goals
    assert result.status == status


def test_overlapping_snapshots_must_agree(tmp_path):
    _write(tmp_path, [_fixture()], "a")
    _write(tmp_path, [_fixture()], "b")
    assert len(load_regulation_scores(tmp_path, {1})) == 1
    _write(tmp_path, [_fixture(fulltime=(0, 0))], "b")
    with pytest.raises(ValueError, match="Conflicting finished snapshots"):
        load_regulation_scores(tmp_path, {1})


def test_canonical_names_follow_existing_team_id_mapping(tmp_path):
    entry = _fixture()
    entry["teams"]["home"] = {"id": 1, "name": "USA"}
    entry["teams"]["away"] = {"id": 2, "name": "Türkiye"}
    _write(tmp_path, [entry])
    result = load_regulation_scores(tmp_path, {1}, team_names={1: "United States", 2: "Turkey"})
    assert result.home.item() == "United States" and result.away.item() == "Turkey"


def test_unfinished_snapshot_does_not_replace_finished_result(tmp_path):
    _write(tmp_path, [_fixture(status="NS", fulltime=(None, None), goals=(None, None))], "a")
    _write(tmp_path, [_fixture()], "b")
    assert len(load_regulation_scores(tmp_path, {1})) == 1
    with pytest.raises(ValueError, match="Missing finished regulation scores.*2"):
        load_regulation_scores(tmp_path, {1, 2})


@pytest.mark.parametrize("fulltime", [(None, 1), (-1, 1), (1.5, 1)])
def test_invalid_fulltime_score_is_rejected(tmp_path, fulltime):
    _write(tmp_path, [_fixture(fulltime=fulltime)])
    with pytest.raises(ValueError, match="Invalid regulation score"):
        load_regulation_scores(tmp_path, {1})


def _matches():
    return pd.DataFrame(
        [
            {
                "match_id": mid,
                "home": "Home",
                "away": "Away",
                "round_label": label,
                "actual_h": h,
                "actual_a": a,
                "actual_outcome": 0 if h > a else 1,
                "model_name": "xgboost",
                "cadence_mode": arm,
                "lambda_h": 1.4,
                "lambda_a": 0.8,
                "p_home": 0.5,
                "p_draw": 0.3,
                "p_away": 0.2,
                "rps": 9.0,
                "nll": 9.0,
                "rmse_h": 9.0,
                "rmse_a": 9.0,
                "inference_run_id": f"{mid}-{arm}",
                "provenance": "live",
            }
            for mid, label, h, a in [(1, "1", 2, 1), (2, "R32", 2, 2)]
            for arm in ("frozen", "per_round")
        ]
    )


def test_rescores_goals_and_outcomes_preserving_all_forecast_and_provenance_columns(tmp_path):
    _write(tmp_path, [_fixture(1), _fixture(2, "PEN", (1, 1), (2, 2))])
    original = _matches()
    before = original.copy(deep=True)
    scores = load_regulation_scores(tmp_path, {1, 2})
    rescored, audit = rescore_matches(original, scores)
    pd.testing.assert_frame_equal(original, before)
    changed = {"actual_h", "actual_a", "actual_outcome", "rps", "nll", "rmse_h", "rmse_a"}
    pd.testing.assert_frame_equal(
        rescored[original.columns.difference(list(changed))],
        original[original.columns.difference(list(changed))],
    )
    assert (rescored.actual_h == 1).all() and (rescored.actual_a == 1).all()
    assert (rescored.actual_outcome == 1).all()
    assert np.allclose(rescored.rps, compute_rps(np.array([[0.5, 0.3, 0.2]] * 4), np.ones(4, int)))
    assert np.allclose(rescored.nll, -poisson.logpmf(1, 1.4) - poisson.logpmf(1, 0.8))
    assert np.allclose(rescored.rmse_h, 0.4) and np.allclose(rescored.rmse_a, 0.2)
    assert audit.outcome_changed.tolist() == [True, False]
    assert audit.goals_changed.all()
    # The existing MD1 tie is preserved, including recomputed metrics.
    assert rescored.loc[rescored.round_label == "1", ["rps", "nll"]].nunique().max() == 1


def test_rejects_disagreement_with_original_final_score_or_team_orientation(tmp_path):
    _write(tmp_path, [_fixture(1), _fixture(2, "PEN", (1, 1), (2, 2))])
    scores = load_regulation_scores(tmp_path, {1, 2})
    matches = _matches()
    matches.loc[matches.match_id == 1, "actual_h"] = 3
    with pytest.raises(ValueError, match="Original scores disagree"):
        rescore_matches(matches, scores)
    matches = _matches()
    matches.loc[matches.match_id == 1, "home"] = "Away"
    with pytest.raises(ValueError, match="Team orientation disagrees"):
        rescore_matches(matches, scores)


def test_comparison_keeps_hypotheses_and_flags_holm_threshold_changes():
    old = pd.DataFrame(
        {"model": ["a", "b"], "n": [32, 32], "p_perm": [0.01, 0.2], "p_holm": [0.03, np.nan]}
    )
    new = pd.DataFrame(
        {"model": ["b", "a"], "n": [32, 32], "p_perm": [0.1, 0.04], "p_holm": [np.nan, 0.12]}
    )
    compared = compare_tables(old, new, ["model"]).set_index("model")
    assert compared.loc["a", "p_holm_lt_05_original"]
    assert not compared.loc["a", "p_holm_lt_05_90min"]
    assert pd.isna(compared.loc["b", "p_holm_lt_05_90min"])
    with pytest.raises(ValueError, match="different hypothesis rows"):
        compare_tables(old, new.iloc[:1], ["model"])
    new.loc[0, "n"] = 31
    with pytest.raises(ValueError, match="different n"):
        compare_tables(old, new, ["model"])


def test_complete_workflow_writes_separate_outputs_with_the_same_test_families(tmp_path):
    fixtures, baseline, tables = (tmp_path / name for name in ("fixtures", "baseline", "tables"))
    baseline.mkdir()
    rows, entries = [], []
    for i, label in enumerate(("1", "2", "3", "R32", "R16", "QF", "SF", "Final")):
        for j in range(2):
            mid = 10 * i + j
            entries.append(
                _fixture(mid, "AET" if i >= 3 else "FT", (1, 1) if i >= 3 else (2, 1), (2, 1))
            )
            for model in rq1_paired.CONTRAST_MODELS:
                for arm in ("frozen", "per_round"):
                    rate = 1.4 + (0.1 if arm == "per_round" and label != "1" else 0)
                    probs = compute_outcome_probs([rate], [0.8])[0]
                    rows.append(
                        _matches().iloc[0].to_dict()
                        | {
                            "match_id": mid,
                            "kickoff_utc": f"2026-06-{i + 11:02d}T19:00:00Z",
                            "round_label": label,
                            "model_name": model,
                            "cadence_mode": arm,
                            "lambda_h": rate,
                            "p_home": probs[0],
                            "p_draw": probs[1],
                            "p_away": probs[2],
                            "rps": compute_rps(probs[None, :], np.array([0]))[0],
                            "nll": -poisson.logpmf(2, rate) - poisson.logpmf(1, 0.8),
                            "rmse_h": abs(2 - rate),
                            "rmse_a": 0.2,
                        }
                    )
    original = pd.DataFrame(rows)
    original_path = tmp_path / "original.csv"
    original.to_csv(original_path, index=False)
    original_bytes = original_path.read_bytes()
    _write(fixtures, entries)
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("api_football_team_id,canonical_team_name\n1,Home\n2,Away\n")
    old_rq1 = rq1_paired.holm_core_families(
        rq1_paired.phase_table(rq1_paired.paired_differences(original), n_boot=3),
        rq1_paired.HOLM_FAMILIES,
    )
    paired = rq3_calibration_drift.load_paired(original)
    old_rq3 = rq1_paired.holm_core_families(
        rq3_calibration_drift.phase_table(paired, n_boot=3, noise_floor_draws=2),
        rq3_calibration_drift.HOLM_FAMILIES,
    )
    old_trend = rq3_calibration_drift.trend_table(paired, n_boot=3).assign(p_holm=np.nan)
    for name, frame in (
        ("rq1_paired_phase", old_rq1),
        ("rq3_calibration_phase", old_rq3),
        ("rq3_trend", old_trend),
    ):
        frame.to_csv(baseline / f"{name}.csv", index=False)
    baseline_bytes = {p: p.read_bytes() for p in baseline.iterdir()}
    report = tmp_path / "report.md"
    main(
        [
            "--rq1",
            str(original_path),
            "--fixtures",
            str(fixtures),
            "--mapping",
            str(mapping),
            "--tables-out",
            str(tables),
            "--data-out",
            str(tmp_path / "data"),
            "--n-boot",
            "3",
        ]
    )
    assert original_path.read_bytes() == original_bytes
    assert all(path.read_bytes() == content for path, content in baseline_bytes.items())
    assert len(pd.read_csv(tmp_path / "data" / "outcome_audit.csv")) == 16
    rq1 = pd.read_csv(tables / "rq1_paired_phase.csv")
    rq3 = pd.read_csv(tables / "rq3_calibration_phase.csv")
    assert rq1.loc[rq1.metric == "rps", "p_holm"].notna().sum() == 9
    assert rq1.loc[rq1.metric == "nll", "p_holm"].notna().sum() == 9
    assert rq3.p_holm.notna().sum() == 18
    bins = pd.read_csv(tmp_path / "data" / "rq3_calibration.csv")
    rq3_calibration_drift.check_against_bins(
        rq3_calibration_drift.load_paired(pd.read_csv(tmp_path / "data" / "rq1_matches.csv")), bins
    )
    assert not report.exists()
    assert not list(tables.glob("*comparison*"))
    comparisons = tmp_path / "historical_comparison"
    write_comparison(
        baseline,
        tables,
        pd.read_csv(tmp_path / "data" / "outcome_audit.csv"),
        report,
        comparison_tables=comparisons,
        n_boot=3,
        seed=0,
    )
    assert len(pd.read_csv(comparisons / "rq3_trend_comparison.csv")) == 12
    assert "training targets" not in report.read_text()
