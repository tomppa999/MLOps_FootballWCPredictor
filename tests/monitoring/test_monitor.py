"""Tests for the WC 2026 monitoring layer.

Covers:
  - per-match RPS / NLL / RMSE_h / RMSE_a from synthetic predictions + actuals
  - pre-kickoff inference run selection (strictly earlier)
  - alert threshold logic per model
  - cold-start guard (< ALERT_WINDOW matches → no alert)
  - lambda orientation when actual home/away differs from prediction pair
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.monitoring import monitor
from src.monitoring.baselines import ALERT_WINDOW, WC2022_RPS_BASELINES


# ---------------------------------------------------------------------------
# Pre-kickoff selection
# ---------------------------------------------------------------------------


def test_select_pre_kickoff_run_picks_strictly_earlier():
    runs = [
        {"run_id": "a", "inference_timestamp": pd.Timestamp("2026-06-15T10:00", tz="UTC")},
        {"run_id": "b", "inference_timestamp": pd.Timestamp("2026-06-15T11:00", tz="UTC")},
        {"run_id": "c", "inference_timestamp": pd.Timestamp("2026-06-15T13:00", tz="UTC")},
    ]
    kickoff = pd.Timestamp("2026-06-15T12:00", tz="UTC")
    chosen = monitor._select_pre_kickoff_run(runs, kickoff)
    assert chosen["run_id"] == "b"


def test_select_pre_kickoff_run_excludes_runs_at_kickoff():
    """A run timestamped exactly at kickoff is post-kickoff (strict <)."""
    runs = [
        {"run_id": "a", "inference_timestamp": pd.Timestamp("2026-06-15T11:00", tz="UTC")},
        {"run_id": "b", "inference_timestamp": pd.Timestamp("2026-06-15T12:00", tz="UTC")},
    ]
    kickoff = pd.Timestamp("2026-06-15T12:00", tz="UTC")
    chosen = monitor._select_pre_kickoff_run(runs, kickoff)
    assert chosen["run_id"] == "a"


def test_select_pre_kickoff_run_returns_none_when_no_earlier_run():
    runs = [
        {"run_id": "a", "inference_timestamp": pd.Timestamp("2026-06-15T13:00", tz="UTC")},
    ]
    kickoff = pd.Timestamp("2026-06-15T12:00", tz="UTC")
    assert monitor._select_pre_kickoff_run(runs, kickoff) is None


def test_select_pre_kickoff_run_handles_empty_list():
    kickoff = pd.Timestamp("2026-06-15T12:00", tz="UTC")
    assert monitor._select_pre_kickoff_run([], kickoff) is None


# ---------------------------------------------------------------------------
# Per-match scoring
# ---------------------------------------------------------------------------


def _make_match(
    home: str = "France",
    away: str = "Germany",
    actual_h: int = 2,
    actual_a: int = 1,
) -> pd.Series:
    outcome = 0 if actual_h > actual_a else (2 if actual_h < actual_a else 1)
    return pd.Series({
        "match_id": 12345,
        "kickoff_utc": pd.Timestamp("2026-06-15T16:00", tz="UTC"),
        "home": home,
        "away": away,
        "actual_h": actual_h,
        "actual_a": actual_a,
        "actual_outcome": outcome,
    })


def _make_predictions(
    pred_home: str,
    pred_away: str,
    rows: list[tuple[str, float, float]],
) -> pd.DataFrame:
    """Build a long-format prediction frame for one fixture pair."""
    out = []
    for model_name, lam_h, lam_a in rows:
        out.append({
            "fixture_id": f"wc2026_pair_{pred_home}_{pred_away}",
            "home_team": pred_home,
            "away_team": pred_away,
            "date_utc": "2026-06-15",
            "model_name": model_name,
            "lambda_h": lam_h,
            "lambda_a": lam_a,
            "p_home": 0.0,
            "p_draw": 0.0,
            "p_away": 0.0,
        })
    return pd.DataFrame(out)


def test_score_one_match_aligned_orientation():
    """Predictions stored with home=actual home → lambdas pass through."""
    match = _make_match("France", "Germany", actual_h=2, actual_a=1)
    preds = _make_predictions(
        "France", "Germany",
        [("xgboost", 1.6, 1.0), ("ridge", 1.4, 1.2)],
    )
    rows = monitor._score_one_match(match, preds, "inf-1")

    assert len(rows) == 2
    xgb = next(r for r in rows if r["model_name"] == "xgboost")
    assert xgb["lambda_h"] == pytest.approx(1.6)
    assert xgb["lambda_a"] == pytest.approx(1.0)
    assert xgb["actual_outcome"] == 0
    assert xgb["rps"] >= 0.0
    assert np.isfinite(xgb["nll"])
    assert xgb["rmse_h"] == pytest.approx(abs(1.6 - 2))
    assert xgb["rmse_a"] == pytest.approx(abs(1.0 - 1))
    assert xgb["inference_run_id"] == "inf-1"


def test_score_one_match_swaps_lambdas_when_orientation_flipped():
    """Predictions store (TeamA, TeamB) in alphabetical order; if the actual
    fixture has TeamB at home, the predicted lambdas must be swapped.
    """
    match = _make_match("Germany", "France", actual_h=1, actual_a=2)
    # Predicted pair stored alphabetically: France-Germany (France=home).
    preds = _make_predictions(
        "France", "Germany",
        [("xgboost", 1.6, 1.0)],
    )
    rows = monitor._score_one_match(match, preds, "inf-2")
    assert len(rows) == 1
    # Lambdas are swapped: home (Germany) gets the original lambda_a.
    assert rows[0]["lambda_h"] == pytest.approx(1.0)
    assert rows[0]["lambda_a"] == pytest.approx(1.6)


def test_score_one_match_returns_empty_when_no_pair_match():
    match = _make_match("Brazil", "Argentina")
    preds = _make_predictions(
        "France", "Germany",
        [("xgboost", 1.6, 1.0)],
    )
    rows = monitor._score_one_match(match, preds, "inf-3")
    assert rows == []


# ---------------------------------------------------------------------------
# Alert threshold
# ---------------------------------------------------------------------------


def _build_long_table(per_model_rps: dict[str, list[float]]) -> pd.DataFrame:
    """Construct a synthetic monitoring DataFrame from {model: [rps_per_match]}."""
    rows = []
    for model_name, rps_list in per_model_rps.items():
        for i, rps in enumerate(rps_list):
            rows.append({
                "match_id": i,
                "kickoff_utc": pd.Timestamp("2026-06-11", tz="UTC")
                + pd.Timedelta(hours=i),
                "model_name": model_name,
                "rps": rps,
                "nll": 2.5,
                "rmse_h": 1.0,
                "rmse_a": 1.0,
            })
    return pd.DataFrame(rows)


def test_alert_threshold_triggers_when_rolling_mean_breaches():
    bad_rps = [WC2022_RPS_BASELINES["xgboost"] * 1.5] * ALERT_WINDOW
    df = _build_long_table({"xgboost": bad_rps})
    breached = monitor.evaluate_alert_threshold(df)
    assert "xgboost" in breached


def test_alert_threshold_silent_when_rolling_mean_below():
    good_rps = [WC2022_RPS_BASELINES["xgboost"] * 0.9] * ALERT_WINDOW
    df = _build_long_table({"xgboost": good_rps})
    breached = monitor.evaluate_alert_threshold(df)
    assert "xgboost" not in breached


def test_alert_threshold_cold_start_guard():
    """Fewer than ALERT_WINDOW scored matches → no alert even if very bad."""
    huge_rps = [0.9] * (ALERT_WINDOW - 1)
    df = _build_long_table({"xgboost": huge_rps})
    breached = monitor.evaluate_alert_threshold(df)
    assert breached == []


def test_alert_threshold_iterates_all_models():
    """Both champion and shadow alerts surface — alerting is per-model."""
    bad = [0.9] * ALERT_WINDOW
    good = [0.10] * ALERT_WINDOW
    df = _build_long_table({"xgboost": bad, "ridge": good, "lstm": bad})
    breached = monitor.evaluate_alert_threshold(df)
    assert set(breached) == {"xgboost", "lstm"}


# ---------------------------------------------------------------------------
# Score → log → alert wrapper (no real MLflow I/O)
# ---------------------------------------------------------------------------


def test_run_monitoring_step_returns_empty_when_no_settled_matches():
    with patch.object(monitor, "parse_wc_settled_matches", return_value=pd.DataFrame()):
        out = monitor.run_monitoring_step()
    assert out.empty
