"""Tests for the WC 2026 monitoring layer.

Covers:
  - per-match RPS / NLL / RMSE_h / RMSE_a from synthetic predictions + actuals
  - pre-kickoff inference run selection (strictly earlier)
  - alert threshold logic per model
  - cold-start guard (< ALERT_WINDOW matches → no alert)
  - lambda orientation when actual home/away differs from prediction pair
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.monitoring import monitor
from src.monitoring.baselines import ALERT_WINDOW, NAIVE_BASELINE_RPS


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


def _build_long_table(
    per_model_rps: dict[str, list[float]],
    *,
    cadence_mode: str = "frozen",
) -> pd.DataFrame:
    """Construct a synthetic monitoring DataFrame from {model: [rps_per_match]}."""
    rows = []
    for model_name, rps_list in per_model_rps.items():
        for i, rps in enumerate(rps_list):
            rows.append({
                "match_id": i,
                "kickoff_utc": pd.Timestamp("2026-06-11", tz="UTC")
                + pd.Timedelta(hours=i),
                "model_name": model_name,
                "cadence_mode": cadence_mode,
                "rps": rps,
                "nll": 2.5,
                "rmse_h": 1.0,
                "rmse_a": 1.0,
            })
    return pd.DataFrame(rows)


def test_alert_threshold_triggers_when_rolling_mean_breaches():
    bad_rps = [NAIVE_BASELINE_RPS + 0.05] * ALERT_WINDOW
    df = _build_long_table({"xgboost": bad_rps})
    breached = monitor.evaluate_alert_threshold(df)
    assert ("frozen", "xgboost") in breached


def test_alert_threshold_silent_when_rolling_mean_below():
    good_rps = [NAIVE_BASELINE_RPS - 0.05] * ALERT_WINDOW
    df = _build_long_table({"xgboost": good_rps})
    breached = monitor.evaluate_alert_threshold(df)
    assert ("frozen", "xgboost") not in breached


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
    assert set(breached) == {("frozen", "xgboost"), ("frozen", "lstm")}


def test_alert_threshold_groups_by_cadence_mode():
    """Same model in two modes is evaluated independently."""
    bad = [0.9] * ALERT_WINDOW
    good = [0.10] * ALERT_WINDOW
    frozen_df = _build_long_table({"xgboost": bad}, cadence_mode="frozen")
    per_round_df = _build_long_table({"xgboost": good}, cadence_mode="per_round")
    df = pd.concat([frozen_df, per_round_df], ignore_index=True)
    breached = monitor.evaluate_alert_threshold(df)
    assert breached == [("frozen", "xgboost")]


# ---------------------------------------------------------------------------
# Per-mode inference run lookup
# ---------------------------------------------------------------------------


def test_list_inference_runs_filters_by_cadence_mode():
    frozen_run = MagicMock()
    frozen_run.info.run_id = "run-frozen"
    frozen_run.data.params = {
        "inference_timestamp": "2026-06-15T10:00:00+00:00",
        "cadence_mode": "frozen",
    }
    per_round_run = MagicMock()
    per_round_run.info.run_id = "run-pr"
    per_round_run.data.params = {
        "inference_timestamp": "2026-06-15T11:00:00+00:00",
        "cadence_mode": "per_round",
    }
    legacy_run = MagicMock()
    legacy_run.info.run_id = "run-legacy"
    legacy_run.data.params = {
        "inference_timestamp": "2026-06-15T09:00:00+00:00",
    }

    mock_exp = MagicMock()
    mock_exp.experiment_id = "exp-1"
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = mock_exp
    mock_client.search_runs.return_value = [frozen_run, per_round_run, legacy_run]

    with (
        patch.object(monitor, "setup_mlflow"),
        patch(
            "src.monitoring.monitor.mlflow.tracking.MlflowClient",
            return_value=mock_client,
        ),
    ):
        frozen_only = monitor._list_inference_runs("frozen")
        pr_only = monitor._list_inference_runs("per_round")
        all_runs = monitor._list_inference_runs(None)

    assert [r["run_id"] for r in frozen_only] == ["run-legacy", "run-frozen"]
    assert [r["run_id"] for r in pr_only] == ["run-pr"]
    assert len(all_runs) == 3


# ---------------------------------------------------------------------------
# Score → log → alert wrapper (no real MLflow I/O)
# ---------------------------------------------------------------------------


def test_run_monitoring_step_returns_empty_when_no_settled_matches():
    with patch.object(monitor, "parse_wc_settled_matches", return_value=pd.DataFrame()):
        out = monitor.run_monitoring_step()
    assert out.empty


def test_run_monitoring_step_invokes_both_cadence_modes():
    with patch.object(
        monitor,
        "score_completed_wc_matches",
        return_value=pd.DataFrame(),
    ) as mock_score:
        monitor.run_monitoring_step()

    assert mock_score.call_count == 2
    modes = {call.kwargs["cadence_mode"] for call in mock_score.call_args_list}
    assert modes == {"frozen", "per_round"}


def test_score_completed_wc_matches_tags_cadence_mode():
    settled = pd.DataFrame([{
        "match_id": 1,
        "kickoff_utc": pd.Timestamp("2026-06-15T16:00", tz="UTC"),
        "home": "France",
        "away": "Germany",
        "actual_h": 2,
        "actual_a": 1,
        "actual_outcome": 0,
    }])
    preds = _make_predictions(
        "France", "Germany",
        [("xgboost", 1.6, 1.0)],
    )
    inference_runs = [
        {
            "run_id": "inf-1",
            "inference_timestamp": pd.Timestamp("2026-06-15T10:00", tz="UTC"),
        },
    ]

    with (
        patch.object(monitor, "parse_wc_settled_matches", return_value=settled),
        patch.object(monitor, "_list_inference_runs", return_value=inference_runs) as mock_list,
        patch.object(monitor, "_load_predictions_all_models", return_value=preds),
    ):
        out = monitor.score_completed_wc_matches(cadence_mode="per_round")

    assert not out.empty
    assert (out["cadence_mode"] == "per_round").all()
    mock_list.assert_called_once_with("per_round")


@patch("src.monitoring.monitor.mlflow")
@patch("src.monitoring.monitor.start_run")
@patch("src.monitoring.monitor.log_run")
@patch("src.monitoring.monitor.setup_mlflow")
@patch("src.monitoring.monitor.get_or_create_experiment")
def test_log_monitoring_run_includes_cadence_mode(
    mock_get_exp,
    mock_setup,
    mock_log_run,
    mock_start_run,
    mock_mlflow,
):
    df = _build_long_table({"xgboost": [0.15, 0.18]})
    df["cadence_mode"] = "per_round"

    fake_run = MagicMock()
    fake_run.info.run_id = "mon_run_1"
    mock_start_run.return_value.__enter__ = MagicMock(return_value=fake_run)
    mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

    monitor.log_monitoring_run(df, cadence_mode="per_round")

    assert mock_start_run.call_args.kwargs["run_name"] == "monitor_per_round_xgboost"
    tags = mock_start_run.call_args.kwargs.get("tags", {})
    assert tags["cadence_mode"] == "per_round"
    params = mock_log_run.call_args.kwargs.get("params", {})
    assert params["cadence_mode"] == "per_round"


# ---------------------------------------------------------------------------
# Last-logged count helper
# ---------------------------------------------------------------------------


def _mock_mlflow_client(search_runs_result: list) -> MagicMock:
    mock_exp = MagicMock()
    mock_exp.experiment_id = "exp-1"
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = mock_exp
    mock_client.search_runs.return_value = search_runs_result
    return mock_client


def test_last_logged_n_scored_matches_returns_0_when_no_runs():
    mock_client = _mock_mlflow_client([])
    with (
        patch.object(monitor, "setup_mlflow"),
        patch("src.monitoring.monitor.mlflow.tracking.MlflowClient", return_value=mock_client),
    ):
        assert monitor._last_logged_n_scored_matches("frozen") == 0


def test_last_logged_n_scored_matches_returns_count_from_latest_run():
    mock_run = MagicMock()
    mock_run.data.params = {"n_scored_matches": "7"}
    mock_client = _mock_mlflow_client([mock_run])
    with (
        patch.object(monitor, "setup_mlflow"),
        patch("src.monitoring.monitor.mlflow.tracking.MlflowClient", return_value=mock_client),
    ):
        assert monitor._last_logged_n_scored_matches("frozen") == 7


def test_last_logged_n_scored_matches_returns_minus1_on_error():
    with patch.object(monitor, "setup_mlflow", side_effect=Exception("network error")):
        assert monitor._last_logged_n_scored_matches("frozen") == -1


def test_last_logged_n_scored_matches_queries_correct_filter():
    """Confirm the MLflow filter string targets the right cadence mode."""
    mock_client = _mock_mlflow_client([])
    with (
        patch.object(monitor, "setup_mlflow"),
        patch("src.monitoring.monitor.mlflow.tracking.MlflowClient", return_value=mock_client),
    ):
        monitor._last_logged_n_scored_matches("per_round")

    filter_used = mock_client.search_runs.call_args.kwargs.get(
        "filter_string"
    ) or mock_client.search_runs.call_args[1].get("filter_string")
    assert "per_round" in filter_used
    assert "monitoring" in filter_used


# ---------------------------------------------------------------------------
# Monitoring gate in run_monitoring_step
# ---------------------------------------------------------------------------


def _make_monitoring_df(n_matches: int = 2) -> pd.DataFrame:
    rows = []
    for i in range(n_matches):
        rows.append({
            "match_id": i,
            "kickoff_utc": pd.Timestamp("2026-06-15", tz="UTC") + pd.Timedelta(hours=i),
            "home": "France",
            "away": "Germany",
            "actual_h": 2,
            "actual_a": 1,
            "actual_outcome": 0,
            "model_name": "xgboost",
            "cadence_mode": "frozen",
            "rps": 0.15,
            "nll": 2.5,
            "rmse_h": 0.4,
            "rmse_a": 0.1,
            "inference_run_id": "inf-1",
            "lambda_h": 1.6,
            "lambda_a": 1.0,
            "p_home": 0.5,
            "p_draw": 0.3,
            "p_away": 0.2,
        })
    return pd.DataFrame(rows)


def test_run_monitoring_step_skips_log_when_count_unchanged():
    df = _make_monitoring_df(n_matches=2)
    with (
        patch.object(monitor, "score_completed_wc_matches", return_value=df),
        patch.object(monitor, "_last_logged_n_scored_matches", return_value=2),
        patch.object(monitor, "log_monitoring_run") as mock_log,
        patch.object(monitor, "evaluate_alert_threshold"),
    ):
        monitor.run_monitoring_step()

    mock_log.assert_not_called()


def test_run_monitoring_step_logs_when_new_matches():
    df = _make_monitoring_df(n_matches=3)
    with (
        patch.object(monitor, "score_completed_wc_matches", return_value=df),
        patch.object(monitor, "_last_logged_n_scored_matches", return_value=2),
        patch.object(monitor, "log_monitoring_run") as mock_log,
        patch.object(monitor, "evaluate_alert_threshold"),
    ):
        monitor.run_monitoring_step()

    assert mock_log.call_count == 2  # one call per cadence mode


def test_run_monitoring_step_logs_when_dagshub_unreachable():
    """last_n=-1 (query error) → fall through to logging."""
    df = _make_monitoring_df(n_matches=2)
    with (
        patch.object(monitor, "score_completed_wc_matches", return_value=df),
        patch.object(monitor, "_last_logged_n_scored_matches", return_value=-1),
        patch.object(monitor, "log_monitoring_run") as mock_log,
        patch.object(monitor, "evaluate_alert_threshold"),
    ):
        monitor.run_monitoring_step()

    assert mock_log.call_count == 2


# ---------------------------------------------------------------------------
# log_batch usage in log_monitoring_run
# ---------------------------------------------------------------------------


@patch("src.monitoring.monitor.mlflow")
@patch("src.monitoring.monitor.start_run")
@patch("src.monitoring.monitor.log_run")
@patch("src.monitoring.monitor.setup_mlflow")
@patch("src.monitoring.monitor.get_or_create_experiment")
def test_log_monitoring_run_uses_log_batch_not_log_metric(
    mock_get_exp,
    mock_setup,
    mock_log_run,
    mock_start_run,
    mock_mlflow,
):
    df = _make_monitoring_df(n_matches=2)

    fake_run = MagicMock()
    fake_run.info.run_id = "mon_run_1"
    mock_start_run.return_value.__enter__ = MagicMock(return_value=fake_run)
    mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    monitor.log_monitoring_run(df, cadence_mode="frozen")

    assert mock_client.log_batch.called
    assert not mock_mlflow.log_metric.called


@patch("src.monitoring.monitor.mlflow")
@patch("src.monitoring.monitor.start_run")
@patch("src.monitoring.monitor.log_run")
@patch("src.monitoring.monitor.setup_mlflow")
@patch("src.monitoring.monitor.get_or_create_experiment")
def test_log_monitoring_run_batch_contains_correct_metric_keys(
    mock_get_exp,
    mock_setup,
    mock_log_run,
    mock_start_run,
    mock_mlflow,
):
    df = _make_monitoring_df(n_matches=2)

    fake_run = MagicMock()
    fake_run.info.run_id = "mon_run_1"
    mock_start_run.return_value.__enter__ = MagicMock(return_value=fake_run)
    mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    # Use real MlflowMetric so we can inspect the batch contents.
    from mlflow.entities import Metric as RealMetric
    with patch("src.monitoring.monitor.MlflowMetric", RealMetric):
        monitor.log_monitoring_run(df, cadence_mode="frozen")

    batch = mock_client.log_batch.call_args.kwargs.get(
        "metrics"
    ) or mock_client.log_batch.call_args[1]["metrics"]
    keys = {m.key for m in batch}
    assert keys == {"rps", "nll", "rmse_h", "rmse_a", "cum_rps"}
    # 2 matches × 5 metrics = 10 entries
    assert len(batch) == 10
