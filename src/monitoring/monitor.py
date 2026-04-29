"""Live WC 2026 monitoring across the champion + 8 shadow candidates.

Scoring, not selection. Every settled WC 2026 match is joined with the most
recent ``predictions_all_models.csv`` artifact whose
``inference_timestamp`` is strictly earlier than that match's kickoff.
Per-match RPS / NLL / RMSE_h / RMSE_a are computed for each of the nine
models. The cumulative long-format leaderboard is logged to per-model
MLflow runs (``monitor_<model_name>``, ``stage=monitoring``).
"""

from __future__ import annotations

import bisect
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import mlflow
import numpy as np
import pandas as pd

from src.inference.features import (
    FINISHED_STATUSES,
    _load_api_id_to_canonical,
)
from src.models.evaluation import (
    compute_outcome_probs,
    compute_rps,
    goals_to_outcome,
)
from src.models.mlflow_utils import (
    EXPERIMENT_NAME,
    get_or_create_experiment,
    log_run,
    setup_mlflow,
    start_run,
)
from src.monitoring.baselines import (
    ALERT_FACTOR,
    ALERT_WINDOW,
    WC2022_RPS_BASELINES,
)

logger = logging.getLogger(__name__)

_FIXTURES_DIR: Final[Path] = Path("data/raw/api_football/fixtures")
_TEAM_MAPPING_PATH: Final[Path] = Path("data/mappings/team_mapping_master_merged.csv")
_WC_SEASONS: Final[frozenset[int]] = frozenset({2025, 2026})
_PREDICTIONS_ALL_MODELS_FILENAME: Final[str] = "predictions_all_models.csv"
_MONITORING_ARTIFACT_FILENAME: Final[str] = "wc2026_monitoring.csv"


# ---------------------------------------------------------------------------
# Settled-match parsing (Bronze)
# ---------------------------------------------------------------------------


def parse_wc_settled_matches(
    fixtures_dir: Path = _FIXTURES_DIR,
    mapping_path: Path = _TEAM_MAPPING_PATH,
) -> pd.DataFrame:
    """Return one row per settled WC 2026 match, with kickoff timestamp.

    A richer sibling of :func:`src.inference.features.parse_wc_results`:
    where the simulation only needs (home, away) → (h_goals, a_goals),
    monitoring also needs ``kickoff_utc`` to enforce the strictly-earlier
    pre-kickoff prediction selection rule.

    Columns: match_id, kickoff_utc, home, away, actual_h, actual_a,
    actual_outcome.
    """
    fixture_files = sorted(fixtures_dir.glob("*/fixtures.json"))
    if not fixture_files:
        logger.info("No fixtures.json files found for monitoring.")
        return pd.DataFrame(
            columns=[
                "match_id", "kickoff_utc", "home", "away",
                "actual_h", "actual_a", "actual_outcome",
            ]
        )

    id_to_name = _load_api_id_to_canonical(mapping_path)
    rows: list[dict] = []
    seen_match_ids: set[int] = set()

    for fp in fixture_files:
        with open(fp) as f:
            data = json.load(f)

        for entry in data.get("response", []):
            fixture = entry.get("fixture", {})
            status = fixture.get("status", {}).get("short", "")
            if status not in FINISHED_STATUSES:
                continue

            league = entry.get("league", {})
            if league.get("id") != 1:
                continue
            if league.get("season") not in _WC_SEASONS:
                continue

            match_id = fixture.get("id")
            if match_id in seen_match_ids:
                continue
            seen_match_ids.add(match_id)

            kickoff_raw = fixture.get("date")
            if not kickoff_raw:
                continue
            kickoff = pd.to_datetime(kickoff_raw, utc=True)

            teams = entry.get("teams", {})
            goals_raw = entry.get("goals", {})
            home_api_id = teams.get("home", {}).get("id")
            away_api_id = teams.get("away", {}).get("id")
            home_name = id_to_name.get(home_api_id, teams.get("home", {}).get("name"))
            away_name = id_to_name.get(away_api_id, teams.get("away", {}).get("name"))

            hg = goals_raw.get("home")
            ag = goals_raw.get("away")
            if hg is None or ag is None:
                continue
            hg, ag = int(hg), int(ag)
            outcome = 0 if hg > ag else (2 if hg < ag else 1)

            rows.append({
                "match_id": match_id,
                "kickoff_utc": kickoff,
                "home": home_name,
                "away": away_name,
                "actual_h": hg,
                "actual_a": ag,
                "actual_outcome": outcome,
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                "match_id", "kickoff_utc", "home", "away",
                "actual_h", "actual_a", "actual_outcome",
            ]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("kickoff_utc").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Pre-kickoff inference run selection
# ---------------------------------------------------------------------------


def _list_inference_runs() -> list[dict]:
    """Return inference runs sorted ascending by ``inference_timestamp``.

    Each entry has ``run_id`` and ``inference_timestamp`` (as a UTC
    ``pd.Timestamp``). Runs without that param are skipped.
    """
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return []

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string='tags.stage = "inference"',
        order_by=["start_time ASC"],
        max_results=5000,
    )
    out: list[dict] = []
    for r in runs:
        ts = r.data.params.get("inference_timestamp")
        if not ts:
            continue
        try:
            ts_parsed = pd.to_datetime(ts, utc=True)
        except (ValueError, TypeError):
            continue
        out.append({"run_id": r.info.run_id, "inference_timestamp": ts_parsed})
    out.sort(key=lambda x: x["inference_timestamp"])
    return out


def _select_pre_kickoff_run(
    inference_runs: list[dict],
    kickoff: pd.Timestamp,
) -> dict | None:
    """Return the most recent inference run strictly before ``kickoff``.

    Picking a later run would leak post-match information — the result
    would already be in Gold by the next inference cycle.
    """
    if not inference_runs:
        return None
    timestamps = [r["inference_timestamp"] for r in inference_runs]
    idx = bisect.bisect_left(timestamps, kickoff) - 1
    if idx < 0:
        return None
    return inference_runs[idx]


def _load_predictions_all_models(run_id: str) -> pd.DataFrame | None:
    """Download and parse ``predictions_all_models.csv`` for ``run_id``.

    Returns ``None`` if the artifact is missing.
    """
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    try:
        local_path = client.download_artifacts(
            run_id, _PREDICTIONS_ALL_MODELS_FILENAME,
        )
    except Exception:  # noqa: BLE001
        return None
    return pd.read_csv(local_path)


# ---------------------------------------------------------------------------
# Per-match scoring
# ---------------------------------------------------------------------------


def _orient_lambdas(
    pred_row: pd.Series,
    actual_home: str,
) -> tuple[float, float]:
    """Align predicted ``lambda_h``/``lambda_a`` with the actual home/away.

    All-pairs predictions are stored with the alphabetically-smaller team as
    "home" (``generate_all_wc_pairings`` uses ``combinations(sorted(teams),
    2)``). When the real fixture flips the assignment, swap the lambdas so
    the predicted-home rate maps to the actual home side.
    """
    pred_home = pred_row["home_team"]
    if pred_home == actual_home:
        return float(pred_row["lambda_h"]), float(pred_row["lambda_a"])
    return float(pred_row["lambda_a"]), float(pred_row["lambda_h"])


def _score_one_match(
    match: pd.Series,
    predictions: pd.DataFrame,
    inference_run_id: str,
) -> list[dict]:
    """Compute per-(match, model) metrics for one settled match.

    Returns a list of dict rows ready to be concatenated into the long-
    format monitoring artifact.
    """
    home_a, away_a = match["home"], match["away"]
    pair = frozenset({home_a, away_a})
    pred_subset = predictions[
        predictions.apply(
            lambda r: frozenset({r["home_team"], r["away_team"]}) == pair,
            axis=1,
        )
    ]
    if pred_subset.empty:
        logger.warning(
            "No prediction rows for %s vs %s in run %s — skipping.",
            home_a, away_a, inference_run_id,
        )
        return []

    actual_h = int(match["actual_h"])
    actual_a = int(match["actual_a"])
    actual_outcome = int(match["actual_outcome"])

    out_rows: list[dict] = []
    for _, pred_row in pred_subset.iterrows():
        lam_h, lam_a = _orient_lambdas(pred_row, home_a)
        lam_h_arr = np.array([lam_h])
        lam_a_arr = np.array([lam_a])

        probs = compute_outcome_probs(lam_h_arr, lam_a_arr)
        rps = float(compute_rps(probs, np.array([actual_outcome]))[0])

        nll = float(
            -(
                _poisson_logpmf(actual_h, lam_h)
                + _poisson_logpmf(actual_a, lam_a)
            )
        )
        rmse_h = float(abs(lam_h - actual_h))
        rmse_a = float(abs(lam_a - actual_a))

        out_rows.append({
            "match_id": match["match_id"],
            "kickoff_utc": match["kickoff_utc"],
            "home": home_a,
            "away": away_a,
            "actual_h": actual_h,
            "actual_a": actual_a,
            "actual_outcome": actual_outcome,
            "model_name": pred_row["model_name"],
            "lambda_h": lam_h,
            "lambda_a": lam_a,
            "p_home": float(probs[0, 0]),
            "p_draw": float(probs[0, 1]),
            "p_away": float(probs[0, 2]),
            "rps": rps,
            "nll": nll,
            "rmse_h": rmse_h,
            "rmse_a": rmse_a,
            "inference_run_id": inference_run_id,
        })
    return out_rows


def _poisson_logpmf(k: int, lam: float) -> float:
    """Numerically safe Poisson log-pmf without scipy import at top-level."""
    from scipy.stats import poisson  # noqa: PLC0415

    lam = max(float(lam), 1e-6)
    return float(poisson.logpmf(int(k), lam))


def score_completed_wc_matches(
    fixtures_dir: Path = _FIXTURES_DIR,
    mapping_path: Path = _TEAM_MAPPING_PATH,
) -> pd.DataFrame:
    """Score every settled WC 2026 match against pre-kickoff predictions.

    Uses the most recent inference run with ``inference_timestamp <
    kickoff`` (strictly) to look up predictions for all 9 models, then
    computes per-match RPS / NLL / RMSE_h / RMSE_a.

    Returns the long-format leaderboard DataFrame (one row per
    settled-match × model). May be empty if no WC 2026 matches have been
    settled yet, or if no pre-kickoff inference run is available.
    """
    settled = parse_wc_settled_matches(fixtures_dir, mapping_path)
    if settled.empty:
        logger.info("No settled WC 2026 matches yet — monitoring no-op.")
        return pd.DataFrame()

    inference_runs = _list_inference_runs()
    if not inference_runs:
        logger.warning("No inference runs found — cannot score matches.")
        return pd.DataFrame()

    predictions_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []

    for _, match in settled.iterrows():
        chosen = _select_pre_kickoff_run(inference_runs, match["kickoff_utc"])
        if chosen is None:
            logger.info(
                "No pre-kickoff inference run for %s vs %s @ %s — skipping.",
                match["home"], match["away"], match["kickoff_utc"],
            )
            continue
        run_id = chosen["run_id"]
        if run_id not in predictions_cache:
            preds = _load_predictions_all_models(run_id)
            if preds is None:
                logger.warning(
                    "Inference run %s has no %s artifact — skipping match %s vs %s.",
                    run_id,
                    _PREDICTIONS_ALL_MODELS_FILENAME,
                    match["home"], match["away"],
                )
                predictions_cache[run_id] = pd.DataFrame()
                continue
            predictions_cache[run_id] = preds
        preds = predictions_cache[run_id]
        if preds.empty:
            continue

        rows.extend(_score_one_match(match, preds, run_id))

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(["kickoff_utc", "model_name"]).reset_index(drop=True)
    logger.info(
        "Monitoring scored %d match-model rows across %d models.",
        len(out),
        out["model_name"].nunique(),
    )
    return out


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------


def evaluate_alert_threshold(
    monitoring_df: pd.DataFrame,
    *,
    window: int = ALERT_WINDOW,
    factor: float = ALERT_FACTOR,
    baselines: dict[str, float] = WC2022_RPS_BASELINES,
) -> list[str]:
    """Iterate every model and warn when rolling-mean RPS breaches threshold.

    Threshold is ``baselines[model] * factor``. Models with fewer than
    ``window`` scored matches are skipped (cold-start guard). Champion and
    shadows are treated identically — promotion remains manual under all
    conditions.

    Returns the list of breaching model names.
    """
    if monitoring_df.empty:
        return []

    breached: list[str] = []
    for model_name, group in monitoring_df.groupby("model_name"):
        baseline = baselines.get(model_name)
        if baseline is None:
            logger.warning(
                "No WC2022 baseline for %s — skipping alert evaluation.",
                model_name,
            )
            continue
        recent = group.sort_values("kickoff_utc").tail(window)
        if len(recent) < window:
            continue
        rolling_rps = float(recent["rps"].mean())
        threshold = baseline * factor
        if rolling_rps > threshold:
            logger.warning(
                "ALERT %s: rolling RPS %.4f over last %d matches > %.4f "
                "(baseline %.4f x %.2f). Manual investigation required.",
                model_name, rolling_rps, window, threshold, baseline, factor,
            )
            breached.append(model_name)
    return breached


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def log_monitoring_run(
    monitoring_df: pd.DataFrame,
) -> dict[str, str]:
    """Write one ``monitor_<model_name>`` run per model in the long table.

    For each model:
      - per-match metrics (rps, nll, rmse_h, rmse_a) are logged with
        ``step=match_index`` so the time series is browsable in MLflow.
      - a derived cumulative mean ``cum_rps`` is logged on the same axis.
      - the full long-format ``wc2026_monitoring.csv`` is attached as an
        artifact (cumulative-overwrite per cycle, so the latest run for
        each model holds the complete leaderboard).

    Returns ``{model_name: run_id}``.
    """
    if monitoring_df.empty:
        return {}

    setup_mlflow()
    get_or_create_experiment()

    out: dict[str, str] = {}
    cycle_ts = datetime.now(timezone.utc).isoformat()

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / _MONITORING_ARTIFACT_FILENAME
        monitoring_df.to_csv(artifact_path, index=False)

        for model_name, group in monitoring_df.groupby("model_name"):
            group = group.sort_values("kickoff_utc").reset_index(drop=True)

            with start_run(
                run_name=f"monitor_{model_name}",
                tags={"stage": "monitoring", "model_name": str(model_name)},
            ) as run:
                log_run(
                    params={
                        "model_name": str(model_name),
                        "n_scored_matches": str(len(group)),
                        "monitoring_cycle_ts": cycle_ts,
                    },
                )
                cum_rps_series = group["rps"].expanding().mean()
                for idx, row in group.iterrows():
                    step = int(idx)
                    mlflow.log_metric("rps", float(row["rps"]), step=step)
                    mlflow.log_metric("nll", float(row["nll"]), step=step)
                    mlflow.log_metric("rmse_h", float(row["rmse_h"]), step=step)
                    mlflow.log_metric("rmse_a", float(row["rmse_a"]), step=step)
                    mlflow.log_metric(
                        "cum_rps", float(cum_rps_series.iloc[step]), step=step,
                    )
                mlflow.log_artifact(str(artifact_path))
                out[str(model_name)] = run.info.run_id

    logger.info("Logged %d monitor_<model> runs.", len(out))
    return out


# ---------------------------------------------------------------------------
# Wrapper invoked from the trigger
# ---------------------------------------------------------------------------


def run_monitoring_step(
    fixtures_dir: Path = _FIXTURES_DIR,
    mapping_path: Path = _TEAM_MAPPING_PATH,
) -> pd.DataFrame:
    """Score → log → alert. Best-effort: callers wrap this in try/except.

    Returns the long-format monitoring DataFrame for inspection / testing.
    """
    monitoring_df = score_completed_wc_matches(fixtures_dir, mapping_path)
    if monitoring_df.empty:
        return monitoring_df
    log_monitoring_run(monitoring_df)
    evaluate_alert_threshold(monitoring_df)
    return monitoring_df
