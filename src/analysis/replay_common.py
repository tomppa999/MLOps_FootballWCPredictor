"""Shared utilities for the WC 2026 D.1 reconstruction pass.

All drivers replay from the ``wc2026-end-of-tournament`` tag.  Never run
``dvc repro`` or ``dvc gc`` during reconstruction.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import mlflow
import numpy as np
import pandas as pd
import yaml

from src.inference.features import (
    FINISHED_STATUSES,
    _KO_NEXT,
    _load_api_id_to_canonical,
    _load_expected_matches_per_round,
    _resolve_ko_winner,
    _round_to_matchday_label,
    build_inference_features,
    generate_all_wc_pairings,
    wc_results_to_gold_rows,
)
from src.models.config import MODEL_FEATURE_SETS
from src.models.data_split import load_gold
from src.models.evaluation import compute_outcome_probs, compute_rps
from src.models.mlflow_utils import (
    SHADOW_MODEL_NAME,
    get_or_create_experiment,
    log_run,
    setup_mlflow,
    start_run,
)
from src.monitoring.monitor import (
    _list_inference_runs,
    _orient_lambdas,
    _poisson_logpmf,
    _select_pre_kickoff_run,
    parse_wc_settled_matches,
)

logger = logging.getLogger(__name__)

REPLAY_TAG: Final[str] = "wc2026-end-of-tournament"
RECONSTRUCTION_EXPERIMENT: Final[str] = "wc2026_reconstruction"
OUTPUT_ROOT: Final[Path] = Path("data/reconstruction")
_FIXTURES_DIR: Final[Path] = Path("data/raw/api_football/fixtures")
_TEAM_MAPPING_PATH: Final[Path] = Path("data/mappings/team_mapping_master_merged.csv")
_WC_SEASONS: Final[frozenset[int]] = frozenset({2025, 2026})

PINNED_FROZEN_SHADOW_VERSIONS: Final[dict[str, int]] = {
    "poisson_glm": 88,
    "bayesian_poisson": 90,
}

BACKFILL_FIXTURES: Final[dict[str, dict[str, Any]]] = {
    "ridge_per_round_md3": {
        "fixture_ids": [1489419, 1539013],
        "model_name": "ridge",
        "cadence_mode": "per_round",
    },
    "random_forest_frozen_r32": {
        "fixture_ids": [1564789],
        "model_name": "random_forest",
        "cadence_mode": "frozen",
        "copy_from_cadence": "per_round",
    },
    "mean_rate_poisson_frozen_md1": {
        "fixture_ids": [1539003],
        "model_name": "mean_rate_poisson",
        "cadence_mode": "frozen",
        "copy_from_cadence": "per_round",
    },
}


@dataclass(frozen=True)
class GoldCommit:
    """A git commit that touched ``dvc.lock`` and its Gold directory hash."""

    commit_sha: str
    commit_time: pd.Timestamp
    gold_hash: str


def _run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def build_gold_commit_index(tag: str = REPLAY_TAG) -> list[GoldCommit]:
    """Walk git history at ``tag`` and index Gold hashes per ``dvc.lock`` commit."""
    raw = _run_git(
        "log", tag, "--format=%H %cI", "--", "dvc.lock",
    )
    if not raw:
        raise RuntimeError(f"No dvc.lock commits found at tag {tag!r}")

    index: list[GoldCommit] = []
    for line in raw.splitlines():
        sha, ts_raw = line.split(" ", 1)
        lock_yaml = _run_git("show", f"{sha}:dvc.lock")
        lock = yaml.safe_load(lock_yaml)
        gold_hash = lock["stages"]["gold"]["outs"][0]["md5"]
        index.append(
            GoldCommit(
                commit_sha=sha,
                commit_time=pd.to_datetime(ts_raw, utc=True),
                gold_hash=gold_hash,
            )
        )
    index.sort(key=lambda c: c.commit_time)
    return index


def resolve_gold_commit(
    kickoff: pd.Timestamp,
    index: list[GoldCommit],
) -> GoldCommit | None:
    """Return the last Gold commit strictly before ``kickoff``."""
    candidates = [c for c in index if c.commit_time < kickoff]
    if not candidates:
        return None
    return candidates[-1]


@lru_cache(maxsize=32)
def _gold_parquet_path_at_commit(commit_sha: str) -> Path:
    """Materialise ``data/gold`` at ``commit_sha`` into a temp dir (cached)."""
    tmp = Path(tempfile.mkdtemp(prefix=f"gold-{commit_sha[:8]}-"))
    out = tmp / "gold"
    subprocess.run(
        ["dvc", "get", ".", "data/gold", "-o", str(out), "--rev", commit_sha],
        check=True,
        capture_output=True,
    )
    return out / "matches.parquet"


def load_gold_at_commit(commit_sha: str) -> pd.DataFrame:
    """Load Gold parquet materialised from a historical git commit."""
    return load_gold(_gold_parquet_path_at_commit(commit_sha))


def augment_gold_for_inference(
    gold_df: pd.DataFrame,
    wc_results: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Append stub WC rows and derive ``reference_date`` like ``run_inference``."""
    wc_gold_rows = wc_results_to_gold_rows(wc_results)
    if wc_gold_rows.empty:
        return gold_df, None
    new_wc_rows = wc_gold_rows[~wc_gold_rows["fixture_id"].isin(gold_df["fixture_id"])]
    augmented = pd.concat([gold_df, new_wc_rows], ignore_index=True)
    augmented = augmented.sort_values("date_utc").reset_index(drop=True)
    reference_date = wc_gold_rows["date_utc"].max() + pd.Timedelta(days=1)
    return augmented, reference_date


def predict_single_model(
    model: Any,
    model_name: str,
    home_team: str,
    away_team: str,
    augmented_gold: pd.DataFrame,
    reference_date: pd.Timestamp | None,
) -> dict[str, float]:
    """Predict one fixture with a loaded pyfunc model."""
    pairings = generate_all_wc_pairings(reference_date=reference_date)
    mask = (
        (pairings["home_team"] == home_team) & (pairings["away_team"] == away_team)
    ) | (
        (pairings["home_team"] == away_team) & (pairings["away_team"] == home_team)
    )
    fixture_rows = pairings.loc[mask]
    if fixture_rows.empty:
        raise ValueError(f"No pairing row for {home_team} vs {away_team}")
    row = fixture_rows.iloc[0]
    if row["home_team"] != home_team:
        row = row.copy()
        row["home_team"], row["away_team"] = home_team, away_team

    features = build_inference_features(fixture_rows, augmented_gold)
    feature_cols = MODEL_FEATURE_SETS[model_name]
    x_df = features[feature_cols].astype(float)
    preds = model.predict(x_df)
    preds = np.atleast_2d(preds)
    lam_h = float(np.clip(preds[0, 0], 1e-6, None))
    lam_a = float(np.clip(preds[0, 1], 1e-6, None))
    probs = compute_outcome_probs(np.array([lam_h]), np.array([lam_a]))
    return {
        "lambda_h": lam_h,
        "lambda_a": lam_a,
        "p_home": float(probs[0, 0]),
        "p_draw": float(probs[0, 1]),
        "p_away": float(probs[0, 2]),
    }


def score_prediction_row(
    match: pd.Series,
    model_name: str,
    pred: dict[str, float],
    *,
    inference_run_id: str = "reconstruction",
    cadence_mode: str = "frozen",
) -> dict[str, Any]:
    """Score one prediction against actuals (monitoring-compatible row)."""
    lam_h, lam_a = pred["lambda_h"], pred["lambda_a"]
    actual_h = int(match["actual_h"])
    actual_a = int(match["actual_a"])
    actual_outcome = int(match["actual_outcome"])
    probs = compute_outcome_probs(np.array([lam_h]), np.array([lam_a]))
    rps = float(compute_rps(probs, np.array([actual_outcome]))[0])
    nll = float(-(_poisson_logpmf(actual_h, lam_h) + _poisson_logpmf(actual_a, lam_a)))
    return {
        "match_id": match["match_id"],
        "kickoff_utc": match["kickoff_utc"],
        "home": match["home"],
        "away": match["away"],
        "actual_h": actual_h,
        "actual_a": actual_a,
        "actual_outcome": actual_outcome,
        "model_name": model_name,
        "lambda_h": lam_h,
        "lambda_a": lam_a,
        "p_home": pred["p_home"],
        "p_draw": pred["p_draw"],
        "p_away": pred["p_away"],
        "rps": rps,
        "nll": nll,
        "rmse_h": float(abs(lam_h - actual_h)),
        "rmse_a": float(abs(lam_a - actual_a)),
        "inference_run_id": inference_run_id,
        "cadence_mode": cadence_mode,
    }


def load_pinned_shadow_model(model_name: str, version: int) -> Any:
    """Load a specific ``wc_shadow`` registry version (read-only)."""
    setup_mlflow()
    uri = f"models:/{SHADOW_MODEL_NAME}/{version}"
    return mlflow.pyfunc.load_model(uri)


def leaderboard_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monitoring-style rows to model-level RPS / NLL / RMSE."""
    if df.empty:
        return pd.DataFrame()
    agg = (
        df.groupby(["model_name", "cadence_mode"], as_index=False)
        .agg(
            matches=("match_id", "count"),
            mean_rps=("rps", "mean"),
            mean_nll=("nll", "mean"),
            mean_rmse_h=("rmse_h", "mean"),
            mean_rmse_a=("rmse_a", "mean"),
        )
    )
    agg["mean_rmse"] = (agg["mean_rmse_h"] + agg["mean_rmse_a"]) / 2.0
    return agg.sort_values("mean_rps")


def compute_advancement_entropy(
    advancement_df: pd.DataFrame,
    *,
    prob_col: str = "p_r32",
) -> float:
    """Shannon entropy of the normalised advancement vector (thesis D.1)."""
    if advancement_df.empty or prob_col not in advancement_df.columns:
        return float("nan")
    p = advancement_df[prob_col].astype(float).to_numpy()
    p_norm = p / 32.0
    p_pos = p_norm[p_norm > 0]
    if p_pos.size == 0:
        return 0.0
    return float(-np.sum(p_pos * np.log(p_pos)))


def ensure_output_dir(subdir: str) -> Path:
    """Create ``data/reconstruction/<subdir>`` and return it."""
    out = OUTPUT_ROOT / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def log_reconstruction_run(
    *,
    strand: str,
    params: dict[str, str],
    metrics: dict[str, float] | None = None,
    artifacts: dict[str, Path] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log a reconstruction driver run to ``wc2026_reconstruction``."""
    setup_mlflow()
    get_or_create_experiment(RECONSTRUCTION_EXPERIMENT)
    run_tags = {"stage": "reconstruction", "strand": strand}
    if tags:
        run_tags.update(tags)
    with start_run(
        run_name=f"reconstruction_{strand}",
        tags=run_tags,
        experiment_name=RECONSTRUCTION_EXPERIMENT,
    ) as run:
        log_run(params=params, metrics=metrics or {}, tags={})
        if artifacts:
            for name, path in artifacts.items():
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path=name)
        return run.info.run_id


def parse_wc_results_before_kickoff(
    max_kickoff: pd.Timestamp,
    *,
    fixtures_dir: Path = _FIXTURES_DIR,
    mapping_path: Path = _TEAM_MAPPING_PATH,
) -> dict[str, Any]:
    """Build partial ``wc_results`` with matches whose kickoff <= ``max_kickoff``."""
    id_to_name = _load_api_id_to_canonical(mapping_path)
    fixture_files = sorted(fixtures_dir.glob("*/fixtures.json"), reverse=True)
    seen_fixture_ids: set[int] = set()
    group_results: dict[tuple[str, str], tuple[int, int]] = {}
    ko_results: dict[frozenset, dict] = {}
    finished_fixtures: list[dict] = []
    finished_per_round: dict[str, int] = {}
    max_group_matchday = 0

    for fp in fixture_files:
        with open(fp) as f:
            data = json.load(f)
        for entry in data.get("response", []):
            fixture = entry.get("fixture", {})
            status = fixture.get("status", {}).get("short", "")
            if status not in FINISHED_STATUSES:
                continue
            league = entry.get("league", {})
            if league.get("id") != 1 or league.get("season") not in _WC_SEASONS:
                continue
            kickoff_raw = fixture.get("date")
            if not kickoff_raw:
                continue
            kickoff = pd.to_datetime(kickoff_raw, utc=True)
            if kickoff > max_kickoff:
                continue

            fixture_id = fixture.get("id")
            if fixture_id is not None and fixture_id in seen_fixture_ids:
                continue
            if fixture_id is not None:
                seen_fixture_ids.add(fixture_id)

            round_str = league.get("round", "")
            round_label = _round_to_matchday_label(round_str)
            teams = entry.get("teams", {})
            goals_raw = entry.get("goals", {})
            home_name = id_to_name.get(
                teams.get("home", {}).get("id"), teams.get("home", {}).get("name"),
            )
            away_name = id_to_name.get(
                teams.get("away", {}).get("id"), teams.get("away", {}).get("name"),
            )
            hg_raw, ag_raw = goals_raw.get("home"), goals_raw.get("away")
            if hg_raw is None or ag_raw is None:
                continue
            home_goals, away_goals = int(hg_raw), int(ag_raw)

            finished_fixtures.append({
                "fixture_id": fixture_id,
                "date_utc": kickoff_raw,
                "home_team": home_name,
                "away_team": away_name,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "is_knockout": not round_str.lower().startswith("group"),
                "round": round_str,
            })
            if round_label is not None:
                finished_per_round[round_label] = finished_per_round.get(round_label, 0) + 1

            round_lower = round_str.lower()
            if round_lower.startswith("group"):
                if (home_name, away_name) not in group_results:
                    group_results[(home_name, away_name)] = (home_goals, away_goals)
                parts = round_str.rsplit(" - ", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    max_group_matchday = max(max_group_matchday, int(parts[1]))
            else:
                stage = _round_to_matchday_label(round_str)
                if stage is not None:
                    key = frozenset({home_name, away_name})
                    if key not in ko_results:
                        ko_results[key] = {
                            "home": home_name,
                            "away": away_name,
                            "home_goals": home_goals,
                            "away_goals": away_goals,
                            "winner": _resolve_ko_winner(
                                home_name,
                                away_name,
                                home_goals,
                                away_goals,
                                teams,
                                entry.get("score", {}),
                            ),
                            "decided_by": status,
                            "stage": stage,
                        }

    from src.inference.features import _MATCHDAY_ORDER

    _expected_per_round = _load_expected_matches_per_round()
    last_completed_matchday = "0"
    for label in _MATCHDAY_ORDER:
        expected_n = _expected_per_round.get(label, 0)
        if expected_n == 0:
            break
        if finished_per_round.get(label, 0) >= expected_n:
            last_completed_matchday = label
        else:
            break

    if last_completed_matchday in _KO_NEXT:
        next_matchday: int | str = _KO_NEXT[last_completed_matchday]
    elif max_group_matchday >= 3:
        next_matchday = "R32"
    elif max_group_matchday > 0:
        next_matchday = max_group_matchday + 1
    else:
        next_matchday = 1

    return {
        "group_results": group_results,
        "ko_results": ko_results,
        "next_matchday": next_matchday,
        "last_completed_matchday": last_completed_matchday,
        "finished_fixtures": finished_fixtures,
    }


def fixture_kickoff(fixture_id: int, settled: pd.DataFrame) -> pd.Timestamp:
    """Look up kickoff for a fixture id from settled matches."""
    row = settled.loc[settled["match_id"] == fixture_id]
    if row.empty:
        raise KeyError(f"fixture_id {fixture_id} not in settled matches")
    return pd.Timestamp(row.iloc[0]["kickoff_utc"])


def match_row_for_fixture(settled: pd.DataFrame, fixture_id: int) -> pd.Series:
    row = settled.loc[settled["match_id"] == fixture_id]
    if row.empty:
        raise KeyError(f"fixture_id {fixture_id} not in settled matches")
    return row.iloc[0]


def pre_kickoff_gold_for_match(
    match: pd.Series,
    gold_index: list[GoldCommit],
) -> GoldCommit:
    commit = resolve_gold_commit(match["kickoff_utc"], gold_index)
    if commit is None:
        raise RuntimeError(
            f"No pre-kickoff Gold commit for {match['home']} vs {match['away']}",
        )
    return commit


def load_monitoring_artifact(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "kickoff_utc" in df.columns:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    return df
