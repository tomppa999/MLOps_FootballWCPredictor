"""Rescore saved forecasts against 90-minute results, leaving legacy outputs intact.

Run through ``dvc repro -s regulation_time``. The existing RQ1/RQ3 implementations
provide the tests, bootstrap intervals and Holm families without modification.
Historical comparison is a separate optional stage, ``regulation_time_comparison``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.analysis import rq1_paired, rq3_calibration_drift
from src.analysis.rq_datasets.build_rq_datasets import _reliability_bins
from src.inference.features import _load_api_id_to_canonical
from src.models.evaluation import compute_rps, goals_to_outcome

PROB_COLS = ["p_home", "p_draw", "p_away"]


def _goal(value: object, *, match_id: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid {label} score for match {match_id}: {value!r}")  # noqa: TRY004 -- invalid JSON field
    if not np.isfinite(value) or value < 0 or int(value) != value:
        raise ValueError(f"Invalid {label} score for match {match_id}: {value!r}")
    return int(value)


def load_regulation_scores(
    fixtures_dir: Path, match_ids: set[int], *, team_names: dict[int, str] | None = None
) -> pd.DataFrame:
    """Require complete, consistent finished snapshots; never use a score fallback."""
    records: dict[int, dict] = {}
    team_names = team_names or {}
    for path in sorted(fixtures_dir.glob("*/fixtures.json")):
        for entry in json.loads(path.read_text()).get("response", []):
            fixture = entry["fixture"]
            mid = fixture["id"]
            status = fixture.get("status", {}).get("short")
            if mid not in match_ids or status not in {"FT", "AET", "PEN"}:
                continue
            fulltime = entry.get("score", {}).get("fulltime") or {}
            goals = entry.get("goals") or {}
            row = {
                "match_id": mid,
                "status": status,
                "home": team_names.get(
                    entry["teams"]["home"].get("id"), entry["teams"]["home"]["name"]
                ),
                "away": team_names.get(
                    entry["teams"]["away"].get("id"), entry["teams"]["away"]["name"]
                ),
            }
            for side, suffix in (("home", "h"), ("away", "a")):
                row[f"regulation_{suffix}"] = _goal(
                    fulltime.get(side), match_id=mid, label="regulation"
                )
                row[f"final_{suffix}"] = _goal(goals.get(side), match_id=mid, label="final")
                if row[f"regulation_{suffix}"] > row[f"final_{suffix}"]:
                    raise ValueError(f"Regulation goals exceed final goals for match {mid}")
                if status == "FT" and row[f"regulation_{suffix}"] != row[f"final_{suffix}"]:
                    raise ValueError(f"FT regulation/final scores disagree for match {mid}")
            if mid in records:
                previous = {k: v for k, v in records[mid].items() if k != "source_file"}
                if previous != row:
                    raise ValueError(f"Conflicting finished snapshots for match {mid}: {path}")
            else:
                records[mid] = row | {"source_file": str(path)}
    missing = match_ids - records.keys()
    if missing:
        raise ValueError(f"Missing finished regulation scores for matches {sorted(missing)}")
    return pd.DataFrame([records[mid] for mid in sorted(records)])


def rescore_matches(
    matches: pd.DataFrame, scores: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a new match table plus a per-match audit; preserve all forecast fields."""
    actual_cols = [
        "match_id",
        "home",
        "away",
        "round_label",
        "actual_h",
        "actual_a",
        "actual_outcome",
    ]
    actuals = matches[actual_cols].drop_duplicates()
    if actuals.match_id.duplicated().any():
        raise ValueError("Inconsistent original outcomes or metadata within a match")
    audit = actuals.merge(
        scores, on="match_id", how="left", validate="one_to_one", suffixes=("", "_raw")
    )
    if audit.regulation_h.isna().any() or audit.regulation_a.isna().any():
        raise ValueError("Missing regulation scores in match join")
    if (audit.home != audit.home_raw).any() or (audit.away != audit.away_raw).any():
        raise ValueError("Team orientation disagrees between saved matches and raw fixtures")
    if (audit.actual_h != audit.final_h).any() or (audit.actual_a != audit.final_a).any():
        raise ValueError("Original scores disagree with finished raw fixture goals")
    if not np.array_equal(audit.actual_outcome, goals_to_outcome(audit.actual_h, audit.actual_a)):
        raise ValueError("Original outcome codes disagree with original goals")
    audit["regulation_outcome"] = goals_to_outcome(audit.regulation_h, audit.regulation_a)
    audit["goals_changed"] = (audit.actual_h != audit.regulation_h) | (
        audit.actual_a != audit.regulation_a
    )
    audit["outcome_changed"] = audit.actual_outcome != audit.regulation_outcome
    audit = (
        audit.drop(columns=["home_raw", "away_raw"]).sort_values("match_id").reset_index(drop=True)
    )

    by_id = audit.set_index("match_id")
    result = matches.copy()
    for target, source in (
        ("actual_h", "regulation_h"),
        ("actual_a", "regulation_a"),
        ("actual_outcome", "regulation_outcome"),
    ):
        result[target] = result.match_id.map(by_id[source]).astype(int)
    probs = result[PROB_COLS].to_numpy(dtype=float)
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(axis=1), 1):
        raise ValueError("Invalid saved outcome probabilities")
    rates = result[["lambda_h", "lambda_a"]].to_numpy(dtype=float)
    if not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("Invalid saved goal rates")
    result["rps"] = compute_rps(probs, result.actual_outcome.to_numpy(dtype=int))
    result["nll"] = -(
        poisson.logpmf(result.actual_h, rates[:, 0].clip(1e-6))
        + poisson.logpmf(result.actual_a, rates[:, 1].clip(1e-6))
    )
    # Preserve the monitoring schema: per-match 'rmse' fields are absolute errors.
    result["rmse_h"] = np.abs(rates[:, 0] - result.actual_h)
    result["rmse_a"] = np.abs(rates[:, 1] - result.actual_a)
    return result, audit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rq1", type=Path, default=Path("data/analysis/rq1_matches.csv"))
    parser.add_argument("--fixtures", type=Path, default=Path("data/raw/api_football/fixtures"))
    parser.add_argument(
        "--mapping", type=Path, default=Path("data/mappings/team_mapping_master_merged.csv")
    )
    parser.add_argument("--data-out", type=Path, default=Path("data/analysis/regulation_time"))
    parser.add_argument("--tables-out", type=Path, default=Path("docs/tables/regulation_time"))
    parser.add_argument("--n-boot", type=int, default=rq1_paired.DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=rq1_paired.DEFAULT_SEED)
    args = parser.parse_args(argv)
    original = pd.read_csv(args.rq1)
    scores = load_regulation_scores(
        args.fixtures, set(original.match_id), team_names=_load_api_id_to_canonical(args.mapping)
    )
    matches, audit = rescore_matches(original, scores)
    args.data_out.mkdir(parents=True, exist_ok=True)
    args.tables_out.mkdir(parents=True, exist_ok=True)
    match_path = args.data_out / "rq1_matches.csv"
    # Preserve full saved forecast precision rather than rounding an intermediate.
    matches.to_csv(match_path, index=False)
    audit.to_csv(args.data_out / "outcome_audit.csv", index=False)
    bins_path = args.data_out / "rq3_calibration.csv"
    _reliability_bins(matches).to_csv(bins_path, index=False)
    print(
        f"Rescored {len(matches)} forecast rows; {int(audit.outcome_changed.sum())} match outcomes changed",
        flush=True,
    )
    common = ["--rq1", str(match_path), "--n-boot", str(args.n_boot), "--seed", str(args.seed)]
    rq1_paired.main(
        common
        + [
            "--diffs-out",
            str(args.data_out / "rq1_paired_diffs.csv"),
            "--phase-out",
            str(args.tables_out / "rq1_paired_phase.csv"),
            "--rounds-out",
            str(args.tables_out / "rq1_paired_rounds.csv"),
            "--coverage-out",
            str(args.tables_out / "rq1_pairing_coverage.csv"),
        ]
    )
    print("Computing RQ3 with the existing 10-bin definition and resampling methods", flush=True)
    rq3_calibration_drift.main(
        common
        + [
            "--rq3",
            str(bins_path),
            "--phase-out",
            str(args.tables_out / "rq3_calibration_phase.csv"),
            "--rounds-out",
            str(args.tables_out / "rq3_calibration_rounds.csv"),
            "--trend-out",
            str(args.tables_out / "rq3_trend.csv"),
        ]
    )


if __name__ == "__main__":
    main()
