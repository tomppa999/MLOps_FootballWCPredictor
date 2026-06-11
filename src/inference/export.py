"""Export scoreline and tournament probabilities as betting CSV files.

Usage (from project root):

    python -m src.inference.export                      # latest frozen run
    python -m src.inference.export --run-id <run_id>    # specific run
    python -m src.inference.export --top-n 5            # top-5 per match (default: 3)
    python -m src.inference.export --out-dir /tmp/bets  # custom output dir
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_STAGE_ORDER = {"Group": 0, "R32": 1, "R16": 2, "QF": 3, "SF": 4, "Final": 5}
_OVER_UNDER_THRESHOLDS = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
_WDL_COLUMNS = (
    "p_home_win",
    "p_draw",
    "p_away_win",
    "p_home_or_draw",
    "p_home_or_away",
    "p_draw_or_away",
)
_BTTS_COLUMNS = (
    "p_btts_y_over25",
    "p_btts_y_under25",
    "p_btts_n_over25",
    "p_btts_n_under25",
)
_TOURNAMENT_COLUMNS = (
    "team",
    "group",
    "p_reach_r32",
    "p_reach_r16",
    "p_reach_qf",
    "p_reach_sf",
    "p_reach_final",
    "p_win_tournament",
    "p_group_first",
    "p_group_last",
)


def _threshold_col(threshold: float) -> str:
    return str(threshold).replace(".", "_")


def _round_prob(value: float) -> float:
    return round(float(value), 4)


def _over_under_columns() -> list[str]:
    cols: list[str] = []
    for threshold in _OVER_UNDER_THRESHOLDS:
        col = _threshold_col(threshold)
        cols.extend([f"p_over_{col}", f"p_under_{col}"])
    return cols


def _rank_columns(top_n: int) -> list[str]:
    cols: list[str] = []
    for rank in range(1, top_n + 1):
        cols.extend([f"rank{rank}_scoreline", f"rank{rank}_prob"])
    return cols


def expected_betting_columns(top_n: int) -> list[str]:
    """Return the full ordered column list for a wide betting export."""
    return [
        "stage",
        "home_team",
        "away_team",
        *_rank_columns(top_n),
        *_WDL_COLUMNS,
        *_over_under_columns(),
        *_BTTS_COLUMNS,
    ]


def expected_tournament_betting_columns() -> list[str]:
    """Return the ordered column list for the tournament betting export."""
    return list(_TOURNAMENT_COLUMNS)


def _compute_match_markets(
    group: pd.DataFrame,
    *,
    stage: str,
    home_team: str,
    away_team: str,
    top_n: int,
) -> dict[str, object]:
    """Aggregate one match's full scoreline distribution into wide market columns."""
    total = group["home_goals"] + group["away_goals"]
    btts = (group["home_goals"] > 0) & (group["away_goals"] > 0)

    p_home = group.loc[group["home_goals"] > group["away_goals"], "probability"].sum()
    p_draw = group.loc[group["home_goals"] == group["away_goals"], "probability"].sum()
    p_away = group.loc[group["home_goals"] < group["away_goals"], "probability"].sum()

    row: dict[str, object] = {
        "stage": stage,
        "home_team": home_team,
        "away_team": away_team,
        "p_home_win": _round_prob(p_home),
        "p_draw": _round_prob(p_draw),
        "p_away_win": _round_prob(p_away),
        "p_home_or_draw": _round_prob(p_home + p_draw),
        "p_home_or_away": _round_prob(p_home + p_away),
        "p_draw_or_away": _round_prob(p_draw + p_away),
    }

    for threshold in _OVER_UNDER_THRESHOLDS:
        col = _threshold_col(threshold)
        row[f"p_over_{col}"] = _round_prob(group.loc[total > threshold, "probability"].sum())
        row[f"p_under_{col}"] = _round_prob(group.loc[total <= threshold, "probability"].sum())

    row["p_btts_y_over25"] = _round_prob(
        group.loc[btts & (total > 2.5), "probability"].sum()
    )
    row["p_btts_y_under25"] = _round_prob(
        group.loc[btts & (total <= 2.5), "probability"].sum()
    )
    row["p_btts_n_over25"] = _round_prob(
        group.loc[~btts & (total > 2.5), "probability"].sum()
    )
    row["p_btts_n_under25"] = _round_prob(
        group.loc[~btts & (total <= 2.5), "probability"].sum()
    )

    top = group.nlargest(top_n, "probability").reset_index(drop=True)
    for rank in range(1, top_n + 1):
        if rank <= len(top):
            score = top.iloc[rank - 1]
            row[f"rank{rank}_scoreline"] = (
                f"{int(score['home_goals'])}-{int(score['away_goals'])}"
            )
            row[f"rank{rank}_prob"] = _round_prob(score["probability"])
        else:
            row[f"rank{rank}_scoreline"] = None
            row[f"rank{rank}_prob"] = None

    return row


def _compute_markets(sl_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Build a wide betting DataFrame for one stage's scoreline distribution."""
    rows: list[dict[str, object]] = []
    for (home_team, away_team), match_sl in sl_df.groupby(["home_team", "away_team"]):
        stage = str(match_sl["stage"].iloc[0]) if "stage" in match_sl.columns else "Group"
        rows.append(
            _compute_match_markets(
                match_sl,
                stage=stage,
                home_team=home_team,
                away_team=away_team,
                top_n=top_n,
            )
        )

    if not rows:
        return pd.DataFrame(columns=expected_betting_columns(top_n))

    out_df = pd.DataFrame(rows)
    return out_df[expected_betting_columns(top_n)]


def _filter_to_champion(
    df: pd.DataFrame,
    champion_model_name: str | None,
    artifact_name: str,
) -> pd.DataFrame:
    """Keep champion rows from stacked multi-model artifacts."""
    if "model_name" not in df.columns:
        return df
    if not champion_model_name:
        raise ValueError(
            f"{artifact_name} has model_name rows, but the MLflow run has no "
            "champion_model_name param."
        )
    filtered = df[df["model_name"] == champion_model_name].drop(columns=["model_name"])
    if filtered.empty:
        raise ValueError(
            f"{artifact_name} has no rows for champion model {champion_model_name!r}."
        )
    return filtered.reset_index(drop=True)


def _load_inference_artifacts(
    run_id: str | None,
) -> tuple[dict[str, pd.DataFrame], str, str | None]:
    """Load export-relevant inference artifacts from MLflow."""
    from src.dashboard.load_artifacts import load_latest_inference_artifacts
    import mlflow

    if run_id is None:
        data, info = load_latest_inference_artifacts()
        return data, info.run_id, info.champion_model_name

    setup_fn = None
    try:
        from src.models.mlflow_utils import setup_mlflow
        setup_fn = setup_mlflow
    except ImportError:
        pass
    if setup_fn:
        setup_fn()
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    champion_model_name = run.data.params.get("champion_model_name") or None
    artifact_dir = Path(client.download_artifacts(run_id, ""))

    data: dict[str, pd.DataFrame] = {}
    for filename in (
        "scoreline_distributions.csv",
        "tournament_probabilities.csv",
        "group_positions.csv",
    ):
        path = artifact_dir / filename
        if not path.exists():
            continue
        key = path.stem
        df = pd.read_csv(path)
        if key in {"tournament_probabilities", "group_positions"}:
            df = _filter_to_champion(df, champion_model_name, filename)
        data[key] = df

    return data, run_id, champion_model_name


def _validate_scoreline_distributions(sl_df: pd.DataFrame | None) -> pd.DataFrame:
    """Validate and normalize scoreline_distributions for export."""
    if sl_df is None or sl_df.empty:
        raise RuntimeError(
            "scoreline_distributions artifact is empty or missing. "
            "Re-run inference to regenerate it."
        )

    required = {"home_team", "away_team", "home_goals", "away_goals", "probability"}
    missing = required - set(sl_df.columns)
    if missing:
        raise ValueError(
            f"scoreline_distributions is missing columns: {missing}. "
            "Re-run inference to regenerate the artifact."
        )

    if "stage" not in sl_df.columns:
        sl_df = sl_df.copy()
        sl_df["stage"] = "Group"

    return sl_df


def _compute_tournament_bets(
    advancement_df: pd.DataFrame,
    group_positions_df: pd.DataFrame,
    group_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Combine champion advancement and group-position probabilities by team."""
    required_adv = {"team", "p_r32", "p_r16", "p_qf", "p_sf", "p_final", "p_winner"}
    missing_adv = required_adv - set(advancement_df.columns)
    if missing_adv:
        raise ValueError(f"tournament_probabilities is missing columns: {missing_adv}.")

    required_gp = {"team", "p_1st", "p_4th"}
    missing_gp = required_gp - set(group_positions_df.columns)
    if missing_gp:
        raise ValueError(f"group_positions is missing columns: {missing_gp}.")

    merged = advancement_df.merge(
        group_positions_df[["team", "p_1st", "p_4th"]],
        on="team",
        how="left",
    )
    if merged[["p_1st", "p_4th"]].isna().any().any():
        raise ValueError("group_positions is missing one or more advancement teams.")

    groups = group_mapping or {}
    out = pd.DataFrame({
        "team": merged["team"],
        "group": merged["team"].map(groups),
        "p_reach_r32": merged["p_r32"].map(_round_prob),
        "p_reach_r16": merged["p_r16"].map(_round_prob),
        "p_reach_qf": merged["p_qf"].map(_round_prob),
        "p_reach_sf": merged["p_sf"].map(_round_prob),
        "p_reach_final": merged["p_final"].map(_round_prob),
        "p_win_tournament": merged["p_winner"].map(_round_prob),
        "p_group_first": merged["p_1st"].map(_round_prob),
        "p_group_last": merged["p_4th"].map(_round_prob),
    })
    return out.sort_values("p_win_tournament", ascending=False).reset_index(drop=True)


def export_betting_csvs(
    output_dir: Path | str = Path("betting/scorelines"),
    run_id: str | None = None,
    top_n: int = 3,
) -> dict[str, Path]:
    """Export matchup and champion-only tournament betting CSVs.

    Args:
        output_dir: Directory to write CSVs into (created if absent).
        run_id: Specific MLflow run ID to load from. When ``None`` the latest
            frozen inference run is used (same run the dashboard shows).
        top_n: How many scorelines to keep per match (sorted by probability).

    Returns:
        Mapping of export label -> written CSV path.
    """
    from src.dashboard.load_artifacts import load_group_mapping

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data, used_run_id, _ = _load_inference_artifacts(run_id)
    sl_df = _validate_scoreline_distributions(data.get("scoreline_distributions"))

    written: dict[str, Path] = {}
    stages = sorted(sl_df["stage"].dropna().unique(), key=lambda s: _STAGE_ORDER.get(s, 99))

    for stage in stages:
        stage_df = sl_df[sl_df["stage"] == stage].copy()
        out_df = _compute_markets(stage_df, top_n=top_n)
        if out_df.empty:
            continue

        filename = f"betting_{stage.lower()}.csv"
        path = out / filename
        out_df.to_csv(path, index=False)
        written[stage] = path
        logger.info("Wrote %d rows → %s", len(out_df), path)

    advancement_df = data.get("tournament_probabilities")
    group_positions_df = data.get("group_positions")
    if advancement_df is None or advancement_df.empty:
        raise RuntimeError(
            "tournament_probabilities artifact is empty or missing. "
            "Re-run inference to regenerate it."
        )
    if group_positions_df is None or group_positions_df.empty:
        raise RuntimeError(
            "group_positions artifact is empty or missing. "
            "Re-run inference to regenerate it."
        )

    tournament_df = _compute_tournament_bets(
        advancement_df,
        group_positions_df,
        group_mapping=load_group_mapping(),
    )
    tournament_path = out / "tournament_bets.csv"
    tournament_df.to_csv(tournament_path, index=False)
    written["Tournament"] = tournament_path
    logger.info("Wrote %d rows → %s", len(tournament_df), tournament_path)

    logger.info("Export complete. run_id=%s, stages=%s", used_run_id, list(written.keys()))
    return written


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Export wide matchup and tournament betting CSVs."
    )
    parser.add_argument("--run-id", default=None, help="MLflow run ID (default: latest frozen)")
    parser.add_argument("--top-n", type=int, default=3, help="Top N scorelines per match")
    parser.add_argument(
        "--out-dir",
        default="betting/scorelines",
        help="Output directory (default: betting/scorelines)",
    )
    args = parser.parse_args()
    written = export_betting_csvs(
        output_dir=args.out_dir,
        run_id=args.run_id,
        top_n=args.top_n,
    )
    for stage, path in written.items():
        print(f"  {stage:10s}  →  {path}")


if __name__ == "__main__":
    _main()
