"""Build the Gold matches table from Silver.

Usage:
    python -m src.gold.build_gold [--data-root DATA_ROOT]

Reads:
    data/silver/matches/season=YYYY/part-0.parquet (or .csv fallback)

Writes:
    data/gold/matches.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.gold.context_features import add_elo_diff, override_neutral_for_2026_hosts
from src.gold.rolling_features import compute_rolling_features
from src.gold.schema import GOLD_COLUMNS, apply_gold_dtypes, validate_gold

logger = logging.getLogger(__name__)


def load_silver(silver_dir: Path) -> pd.DataFrame:
    """Load all Silver partitions into a single DataFrame."""
    season_dirs = sorted(silver_dir.glob("season=*"))
    if not season_dirs:
        raise FileNotFoundError(f"No season partitions found under {silver_dir}")

    frames: list[pd.DataFrame] = []
    for sdir in season_dirs:
        parquet = sdir / "part-0.parquet"
        csv = sdir / "part-0.csv"
        if parquet.exists():
            part = pd.read_parquet(parquet)
        elif csv.exists():
            part = pd.read_csv(csv)
        else:
            logger.warning("No data file in %s, skipping", sdir)
            continue

        season_str = sdir.name.split("=")[1]
        if "season" not in part.columns:
            part["season"] = int(season_str)

        logger.info("Loaded %d rows from %s", len(part), sdir.name)
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    logger.info("Total Silver rows loaded: %d", len(df))
    return df


def build_gold(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a Silver DataFrame into a Gold DataFrame."""
    df = df.copy()

    # Filter to finished matches only
    finished_mask = df["match_status"].isin({"FT", "AET", "PEN"})
    n_dropped = (~finished_mask).sum()
    if n_dropped:
        logger.info("Dropped %d non-finished matches", n_dropped)
    df = df[finished_mask].reset_index(drop=True)

    # Drop rows with missing goals (can't be targets)
    goals_null = df["home_goals"].isna() | df["away_goals"].isna()
    if goals_null.any():
        logger.warning("Dropping %d rows with null goals", goals_null.sum())
        df = df[~goals_null].reset_index(drop=True)

    # Rolling features (time-aware) — includes match indices and tactical profiles
    logger.info("Computing rolling features...")
    df = compute_rolling_features(df)

    # Context features
    logger.info("Adding context features...")
    df = add_elo_diff(df)
    df = override_neutral_for_2026_hosts(df)

    # Select and reorder to Gold schema
    for col in GOLD_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"Expected Gold column '{col}' not found after transforms")

    df = df[GOLD_COLUMNS].copy()
    df = apply_gold_dtypes(df)

    # Sort chronologically
    df = df.sort_values(["date_utc", "fixture_id"]).reset_index(drop=True)

    # Validate
    errors = validate_gold(df)
    if errors:
        for err in errors:
            logger.error("Gold validation: %s", err)
        raise ValueError(f"Gold validation failed: {errors}")

    logger.info("Gold table: %d rows, %d columns", len(df), len(df.columns))
    return df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build Gold matches table from Silver")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <data-root>/gold/matches.parquet)",
    )
    args = parser.parse_args()

    silver_dir = args.data_root / "silver" / "matches"
    output_path = args.output or (args.data_root / "gold" / "matches.parquet")

    df = load_silver(silver_dir)
    gold = build_gold(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Gold table written to %s", output_path)


if __name__ == "__main__":
    main()
