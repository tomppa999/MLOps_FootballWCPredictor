"""Parse Elo TSV files into a team-centric history and join to fixtures.

Elo TSV format (tab-separated, no header):
  0: year  1: month  2: day  3: home_code  4: away_code
  5: home_goals  6: away_goals  7: competition  8: neutral_venue
  9: home_elo_change  10: home_elo_post  11: away_elo_post
  12: home_rank_change  13: away_rank_change  14: home_rank  15: away_rank

Pre-match Elo derivation:
  home_elo_pre = home_elo_post - home_elo_change
  away_elo_pre = away_elo_post + home_elo_change
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fixture data starts at 2018; one year of runway for merge_asof fallback.
ELO_MIN_YEAR = 2017


def _safe_int(value: str) -> int | None:
    """Parse an integer, returning None for non-numeric strings like '−'."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def parse_single_tsv(path: Path) -> pd.DataFrame:
    """Parse one Elo TSV file into a DataFrame of match rows.

    Returns columns: date, home_code, away_code, is_neutral_elo,
        home_elo_pre, away_elo_pre, home_elo_post, away_elo_post
    """
    rows: list[dict] = []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue

            elo_change = _safe_int(parts[9])
            home_elo_post = _safe_int(parts[10])
            away_elo_post = _safe_int(parts[11])

            if elo_change is None or home_elo_post is None or away_elo_post is None:
                continue

            year = _safe_int(parts[0])
            if year is None or year < ELO_MIN_YEAR:
                continue

            try:
                date = pd.Timestamp(year, int(parts[1]), int(parts[2]))
            except (ValueError, TypeError):
                logger.warning("%s line %d: invalid date, skipping", path.name, line_num)
                continue

            neutral_venue = parts[8].strip() if len(parts) > 8 else ""

            rows.append(
                {
                    "date": date,
                    "home_code": parts[3].strip(),
                    "away_code": parts[4].strip(),
                    "is_neutral_elo": neutral_venue != "",
                    "elo_change": elo_change,
                    "home_elo_post": home_elo_post,
                    "away_elo_post": away_elo_post,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["home_elo_pre"] = df["home_elo_post"] - df["elo_change"]
    df["away_elo_pre"] = df["away_elo_post"] + df["elo_change"]
    return df


def build_elo_match_table(elo_dir: Path) -> pd.DataFrame:
    """Parse all Elo TSVs into a deduplicated match-level table.

    Returns one row per unique (date, home_code, away_code) match.
    """
    tsv_files = sorted(elo_dir.glob("*.tsv"))
    if not tsv_files:
        logger.warning("No Elo TSV files found in %s", elo_dir)
        return pd.DataFrame()

    logger.info("Parsing %d Elo TSV files from %s", len(tsv_files), elo_dir)
    parts = []
    for tsv_path in tsv_files:
        df = parse_single_tsv(tsv_path)
        if not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    dedup = combined.drop_duplicates(subset=["date", "home_code", "away_code"], keep="first")
    dedup = dedup.sort_values("date").reset_index(drop=True)

    logger.info(
        "Elo match table: %d unique matches (from %d raw rows)",
        len(dedup),
        len(combined),
    )
    return dedup[
        [
            "date",
            "home_code",
            "away_code",
            "is_neutral_elo",
            "home_elo_pre",
            "away_elo_pre",
            "home_elo_post",
            "away_elo_post",
        ]
    ]


def _build_team_elo_series(elo_match_table: pd.DataFrame) -> pd.DataFrame:
    """Reshape match table into per-team (elo_code, date, elo_post) time series.

    The elo_post from a team's last game is their effective "current" Elo
    entering any subsequent game.
    """
    if elo_match_table.empty:
        return pd.DataFrame(columns=["elo_code", "date", "elo_post"])

    home = elo_match_table[["date", "home_code", "home_elo_post"]].rename(
        columns={"home_code": "elo_code", "home_elo_post": "elo_post"}
    )
    away = elo_match_table[["date", "away_code", "away_elo_post"]].rename(
        columns={"away_code": "elo_code", "away_elo_post": "elo_post"}
    )

    series = pd.concat([home, away], ignore_index=True)
    series = series.drop_duplicates(subset=["elo_code", "date"], keep="last")
    series = series.sort_values(["elo_code", "date"]).reset_index(drop=True)
    return series


def load_elo_code_map(
    mapping_path: Path,
) -> dict[str, list[tuple[pd.Timestamp | None, pd.Timestamp | None, str]]]:
    """Load elo_code_map.csv and return a time-range-aware code structure.

    Returns a dict mapping country_code to a list of
    (valid_from, valid_to, elo_code) tuples where valid_from and valid_to
    are pd.Timestamp or None (meaning open-ended).

    For teams with a single code (no valid_from/valid_to), returns one tuple
    with both bounds set to None.  For teams with historical renames (e.g. North
    Macedonia MK → NM), returns one tuple per period so the correct code can be
    resolved per match date.
    """
    df = pd.read_csv(mapping_path, keep_default_na=False, na_values=[""])
    df = df.dropna(subset=["elo_code"])
    df = df[df["elo_code"] != ""]

    result: dict[str, list[tuple[pd.Timestamp | None, pd.Timestamp | None, str]]] = {}
    for _, row in df.iterrows():
        cc = row["country_code"]
        code = row["elo_code"]
        vf_raw = row.get("valid_from")
        vt_raw = row.get("valid_to")
        vf: pd.Timestamp | None = pd.Timestamp(vf_raw) if vf_raw and pd.notna(vf_raw) else None
        vt: pd.Timestamp | None = pd.Timestamp(vt_raw) if vt_raw and pd.notna(vt_raw) else None
        result.setdefault(cc, []).append((vf, vt, code))

    return result


def _resolve_elo_code(
    country_code: str | None,
    match_date: pd.Timestamp,
    code_ranges: dict[str, list[tuple[pd.Timestamp | None, pd.Timestamp | None, str]]],
) -> str | None:
    """Return the elo_code active on match_date for a given country_code.

    Iterates through all (valid_from, valid_to, elo_code) ranges for the
    country and returns the first whose interval contains match_date.
    Returns None if the country is unknown or no interval matches.
    """
    if not country_code:
        return None
    ranges = code_ranges.get(country_code)
    if not ranges:
        logger.warning("No elo_code mapping found for country_code=%r on %s", country_code, match_date.date())
        return None
    for valid_from, valid_to, elo_code in ranges:
        from_ok = valid_from is None or match_date >= valid_from
        to_ok = valid_to is None or match_date <= valid_to
        if from_ok and to_ok:
            return elo_code
    return None


def join_elo_to_fixtures(
    fixtures: pd.DataFrame,
    elo_dir: Path,
    elo_code_map_path: Path,
) -> pd.DataFrame:
    """Join pre-match Elo values and neutral venue flags to a fixtures DataFrame.

    Expects fixtures to have: home_country_code, away_country_code, date_utc,
        competition_tier (for heuristic neutral fallback).

    Strategy:
      1. Exact join on (date, home_elo_code, away_elo_code) to get true pre-match
         Elo, post-match Elo, and neutral flag from Elo data.
      2. For unmatched fixtures, fall back to merge_asof on each team's Elo series
         (last elo_post before match date) for pre-match Elo estimate.
      3. Neutral flag falls back to tier-based heuristic when not from Elo.
    """
    fixtures = fixtures.copy()
    elo_match_table = build_elo_match_table(elo_dir)

    if elo_match_table.empty:
        logger.warning("Empty Elo match table; Elo columns will be NaN")
        fixtures["home_elo_pre"] = np.nan
        fixtures["away_elo_pre"] = np.nan
        fixtures["home_elo_post"] = np.nan
        fixtures["away_elo_post"] = np.nan
        fixtures["is_neutral"] = fixtures["competition_tier"].isin([1, 2])
        fixtures["neutral_source"] = "heuristic"
        return fixtures

    # Map country_code → elo_code (date-aware to handle historical renames)
    code_ranges = load_elo_code_map(elo_code_map_path)
    fixtures["_match_date"] = pd.to_datetime(fixtures["date_utc"])
    fixtures["_home_elo_code"] = [
        _resolve_elo_code(cc, dt, code_ranges)
        for cc, dt in zip(fixtures["home_country_code"], fixtures["_match_date"])
    ]
    fixtures["_away_elo_code"] = [
        _resolve_elo_code(cc, dt, code_ranges)
        for cc, dt in zip(fixtures["away_country_code"], fixtures["_match_date"])
    ]

    # --- Step 1: Exact join for matches present in Elo data ---
    elo_for_join = elo_match_table.rename(
        columns={
            "date": "_match_date",
            "home_code": "_home_elo_code",
            "away_code": "_away_elo_code",
        }
    )

    merged = fixtures.merge(
        elo_for_join,
        on=["_match_date", "_home_elo_code", "_away_elo_code"],
        how="left",
        indicator="_elo_match",
    )

    exact_mask = merged["_elo_match"] == "both"
    n_exact = exact_mask.sum()
    logger.info(
        "Elo exact match: %d / %d fixtures (%.1f%%)",
        n_exact,
        len(merged),
        100 * n_exact / max(len(merged), 1),
    )

    # Store exact-match results
    fixtures["home_elo_pre"] = merged["home_elo_pre"].where(exact_mask, np.nan).values
    fixtures["away_elo_pre"] = merged["away_elo_pre"].where(exact_mask, np.nan).values
    fixtures["home_elo_post"] = merged["home_elo_post"].where(exact_mask, np.nan).values
    fixtures["away_elo_post"] = merged["away_elo_post"].where(exact_mask, np.nan).values
    fixtures["_elo_neutral"] = merged["is_neutral_elo"].where(exact_mask, pd.NA).values
    fixtures["_exact_matched"] = exact_mask.values

    # --- Step 2: Fallback via merge_asof for unmatched fixtures ---
    needs_fallback = ~fixtures["_exact_matched"]
    if needs_fallback.any():
        team_series = _build_team_elo_series(elo_match_table)
        _asof_fill_elo(fixtures, team_series, needs_fallback)

    # --- Step 3: Neutral flag ---
    heuristic_neutral = fixtures["competition_tier"].isin([1, 2])
    fixtures["is_neutral"] = fixtures["_elo_neutral"].where(
        fixtures["_exact_matched"], heuristic_neutral
    ).astype("boolean")
    fixtures["neutral_source"] = np.where(fixtures["_exact_matched"], "elo", "heuristic")

    # Clean up
    drop_cols = [c for c in fixtures.columns if c.startswith("_")]
    fixtures.drop(columns=drop_cols, inplace=True)

    return fixtures


def _asof_fill_elo(
    fixtures: pd.DataFrame,
    team_series: pd.DataFrame,
    mask: pd.Series,
) -> None:
    """Fill home_elo_pre / away_elo_pre for fixtures[mask] using merge_asof.

    Uses elo_post from the team's most recent prior match as the pre-match Elo.
    Modifies fixtures in place.
    """
    subset_idx = fixtures.index[mask]
    subset = fixtures.loc[subset_idx].copy()
    if subset.empty:
        return

    for side in ("home", "away"):
        elo_code_col = f"_{side}_elo_code"
        elo_pre_col = f"{side}_elo_pre"

        side_df = subset[["_match_date", elo_code_col]].copy()
        side_df = side_df.rename(columns={elo_code_col: "elo_code", "_match_date": "date"})
        side_df = side_df.sort_values("date")

        ts = team_series.sort_values("date")

        joined = pd.merge_asof(
            side_df.reset_index(),
            ts,
            on="date",
            by="elo_code",
            direction="backward",
            allow_exact_matches=False,
        ).set_index("index")

        # elo_post from the last prior game is the effective pre-match Elo.
        # Only fill where currently NaN.
        current = fixtures.loc[subset_idx, elo_pre_col]
        fallback = joined["elo_post"].reindex(subset_idx)
        fixtures.loc[subset_idx, elo_pre_col] = current.where(current.notna(), fallback)

    n_filled = mask.sum() - fixtures.loc[subset_idx, "home_elo_pre"].isna().sum()
    logger.info("Elo asof fallback filled %d / %d unmatched fixtures", n_filled, mask.sum())
