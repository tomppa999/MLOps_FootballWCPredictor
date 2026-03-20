from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ELO_DIR = PROJECT_ROOT / "data" / "raw" / "elo"
OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_FILE = OUTPUT_DIR / "elo_code_profile.csv"

TSV_COLUMNS = [
    "year",
    "month",
    "day",
    "home_code",
    "away_code",
    "home_score",
    "away_score",
    "tournament_code",
    "venue_country_code",
    "rating_change_home",
    "elo_home_post",
    "elo_away_post",
    "rank_change_home",
    "rank_change_away",
    "rank_home_post",
    "rank_away_post",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2018-01-01",
        help="Only matches on/after this date are used for active profiling.",
    )
    return parser.parse_args()


def load_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=TSV_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )

    if df.shape[1] != len(TSV_COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(TSV_COLUMNS)} columns, got {df.shape[1]}"
        )

    for col in ["year", "month", "day"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["match_date"] = pd.to_datetime(
        df[["year", "month", "day"]],
        errors="coerce",
    )

    return df


def summarize_codes(df: pd.DataFrame) -> pd.Series:
    codes = pd.concat([df["home_code"], df["away_code"]], ignore_index=True)
    codes = (
        codes.astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )

    if codes.empty:
        return pd.Series(dtype="int64")

    return codes.value_counts()


def choose_primary_code(code_counts: pd.Series, row_count: int) -> tuple[str, bool]:
    """
    Returns:
        primary_code
        multiple_codes_flag

    Logic:
    - self-code should usually appear close to once per row
    - if top two codes both cover a meaningful share of rows, flag multiple
    """
    if code_counts.empty or row_count == 0:
        return "", False

    primary_code = str(code_counts.index[0])

    if len(code_counts) == 1:
        return primary_code, False

    top1 = int(code_counts.iloc[0])
    top2 = int(code_counts.iloc[1])

    top1_share = top1 / row_count
    top2_share = top2 / row_count

    multiple_codes_flag = (
        top1_share >= 0.30
        and top2_share >= 0.30
        and top2 >= max(3, int(0.15 * row_count))
    )

    return primary_code, multiple_codes_flag


def profile_file(path: Path, cutoff: pd.Timestamp) -> dict:
    slug = path.stem
    df = load_tsv(path)

    row_count_total = len(df)
    df_valid_dates = df[df["match_date"].notna()].copy()

    last_match_date = (
        df_valid_dates["match_date"].max().date().isoformat()
        if not df_valid_dates.empty
        else ""
    )

    df_since = df_valid_dates[df_valid_dates["match_date"] >= cutoff].copy()
    row_count_since_cutoff = len(df_since)

    if row_count_since_cutoff == 0:
        return {
            "elo_slug": slug,
            "row_count_total": row_count_total,
            "row_count_since_cutoff": 0,
            "last_match_date": last_match_date,
            "active_since_cutoff": False,
            "candidate_codes_since_cutoff": "",
            "candidate_code_counts_since_cutoff": "",
            "primary_elo_code": "",
            "primary_elo_code_count": 0,
            "primary_elo_code_share_of_rows": 0.0,
            "multiple_codes_flag": False,
            "review_needed": True,
            "notes": "No matches since cutoff",
        }

    code_counts = summarize_codes(df_since)
    primary_code, multiple_codes_flag = choose_primary_code(
        code_counts=code_counts,
        row_count=row_count_since_cutoff,
    )

    primary_count = int(code_counts.iloc[0]) if not code_counts.empty else 0
    primary_share = (
        round(primary_count / row_count_since_cutoff, 4)
        if row_count_since_cutoff > 0
        else 0.0
    )

    candidate_codes = list(map(str, code_counts.index.tolist()))
    candidate_code_counts = {str(k): int(v) for k, v in code_counts.to_dict().items()}

    review_needed = primary_share < 1.0

    notes = []
    if multiple_codes_flag:
        notes.append("Multiple likely self-codes since cutoff")
    if primary_share < 0.8:
        notes.append("Primary code not dominant enough")
    if not notes:
        notes.append("Looks clean")

    return {
        "elo_slug": slug,
        "row_count_total": row_count_total,
        "row_count_since_cutoff": row_count_since_cutoff,
        "last_match_date": last_match_date,
        "active_since_cutoff": True,
        "candidate_codes_since_cutoff": "|".join(candidate_codes),
        "candidate_code_counts_since_cutoff": json.dumps(
            candidate_code_counts,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "primary_elo_code": primary_code,
        "primary_elo_code_count": primary_count,
        "primary_elo_code_share_of_rows": primary_share,
        "multiple_codes_flag": multiple_codes_flag,
        "review_needed": review_needed,
        "notes": "; ".join(notes),
    }


def main() -> None:
    args = parse_args()
    cutoff = pd.Timestamp(args.cutoff)

    if not ELO_DIR.exists():
        raise FileNotFoundError(f"Elo directory not found: {ELO_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tsv_files = sorted(ELO_DIR.glob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found in: {ELO_DIR}")

    rows = []
    for path in tsv_files:
        try:
            rows.append(profile_file(path, cutoff=cutoff))
        except Exception as e:
            rows.append(
                {
                    "elo_slug": path.stem,
                    "row_count_total": "",
                    "row_count_since_cutoff": "",
                    "last_match_date": "",
                    "active_since_cutoff": "",
                    "candidate_codes_since_cutoff": "",
                    "candidate_code_counts_since_cutoff": "",
                    "primary_elo_code": "",
                    "primary_elo_code_count": "",
                    "primary_elo_code_share_of_rows": "",
                    "multiple_codes_flag": "",
                    "review_needed": True,
                    "notes": f"ERROR: {e}",
                }
            )

    out = pd.DataFrame(rows).sort_values(
        by=["review_needed", "active_since_cutoff", "elo_slug"],
        ascending=[False, False, True],
    )

    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote: {OUTPUT_FILE}")
    print(f"Rows: {len(out)}")
    print(f"Needs review: {int(out['review_needed'].fillna(True).sum())}")


if __name__ == "__main__":
    main()