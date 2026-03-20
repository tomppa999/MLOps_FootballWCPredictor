from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEAM_MAPPING_FILE = PROJECT_ROOT / "data" / "mappings" / "team_mapping_master_merged.csv"
PROFILE_FILE = PROJECT_ROOT / "data" / "interim" / "elo_code_profile.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "mappings" / "elo_code_map.csv"


SPECIAL_CASES = [
    {
        "elo_slug": "North_Macedonia",
        "elo_code": "MK",
        "valid_from": "",
        "valid_to": "2018-12-31",
        "notes": "Historical code before 2019-01",
    },
    {
        "elo_slug": "North_Macedonia",
        "elo_code": "NM",
        "valid_from": "2019-01-01",
        "valid_to": "",
        "notes": "Code used since 2019-01",
    },
    {
        "elo_slug": "Eswatini",
        "elo_code": "SZ",
        "valid_from": "",
        "valid_to": "2018-03-31",
        "notes": "Historical code until 2018-03",
    },
    {
        "elo_slug": "Eswatini",
        "elo_code": "SW",
        "valid_from": "2018-04-01",
        "valid_to": "",
        "notes": "Code used since 2018-04",
    },
]


def main() -> None:
    team_mapping = pd.read_csv(TEAM_MAPPING_FILE, sep=None, engine="python")
    profile = pd.read_csv(PROFILE_FILE)

    base = team_mapping[
        ["elo_slug", "canonical_team_name", "country_code"]
    ].drop_duplicates()

    profile_base = profile[["elo_slug", "primary_elo_code"]].copy()
    profile_base = profile_base.rename(columns={"primary_elo_code": "elo_code"})
    profile_base["valid_from"] = ""
    profile_base["valid_to"] = ""
    profile_base["notes"] = "Primary code from elo_code_profile.csv"

    # remove rows that are handled via explicit special cases
    special_slugs = {row["elo_slug"] for row in SPECIAL_CASES}
    profile_base = profile_base[~profile_base["elo_slug"].isin(special_slugs)]

    special_df = pd.DataFrame(SPECIAL_CASES)

    elo_code_map = pd.concat([profile_base, special_df], ignore_index=True)

    elo_code_map = base.merge(elo_code_map, on="elo_slug", how="inner")

    elo_code_map = elo_code_map[
        [
            "elo_slug",
            "elo_code",
            "canonical_team_name",
            "country_code",
            "valid_from",
            "valid_to",
            "notes",
        ]
    ].sort_values(["elo_slug", "valid_from", "elo_code"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    elo_code_map.to_csv(OUTPUT_FILE, index=False)

    print(f"Wrote {len(elo_code_map)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()