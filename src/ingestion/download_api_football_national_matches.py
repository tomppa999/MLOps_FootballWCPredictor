#!/usr/bin/env python3
"""Download API-Football national-team fixtures and statistics.

Uses the /leagues endpoint to discover which seasons actually exist for each
competition, then fetches fixtures only for those (league, season) pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from src.ingestion.api_football_client import ApiFootballClient


DEFAULT_OUTPUT_DIR = Path("data/raw/api_football")
DEFAULT_TOURNAMENTS_CSV = Path("data/mappings/api_football_tournaments.csv")
DEFAULT_MIN_SEASON = 2018


@dataclass(frozen=True)
class DateWindow:
    start_date: date
    end_date: date

    @property
    def label(self) -> str:
        return f"{self.start_date.isoformat()}_{self.end_date.isoformat()}"


@dataclass(frozen=True)
class SeasonInfo:
    year: int
    start: str
    end: str
    has_fixture_statistics: bool


@dataclass
class CompetitionInfo:
    league_id: int
    league_name: str
    country: Optional[str]
    seasons: List[SeasonInfo] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download API-Football national-team fixtures and statistics using curated competition IDs."
    )
    parser.add_argument("--start-date", type=str, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="End date in YYYY-MM-DD")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=None,
        help="Optional incremental mode. Pull from today-lookback_days through today (UTC).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory for raw API-Football files.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://v3.football.api-sports.io",
        help="API base URL.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help="Sleep between API calls.",
    )
    parser.add_argument(
        "--tournaments-csv",
        type=Path,
        default=DEFAULT_TOURNAMENTS_CSV,
        help="Curated CSV of national-team competitions.",
    )
    parser.add_argument(
        "--min-season",
        type=int,
        default=DEFAULT_MIN_SEASON,
        help="Earliest season year to include (filters discovered seasons).",
    )
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_requested_window(
    start_date_str: Optional[str],
    end_date_str: Optional[str],
    lookback_days: Optional[int],
) -> DateWindow:
    has_explicit_window = start_date_str is not None or end_date_str is not None
    has_lookback = lookback_days is not None

    if has_explicit_window and has_lookback:
        raise ValueError("Use either --start-date/--end-date or --lookback-days, not both.")

    if has_lookback:
        if lookback_days < 0:
            raise ValueError("--lookback-days must be >= 0.")
        end_date = utc_now().date()
        start_date = end_date - timedelta(days=lookback_days)
        return DateWindow(start_date=start_date, end_date=end_date)

    if not start_date_str or not end_date_str:
        raise ValueError("Provide both --start-date and --end-date, or use --lookback-days.")

    start_date = parse_iso_date(start_date_str)
    end_date = parse_iso_date(end_date_str)

    if end_date < start_date:
        raise ValueError("--end-date must be >= --start-date.")

    return DateWindow(start_date=start_date, end_date=end_date)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def extract_fixture_id(fixture_item: Dict[str, Any]) -> Optional[int]:
    return fixture_item.get("fixture", {}).get("id")


def extract_fixture_date(fixture_item: Dict[str, Any]) -> Optional[date]:
    raw_dt = fixture_item.get("fixture", {}).get("date")
    if not raw_dt:
        return None
    try:
        return datetime.fromisoformat(raw_dt.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def fixture_in_window(fixture_item: Dict[str, Any], window: DateWindow) -> bool:
    fixture_date = extract_fixture_date(fixture_item)
    if fixture_date is None:
        return False
    return window.start_date <= fixture_date <= window.end_date


# ---------------------------------------------------------------------------
# CSV loading -- only league_id and name are required
# ---------------------------------------------------------------------------

def load_competitions(tournaments_csv: Path) -> List[CompetitionInfo]:
    if not tournaments_csv.exists():
        raise FileNotFoundError(f"Tournaments CSV not found: {tournaments_csv}")

    df = pd.read_csv(tournaments_csv, sep=";")
    df = df.dropna(how="all")

    id_col = "league_id" if "league_id" in df.columns else "ID (V3)"
    name_col = "name" if "name" in df.columns else "Name"

    df[name_col] = df[name_col].astype("string").str.strip()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")

    df = df[df[name_col].notna() & df[name_col].ne("")]
    df = df[df[id_col].notna()]
    df[id_col] = df[id_col].astype(int)
    df = df.drop_duplicates(subset=[id_col]).reset_index(drop=True)

    country_col = "country" if "country" in df.columns else "Country"
    competitions: List[CompetitionInfo] = []
    for _, row in df.iterrows():
        country = None
        if country_col in row.index and pd.notna(row.get(country_col)):
            country = str(row[country_col])
        competitions.append(
            CompetitionInfo(
                league_id=int(row[id_col]),
                league_name=str(row[name_col]),
                country=country,
            )
        )
    return competitions


# ---------------------------------------------------------------------------
# Season discovery via /leagues
# ---------------------------------------------------------------------------

def discover_seasons(
    client: ApiFootballClient,
    competition: CompetitionInfo,
    min_season: int,
) -> List[SeasonInfo]:
    """Call /leagues?id=X and return the available seasons >= min_season."""
    payload = client.get_league(competition.league_id)
    entries = payload.get("response", [])
    if not entries:
        print(
            f"[DISCOVER] league_id={competition.league_id} | "
            f"league_name={competition.league_name} | no data from /leagues"
        )
        return []

    raw_seasons = entries[0].get("seasons", [])
    seasons: List[SeasonInfo] = []
    for s in raw_seasons:
        year = s.get("year")
        if year is None or year < min_season:
            continue
        coverage = s.get("coverage", {}).get("fixtures", {})
        seasons.append(
            SeasonInfo(
                year=year,
                start=s.get("start", ""),
                end=s.get("end", ""),
                has_fixture_statistics=bool(coverage.get("statistics_fixtures", False)),
            )
        )

    season_years = [s.year for s in seasons]
    print(
        f"[DISCOVER] league_id={competition.league_id} | "
        f"league_name={competition.league_name} | "
        f"seasons={season_years}"
    )
    return seasons


# ---------------------------------------------------------------------------
# Fixture fetching
# ---------------------------------------------------------------------------

def fetch_all_fixtures_for_league_season(
    client: ApiFootballClient,
    league_id: int,
    season: int,
    league_name: str,
) -> List[Dict[str, Any]]:
    payload = client.get_fixtures_by_league_and_season(
        league_id=league_id,
        season=season,
    )
    fixtures = list(payload.get("response", []))
    print(
        f"[FETCH] league_id={league_id} | league_name={league_name} | "
        f"season={season} | fixtures={len(fixtures)}"
    )
    return fixtures


# ---------------------------------------------------------------------------
# Statistics fetching
# ---------------------------------------------------------------------------

def save_fixture_statistics(
    client: ApiFootballClient,
    output_dir: Path,
    window_label: str,
    fixture_ids: List[int],
) -> tuple[int, List[int]]:
    stats_dir = output_dir / "statistics" / window_label
    ensure_dir(stats_dir)

    saved = 0
    missing_or_failed: List[int] = []

    for idx, fixture_id in enumerate(fixture_ids, start=1):
        out_path = stats_dir / f"{fixture_id}.json"
        try:
            payload = client.get_fixture_statistics(fixture_id)
            write_json(out_path, payload)
            saved += 1

            if idx % 25 == 0 or idx == len(fixture_ids):
                print(f"[STATS] progress={idx}/{len(fixture_ids)} | saved={saved}")
        except Exception as exc:
            missing_or_failed.append(fixture_id)
            print(f"[STATS-ERROR] fixture_id={fixture_id} | error={exc}")

    return saved, missing_or_failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key:
        print("Missing required API_FOOTBALL_KEY in environment or .env", file=sys.stderr)
        return 1

    try:
        requested_window = build_requested_window(
            start_date_str=args.start_date,
            end_date_str=args.end_date,
            lookback_days=args.lookback_days,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        competitions = load_competitions(args.tournaments_csv)
    except Exception as exc:
        print(f"Failed to load tournaments CSV: {exc}", file=sys.stderr)
        return 1

    print(
        f"[START] competitions={len(competitions)} | "
        f"min_season={args.min_season} | "
        f"window={requested_window.start_date} to {requested_window.end_date}"
    )

    client = ApiFootballClient(
        api_key=api_key,
        base_url=args.base_url,
        sleep_seconds=args.sleep_seconds,
    )

    ensure_dir(args.output_dir)
    ensure_dir(args.output_dir / "fixtures")
    ensure_dir(args.output_dir / "statistics")
    ensure_dir(args.output_dir / "runs")

    run_started_at = utc_now()
    run_id = run_started_at.strftime("%Y-%m-%dT%H-%M-%SZ")
    window_label = requested_window.label

    # --- Phase 1: discover seasons via /leagues ---
    print("[PHASE-1] Discovering available seasons per competition ...")
    total_pairs = 0
    for comp in competitions:
        comp.seasons = discover_seasons(client, comp, args.min_season)
        total_pairs += len(comp.seasons)
    print(f"[PHASE-1] Done. Total (league, season) pairs to fetch: {total_pairs}")

    # --- Phase 2: fetch fixtures for each discovered pair ---
    print("[PHASE-2] Fetching fixtures ...")
    kept_fixtures_by_id: Dict[int, Dict[str, Any]] = {}
    stats_eligible_fixture_ids: List[int] = []
    attempted_league_season_pairs: List[Dict[str, Any]] = []
    failed_league_season_pairs: List[Dict[str, Any]] = []
    total_raw_fixtures_seen = 0

    for comp in competitions:
        if not comp.seasons:
            continue

        for season_info in comp.seasons:
            pair_record: Dict[str, Any] = {
                "league_id": comp.league_id,
                "league_name": comp.league_name,
                "season": season_info.year,
                "has_fixture_statistics": season_info.has_fixture_statistics,
            }

            try:
                raw_fixtures = fetch_all_fixtures_for_league_season(
                    client=client,
                    league_id=comp.league_id,
                    season=season_info.year,
                    league_name=comp.league_name,
                )
                pair_record["raw_fixture_count"] = len(raw_fixtures)
                attempted_league_season_pairs.append(pair_record)
                total_raw_fixtures_seen += len(raw_fixtures)

                kept_for_this_pair = 0

                for fixture_item in raw_fixtures:
                    if not fixture_in_window(fixture_item, requested_window):
                        continue

                    kept_for_this_pair += 1
                    fixture_id = extract_fixture_id(fixture_item)
                    if fixture_id is None:
                        continue

                    kept_fixtures_by_id[fixture_id] = fixture_item
                    if season_info.has_fixture_statistics:
                        stats_eligible_fixture_ids.append(fixture_id)

                print(
                    f"[FILTER] league_id={comp.league_id} | season={season_info.year} | "
                    f"raw={len(raw_fixtures)} | in_window={kept_for_this_pair} | "
                    f"stats_available={season_info.has_fixture_statistics}"
                )

            except Exception as exc:
                pair_record["error"] = str(exc)
                failed_league_season_pairs.append(pair_record)
                print(
                    f"[ERROR] league_id={comp.league_id} | league_name={comp.league_name} | "
                    f"season={season_info.year} | error={exc}"
                )

    kept_fixtures = sorted(
        kept_fixtures_by_id.values(),
        key=lambda x: (
            x.get("fixture", {}).get("date", ""),
            x.get("fixture", {}).get("id", 0),
        ),
    )
    kept_fixture_ids = [
        fid for fid in (extract_fixture_id(x) for x in kept_fixtures) if fid is not None
    ]

    # --- Phase 3: fetch statistics (only for coverage-eligible fixtures) ---
    eligible_set = set(stats_eligible_fixture_ids)
    ids_to_fetch_stats = [fid for fid in kept_fixture_ids if fid in eligible_set]

    print(
        f"[PHASE-3] Requesting statistics for {len(ids_to_fetch_stats)} fixtures "
        f"(skipped {len(kept_fixture_ids) - len(ids_to_fetch_stats)} without coverage)"
    )

    stats_saved, stats_failed_fixture_ids = save_fixture_statistics(
        client=client,
        output_dir=args.output_dir,
        window_label=window_label,
        fixture_ids=ids_to_fetch_stats,
    )

    print(f"[STATS] saved={stats_saved} | failed={len(stats_failed_fixture_ids)}")
    print(f"[WRITE] writing fixtures for window {window_label}")

    # --- Write outputs ---
    fixtures_dir = args.output_dir / "fixtures" / window_label
    ensure_dir(fixtures_dir)

    fixtures_payload = {
        "requested_window": {
            "start_date": requested_window.start_date.isoformat(),
            "end_date": requested_window.end_date.isoformat(),
            "label": requested_window.label,
        },
        "response": kept_fixtures,
    }
    write_json(fixtures_dir / "fixtures.json", fixtures_payload)

    discovered_seasons_summary = {
        comp.league_name: [s.year for s in comp.seasons] for comp in competitions
    }

    chunk_manifest = {
        "run_id": run_id,
        "source": "api_football",
        "window": {
            "start_date": requested_window.start_date.isoformat(),
            "end_date": requested_window.end_date.isoformat(),
            "label": requested_window.label,
        },
        "tournaments_csv": str(args.tournaments_csv),
        "min_season": args.min_season,
        "discovered_seasons": discovered_seasons_summary,
        "competition_count": len(competitions),
        "league_season_pairs_attempted": len(attempted_league_season_pairs),
        "league_season_pairs_failed": len(failed_league_season_pairs),
        "raw_fixture_count_total": total_raw_fixtures_seen,
        "kept_fixture_count": len(kept_fixtures),
        "statistics_eligible": len(ids_to_fetch_stats),
        "statistics_saved": stats_saved,
        "statistics_failed_fixture_ids": stats_failed_fixture_ids,
        "paths": {
            "fixtures_file": str(fixtures_dir / "fixtures.json"),
            "statistics_dir": str(args.output_dir / "statistics" / window_label),
        },
        "failed_league_season_pairs": failed_league_season_pairs,
    }
    write_json(fixtures_dir / "manifest.json", chunk_manifest)

    run_manifest = {
        "run_id": run_id,
        "source": "api_football",
        "requested_window": {
            "start_date": requested_window.start_date.isoformat(),
            "end_date": requested_window.end_date.isoformat(),
            "label": requested_window.label,
        },
        "started_at_utc": run_started_at.isoformat(),
        "finished_at_utc": utc_now().isoformat(),
        "tournaments_csv": str(args.tournaments_csv),
        "min_season": args.min_season,
        "summary": {
            "competition_count": len(competitions),
            "league_season_pairs_discovered": total_pairs,
            "league_season_pairs_attempted": len(attempted_league_season_pairs),
            "league_season_pairs_failed": len(failed_league_season_pairs),
            "raw_fixture_count_total": total_raw_fixtures_seen,
            "kept_fixture_count_total": len(kept_fixtures),
            "statistics_eligible": len(ids_to_fetch_stats),
            "statistics_saved_total": stats_saved,
            "statistics_failed_total": len(stats_failed_fixture_ids),
        },
    }

    run_manifest_path = args.output_dir / "runs" / f"{run_id}_run_manifest.json"
    write_json(run_manifest_path, run_manifest)

    print(json.dumps(run_manifest["summary"], indent=2))
    print(f"Fixtures written to: {fixtures_dir / 'fixtures.json'}")
    print(f"Run manifest written to: {run_manifest_path}")
    print("[FINISHED] api-football ingestion completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
