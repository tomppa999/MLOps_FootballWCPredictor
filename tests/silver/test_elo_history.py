"""Tests for src.silver.elo_history."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.silver.elo_history import (
    _build_team_elo_series,
    _resolve_elo_code,
    build_elo_match_table,
    load_elo_code_map,
    parse_single_tsv,
)

SAMPLES_DIR = Path("data_samples/raw/elo")


class TestParseSingleTsv:
    def test_argentina_sample_parses(self):
        df = parse_single_tsv(SAMPLES_DIR / "Argentina.tsv")
        assert not df.empty
        assert "date" in df.columns
        assert "home_elo_pre" in df.columns
        assert "away_elo_pre" in df.columns

    def test_pre_match_elo_derivation(self):
        """Verify: home_elo_pre = home_elo_post - elo_change."""
        df = parse_single_tsv(SAMPLES_DIR / "Argentina.tsv")
        np.testing.assert_array_equal(
            df["home_elo_pre"].values,
            (df["home_elo_post"] - df["elo_change"]).values,
        )
        np.testing.assert_array_equal(
            df["away_elo_pre"].values,
            (df["away_elo_post"] + df["elo_change"]).values,
        )

    def test_consecutive_matches_elo_continuity(self):
        """For Argentina, the post-match Elo of match N should equal the
        pre-match Elo of match N+1 (regardless of home/away side)."""
        df = parse_single_tsv(SAMPLES_DIR / "Argentina.tsv")
        # Build AR's Elo timeline: extract their post/pre per match
        ar_rows = []
        for _, row in df.iterrows():
            if row["home_code"] == "AR":
                ar_rows.append((row["date"], row["home_elo_pre"], row["home_elo_post"]))
            elif row["away_code"] == "AR":
                ar_rows.append((row["date"], row["away_elo_pre"], row["away_elo_post"]))
        assert len(ar_rows) > 5
        for i in range(1, min(10, len(ar_rows))):
            prev_post = ar_rows[i - 1][2]
            curr_pre = ar_rows[i][1]
            assert prev_post == curr_pre, (
                f"Match {i}: prev post={prev_post} != curr pre={curr_pre}"
            )

    def test_neutral_venue_detection(self):
        df = parse_single_tsv(SAMPLES_DIR / "Argentina.tsv")
        # WC 2022 matches in Qatar should be marked neutral
        wc_2022 = df[(df["date"] >= "2022-11-01") & (df["date"] <= "2022-12-31")]
        assert wc_2022["is_neutral_elo"].any()

    def test_croatia_sample_parses(self):
        df = parse_single_tsv(SAMPLES_DIR / "Croatia.tsv")
        assert not df.empty
        assert len(df) >= 100


class TestBuildEloMatchTable:
    def test_deduplicates_across_files(self):
        table = build_elo_match_table(SAMPLES_DIR)
        # AR vs HR and HR vs AR matches should appear only once
        dupes = table.duplicated(subset=["date", "home_code", "away_code"])
        assert not dupes.any()

    def test_sorted_by_date(self):
        table = build_elo_match_table(SAMPLES_DIR)
        assert table["date"].is_monotonic_increasing


class TestBuildTeamEloSeries:
    def test_series_structure(self):
        table = build_elo_match_table(SAMPLES_DIR)
        series = _build_team_elo_series(table)
        assert "elo_code" in series.columns
        assert "date" in series.columns
        assert "elo_post" in series.columns

    def test_argentina_in_series(self):
        table = build_elo_match_table(SAMPLES_DIR)
        series = _build_team_elo_series(table)
        ar = series[series["elo_code"] == "AR"]
        assert not ar.empty


class TestLoadEloCodeMap:
    def test_loads_and_returns_dict(self):
        code_ranges = load_elo_code_map(Path("data/mappings/elo_code_map.csv"))
        assert isinstance(code_ranges, dict)
        # Single-code teams resolve to their code at any date
        ref = pd.Timestamp("2022-01-01")
        assert _resolve_elo_code("ARG", ref, code_ranges) == "AR"
        assert _resolve_elo_code("HRV", ref, code_ranges) == "HR"
        assert _resolve_elo_code("FRA", ref, code_ranges) == "FR"

    def test_north_macedonia_historical_code(self):
        """MKD should resolve to MK before 2019-01-01 and NM from 2019-01-01."""
        code_ranges = load_elo_code_map(Path("data/mappings/elo_code_map.csv"))
        assert _resolve_elo_code("MKD", pd.Timestamp("2018-06-01"), code_ranges) == "MK"
        assert _resolve_elo_code("MKD", pd.Timestamp("2019-01-01"), code_ranges) == "NM"
        assert _resolve_elo_code("MKD", pd.Timestamp("2022-01-01"), code_ranges) == "NM"

    def test_unknown_country_returns_none(self):
        code_ranges = load_elo_code_map(Path("data/mappings/elo_code_map.csv"))
        assert _resolve_elo_code("ZZZ", pd.Timestamp("2022-01-01"), code_ranges) is None
        assert _resolve_elo_code(None, pd.Timestamp("2022-01-01"), code_ranges) is None
