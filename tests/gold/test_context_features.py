"""Tests for src.gold.context_features."""

import numpy as np
import pandas as pd
import pytest

from src.gold.context_features import add_elo_diff, override_neutral_for_2026_hosts


class TestEloDiff:
    def test_basic_diff(self):
        df = pd.DataFrame({
            "home_elo_pre": [1800.0, 1500.0],
            "away_elo_pre": [1600.0, 1700.0],
        })
        result = add_elo_diff(df)
        assert result["elo_diff"].tolist() == pytest.approx([200.0, -200.0])

    def test_nan_propagates(self):
        df = pd.DataFrame({
            "home_elo_pre": [np.nan],
            "away_elo_pre": [1600.0],
        })
        result = add_elo_diff(df)
        assert pd.isna(result["elo_diff"].iloc[0])

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({
            "home_elo_pre": [1800.0],
            "away_elo_pre": [1600.0],
        })
        _ = add_elo_diff(df)
        assert "elo_diff" not in df.columns


class TestNeutralOverride2026:
    def _make_df(
        self,
        season: int = 2026,
        competition_tier: int = 1,
        is_knockout: bool = False,
        home_team: str = "United States",
        is_neutral: bool = True,
    ) -> pd.DataFrame:
        return pd.DataFrame({
            "season": [season],
            "competition_tier": [competition_tier],
            "is_knockout": [is_knockout],
            "home_team": [home_team],
            "is_neutral": [is_neutral],
        })

    def test_usa_group_stage_overridden(self):
        result = override_neutral_for_2026_hosts(self._make_df(home_team="United States"))
        assert result["is_neutral"].iloc[0] == False  # noqa: E712

    def test_canada_group_stage_overridden(self):
        result = override_neutral_for_2026_hosts(self._make_df(home_team="Canada"))
        assert result["is_neutral"].iloc[0] == False  # noqa: E712

    def test_mexico_group_stage_overridden(self):
        result = override_neutral_for_2026_hosts(self._make_df(home_team="Mexico"))
        assert result["is_neutral"].iloc[0] == False  # noqa: E712

    def test_non_host_stays_neutral(self):
        result = override_neutral_for_2026_hosts(self._make_df(home_team="Germany"))
        assert result["is_neutral"].iloc[0] == True  # noqa: E712

    def test_knockout_stays_neutral(self):
        result = override_neutral_for_2026_hosts(self._make_df(is_knockout=True))
        assert result["is_neutral"].iloc[0] == True  # noqa: E712

    def test_other_season_unchanged(self):
        result = override_neutral_for_2026_hosts(self._make_df(season=2022))
        assert result["is_neutral"].iloc[0] == True  # noqa: E712

    def test_non_wc_tier_unchanged(self):
        result = override_neutral_for_2026_hosts(self._make_df(competition_tier=3))
        assert result["is_neutral"].iloc[0] == True  # noqa: E712

    def test_already_not_neutral_stays_false(self):
        result = override_neutral_for_2026_hosts(
            self._make_df(home_team="United States", is_neutral=False)
        )
        assert result["is_neutral"].iloc[0] == False  # noqa: E712

    def test_does_not_mutate_input(self):
        df = self._make_df()
        original_val = df["is_neutral"].iloc[0]
        _ = override_neutral_for_2026_hosts(df)
        assert df["is_neutral"].iloc[0] == original_val
