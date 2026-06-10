"""Tests for src.inference.export."""

import pandas as pd
import pytest

from src.inference.export import (
    _OVER_UNDER_THRESHOLDS,
    _compute_markets,
    expected_betting_columns,
)


def _sample_scoreline_df() -> pd.DataFrame:
    """Small distribution for France vs Germany that sums to 1.0."""
    return pd.DataFrame(
        [
            {
                "stage": "Group",
                "home_team": "France",
                "away_team": "Germany",
                "home_goals": 1,
                "away_goals": 0,
                "probability": 0.30,
            },
            {
                "stage": "Group",
                "home_team": "France",
                "away_team": "Germany",
                "home_goals": 1,
                "away_goals": 1,
                "probability": 0.25,
            },
            {
                "stage": "Group",
                "home_team": "France",
                "away_team": "Germany",
                "home_goals": 0,
                "away_goals": 0,
                "probability": 0.20,
            },
            {
                "stage": "Group",
                "home_team": "France",
                "away_team": "Germany",
                "home_goals": 2,
                "away_goals": 1,
                "probability": 0.15,
            },
            {
                "stage": "Group",
                "home_team": "France",
                "away_team": "Germany",
                "home_goals": 0,
                "away_goals": 2,
                "probability": 0.10,
            },
        ]
    )


def test_wide_schema():
    out = _compute_markets(_sample_scoreline_df(), top_n=3)
    assert list(out.columns) == expected_betting_columns(3)
    assert len(out) == 1
    assert out.iloc[0]["home_team"] == "France"
    assert out.iloc[0]["away_team"] == "Germany"


def test_wdl_sum_to_one():
    out = _compute_markets(_sample_scoreline_df(), top_n=3)
    row = out.iloc[0]
    wdl_sum = row["p_home_win"] + row["p_draw"] + row["p_away_win"]
    assert wdl_sum == pytest.approx(1.0, abs=1e-4)
    assert row["p_home_or_draw"] == pytest.approx(row["p_home_win"] + row["p_draw"], abs=1e-4)
    assert row["p_home_or_away"] == pytest.approx(row["p_home_win"] + row["p_away_win"], abs=1e-4)
    assert row["p_draw_or_away"] == pytest.approx(row["p_draw"] + row["p_away_win"], abs=1e-4)


def test_over_under_complement():
    out = _compute_markets(_sample_scoreline_df(), top_n=3)
    row = out.iloc[0]
    for threshold in _OVER_UNDER_THRESHOLDS:
        col = str(threshold).replace(".", "_")
        over = row[f"p_over_{col}"]
        under = row[f"p_under_{col}"]
        assert over + under == pytest.approx(1.0, abs=1e-4)


def test_btts_combos_sum_to_one():
    out = _compute_markets(_sample_scoreline_df(), top_n=3)
    row = out.iloc[0]
    combo_sum = (
        row["p_btts_y_over25"]
        + row["p_btts_y_under25"]
        + row["p_btts_n_over25"]
        + row["p_btts_n_under25"]
    )
    assert combo_sum == pytest.approx(1.0, abs=1e-4)
    assert row["p_btts_y_over25"] == pytest.approx(0.15, abs=1e-4)
    assert row["p_btts_y_under25"] == pytest.approx(0.25, abs=1e-4)
    assert row["p_btts_n_over25"] == pytest.approx(0.0, abs=1e-4)
    assert row["p_btts_n_under25"] == pytest.approx(0.6, abs=1e-4)


def test_top_n_columns():
    out = _compute_markets(_sample_scoreline_df(), top_n=5)
    assert list(out.columns) == expected_betting_columns(5)
    row = out.iloc[0]
    assert row["rank1_scoreline"] == "1-0"
    assert row["rank1_prob"] == pytest.approx(0.30, abs=1e-4)
    assert row["rank5_scoreline"] == "0-2"
    assert row["rank5_prob"] == pytest.approx(0.10, abs=1e-4)
