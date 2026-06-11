"""Tests for src.inference.export."""

import pandas as pd
import pytest

from src.inference.export import (
    _OVER_UNDER_THRESHOLDS,
    _compute_markets,
    _compute_tournament_bets,
    _filter_to_champion,
    expected_betting_columns,
    expected_tournament_betting_columns,
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


def _sample_advancement_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "France",
                "p_group": 1.0,
                "p_r32": 0.90,
                "p_r16": 0.70,
                "p_qf": 0.50,
                "p_sf": 0.30,
                "p_final": 0.20,
                "p_winner": 0.12,
            },
            {
                "team": "Germany",
                "p_group": 1.0,
                "p_r32": 0.80,
                "p_r16": 0.60,
                "p_qf": 0.40,
                "p_sf": 0.20,
                "p_final": 0.10,
                "p_winner": 0.05,
            },
        ]
    )


def _sample_group_positions_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "France",
                "p_1st": 0.55,
                "p_2nd": 0.25,
                "p_3rd_qualify": 0.10,
                "p_3rd_elim": 0.05,
                "p_4th": 0.05,
            },
            {
                "team": "Germany",
                "p_1st": 0.35,
                "p_2nd": 0.30,
                "p_3rd_qualify": 0.20,
                "p_3rd_elim": 0.10,
                "p_4th": 0.05,
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


def test_tournament_bets_schema_and_values():
    out = _compute_tournament_bets(
        _sample_advancement_df(),
        _sample_group_positions_df(),
        group_mapping={"France": "D", "Germany": "E"},
    )
    assert list(out.columns) == expected_tournament_betting_columns()
    assert "model_name" not in out.columns

    row = out[out["team"] == "France"].iloc[0]
    assert row["group"] == "D"
    assert row["p_reach_r32"] == pytest.approx(0.90, abs=1e-4)
    assert row["p_reach_final"] == pytest.approx(0.20, abs=1e-4)
    assert row["p_win_tournament"] == pytest.approx(0.12, abs=1e-4)
    assert row["p_group_first"] == pytest.approx(0.55, abs=1e-4)
    assert row["p_group_last"] == pytest.approx(0.05, abs=1e-4)


def test_filter_to_champion_drops_model_name():
    stacked = pd.DataFrame(
        [
            {"model_name": "champion", "team": "France", "p_winner": 0.12},
            {"model_name": "shadow", "team": "France", "p_winner": 0.08},
        ]
    )
    out = _filter_to_champion(stacked, "champion", "tournament_probabilities.csv")
    assert list(out.columns) == ["team", "p_winner"]
    assert len(out) == 1
    assert out.iloc[0]["p_winner"] == pytest.approx(0.12, abs=1e-4)
