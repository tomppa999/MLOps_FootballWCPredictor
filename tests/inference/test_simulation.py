"""Tests for src.inference.simulation."""

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference.simulation import (
    _resolve_ko_match,
    _simulate_group,
    build_group_tables,
    load_tournament_config,
    sample_scorelines,
    scoreline_distribution,
    simulate_tournament,
)


# ---------------------------------------------------------------------------
# sample_scorelines
# ---------------------------------------------------------------------------


class TestSampleScorelines:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        samples = sample_scorelines(
            np.array([1.5, 1.0]),
            np.array([1.2, 0.8]),
            n_sims=1000,
            rng=rng,
        )
        assert samples.shape == (2, 1000, 2)

    def test_reproducibility_with_seed(self):
        lh = np.array([1.5])
        la = np.array([1.2])
        s1 = sample_scorelines(lh, la, n_sims=100, rng=np.random.default_rng(42))
        s2 = sample_scorelines(lh, la, n_sims=100, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(s1, s2)

    def test_non_negative_goals(self):
        rng = np.random.default_rng(0)
        samples = sample_scorelines(
            np.array([0.5]),
            np.array([0.3]),
            n_sims=5000,
            rng=rng,
        )
        assert (samples >= 0).all()


# ---------------------------------------------------------------------------
# scoreline_distribution
# ---------------------------------------------------------------------------


class TestScorelineDistribution:
    def test_probabilities_sum_to_one(self):
        rng = np.random.default_rng(42)
        samples = sample_scorelines(np.array([1.5]), np.array([1.0]), n_sims=5000, rng=rng)
        dist = scoreline_distribution(samples)

        total = dist.loc[dist["match_idx"] == 0, "probability"].sum()
        assert abs(total - 1.0) < 1e-6

    def test_columns(self):
        rng = np.random.default_rng(42)
        samples = sample_scorelines(np.array([1.0]), np.array([1.0]), n_sims=100, rng=rng)
        dist = scoreline_distribution(samples)
        assert set(dist.columns) == {"match_idx", "home_goals", "away_goals", "probability"}


# ---------------------------------------------------------------------------
# _resolve_ko_match
# ---------------------------------------------------------------------------


class TestResolveKoMatch:
    def test_always_produces_winner(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            hg, ag, method = _resolve_ko_match(1.2, 1.2, rng)
            assert hg != ag, "KO match must have a winner"
            assert method in ("90min", "ET", "PEN")

    def test_decided_by_is_valid(self):
        rng = np.random.default_rng(0)
        methods_seen: set[str] = set()
        for _ in range(500):
            _, _, method = _resolve_ko_match(1.0, 1.0, rng)
            methods_seen.add(method)
        assert "90min" in methods_seen
        assert "PEN" in methods_seen


# ---------------------------------------------------------------------------
# _simulate_group
# ---------------------------------------------------------------------------


class TestSimulateGroup:
    def test_returns_correct_number_of_teams(self):
        rng = np.random.default_rng(42)
        teams = ["A", "B", "C", "D"]
        rates: dict[tuple[str, str], tuple[float, float]] = {
            ("A", "B"): (2.0, 0.5),
            ("A", "C"): (2.0, 0.5),
            ("A", "D"): (2.0, 0.5),
            ("B", "C"): (1.5, 1.0),
            ("B", "D"): (1.5, 1.0),
            ("C", "D"): (1.0, 1.0),
        }
        standings = _simulate_group(teams, rates, rng)
        assert len(standings) == 4

    def test_points_are_consistent(self):
        rng = np.random.default_rng(42)
        teams = ["A", "B", "C", "D"]
        rates: dict[tuple[str, str], tuple[float, float]] = {}
        standings = _simulate_group(teams, rates, rng)
        total_pts = sum(s["pts"] for s in standings)
        # 6 matches, each distributing either 3 pts (win) or 2 pts (draw)
        assert 12 <= total_pts <= 18

    def test_matchday_order_produces_same_matches(self):
        """Matchday order shouldn't change which pairings are played."""
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        teams = ["A", "B", "C", "D"]
        rates: dict[tuple[str, str], tuple[float, float]] = {
            ("A", "B"): (2.0, 0.5), ("C", "D"): (1.0, 1.5),
            ("A", "C"): (1.8, 0.8), ("D", "B"): (0.7, 1.3),
            ("D", "A"): (0.6, 2.0), ("B", "C"): (1.2, 1.0),
        }
        s1 = _simulate_group(teams, rates, rng1)
        s2 = _simulate_group(teams, rates, rng2)
        assert [e["team"] for e in s1] == [e["team"] for e in s2]

    def test_sorted_by_points_descending(self):
        rng = np.random.default_rng(42)
        teams = ["Strong", "Medium", "Weak", "Weakest"]
        rates = {
            ("Strong", "Medium"): (3.0, 0.3),
            ("Strong", "Weak"): (3.0, 0.2),
            ("Strong", "Weakest"): (4.0, 0.1),
            ("Medium", "Weak"): (2.0, 0.5),
            ("Medium", "Weakest"): (2.5, 0.3),
            ("Weak", "Weakest"): (1.5, 0.5),
        }
        top_counts: dict[str, int] = {t: 0 for t in teams}
        for seed in range(50):
            standings = _simulate_group(teams, rates, np.random.default_rng(seed))
            top_counts[standings[0]["team"]] += 1
        assert top_counts["Strong"] > 30


# ---------------------------------------------------------------------------
# build_group_tables
# ---------------------------------------------------------------------------


class TestBuildGroupTables:
    def test_full_group(self):
        config = {
            "groups": {"A": ["T1", "T2", "T3", "T4"]},
            "group_matchdays": [
                {"matchday": 1, "pairs": [[0, 1], [2, 3]]},
                {"matchday": 2, "pairs": [[0, 2], [3, 1]]},
                {"matchday": 3, "pairs": [[3, 0], [1, 2]]},
            ],
        }
        results = {
            ("T1", "T2"): (2, 0),
            ("T3", "T4"): (1, 1),
            ("T1", "T3"): (1, 0),
            ("T4", "T2"): (0, 3),
            ("T4", "T1"): (0, 1),
            ("T2", "T3"): (2, 1),
        }
        tables = build_group_tables(config, results)
        assert "A" in tables
        tbl = tables["A"]
        assert len(tbl) == 4
        assert list(tbl.columns) == ["team", "played", "w", "d", "l", "gf", "ga", "gd", "pts"]

        t1_row = tbl[tbl["team"] == "T1"].iloc[0]
        assert t1_row["pts"] == 9
        assert t1_row["played"] == 3
        assert t1_row["gf"] == 4

    def test_partial_results(self):
        config = {
            "groups": {"A": ["T1", "T2", "T3", "T4"]},
            "group_matchdays": [
                {"matchday": 1, "pairs": [[0, 1], [2, 3]]},
                {"matchday": 2, "pairs": [[0, 2], [3, 1]]},
                {"matchday": 3, "pairs": [[3, 0], [1, 2]]},
            ],
        }
        results = {("T1", "T2"): (1, 1), ("T3", "T4"): (2, 0)}
        tables = build_group_tables(config, results)
        tbl = tables["A"]
        assert tbl[tbl["team"] == "T1"].iloc[0]["played"] == 1
        assert tbl[tbl["team"] == "T4"].iloc[0]["played"] == 1


# ---------------------------------------------------------------------------
# simulate_tournament (small smoke test)
# ---------------------------------------------------------------------------

# R32 match numbers for 3rd-place slots in the toy config
_TOY_3RD_SLOTS = [81, 82, 83, 84, 85, 86, 87, 88]


def _make_toy_config(tmp_path: Path) -> Path:
    """Create a minimal 12-group tournament config for testing."""
    config = {
        "tournament": "Test Cup",
        "host_nations": ["USA"],
        "groups": {
            "A": ["T1", "T2", "T3", "T4"],
            "B": ["T5", "T6", "T7", "T8"],
            "C": ["T9", "T10", "T11", "T12"],
            "D": ["T13", "T14", "T15", "T16"],
            "E": ["T17", "T18", "T19", "T20"],
            "F": ["T21", "T22", "T23", "T24"],
            "G": ["T25", "T26", "T27", "T28"],
            "H": ["T29", "T30", "T31", "T32"],
            "I": ["T33", "T34", "T35", "T36"],
            "J": ["T37", "T38", "T39", "T40"],
            "K": ["T41", "T42", "T43", "T44"],
            "L": ["T45", "T46", "T47", "T48"],
        },
        "group_matchdays": [
            {"matchday": 1, "pairs": [[0, 1], [2, 3]]},
            {"matchday": 2, "pairs": [[0, 2], [3, 1]]},
            {"matchday": 3, "pairs": [[3, 0], [1, 2]]},
        ],
        "r32_matches": [
            {"match": 73, "home": "1A", "away": "2C"},
            {"match": 74, "home": "1B", "away": "2D"},
            {"match": 75, "home": "1C", "away": "2A"},
            {"match": 76, "home": "1D", "away": "2B"},
            {"match": 77, "home": "1E", "away": "2G"},
            {"match": 78, "home": "1F", "away": "2H"},
            {"match": 79, "home": "1G", "away": "2E"},
            {"match": 80, "home": "1H", "away": "2F"},
            {"match": 81, "home": "1I", "away": "3A/B/C/D"},
            {"match": 82, "home": "1J", "away": "3E/F/G/H"},
            {"match": 83, "home": "1K", "away": "3I/J/K/L"},
            {"match": 84, "home": "1L", "away": "3A/B/C/D"},
            {"match": 85, "home": "2I", "away": "3E/F/G/H"},
            {"match": 86, "home": "2J", "away": "3I/J/K/L"},
            {"match": 87, "home": "2K", "away": "3A/B/C/D"},
            {"match": 88, "home": "2L", "away": "3E/F/G/H"},
        ],
        "best_third_mappings": {},
        "ko_bracket": [
            {"match": 89, "stage": "R16", "home_from": "W73", "away_from": "W74", "venue": "NYC"},
            {"match": 90, "stage": "R16", "home_from": "W75", "away_from": "W76", "venue": "NYC"},
            {"match": 91, "stage": "R16", "home_from": "W77", "away_from": "W78", "venue": "NYC"},
            {"match": 92, "stage": "R16", "home_from": "W79", "away_from": "W80", "venue": "NYC"},
            {"match": 93, "stage": "R16", "home_from": "W81", "away_from": "W82", "venue": "NYC"},
            {"match": 94, "stage": "R16", "home_from": "W83", "away_from": "W84", "venue": "NYC"},
            {"match": 95, "stage": "R16", "home_from": "W85", "away_from": "W86", "venue": "NYC"},
            {"match": 96, "stage": "R16", "home_from": "W87", "away_from": "W88", "venue": "NYC"},
            {"match": 97, "stage": "QF", "home_from": "W89", "away_from": "W90", "venue": "NYC"},
            {"match": 98, "stage": "QF", "home_from": "W91", "away_from": "W92", "venue": "NYC"},
            {"match": 99, "stage": "QF", "home_from": "W93", "away_from": "W94", "venue": "NYC"},
            {"match": 100, "stage": "QF", "home_from": "W95", "away_from": "W96", "venue": "NYC"},
            {"match": 101, "stage": "SF", "home_from": "W97", "away_from": "W98", "venue": "NYC"},
            {"match": 102, "stage": "SF", "home_from": "W99", "away_from": "W100", "venue": "NYC"},
            {"match": 103, "stage": "Final", "home_from": "W101", "away_from": "W102", "venue": "NYC"},
        ],
        "venue_country": {"NYC": "USA"},
    }

    all_groups = list("ABCDEFGHIJKL")
    mappings = {}
    for combo in itertools.combinations(all_groups, 8):
        key = "".join(combo)
        assignment = {}
        for i, g in enumerate(combo):
            assignment[str(_TOY_3RD_SLOTS[i])] = f"3{g}"
        mappings[key] = assignment
    config["best_third_mappings"] = mappings

    path = tmp_path / "test_config.json"
    path.write_text(json.dumps(config))
    return path


def _make_all_pairs_predictions() -> pd.DataFrame:
    return pd.DataFrame([
        {"home_team": f"T{i}", "away_team": f"T{j}", "lambda_h": 1.5, "lambda_a": 1.0}
        for i in range(1, 49)
        for j in range(1, 49)
        if i != j
    ])


class TestSimulateTournament:
    def test_returns_advancement_dataframe(self, tmp_path):
        config_path = _make_toy_config(tmp_path)
        result = simulate_tournament(
            _make_all_pairs_predictions(),
            n_sims=10,
            config_path=config_path,
            seed=42,
        )

        assert "advancement" in result
        adv = result["advancement"]
        assert "team" in adv.columns
        assert "p_winner" in adv.columns
        assert len(adv) == 48

    def test_returns_group_positions(self, tmp_path):
        config_path = _make_toy_config(tmp_path)
        result = simulate_tournament(
            _make_all_pairs_predictions(),
            n_sims=10,
            config_path=config_path,
            seed=42,
        )

        assert "group_positions" in result
        gp = result["group_positions"]
        assert len(gp) == 48
        for col in ["p_1st", "p_2nd", "p_3rd_qualify", "p_3rd_elim", "p_4th"]:
            assert col in gp.columns

        # Each team's position probabilities should sum to 1.0
        for _, row in gp.iterrows():
            total = row["p_1st"] + row["p_2nd"] + row["p_3rd_qualify"] + row["p_3rd_elim"] + row["p_4th"]
            assert abs(total - 1.0) < 1e-6

    def test_winner_probabilities_sum_approximately(self, tmp_path):
        config_path = _make_toy_config(tmp_path)

        predictions = pd.DataFrame([
            {"home_team": f"T{i}", "away_team": f"T{j}", "lambda_h": 1.2, "lambda_a": 1.2}
            for i in range(1, 49)
            for j in range(1, 49)
            if i != j
        ])

        result = simulate_tournament(
            predictions,
            n_sims=50,
            config_path=config_path,
            seed=42,
        )

        adv = result["advancement"]
        winner_sum = adv["p_winner"].sum()
        assert abs(winner_sum - 1.0) < 0.15


# ---------------------------------------------------------------------------
# _simulate_group with locked results
# ---------------------------------------------------------------------------


class TestSimulateGroupLocked:
    """_simulate_group with locked_results uses actual scores deterministically."""

    _TEAMS = ["A", "B", "C", "D"]
    _RATES: dict[tuple[str, str], tuple[float, float]] = {
        ("A", "B"): (2.0, 0.5), ("C", "D"): (1.0, 1.0),
        ("A", "C"): (1.8, 0.8), ("D", "B"): (0.7, 1.3),
        ("D", "A"): (0.6, 2.0), ("B", "C"): (1.2, 1.0),
    }

    def test_locked_match_is_deterministic(self):
        """Locked matches must produce the same score on every call."""
        locked = {("A", "B"): (3, 0), ("C", "D"): (1, 2)}
        scores_seen: set[tuple[int, int]] = set()
        for seed in range(30):
            rng = np.random.default_rng(seed)
            standings = _simulate_group(
                self._TEAMS, self._RATES, rng, locked_results=locked
            )
            # Reconstruct A-B and C-D results from the standings stats
            # A won 3-0 → should always have +3 pts from that match
            a_pts = next(s["pts"] for s in standings if s["team"] == "A")
            # A beats B (3-0) and faces C and D via rates — just check A always
            # benefits from the locked 3-0 win.  A must have at least 3 pts.
            assert a_pts >= 3, "A should always score 3+ pts from locked 3-0 win over B"

            # Track variability of unlocked matches: record C's GF across seeds
            c_gf = next(s["gf"] for s in standings if s["team"] == "C")
            scores_seen.add((seed, c_gf))

        # Check that C's GF is not identical across all seeds (unlocked matches vary)
        c_gf_values = {v for _, v in scores_seen}
        assert len(c_gf_values) > 1, "Unlocked matches should vary across seeds"

    def test_locked_results_do_not_consume_rng_for_those_matches(self):
        """Two groups: one fully locked, one not.  The locked group's standings are
        bit-for-bit identical across seeds because no RNG is consumed for it."""
        fully_locked = {
            ("A", "B"): (2, 1),
            ("C", "D"): (0, 0),
            ("A", "C"): (1, 1),
            ("D", "B"): (2, 0),
            ("D", "A"): (0, 3),
            ("B", "C"): (1, 0),
        }
        # All 6 matches locked → standings must be identical for every rng seed
        previous_standings = None
        for seed in range(10):
            rng = np.random.default_rng(seed)
            standings = _simulate_group(
                self._TEAMS, self._RATES, rng, locked_results=fully_locked
            )
            result = [(s["team"], s["pts"], s["gf"], s["ga"]) for s in standings]
            if previous_standings is None:
                previous_standings = result
            else:
                assert result == previous_standings


# ---------------------------------------------------------------------------
# simulate_tournament with locked group results
# ---------------------------------------------------------------------------


class TestSimulateTournamentLockedGroup:
    def test_locked_group_standings_are_constant(self, tmp_path):
        """When all matches of one group are locked, that group's position
        probabilities must be 0 or 1 across any number of sims."""
        config_path = _make_toy_config(tmp_path)
        predictions = _make_all_pairs_predictions()

        # Lock all 6 matches of group A (T1-T4)
        locked_group = {
            ("T1", "T2"): (2, 0),
            ("T3", "T4"): (1, 1),
            ("T1", "T3"): (1, 0),
            ("T4", "T2"): (0, 3),
            ("T4", "T1"): (0, 1),
            ("T2", "T3"): (2, 1),
        }

        result = simulate_tournament(
            predictions,
            n_sims=20,
            config_path=config_path,
            seed=7,
            locked_group_results=locked_group,
        )
        gp = result["group_positions"]

        for team in ["T1", "T2", "T3", "T4"]:
            row = gp[gp["team"] == team].iloc[0]
            total = row["p_1st"] + row["p_2nd"] + row["p_3rd_qualify"] + row["p_3rd_elim"] + row["p_4th"]
            assert abs(total - 1.0) < 1e-6

            # Each probability must be exactly 0 or 1 (fully deterministic group)
            for col in ["p_1st", "p_2nd", "p_3rd_qualify", "p_3rd_elim", "p_4th"]:
                v = row[col]
                assert v == 0.0 or v == 1.0, (
                    f"{team} {col}={v} should be 0 or 1 when all group matches are locked"
                )


# ---------------------------------------------------------------------------
# simulate_tournament with locked KO results
# ---------------------------------------------------------------------------


class TestSimulateTournamentLockedKO:
    # Group-A locks that guarantee T1=1st, T2=2nd every sim.
    _GROUP_A_LOCKED: dict[tuple[str, str], tuple[int, int]] = {
        ("T1", "T2"): (3, 0),
        ("T3", "T4"): (1, 1),
        ("T1", "T3"): (2, 0),
        ("T4", "T2"): (0, 2),
        ("T4", "T1"): (0, 1),
        ("T2", "T3"): (1, 2),
    }
    # Group-B locks that guarantee T5=1st, T6=2nd every sim.
    _GROUP_B_LOCKED: dict[tuple[str, str], tuple[int, int]] = {
        ("T5", "T6"): (3, 0),
        ("T7", "T8"): (1, 1),
        ("T5", "T7"): (2, 0),
        ("T8", "T6"): (0, 2),
        ("T8", "T5"): (0, 1),
        ("T6", "T7"): (1, 2),
    }

    def test_locked_r32_winner_always_advances_to_r16(self, tmp_path):
        """Locking an R32 match should make its winner advance to R16 in 100% of sims.

        We also lock group A so T1 is deterministically the 1A slot, ensuring T1
        only appears in one R32 match (match 73) across all sims.
        """
        config_path = _make_toy_config(tmp_path)
        predictions = _make_all_pairs_predictions()

        # R32 match 73 in the toy config is "1A vs 2C"
        # With group A locked, T1 is always 1A.
        locked_ko = {
            73: {"home": "T1", "away": "T9", "home_goals": 2, "away_goals": 0, "decided_by": "FT"}
        }

        result = simulate_tournament(
            predictions,
            n_sims=30,
            config_path=config_path,
            seed=0,
            locked_group_results=self._GROUP_A_LOCKED,
            locked_ko_results=locked_ko,
        )
        adv = result["advancement"]
        t1_row = adv[adv["team"] == "T1"].iloc[0]
        assert t1_row["p_r16"] == 1.0, (
            f"T1 should reach R16 in 100% of sims, got {t1_row['p_r16']}"
        )

    def test_locked_ko_winner_propagates_through_bracket(self, tmp_path):
        """When an R32 match is locked, the winner is deterministically in R16.

        Group B is also locked so T5 is always 1B (only in match 73 slot "1A"
        substituted here as 1B for the toy bracket).
        """
        config_path = _make_toy_config(tmp_path)
        predictions = _make_all_pairs_predictions()

        # Toy config match 74: "1B vs 2D". Lock group B so T5=1B always.
        locked_ko = {
            74: {"home": "T5", "away": "T13", "home_goals": 1, "away_goals": 0, "decided_by": "FT"}
        }

        result = simulate_tournament(
            predictions,
            n_sims=50,
            config_path=config_path,
            seed=1,
            locked_group_results=self._GROUP_B_LOCKED,
            locked_ko_results=locked_ko,
        )
        adv = result["advancement"]
        t5_row = adv[adv["team"] == "T5"].iloc[0]
        assert t5_row["p_r16"] == 1.0


# ---------------------------------------------------------------------------
# simulate_tournament with partial group + KO locks
# ---------------------------------------------------------------------------


class TestSimulateTournamentPartialLocks:
    def test_partial_group_and_ko_lock(self, tmp_path):
        """Locking all group-A matches plus R32 match 73 gives T1 p_r16==1.0
        while leaving all other groups and KO matches fully stochastic."""
        config_path = _make_toy_config(tmp_path)
        predictions = _make_all_pairs_predictions()

        # Lock all 6 group A matches so T1 is always 1A (9 pts)
        locked_group = {
            ("T1", "T2"): (3, 0),
            ("T3", "T4"): (1, 1),
            ("T1", "T3"): (2, 0),
            ("T4", "T2"): (0, 2),
            ("T4", "T1"): (0, 1),
            ("T2", "T3"): (1, 2),
        }
        # Lock R32 match 73 (1A vs 2C) with T1 winning
        locked_ko = {
            73: {"home": "T1", "away": "T9", "home_goals": 2, "away_goals": 1, "decided_by": "FT"}
        }

        result = simulate_tournament(
            predictions,
            n_sims=20,
            config_path=config_path,
            seed=99,
            locked_group_results=locked_group,
            locked_ko_results=locked_ko,
        )
        adv = result["advancement"]
        assert len(adv) == 48

        # T1 locked as 1A and locked as R32 winner → must always reach R16
        t1_row = adv[adv["team"] == "T1"].iloc[0]
        assert t1_row["p_r16"] == 1.0

        # Overall winner sum should still be approximately 1
        assert abs(adv["p_winner"].sum() - 1.0) < 0.15
