"""Monte Carlo tournament simulation for the 2026 FIFA World Cup.

Two layers:
  1. Per-match: sample Poisson scorelines from predicted rates.
  2. Full tournament: group stage → best-third ranking → R32 → KO bracket.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TOURNAMENT_CONFIG_PATH: Final[Path] = Path("data/tournament/wc2026.json")
ET_SCALE: Final[float] = 30 / 90  # extra time is 30 min vs 90 min regulation

GROUP_POSITIONS = ("1st", "2nd", "3rd_qualify", "3rd_elim", "4th")

# Default matchday schedule (indices into the 4-team group array).
# MD1: 1v2, 3v4 | MD2: 1v3, 4v2 | MD3: 4v1, 2v3
_DEFAULT_MATCHDAYS: list[list[list[int]]] = [
    [[0, 1], [2, 3]],
    [[0, 2], [3, 1]],
    [[3, 0], [1, 2]],
]


# ---------------------------------------------------------------------------
# Per-match Monte Carlo
# ---------------------------------------------------------------------------


def sample_scorelines(
    lambda_h: np.ndarray,
    lambda_a: np.ndarray,
    n_sims: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw n_sims independent Poisson scorelines per match.

    Args:
        lambda_h: shape (n_matches,) expected home goals.
        lambda_a: shape (n_matches,) expected away goals.
        n_sims: number of simulations.
        rng: numpy random generator (for reproducibility).

    Returns:
        Array of shape (n_matches, n_sims, 2) — [home_goals, away_goals].
    """
    if rng is None:
        rng = np.random.default_rng()

    lambda_h = np.atleast_1d(lambda_h).clip(1e-6)
    lambda_a = np.atleast_1d(lambda_a).clip(1e-6)
    n = len(lambda_h)

    home = rng.poisson(lambda_h[:, None], size=(n, n_sims))
    away = rng.poisson(lambda_a[:, None], size=(n, n_sims))
    return np.stack([home, away], axis=-1)


def scoreline_distribution(samples: np.ndarray) -> pd.DataFrame:
    """Aggregate sampled scorelines into probability tables per match.

    Args:
        samples: shape (n_matches, n_sims, 2).

    Returns:
        DataFrame with columns: match_idx, home_goals, away_goals, probability.
    """
    n_matches, n_sims, _ = samples.shape
    records: list[dict] = []
    for i in range(n_matches):
        scores, counts = np.unique(samples[i], axis=0, return_counts=True)
        for score, count in zip(scores, counts):
            records.append({
                "match_idx": i,
                "home_goals": int(score[0]),
                "away_goals": int(score[1]),
                "probability": count / n_sims,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tournament simulation
# ---------------------------------------------------------------------------


def load_tournament_config(path: Path = TOURNAMENT_CONFIG_PATH) -> dict:
    """Load the tournament configuration JSON."""
    with open(path) as f:
        return json.load(f)


def _resolve_ko_match(
    lambda_h: float,
    lambda_a: float,
    rng: np.random.Generator,
) -> tuple[int, int, str]:
    """Simulate a single KO match: 90 min → ET → penalties.

    Returns (home_goals_total, away_goals_total, decided_by).
    decided_by is one of "90min", "ET", "PEN".
    """
    h90 = rng.poisson(lambda_h)
    a90 = rng.poisson(lambda_a)
    if h90 != a90:
        return h90, a90, "90min"

    h_et = rng.poisson(lambda_h * ET_SCALE)
    a_et = rng.poisson(lambda_a * ET_SCALE)
    h_total = h90 + h_et
    a_total = a90 + a_et
    if h_total != a_total:
        return h_total, a_total, "ET"

    if rng.random() < 0.5:
        return h_total + 1, a_total, "PEN"
    return h_total, a_total + 1, "PEN"


def _simulate_group(
    teams: list[str],
    match_rates: dict[tuple[str, str], tuple[float, float]],
    rng: np.random.Generator,
    matchdays: list[list[list[int]]] | None = None,
    locked_results: dict[tuple[str, str], tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Simulate round-robin within one group and return standings.

    Matches are played in matchday order so that inference rows are generated
    in the correct sequence.

    If ``locked_results`` is provided, any match whose (home, away) key appears
    in the dict uses the actual score deterministically instead of sampling from
    Poisson — zero RNG cost for those matches.

    Returns a list of dicts sorted by FIFA rules:
      points -> GD -> GF -> head-to-head -> random tiebreak.
    """
    if matchdays is None:
        matchdays = _DEFAULT_MATCHDAYS

    locked = locked_results or {}

    stats: dict[str, dict[str, int]] = {
        t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "h2h_pts": 0}
        for t in teams
    }
    h2h: dict[tuple[str, str], tuple[int, int]] = {}

    for md_pairs in matchdays:
        for h_idx, a_idx in md_pairs:
            home, away = teams[h_idx], teams[a_idx]
            if (home, away) in locked:
                hg, ag = locked[(home, away)]
            else:
                rates = match_rates.get((home, away))
                if rates is None:
                    rates = (1.2, 1.2)
                lh, la = rates
                hg = rng.poisson(lh)
                ag = rng.poisson(la)

            h2h[(home, away)] = (hg, ag)

            stats[home]["gf"] += hg
            stats[home]["ga"] += ag
            stats[away]["gf"] += ag
            stats[away]["ga"] += hg

            if hg > ag:
                stats[home]["pts"] += 3
            elif hg == ag:
                stats[home]["pts"] += 1
                stats[away]["pts"] += 1
            else:
                stats[away]["pts"] += 3

    for t in teams:
        stats[t]["gd"] = stats[t]["gf"] - stats[t]["ga"]

    for i, t1 in enumerate(teams):
        for t2 in teams[i + 1:]:
            key = (t1, t2) if (t1, t2) in h2h else (t2, t1)
            hg, ag = h2h.get(key, (0, 0))
            if key[0] == t1:
                t1_goals, t2_goals = hg, ag
            else:
                t1_goals, t2_goals = ag, hg
            if t1_goals > t2_goals:
                stats[t1]["h2h_pts"] += 3
            elif t1_goals == t2_goals:
                stats[t1]["h2h_pts"] += 1
                stats[t2]["h2h_pts"] += 1
            else:
                stats[t2]["h2h_pts"] += 3

    standings = [
        {"team": t, **stats[t], "random": rng.random()}
        for t in teams
    ]
    standings.sort(
        key=lambda x: (x["pts"], x["gd"], x["gf"], x["h2h_pts"], x["random"]),
        reverse=True,
    )
    return standings


def _rank_third_place(
    all_thirds: list[dict[str, Any]],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Rank all 12 third-place teams; top 8 qualify.

    Same criteria as group tiebreaker: points -> GD -> GF -> random.
    """
    for t in all_thirds:
        t["random"] = rng.random()
    all_thirds.sort(
        key=lambda x: (x["pts"], x["gd"], x["gf"], x["random"]),
        reverse=True,
    )
    return all_thirds[:8]


def simulate_tournament(
    predictions_df: pd.DataFrame,
    n_sims: int = 10_000,
    config_path: Path = TOURNAMENT_CONFIG_PATH,
    seed: int | None = None,
    locked_group_results: dict[tuple[str, str], tuple[int, int]] | None = None,
    locked_ko_results: dict[int, dict] | None = None,
) -> dict[str, Any]:
    """Run n_sims full 2026 WC tournament simulations.

    Already-played matches can be locked so their actual result is used
    deterministically every sim, consuming no RNG draws.

    Args:
        predictions_df: DataFrame with columns home_team, away_team, lambda_h, lambda_a.
        n_sims: number of tournament simulations.
        config_path: path to wc2026.json.
        seed: random seed for reproducibility.
        locked_group_results: (home, away) -> (home_goals, away_goals) for finished
            group matches.  Passed read-only to each _simulate_group() call.
        locked_ko_results: match_num -> {home, away, home_goals, away_goals, decided_by}
            for finished KO matches.  Checked once per match per sim before any Poisson
            draw; locked matches skip _resolve_ko_match() entirely.

    Returns:
        Dict with:
          - "advancement": per-team probabilities per round.
          - "group_positions": per-team probabilities for 1st/2nd/3rd(q)/3rd(e)/4th.
          - "n_sims": simulation count.
    """
    rng = np.random.default_rng(seed)
    config = load_tournament_config(config_path)
    groups = config["groups"]
    best_third_mappings = config["best_third_mappings"]
    ko_bracket = config["ko_bracket"]
    venue_country = config["venue_country"]
    host_nations = set(config["host_nations"])

    matchdays_cfg = config.get("group_matchdays")
    matchdays: list[list[list[int]]] | None = None
    if matchdays_cfg:
        matchdays = [md["pairs"] for md in matchdays_cfg]

    rate_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for _, row in predictions_df.iterrows():
        rate_lookup[(row["home_team"], row["away_team"])] = (
            float(row["lambda_h"]),
            float(row["lambda_a"]),
        )
    for _, row in predictions_df.iterrows():
        if (row["away_team"], row["home_team"]) not in rate_lookup:
            rate_lookup[(row["away_team"], row["home_team"])] = (
                float(row["lambda_a"]),
                float(row["lambda_h"]),
            )

    stages = ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"]
    all_teams = {t for teams in groups.values() for t in teams}
    advancement_counts: dict[str, dict[str, int]] = {
        team: {s: 0 for s in stages} for team in all_teams
    }
    for team in all_teams:
        advancement_counts[team]["Group"] = n_sims

    group_position_counts: dict[str, dict[str, int]] = {
        team: {pos: 0 for pos in GROUP_POSITIONS} for team in all_teams
    }

    r32_matches = config["r32_matches"]

    for _ in range(n_sims):
        # --- Group stage ---
        group_results: dict[str, list[dict]] = {}
        all_thirds: list[dict] = []

        for group_letter, teams in groups.items():
            standings = _simulate_group(
                teams, rate_lookup, rng, matchdays, locked_group_results
            )
            group_results[group_letter] = standings

            for rank, entry in enumerate(standings):
                entry["group"] = group_letter
                entry["rank"] = rank + 1

            for entry in standings[:2]:
                advancement_counts[entry["team"]]["R32"] += 1

            all_thirds.append(standings[2])

        # --- Best third-place ranking ---
        qualifying_thirds = _rank_third_place(all_thirds, rng)
        qualifying_third_teams = {t["team"] for t in qualifying_thirds}
        qualifying_groups = sorted(t["group"] for t in qualifying_thirds)
        combo_key = "".join(qualifying_groups)

        third_slot_map = best_third_mappings.get(combo_key, {})
        third_by_group: dict[str, str] = {
            t["group"]: t["team"] for t in qualifying_thirds
        }

        for t in qualifying_thirds:
            advancement_counts[t["team"]]["R32"] += 1

        # --- Accumulate group position counts ---
        for group_letter, standings in group_results.items():
            for rank_idx, entry in enumerate(standings):
                team = entry["team"]
                if rank_idx == 0:
                    group_position_counts[team]["1st"] += 1
                elif rank_idx == 1:
                    group_position_counts[team]["2nd"] += 1
                elif rank_idx == 2:
                    if team in qualifying_third_teams:
                        group_position_counts[team]["3rd_qualify"] += 1
                    else:
                        group_position_counts[team]["3rd_elim"] += 1
                else:
                    group_position_counts[team]["4th"] += 1

        # --- Resolve R32 matchups ---
        r32_winners: dict[int, str] = {}
        for m in r32_matches:
            match_num = m["match"]
            home_slot = m["home"]
            away_slot = m["away"]

            if locked_ko_results and match_num in locked_ko_results:
                locked = locked_ko_results[match_num]
                winner = locked["home"] if locked["home_goals"] > locked["away_goals"] else locked["away"]
                r32_winners[match_num] = winner
                continue

            home_team = _resolve_slot(
                home_slot, group_results, third_by_group, third_slot_map, match_num
            )
            away_team = _resolve_slot(
                away_slot, group_results, third_by_group, third_slot_map, match_num
            )

            if home_team is None or away_team is None:
                r32_winners[match_num] = home_team or away_team or "Unknown"
                continue

            rates = rate_lookup.get((home_team, away_team), (1.2, 1.2))
            hg, ag, _ = _resolve_ko_match(rates[0], rates[1], rng)
            winner = home_team if hg > ag else away_team
            r32_winners[match_num] = winner

        for team in r32_winners.values():
            if team in advancement_counts:
                advancement_counts[team]["R16"] += 1

        # --- R16 through Final ---
        ko_winners: dict[int, str] = dict(r32_winners)
        stage_map = {"R16": "QF", "QF": "SF", "SF": "Final", "Final": "Winner"}

        for ko_match in ko_bracket:
            match_num = ko_match["match"]
            stage = ko_match["stage"]
            venue = ko_match.get("venue", "")

            if locked_ko_results and match_num in locked_ko_results:
                locked = locked_ko_results[match_num]
                winner = locked["home"] if locked["home_goals"] > locked["away_goals"] else locked["away"]
                ko_winners[match_num] = winner
                next_stage = stage_map.get(stage)
                if next_stage and winner in advancement_counts:
                    advancement_counts[winner][next_stage] += 1
                continue

            home_ref = ko_match["home_from"]
            away_ref = ko_match["away_from"]

            home_team = ko_winners.get(int(home_ref[1:]))
            away_team = ko_winners.get(int(away_ref[1:]))

            if home_team is None or away_team is None:
                winner = home_team or away_team or "Unknown"
                ko_winners[match_num] = winner
                continue

            rates = rate_lookup.get((home_team, away_team), (1.2, 1.2))
            lh, la = rates

            venue_host = venue_country.get(venue)
            if venue_host and venue_host in host_nations:
                if away_team == venue_host:
                    lh, la = la, lh

            hg, ag, _ = _resolve_ko_match(lh, la, rng)
            winner = home_team if hg > ag else away_team
            ko_winners[match_num] = winner

            next_stage = stage_map.get(stage)
            if next_stage and winner in advancement_counts:
                advancement_counts[winner][next_stage] += 1

    # --- Build output DataFrames ---
    adv_records = []
    for team, counts in advancement_counts.items():
        row = {"team": team}
        for stage in stages:
            row[f"p_{stage.lower()}"] = counts[stage] / n_sims
        adv_records.append(row)

    advancement_df = pd.DataFrame(adv_records).sort_values("p_winner", ascending=False)
    advancement_df = advancement_df.reset_index(drop=True)

    gp_records = []
    for team, counts in group_position_counts.items():
        row = {"team": team}
        for pos in GROUP_POSITIONS:
            row[f"p_{pos}"] = counts[pos] / n_sims
        gp_records.append(row)

    group_positions_df = pd.DataFrame(gp_records).sort_values("p_1st", ascending=False)
    group_positions_df = group_positions_df.reset_index(drop=True)

    logger.info(
        "Tournament simulation complete: %d sims, %d teams tracked.",
        n_sims,
        len(advancement_df),
    )

    return {
        "advancement": advancement_df,
        "group_positions": group_positions_df,
        "n_sims": n_sims,
    }


# ---------------------------------------------------------------------------
# Group table builder (single-prediction / actual results path)
# ---------------------------------------------------------------------------


def build_group_tables(
    config: dict,
    match_results: dict[tuple[str, str], tuple[int, int]],
) -> dict[str, pd.DataFrame]:
    """Build group standings tables from actual or predicted match results.

    Args:
        config: tournament config (from load_tournament_config).
        match_results: mapping of (home, away) -> (home_goals, away_goals)
            for every group match played so far.

    Returns:
        Dict of group_letter -> DataFrame with columns:
          team, played, w, d, l, gf, ga, gd, pts
        sorted by FIFA ranking rules.
    """
    groups = config["groups"]
    matchdays_cfg = config.get("group_matchdays")
    matchdays: list[list[list[int]]] = (
        [md["pairs"] for md in matchdays_cfg]
        if matchdays_cfg
        else _DEFAULT_MATCHDAYS
    )

    tables: dict[str, pd.DataFrame] = {}
    for group_letter, teams in groups.items():
        stats: dict[str, dict[str, int]] = {
            t: {"played": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "gd": 0, "pts": 0}
            for t in teams
        }

        for md_pairs in matchdays:
            for h_idx, a_idx in md_pairs:
                home, away = teams[h_idx], teams[a_idx]
                result = match_results.get((home, away))
                if result is None:
                    continue
                hg, ag = result

                for t in (home, away):
                    stats[t]["played"] += 1

                stats[home]["gf"] += hg
                stats[home]["ga"] += ag
                stats[away]["gf"] += ag
                stats[away]["ga"] += hg

                if hg > ag:
                    stats[home]["w"] += 1
                    stats[home]["pts"] += 3
                    stats[away]["l"] += 1
                elif hg == ag:
                    stats[home]["d"] += 1
                    stats[home]["pts"] += 1
                    stats[away]["d"] += 1
                    stats[away]["pts"] += 1
                else:
                    stats[away]["w"] += 1
                    stats[away]["pts"] += 3
                    stats[home]["l"] += 1

        for t in teams:
            stats[t]["gd"] = stats[t]["gf"] - stats[t]["ga"]

        rows = [{"team": t, **stats[t]} for t in teams]
        rows.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
        tables[group_letter] = pd.DataFrame(rows)

    return tables


def _resolve_slot(
    slot: str,
    group_results: dict[str, list[dict]],
    third_by_group: dict[str, str],
    third_slot_map: dict[str, str],
    match_num: int,
) -> str | None:
    """Resolve a bracket slot like '1A', '2B', '3X/Y/Z' to a team name."""
    if slot.startswith("1") and len(slot) == 2:
        group = slot[1]
        standings = group_results.get(group, [])
        return standings[0]["team"] if standings else None
    elif slot.startswith("2") and len(slot) == 2:
        group = slot[1]
        standings = group_results.get(group, [])
        return standings[1]["team"] if len(standings) > 1 else None
    elif slot.startswith("3"):
        assigned = third_slot_map.get(str(match_num))
        if assigned:
            group_letter = assigned[1]  # e.g. "3A" -> "A"
            return third_by_group.get(group_letter)
        return None
    return None
