"""Tests for host-advantage orientation in the tournament simulation.

Covers the fix for the host-rate "scramble" bug: home advantage is baked into
predictions at predict time (host stored as ``home_team``, ``is_neutral=False``),
and the simulation must orient those stored rates onto bracket slots WITHOUT
swapping (which previously handed the host the opponent's rate).

See docs/notes/wc_live.md (MD3, Jun 27) for the root-cause writeup.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.gold.context_features import WC_2026_HOSTS
from src.inference.features import generate_all_wc_pairings
from src.inference.simulation import (
    _ko_match_rates,
    _resolve_ko_match,
    simulate_tournament,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OFFLINE_PREDICTIONS = _REPO_ROOT / "src" / "dashboard" / "_offline_cache" / "predictions.csv"

# Stored, home-oriented rates (host always in home_team, is_neutral=False).
# Spain/US uses the real verified prediction (home=United States).
_RATE_LOOKUP: dict[tuple[str, str], tuple[float, float]] = {
    # Single host: US home rate 0.665, Spain away rate 1.968.
    ("United States", "Spain"): (0.665, 1.968),
    ("Spain", "United States"): (1.968, 0.665),  # reverse-fill mirror
    # Host vs host: each host's own home-oriented prediction.
    ("United States", "Mexico"): (1.5, 1.0),  # US at home vs Mexico
    ("Mexico", "United States"): (1.4, 0.9),  # Mexico at home vs US
    # Neutral pair (both orientations present, mirror of each other).
    ("France", "Brazil"): (1.3, 1.1),
    ("Brazil", "France"): (1.1, 1.3),
}

_HOST_NATIONS = {"United States", "Canada", "Mexico"}
_VENUE_COUNTRY = {
    "Dallas": "United States",
    "Mexico City": "Mexico",
    "Toronto": "Canada",
}


def _away_win_prob(lh: float, la: float, n: int = 20000, seed: int = 0) -> float:
    """Monte-Carlo P(away team wins) for a KO match with given rates."""
    rng = np.random.default_rng(seed)
    away_wins = 0
    for _ in range(n):
        hg, ag, _ = _resolve_ko_match(lh, la, rng)
        if ag > hg:
            away_wins += 1
    return away_wins / n


def _rates(home: str, away: str, venue: str) -> tuple[float, float]:
    return _ko_match_rates(
        home, away, venue, _RATE_LOOKUP, _VENUE_COUNTRY, _HOST_NATIONS
    )


# ---------------------------------------------------------------------------
# The four orientation cases
# ---------------------------------------------------------------------------


class TestKoMatchRatesOrientation:
    def test_case1_nonhost_home_vs_host_away(self):
        # Bracket home=Spain, away=US, US venue — the case the swap broke.
        lh, la = _rates("Spain", "United States", "Dallas")
        assert lh == pytest.approx(1.968)  # Spain scores its own rate
        assert la == pytest.approx(0.665)  # US keeps its (home-boosted) rate

        p_us = _away_win_prob(lh, la)
        assert p_us < 0.30, "US must remain the underdog vs Spain"

    def test_case2_host_home_vs_nonhost_away(self):
        # Bracket home=US, away=Spain, US venue. Same per-team rates as case 1.
        lh, la = _rates("United States", "Spain", "Dallas")
        assert lh == pytest.approx(0.665)
        assert la == pytest.approx(1.968)

        # P(US win) is symmetric with case 1 (US is now the home slot).
        p_us_home = 1.0 - _away_win_prob(lh, la)
        p_us_away = _away_win_prob(*_rates("Spain", "United States", "Dallas"))
        assert p_us_home == pytest.approx(p_us_away, abs=0.02)

    def test_case3_host_vs_host_follows_venue(self):
        # Bracket home=US, away=Mexico. The venue host gets its home rate.
        lh_mx, la_mx = _rates("United States", "Mexico", "Mexico City")
        assert la_mx == pytest.approx(1.4)  # Mexico (away slot) at home in Mexico
        assert lh_mx == pytest.approx(0.9)  # US gets its away rate vs Mexico-home

        lh_us, la_us = _rates("United States", "Mexico", "Dallas")
        assert lh_us == pytest.approx(1.5)  # US (home slot) at home in the US
        assert la_us == pytest.approx(1.0)

        # Flipping the venue flips which host is boosted.
        assert lh_us > lh_mx and la_mx > la_us

    def test_case4_nonhost_vs_nonhost_neutral(self):
        # Neutral pair: stored rate used directly, orientation-symmetric.
        lh, la = _rates("France", "Brazil", "Dallas")
        assert (lh, la) == pytest.approx((1.3, 1.1))

        lh_rev, la_rev = _rates("Brazil", "France", "Dallas")
        assert (lh_rev, la_rev) == pytest.approx((1.1, 1.3))


# ---------------------------------------------------------------------------
# Guards / regressions
# ---------------------------------------------------------------------------


class TestHostAdvantageGuards:
    def test_no_swap_residue(self):
        # The old bug would have returned (0.665, 1.968) for bracket
        # home=Spain/away=US, giving US the favourite's rate.
        lh, la = _rates("Spain", "United States", "Dallas")
        p_us = _away_win_prob(lh, la)

        # What the swap would have produced (US handed Spain's rate):
        p_us_swapped = _away_win_prob(0.665, 1.968)
        assert p_us_swapped > 0.5, "sanity: the buggy swap makes US a favourite"
        assert p_us < 0.5 < p_us_swapped
        assert abs(p_us - p_us_swapped) > 0.3

    def test_r32_and_ko_paths_share_one_helper(self):
        # Structural guard that both KO loops orient via _ko_match_rates and
        # that the destructive swap is gone (R32 previously had no host logic).
        source = (
            _REPO_ROOT / "src" / "inference" / "simulation.py"
        ).read_text()
        assert "lh, la = la, lh" not in source, "rate swap must be deleted"
        assert source.count("_ko_match_rates(") >= 3  # def + R32 call + KO call

    def test_host_vs_host_venue_fallback_to_home_slot(self):
        # Two hosts but the venue is neither — fall back to bracket home_team.
        lh, la = _rates("United States", "Mexico", "Toronto")
        assert lh == pytest.approx(1.5)  # US is the bracket home_team
        assert la == pytest.approx(1.0)

    def test_zero_hosts_ignores_venue_country(self):
        # A non-host pair at a host venue stays neutral (no boost reassigned).
        lh, la = _rates("France", "Brazil", "Dallas")
        assert (lh, la) == pytest.approx((1.3, 1.1))


class TestUpstreamPositionalGuard:
    def test_single_host_always_in_home_team(self):
        pairings = generate_all_wc_pairings()
        for home, away in zip(pairings["home_team"], pairings["away_team"]):
            home_host = home in WC_2026_HOSTS
            away_host = away in WC_2026_HOSTS
            if home_host != away_host:  # exactly one host
                assert home_host, f"host must be home_team: {home} vs {away}"

    def test_host_vs_host_emits_both_orientations(self):
        pairings = generate_all_wc_pairings()
        oriented = set(zip(pairings["home_team"], pairings["away_team"]))
        hosts = sorted(WC_2026_HOSTS)
        for i, a in enumerate(hosts):
            for b in hosts[i + 1:]:
                assert (a, b) in oriented and (b, a) in oriented, (
                    f"host-vs-host {a}/{b} must appear in both orientations"
                )

    def test_host_vs_host_rows_are_not_neutral(self):
        pairings = generate_all_wc_pairings()
        host_rows = pairings[
            pairings["home_team"].isin(WC_2026_HOSTS)
            & pairings["away_team"].isin(WC_2026_HOSTS)
        ]
        assert not host_rows.empty
        assert not host_rows["is_neutral"].any()


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


class TestEndToEndSmoke:
    @pytest.mark.skipif(
        not _OFFLINE_PREDICTIONS.exists(),
        reason="offline-cache predictions snapshot not available",
    )
    def test_us_does_not_outrank_elite_teams(self):
        preds = pd.read_csv(_OFFLINE_PREDICTIONS)
        preds = preds[["home_team", "away_team", "lambda_h", "lambda_a"]]

        result = simulate_tournament(preds, n_sims=1500, seed=42)
        adv = result["advancement"].set_index("team")["p_winner"]

        for elite in ("France", "Argentina"):
            if elite in adv.index and "United States" in adv.index:
                assert adv["United States"] < adv[elite], (
                    f"US p_winner ({adv['United States']:.4f}) should be below "
                    f"{elite} ({adv[elite]:.4f}) — host boost must not flip favouritism"
                )
