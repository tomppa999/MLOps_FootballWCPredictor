import numpy as np
import pandas as pd
import pytest

from src.analysis.poisson_joint_prepare import fixture_rates, outcome_probs
from src.inference.simulation import _draw_goals, _ko_match_rates, _resolve_ko_match


def test_outcomes_use_only_unique_components_and_handle_zero_rates():
    p = outcome_probs(np.array([0, 1, 0, 2]), np.array([0, 0, 1, 2]))
    e = np.exp(-1)
    np.testing.assert_allclose(p[:3], [[0, 1, 0], [1 - e, e, 0], [0, e, 1 - e]])
    assert p[3, 0] == pytest.approx(p[3, 2])
    np.testing.assert_allclose(p.sum(axis=1), 1)
    with pytest.raises(ValueError):
        outcome_probs([1], [-1])


def test_goal_draws_share_covariance():
    rng = np.random.default_rng(41)
    draws = np.array([_draw_goals(rng, 2.0, 1.5, 0.6) for _ in range(50_000)])
    np.testing.assert_allclose(draws.mean(axis=0), [2, 1.5], atol=0.025)
    assert np.cov(draws.T)[0, 1] == pytest.approx(0.6, abs=0.035)


def test_extra_time_scales_all_three_rates_after_shared_regulation_draw():
    class RecordedDraws:
        def __init__(self):
            self.rates = []
            self.draws = iter([0, 0, 2, 1, 0, 1])

        def poisson(self, rate):
            self.rates.append(rate)
            return next(self.draws)

    rng = RecordedDraws()
    assert _resolve_ko_match(3, 2, rng, 1.5) == (4, 3, "ET")
    np.testing.assert_allclose(rng.rates, [1.5, 0.5, 1.5, 0.5, 1 / 6, 0.5])


def test_reversed_host_uses_its_own_shared_rate():
    rates = {("A", "B"): (2, 1, 0.8), ("B", "A"): (3, 1.5, 0.3)}
    assert _ko_match_rates("A", "B", "v", rates, {"v": "B"}, {"A", "B"}) == (1.5, 3, 0.3)


def test_fixture_orientation_preserves_strand_and_live_source_choice():
    rates = pd.DataFrame(
        [
            ("B", "A", 3.0, 1.5, 0.3),
            ("A", "B", 2.0, 1.0, 0.8),
        ],
        columns=["home_team", "away_team", "lambda_h", "lambda_a", "lambda_shared"],
    )
    assert fixture_rates(rates, "A", "B", prefer_home=False) == (1.5, 3.0, 0.3)
    assert fixture_rates(rates, "A", "B", prefer_home=True) == (2.0, 1.0, 0.8)


def test_skellam_agrees_with_wins_draws_and_losses_on_score_grid():
    from scipy.stats import poisson

    grid = np.outer(poisson.pmf(np.arange(40), 1.8), poisson.pmf(np.arange(40), 0.65))
    np.testing.assert_allclose(
        outcome_probs(1.8, 0.65)[0],
        [np.tril(grid, -1).sum(), grid.trace(), np.triu(grid, 1).sum()],
        atol=1e-14,
        rtol=0,
    )


def test_joint_entry_point_preserves_original_draw_sequence_at_zero_covariance():
    from src.inference.features import generate_all_wc_pairings
    from src.inference.simulation import simulate_tournament

    pairs = generate_all_wc_pairings()[["home_team", "away_team"]]
    rates = pairs.assign(lambda_h=1.8, lambda_a=0.9, lambda_shared=0.0)
    old = simulate_tournament(rates, n_sims=25, seed=91)
    joint = simulate_tournament(rates, n_sims=25, seed=91, joint_poisson=True)
    for key in ("advancement", "group_positions", "ko_pairings", "ko_slot_pairings"):
        pd.testing.assert_frame_equal(old[key], joint[key])
    with pytest.raises(ValueError, match="missing a team pairing"):
        simulate_tournament(rates.head(1), n_sims=1, joint_poisson=True)
    with pytest.raises(ValueError, match="Invalid bivariate"):
        simulate_tournament(rates.assign(lambda_shared=1.0), n_sims=1, joint_poisson=True)


@pytest.mark.parametrize("order", [(88, 104), (104, 88)])
def test_model_versions_get_distinct_rates_on_same_cached_features(monkeypatch, order):
    from types import SimpleNamespace

    from src.analysis import poisson_joint_prepare as prep

    history = object.__new__(prep.HistoricalRates)
    t = pd.Timestamp("2026-07-01", tz="UTC")
    commit = SimpleNamespace(
        commit_sha="commit", commit_time=t - pd.Timedelta(days=1), gold_hash="gold"
    )
    history.created = dict.fromkeys(order, commit.commit_time)
    n = len(prep.FEATURES)
    history.models = {}
    for version, shared in [(88, 0.1), (104, 0.3)]:
        history.models[version] = {
            "mean_": np.zeros(n),
            "scale_": np.ones(n),
            "_coef1": np.zeros(n + 1),
            "_coef2": np.zeros(n + 1),
            "_coef3": np.r_[np.log(shared), np.zeros(n)],
        }
    pairs = pd.DataFrame([{"home_team": "A", "away_team": "B", **dict.fromkeys(prep.FEATURES, 0)}])
    monkeypatch.setattr(history, "gold", lambda _: (pd.DataFrame(), b"signature"))
    monkeypatch.setattr(prep.rc, "augment_gold_for_inference", lambda gold, wc: (gold, t))
    monkeypatch.setattr(prep, "generate_all_wc_pairings", lambda reference_date: pairs)
    monkeypatch.setattr(prep, "build_inference_features", lambda pairs, gold: pairs)
    history.feature_batches = {}
    for version in order:
        rates = history.predict(commit, {"finished_fixtures": []}, version, t)
        assert rates.lambda_shared.iloc[0] == pytest.approx({88: 0.1, 104: 0.3}[version])
    assert len(history.feature_batches) == 1
    with pytest.raises(ValueError, match="Future"):
        history.predict(commit, {"finished_fixtures": []}, 88, commit.commit_time)


def test_legacy_tournament_accepts_numeric_text_rates():
    from src.inference.features import generate_all_wc_pairings
    from src.inference.simulation import simulate_tournament

    pairs = generate_all_wc_pairings()[["home_team", "away_team"]]
    numeric = pairs.assign(lambda_h=1.7, lambda_a=1.1)
    text = pairs.assign(lambda_h="1.7", lambda_a="1.1")
    expected = simulate_tournament(numeric, n_sims=5, seed=123)
    actual = simulate_tournament(text, n_sims=5, seed=123)
    pd.testing.assert_frame_equal(actual["advancement"], expected["advancement"])


def test_unrecognized_match_provenance_cannot_fall_back_to_frozen():
    from src.analysis.poisson_joint_prepare import match_source

    with pytest.raises(ValueError, match="provenance"):
        match_source("per_round", "2", "unknown")
    with pytest.raises(ValueError, match="provenance"):
        match_source("per_round", "2", "md1_frozen_copied")
    assert match_source("per_round", "1", "md1_frozen_copied") == "frozen"


@pytest.mark.parametrize(
    ("label", "timestamp", "winners", "next_pair"),
    [
        ("r32_pre_japan_brazil", "2026-06-29T16:00Z", [], ("Brazil", "Japan")),
        ("r32_japan_brazil_locked", "2026-06-29T19:05Z", ["Brazil"], ("Germany", "Paraguay")),
        (
            "r32_germany_paraguay_locked",
            "2026-06-29T23:30Z",
            ["Brazil", "Paraguay"],
            ("Netherlands", "Morocco"),
        ),
        ("r16_brazil_norway_locked", "2026-07-05T22:30Z", ["Norway"], ("Mexico", "England")),
    ],
)
def test_public_outage_state_adds_only_finished_games(
    monkeypatch, label, timestamp, winners, next_pair
):
    from src.analysis import replay_common as rc
    from src.analysis.rq2_outages import SNAPSHOTS, public_results

    fixtures = []
    settlement = {}
    for fid, t, stage, home, away, hg, ag, first_seen in [
        (1562344, "06-29T17:00", "R32", "Brazil", "Japan", 2, 1, "06-30T07:31"),
        (1565176, "06-29T20:30", "R32", "Germany", "Paraguay", 1, 1, "06-30T07:31"),
        (3, "06-30T01:00", "R32", "Netherlands", "Morocco", 1, 1, "06-30T07:31"),
        (1568100, "07-05T20:00", "R16", "Brazil", "Norway", 1, 2, "07-06T18:40"),
        (5, "07-06T01:00", "R16", "Mexico", "England", 2, 3, "07-06T18:40"),
    ]:
        kickoff = pd.Timestamp(f"2026-{t}Z")
        fixtures.append(
            rc.FinishedFixture(
                fid,
                kickoff,
                str(kickoff),
                stage,
                stage,
                home,
                away,
                hg,
                ag,
                "PEN" if hg == ag else "FT",
                {},
                {"penalty": {"home": 3, "away": 4}} if hg == ag else {},
            )
        )
        settlement[fid] = pd.Timestamp(f"2026-{first_seen}Z")
    monkeypatch.setattr(rc, "load_fixture_settlement", lambda: settlement)
    monkeypatch.setattr(rc, "load_finished_wc_fixtures", lambda *args: fixtures)
    cutoff = SNAPSHOTS[label]
    assert cutoff == pd.Timestamp(timestamp)
    historical = rc.parse_wc_results_before_kickoff(cutoff, settle_delta=rc.SETTLE_DELTA)
    public = public_results(cutoff)
    added = public["ko_results"].keys() - historical["ko_results"].keys()
    assert sorted(public["ko_results"][pair]["winner"] for pair in added) == sorted(winners)
    assert frozenset(next_pair) not in public["ko_results"]
    assert public["group_results"] == historical["group_results"]
    assert historical == rc.parse_wc_results_before_kickoff(cutoff, settle_delta=rc.SETTLE_DELTA)


def test_locked_winner_and_loser_must_have_exact_advancement():
    from src.analysis.rq2_outage_comparison import check_advancement

    frame = pd.DataFrame({"team": ["Paraguay", "Germany"], "p_r16": [1.0, 0.0]})
    result = {"home": "Germany", "away": "Paraguay", "winner": "Paraguay", "stage": "R32"}
    check_advancement(frame, [result])
    with pytest.raises(ValueError, match="Paraguay"):
        check_advancement(frame.assign(p_r16=[0.6, 0.4]), [result])
