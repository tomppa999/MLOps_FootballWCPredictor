"""Unit tests for RQ3 calibration-drift analysis (synthetic input, no I/O)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.rq1_paired import CONTRAST_MODELS, SUMMARY_COLUMNS
from src.analysis.rq3_calibration_drift import (
    BLOCK_TO_PERIOD,
    CELL_METRICS,
    MIN_INTERPRETABLE_N,
    PERIOD_ORDER,
    calibration_stats,
    check_against_bins,
    ece_from_reliability_bins,
    load_paired,
    phase_table,
    round_table,
    summarise_set_stat,
    trend_table,
)
from src.analysis.rq_datasets.build_rq_datasets import _reliability_bins
from src.analysis.rq_datasets.paths import N_PROBABILITY_BINS


def _perfect_probs(outcomes: np.ndarray) -> np.ndarray:
    """One-hot probabilities matching outcomes → perfect calibration within bins."""
    return np.eye(3, dtype=float)[outcomes]


def test_calibration_stats_perfect() -> None:
    outcomes = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
    probs = _perfect_probs(outcomes)
    stats = calibration_stats(probs, outcomes, n_bins=10)
    assert stats["n"] == 10
    assert stats["ece"] == pytest.approx(0.0, abs=1e-12)
    assert stats["rel"] == pytest.approx(0.0, abs=1e-12)
    assert stats["bs"] == pytest.approx(0.0, abs=1e-12)
    # Constant within bins (one-hot) ⇒ Brier identity exact.
    assert stats["bs_resid"] == pytest.approx(0.0, abs=1e-12)
    assert stats["bs"] == pytest.approx(
        stats["rel"] - stats["res"] + stats["unc"], abs=1e-12
    )


def test_brier_identity_constant_within_bins() -> None:
    """Forecasts constant in each bin → BS = REL − RES + UNC exactly."""
    # All forecasts in the same mid-bin for each class pattern.
    probs = np.tile(np.array([0.55, 0.25, 0.20]), (20, 1))
    rng = np.random.default_rng(0)
    outcomes = rng.choice(3, size=20, p=[0.55, 0.25, 0.20])
    stats = calibration_stats(probs, outcomes, n_bins=10)
    assert stats["bs_resid"] == pytest.approx(0.0, abs=1e-12)
    assert stats["bs"] == pytest.approx(
        stats["rel"] - stats["res"] + stats["unc"], abs=1e-12
    )


def test_per_match_ece_matches_reliability_bins() -> None:
    rng = np.random.default_rng(2)
    n = 40
    rows = []
    for i in range(n):
        p = rng.dirichlet([2, 1, 1])
        y = int(rng.choice(3, p=p))
        rows.append({
            "match_id": 1000 + i,
            "kickoff_utc": f"2026-06-{(i % 28) + 1:02d} 19:00:00+00:00",
            "round_label": "2",
            "cadence_mode": "frozen",
            "model_name": "xgboost",
            "actual_outcome": y,
            "p_home": float(p[0]),
            "p_draw": float(p[1]),
            "p_away": float(p[2]),
            "rps": 0.1,
            "nll": 1.0,
        })
    rq1 = pd.DataFrame(rows)
    probs = rq1[["p_home", "p_draw", "p_away"]].to_numpy()
    outcomes = rq1["actual_outcome"].to_numpy()
    ece_pm = calibration_stats(probs, outcomes)["ece"]

    bins = _reliability_bins(rq1, n_bins=N_PROBABILITY_BINS)
    ece_bin = ece_from_reliability_bins(bins)
    assert ece_pm == pytest.approx(ece_bin, abs=1e-12)


def _synthetic_rq1(
    *,
    n_per_period: dict[str, int] | None = None,
    degrade_frozen: bool = False,
    seed: int = 1,
) -> pd.DataFrame:
    """Fully paired synthetic rq1 with probabilities.

    MD1 arms are identical. From MD2 onward, contrast models may diverge.
    If ``degrade_frozen``, frozen arm drifts toward uniform over periods.
    """
    if n_per_period is None:
        n_per_period = {
            "1": 4, "2": 4, "3": 4, "R32": 4, "R16": 3, "QF": 2, "SF": 1, "Final": 1,
        }
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    mid = 1000
    day = 0
    period_of = {
        "1": "MD1", "2": "MD2", "3": "MD3", "R32": "R32",
        "R16": "R16+", "QF": "R16+", "SF": "R16+", "Final": "R16+",
    }
    period_idx = {p: i for i, p in enumerate(PERIOD_ORDER)}

    for round_label, n in n_per_period.items():
        for _ in range(n):
            mid += 1
            day += 1
            kickoff = f"2026-06-{(day % 28) + 1:02d} 12:00:00+00:00"
            # True outcome drawn from a peaked distribution.
            true_p = np.array([0.55, 0.25, 0.20])
            outcome = int(rng.choice(3, p=true_p))
            for model in CONTRAST_MODELS:
                # Base well-calibrated forecast shared by both arms on MD1.
                base = true_p + rng.normal(0, 0.02, size=3)
                base = np.clip(base, 0.01, None)
                base = base / base.sum()
                for cadence in ("frozen", "per_round"):
                    if round_label == "1":
                        p = base  # placebo: identical arms
                    elif cadence == "frozen" and degrade_frozen:
                        # Systematically overconfident on wrong classes as
                        # the period index grows (clear ECE degradation).
                        t = period_idx[period_of[round_label]] / 4.0
                        wrong = np.array([0.10, 0.20, 0.70])  # peaked on away
                        p = (1.0 - t) * true_p + t * wrong
                        p = p + rng.normal(0, 0.01, size=3)
                        p = np.clip(p, 0.01, None)
                        p = p / p.sum()
                    else:
                        p = true_p + rng.normal(0, 0.02, size=3)
                        p = np.clip(p, 0.01, None)
                        p = p / p.sum()
                    rows.append({
                        "match_id": mid,
                        "kickoff_utc": kickoff,
                        "round_label": round_label,
                        "cadence_mode": cadence,
                        "model_name": model,
                        "actual_outcome": outcome,
                        "p_home": float(p[0]),
                        "p_draw": float(p[1]),
                        "p_away": float(p[2]),
                        "rps": 0.1,
                        "nll": 1.0,
                    })
    return pd.DataFrame(rows)


def test_identical_arms_give_zero_diff_and_p_one() -> None:
    """When both arms have the same probabilities, d ≡ 0 and p ≈ 1."""
    df = _synthetic_rq1(degrade_frozen=False, seed=3)
    # Force per_round := frozen for every row.
    frozen = df[df["cadence_mode"] == "frozen"].copy()
    per = frozen.copy()
    per["cadence_mode"] = "per_round"
    df = pd.concat([frozen, per], ignore_index=True)

    paired = load_paired(df)
    rng = np.random.default_rng(0)
    cell = paired[paired["model_name"] == "xgboost"]
    frozen_p = cell[[f"{c}_frozen" for c in ("p_home", "p_draw", "p_away")]].to_numpy()
    per_p = cell[[f"{c}_per_round" for c in ("p_home", "p_draw", "p_away")]].to_numpy()
    outcomes = cell["actual_outcome"].to_numpy()

    for metric in ("ece", "rel", "res"):
        s = summarise_set_stat(
            frozen_p, per_p, outcomes, metric=metric, rng=rng, n_boot=200
        )
        assert s["mean_diff"] == pytest.approx(0.0, abs=1e-12)
        assert s["p_perm"] == pytest.approx(1.0, abs=1e-12)


def test_injected_frozen_degradation_gives_positive_slope() -> None:
    """Frozen drifting toward uniform → positive ECE slope and small p_frozen."""
    df = _synthetic_rq1(
        n_per_period={
            "1": 12, "2": 12, "3": 12, "R32": 12, "R16": 6, "QF": 3, "SF": 2, "Final": 1,
        },
        degrade_frozen=True,
        seed=5,
    )
    paired = load_paired(df)
    trend = trend_table(paired, n_boot=500, seed=0)
    row = trend[
        (trend["variant"] == "all5")
        & (trend["model_name"] == "xgboost")
        & (trend["metric"] == "ece")
    ].iloc[0]
    assert row["slope_frozen"] > 0
    assert row["p_frozen"] < 0.1
    assert row["slope_diff"] > 0  # frozen degrades more than per_round


def test_period_permutation_preserves_sizes() -> None:
    """Sanity: shuffling period labels keeps the per-period counts."""
    labels = np.array(["MD1"] * 4 + ["MD2"] * 4 + ["MD3"] * 4 + ["R32"] * 4 + ["R16+"] * 4)
    rng = np.random.default_rng(7)
    for _ in range(20):
        shuffled = rng.permutation(labels)
        for p in PERIOD_ORDER:
            assert int((shuffled == p).sum()) == int((labels == p).sum())


def test_md1_mismatch_raises() -> None:
    df = _synthetic_rq1()
    m = (
        (df["model_name"] == "xgboost")
        & (df["cadence_mode"] == "per_round")
        & (df["round_label"] == "1")
    )
    df.loc[m, "p_home"] = df.loc[m, "p_home"] + 0.05
    # Renormalise so probs still sum ~1 (not required for the assert).
    with pytest.raises(ValueError, match="MD1 frozen/per_round differ"):
        load_paired(df)


def test_interpretable_false_for_small_n() -> None:
    df = _synthetic_rq1(
        n_per_period={
            "1": 4, "2": 4, "3": 4, "R32": 4, "R16": 3, "QF": 2, "SF": 1, "Final": 1,
        }
    )
    paired = load_paired(df)
    rounds = round_table(paired, n_boot=50, seed=0, noise_floor_draws=5)
    r16 = rounds[(rounds["block"] == "R16") & (rounds["grain"] == "round")]
    assert (r16["n"] == 3).all()
    assert (r16["interpretable"] == False).all()  # noqa: E712
    assert MIN_INTERPRETABLE_N == 9

    group = phase_table(paired, n_boot=50, seed=0, noise_floor_draws=5)
    # MD2+MD3 = 8 matches → still not interpretable under n > 8 rule.
    g = group[(group["phase_scheme"] == "group_ko") & (group["phase"] == "group")]
    assert (g["n"] == 8).all()
    assert (g["interpretable"] == False).all()  # noqa: E712


def test_check_against_bins_passes_on_synthetic() -> None:
    df = _synthetic_rq1()
    paired = load_paired(df)
    # Build a fake rq3 reliability table from the long form.
    long_rows = []
    for cadence in ("frozen", "per_round"):
        sub = df[df["cadence_mode"] == cadence]
        bins = _reliability_bins(sub)
        long_rows.append(bins)
    rq3 = pd.concat(long_rows, ignore_index=True)
    check_against_bins(paired, rq3)


def test_round_and_phase_column_contract() -> None:
    df = _synthetic_rq1()
    paired = load_paired(df)
    assert set(paired["period"]) <= set(PERIOD_ORDER)
    assert set(BLOCK_TO_PERIOD.values()) == set(PERIOD_ORDER)

    rounds = round_table(paired, n_boot=30, seed=0, noise_floor_draws=3)
    for col in SUMMARY_COLUMNS:
        assert col in rounds.columns
    assert set(rounds["metric"]) == set(CELL_METRICS)

    phase = phase_table(paired, n_boot=30, seed=0, noise_floor_draws=3)
    assert set(phase["phase_scheme"]) == {"group_ko", "period5"}
    assert "all" in set(phase.loc[phase["phase_scheme"] == "group_ko", "phase"])
