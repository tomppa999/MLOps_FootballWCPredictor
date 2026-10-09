"""Unit tests for the RQ1 paired cadence comparison (synthetic input, no I/O)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.rq1_paired import (
    CONTRAST_MODELS,
    METRICS,
    PHASE_SCHEMES,
    SUMMARY_COLUMNS,
    paired_differences,
    pairing_coverage,
    phase_table,
    round_table,
    summarise,
)

ROUNDS: dict[str, int] = {
    "1": 3, "2": 3, "3": 3, "R32": 2, "R16": 2, "QF": 1, "SF": 1, "3rd_place": 1, "Final": 1,
}
ALL_MODELS = ("mean_rate_poisson", "negbin_glm", "ridge", "random_forest", "sarimax", *CONTRAST_MODELS)


N_MATCHES = sum(ROUNDS.values())  # 17
N_MD1 = ROUNDS["1"]  # 3 placebo matches, tied by construction
N_TREATED = N_MATCHES - N_MD1  # 14
N_GROUP, N_KO = 6, 8  # MD2+MD3 / R32..Final


def _rq1_frame(*, shift: float = -0.01, noise: float = 0.0, seed: int = 1) -> pd.DataFrame:
    """Fully paired synthetic rq1_matches.

    per_round = frozen + shift (+ noise) on contrast models from MD2 onward;
    MD1 is tied in both arms, mirroring the source-level premature-refit fix.
    """
    rng = np.random.default_rng(seed)
    rows = []
    mid = 1000
    day = 0
    for round_label, n in ROUNDS.items():
        for _ in range(n):
            mid += 1
            day += 1
            kickoff = f"2026-06-{10 + day:02d} 19:00:00+00:00"
            for model in ALL_MODELS:
                base = {m: float(rng.uniform(0.05, 0.3)) for m in ("rps", "nll", "rmse_h", "rmse_a")}
                for cadence in ("frozen", "per_round"):
                    s = 0.0
                    if cadence == "per_round" and model in CONTRAST_MODELS and round_label != "1":
                        s = shift + (float(rng.normal(0, noise)) if noise else 0.0)
                    rows.append({
                        "match_id": mid, "kickoff_utc": kickoff, "round_label": round_label,
                        "cadence_mode": cadence, "model_name": model,
                        "rps": base["rps"] + s, "nll": base["nll"] + s,
                        "rmse_h": base["rmse_h"] + s, "rmse_a": base["rmse_a"] + s,
                    })
    return pd.DataFrame(rows)


def test_paired_differences_sign_and_shape() -> None:
    diffs = paired_differences(_rq1_frame(shift=-0.01))
    assert len(diffs) == len(CONTRAST_MODELS) * N_MATCHES
    md1 = diffs["block"] == "1"
    for metric in METRICS:
        # per_round - frozen; negative means per_round better
        assert np.allclose(diffs.loc[~md1, f"d_{metric}"], -0.01)
        assert np.allclose(diffs.loc[md1, f"d_{metric}"], 0.0)
    assert set(diffs["block"]) == {"1", "2", "3", "R32", "R16", "QF", "SF", "Final"}
    assert (diffs.loc[diffs["round_label"] == "3rd_place", "block"] == "Final").all()


def test_paired_differences_rejects_md1_divergence() -> None:
    df = _rq1_frame()
    m = (df["model_name"] == "xgboost") & (df["cadence_mode"] == "per_round") & (df["round_label"] == "1")
    df.loc[m, "rps"] += 0.01
    with pytest.raises(ValueError, match="MD1 frozen/per_round differ"):
        paired_differences(df)


def test_paired_differences_rejects_unpaired_rows() -> None:
    df = _rq1_frame()
    drop = (df["model_name"] == "xgboost") & (df["cadence_mode"] == "frozen") & (df["match_id"] == 1001)
    with pytest.raises(ValueError, match="unpaired"):
        paired_differences(df[~drop])


def test_paired_differences_rejects_duplicates() -> None:
    df = _rq1_frame()
    extra = df[df["model_name"] == "xgboost"].iloc[:1]
    with pytest.raises(ValueError, match="duplicate"):
        paired_differences(pd.concat([df, extra], ignore_index=True))


def test_pairing_coverage_matches_contrast_models() -> None:
    cov = pairing_coverage(_rq1_frame())
    assert set(cov["model_name"]) == set(ALL_MODELS)
    contrast = cov[cov["is_contrast"]]
    non = cov[~cov["is_contrast"]]
    assert (contrast["n_pairs"] == N_MATCHES).all()
    assert (contrast["n_diff"] == N_TREATED).all()
    assert (non["n_diff"] == 0).all()
    assert (non["frac_diff"] == 0).all()


def test_pairing_coverage_raises_when_a_degenerate_model_differs() -> None:
    df = _rq1_frame()
    m = (df["model_name"] == "ridge") & (df["cadence_mode"] == "per_round") & (df["round_label"] != "1")
    df.loc[m, "rps"] += 0.05
    with pytest.raises(ValueError, match="CONTRAST_MODELS disagrees"):
        pairing_coverage(df)


def test_summarise_recovers_effect_and_is_deterministic() -> None:
    rng = np.random.default_rng(123)
    d = rng.normal(-0.02, 0.01, size=80)
    a = summarise(d, np.random.default_rng(0), n_boot=2000, frozen=d, per_round=d)
    b = summarise(d, np.random.default_rng(0), n_boot=2000, frozen=d, per_round=d)
    assert a == b
    assert a["n"] == 80
    assert a["ci_lo"] < a["mean_diff"] < a["ci_hi"]
    assert a["ci_hi"] < 0  # clear effect: CI excludes zero
    assert a["p_perm"] < 0.01
    assert a["frac_improved"] > 0.9
    assert a["frac_tied"] == 0.0


def test_summarise_null_effect_has_large_p() -> None:
    d = np.array([0.01, -0.01] * 40)
    s = summarise(d, np.random.default_rng(0), n_boot=2000)
    assert s["mean_diff"] == pytest.approx(0.0)
    assert s["p_perm"] > 0.9
    assert s["ci_lo"] < 0 < s["ci_hi"]


def test_summarise_all_tied() -> None:
    s = summarise(np.zeros(10), np.random.default_rng(0), n_boot=500)
    assert s["mean_diff"] == 0.0
    assert s["frac_tied"] == 1.0
    assert s["frac_improved"] == 0.0
    assert s["p_perm"] == 1.0


def test_summarise_single_observation_has_nan_interval() -> None:
    s = summarise(np.array([-0.3]), np.random.default_rng(0), n_boot=100)
    assert s["n"] == 1 and s["mean_diff"] == -0.3
    assert np.isnan(s["ci_lo"]) and np.isnan(s["p_perm"])


def test_phase_table_counts_and_schemes() -> None:
    diffs = paired_differences(_rq1_frame())
    tab = phase_table(diffs, n_boot=200)
    assert list(tab.columns) == ["phase_scheme", "phase", "model_name", "metric", *SUMMARY_COLUMNS]
    # MD1 (placebo) is excluded from every phase cell.
    gk = tab[(tab["phase_scheme"] == "group_ko") & (tab["metric"] == "rps")]
    assert (gk.loc[gk["phase"] == "group", "n"] == N_GROUP).all()
    assert (gk.loc[gk["phase"] == "KO", "n"] == N_KO).all()
    r3 = tab[(tab["phase_scheme"] == "group_r32_r16plus") & (tab["metric"] == "rps")]
    assert (r3.loc[r3["phase"] == "group", "n"] == N_GROUP).all()
    assert (r3.loc[r3["phase"] == "R32", "n"] == 2).all()
    assert (r3.loc[r3["phase"] == "R16+", "n"] == 6).all()
    # Every scheme carries an "all" row over the union of its phases = treated matches.
    all_rows = tab[tab["phase"] == "all"].set_index(["phase_scheme", "model_name", "metric"])["n"]
    assert (all_rows == N_TREATED).all()
    expected_rows = sum(len(p) + 1 for p, _ in PHASE_SCHEMES.values()) * len(CONTRAST_MODELS) * len(METRICS)
    assert len(tab) == expected_rows
    # Constant shift on treated matches: mean_diff is the shift everywhere.
    assert np.allclose(tab["mean_diff"], -0.01)
    assert np.allclose(tab["mean_per_round"] - tab["mean_frozen"], -0.01)


def test_round_table_round_and_cumulative_counts() -> None:
    diffs = paired_differences(_rq1_frame())
    tab = round_table(diffs, n_boot=200)
    rps_x = tab[(tab["metric"] == "rps") & (tab["model_name"] == "xgboost")]
    rnd = rps_x[rps_x["grain"] == "round"].set_index("block")["n"]
    cum = rps_x[rps_x["grain"] == "cumulative"].set_index("block")["n"]
    assert rnd.to_dict() == {"1": 3, "2": 3, "3": 3, "R32": 2, "R16": 2, "QF": 1, "SF": 1, "Final": 2}
    assert cum.to_dict() == {"1": 3, "2": 6, "3": 9, "R32": 11, "R16": 13, "QF": 14, "SF": 15, "Final": 17}
    assert list(rnd.index) == ["1", "2", "3", "R32", "R16", "QF", "SF", "Final"]
    # MD1 stays in the round table as an exact tie; cumulative rows dilute accordingly.
    md1 = rps_x[(rps_x["block"] == "1") & (rps_x["grain"] == "round")].iloc[0]
    assert md1["mean_diff"] == 0.0 and md1["frac_tied"] == 1.0
    cum2 = rps_x[(rps_x["block"] == "2") & (rps_x["grain"] == "cumulative")].iloc[0]
    assert cum2["mean_diff"] == pytest.approx(-0.01 * 3 / 6)


def test_tables_are_deterministic_under_seed() -> None:
    diffs = paired_differences(_rq1_frame(shift=-0.003, noise=0.02, seed=7))
    a = phase_table(diffs, n_boot=300, seed=42)
    b = phase_table(diffs, n_boot=300, seed=42)
    c = phase_table(diffs, n_boot=300, seed=43)
    pd.testing.assert_frame_equal(a, b)
    assert not a["ci_lo"].equals(c["ci_lo"])
