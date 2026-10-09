"""Unit tests for the RQ2 paired entropy comparison (synthetic input, no I/O)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.rq1_paired import SUMMARY_COLUMNS
from src.analysis.rq2_paired import (
    GRAINS,
    MATCHDAY_ORDER,
    ROSTER_MODELS,
    STAGE_RESOLVED_FROM,
    STAGES,
    WINDOWS,
    mark_states,
    matchday_table,
    pair_cycles,
    pairing_coverage,
    stage_table,
)

# Compact tournament: placebo (1,2), treated unresolved, then resolved labels.
# Each label has 2 live cycles; one extra frozen with no twin; one synthetic pair.
_LABEL_SCHEDULE: list[tuple[str, int]] = [
    ("1", 1),
    ("2", 2),
    ("3", 2),
    ("R32", 2),
    ("R16", 2),
    ("QF", 2),
    ("SF", 2),
    ("Final", 2),
    ("Complete", 2),
]


def _base_entropy(label: str, cadence: str, model: str, state: int) -> dict[str, float]:
    """Deterministic entropy vector that zeros out after resolution labels.

    Treated models get a constant frozen−per_round gap of +0.05 on unresolved
    stages; mean_rate stays flat; placebo carries a small residue.
    """
    label_rank = {lab: i for i, lab in enumerate(MATCHDAY_ORDER)}
    rank = label_rank[label]
    # Floor levels that decrease with tournament progress.
    levels = {
        "entropy_r32": 3.5 - 0.01 * state,
        "entropy_r16": 3.2 - 0.01 * state,
        "entropy_qf": 2.8 - 0.01 * state,
        "entropy_sf": 2.2 - 0.01 * state,
        "entropy_final": 1.5 - 0.01 * state,
        "entropy_winner": 1.0 - 0.01 * state,
    }
    # Lock stages once the resolution label is reached (identical across cadences).
    for stage, resolved_from in STAGE_RESOLVED_FROM.items():
        if rank >= label_rank[resolved_from]:
            levels[stage] = 0.0 if stage == "entropy_winner" and label == "Complete" else levels[stage]
            # Force bit-identical across cadences once locked.
            if cadence == "per_round":
                pass  # same base; gap applied below only if unresolved

    gap = 0.0
    if model != "mean_rate_poisson":
        if label in ("1", "2"):
            gap = 0.01  # placebo residue
        else:
            gap = 0.05  # treated effect

    out = {}
    for stage, base in levels.items():
        resolved_from = STAGE_RESOLVED_FROM[stage]
        locked = rank >= label_rank[resolved_from]
        if locked:
            out[stage] = float(base if stage != "entropy_winner" or label != "Complete" else 0.0)
            if stage != "entropy_winner" or label != "Complete":
                # Keep a locked non-zero level that is identical across arms.
                out[stage] = float(base)
        elif cadence == "frozen":
            out[stage] = float(base + gap)
        else:
            out[stage] = float(base)
    return out


def _rq2_frame() -> pd.DataFrame:
    """Fully twin-paired synthetic rq2_entropy with one orphan frozen and one synthetic."""
    rows: list[dict] = []
    t0 = pd.Timestamp("2026-06-11 16:00:00", tz="UTC")
    cycle_f = 0
    cycle_p = 0
    state = 0
    run = 0

    for label, n_cycles in _LABEL_SCHEDULE:
        for c in range(n_cycles):
            # Two repeats of the same state inside each label (tests state grain).
            if c == 0:
                state += 1
            ts_f = t0 + pd.Timedelta(hours=2 * (cycle_f + cycle_p) + cycle_f)
            ts_p = ts_f + pd.Timedelta(minutes=9)
            for model in ROSTER_MODELS:
                ent_f = _base_entropy(label, "frozen", model, state)
                ent_p = _base_entropy(label, "per_round", model, state)
                rows.append({
                    "inference_run_id": f"f{run}",
                    "inference_timestamp": ts_f.isoformat(),
                    "cadence_mode": "frozen",
                    "model_name": model,
                    "matchday_label": label,
                    "cycle_index": cycle_f,
                    "synthetic": False,
                    "snapshot_label": "",
                    "provenance": "reconstructed_frozen_shadow",
                    **ent_f,
                })
                rows.append({
                    "inference_run_id": f"p{run}",
                    "inference_timestamp": ts_p.isoformat(),
                    "cadence_mode": "per_round",
                    "model_name": model,
                    "matchday_label": label,
                    "cycle_index": cycle_p,
                    "synthetic": False,
                    "snapshot_label": "",
                    "provenance": "live_replay",
                    **ent_p,
                })
            cycle_f += 1
            cycle_p += 1
            run += 1

    # Orphan frozen cycle (no per_round twin within 30 min).
    ts_orphan = t0 + pd.Timedelta(hours=500)
    for model in ROSTER_MODELS:
        ent = _base_entropy("SF", "frozen", model, state)
        rows.append({
            "inference_run_id": "orphan",
            "inference_timestamp": ts_orphan.isoformat(),
            "cadence_mode": "frozen",
            "model_name": model,
            "matchday_label": "SF",
            "cycle_index": cycle_f,
            "synthetic": False,
            "snapshot_label": "",
            "provenance": "reconstructed_frozen_shadow",
            **ent,
        })
    cycle_f += 1

    # Synthetic snapshot pair (identical timestamps) — must be dropped.
    ts_syn = pd.Timestamp("2026-06-29 16:00:00", tz="UTC")
    for cadence, ci in (("frozen", cycle_f), ("per_round", cycle_p)):
        for model in ROSTER_MODELS:
            ent = _base_entropy("R32", cadence, model, state)
            rows.append({
                "inference_run_id": f"syn_{cadence}",
                "inference_timestamp": ts_syn.isoformat(),
                "cadence_mode": cadence,
                "model_name": model,
                "matchday_label": "R32",
                "cycle_index": ci,
                "synthetic": True,
                "snapshot_label": "r32_pre_japan_brazil",
                "provenance": "reconstructed_snapshot",
                **ent,
            })

    return pd.DataFrame(rows)


def test_pair_cycles_shape_and_mapping() -> None:
    df = _rq2_frame()
    pairs = pair_cycles(df)
    n_live_cycles = sum(n for _, n in _LABEL_SCHEDULE)
    assert pairs["cycle_index_per_round"].nunique() == n_live_cycles
    assert len(pairs) == n_live_cycles * len(ROSTER_MODELS)
    # 1:1 twin mapping.
    assert pairs["cycle_index_frozen"].nunique() == n_live_cycles
    assert (pairs.groupby("cycle_index_per_round")["cycle_index_frozen"].nunique() == 1).all()
    # Synthetic dropped; orphan frozen not present.
    assert pairs["offset_min"].min() > 0
    assert not pairs["cycle_index_frozen"].isna().any()


def test_pair_cycles_drops_synthetic_and_orphan() -> None:
    df = _rq2_frame()
    pairs = pair_cycles(df)
    syn_ids = set(df.loc[df["synthetic"], "cycle_index"])
    assert not pairs["cycle_index_per_round"].isin(syn_ids).any()
    assert not pairs["cycle_index_frozen"].isin(syn_ids).any()
    orphan_ci = df.loc[df["inference_run_id"] == "orphan", "cycle_index"].iloc[0]
    assert orphan_ci not in set(pairs["cycle_index_frozen"])


def test_pair_cycles_raises_on_label_mismatch() -> None:
    df = _rq2_frame()
    # Shift one per_round label so the twin lands on a different matchday.
    m = (df["cadence_mode"] == "per_round") & (df["cycle_index"] == 0)
    df.loc[m, "matchday_label"] = "2"
    with pytest.raises(ValueError, match="matchday_label mismatch"):
        pair_cycles(df)


def test_pair_cycles_raises_on_duplicate_twin() -> None:
    df = _rq2_frame()
    # Insert a second per_round cycle that asof-matches the same frozen.
    extra = df[(df["cadence_mode"] == "per_round") & (df["cycle_index"] == 0)].copy()
    extra["cycle_index"] = df["cycle_index"].max() + 1
    extra["inference_run_id"] = "dup"
    # 1 minute after the original per_round so it still maps to the same frozen.
    ts = pd.to_datetime(extra["inference_timestamp"].iloc[0], utc=True) + pd.Timedelta(minutes=1)
    extra["inference_timestamp"] = ts.isoformat()
    with pytest.raises(ValueError, match="multiple per_round"):
        pair_cycles(pd.concat([df, extra], ignore_index=True))


def test_pair_cycles_raises_on_resolved_violation() -> None:
    df = _rq2_frame()
    # Break the lock: at Complete, make frozen winner entropy differ.
    m = (
        (df["cadence_mode"] == "frozen")
        & (df["matchday_label"] == "Complete")
        & (df["model_name"] == "xgboost")
        & (~df["synthetic"])
    )
    df.loc[m, "entropy_winner"] = 0.5
    with pytest.raises(ValueError, match="entropy_winner differs"):
        pair_cycles(df)


def test_sign_orientation_and_frac_improved() -> None:
    pairs = mark_states(pair_cycles(_rq2_frame()))
    # Treated unresolved winner: d = +0.05 for fitted models.
    treated = pairs[~pairs["matchday_label"].isin({"1", "2"})]
    unresolved = treated[treated["matchday_label"] != "Complete"]
    xgb = unresolved[unresolved["model_name"] == "xgboost"]
    assert np.allclose(xgb["d_entropy_winner"], 0.05)

    tab = stage_table(pairs, n_boot=500, seed=0)
    cell = tab[
        (tab["window"] == "treated_unresolved")
        & (tab["grain"] == "cycle")
        & (tab["model_name"] == "xgboost")
        & (tab["stage"] == "entropy_winner")
    ].iloc[0]
    assert cell["mean_diff"] == pytest.approx(0.05)
    assert cell["ci_lo"] > 0
    assert cell["frac_improved"] == pytest.approx(1.0)
    assert cell["p_perm"] < 0.01


def test_state_grain_dedups_repeats() -> None:
    pairs = mark_states(pair_cycles(_rq2_frame()))
    for model in ROSTER_MODELS:
        g = pairs[pairs["model_name"] == model]
        n_states = int(g["is_new_state"].sum())
        # One new state per label (repeats within label share the vector).
        assert n_states == len(_LABEL_SCHEDULE)
        tab = stage_table(pairs, n_boot=100, seed=0)
        all_state = tab[
            (tab["window"] == "all")
            & (tab["grain"] == "state")
            & (tab["model_name"] == model)
            & (tab["stage"] == "entropy_winner")
        ].iloc[0]
        assert all_state["n"] == n_states


def test_window_counts() -> None:
    pairs = mark_states(pair_cycles(_rq2_frame()))
    tab = stage_table(pairs, n_boot=100, seed=0)
    xgb = tab[
        (tab["grain"] == "cycle")
        & (tab["model_name"] == "xgboost")
        & (tab["stage"] == "entropy_winner")
    ].set_index("window")
    n_placebo = sum(n for lab, n in _LABEL_SCHEDULE if lab in {"1", "2"})
    n_all = sum(n for _, n in _LABEL_SCHEDULE)
    n_treated = n_all - n_placebo
    assert xgb.loc["all", "n"] == n_all
    assert xgb.loc["placebo", "n"] == n_placebo
    assert xgb.loc["treated", "n"] == n_treated
    # treated_unresolved for winner excludes Complete.
    n_complete = sum(n for lab, n in _LABEL_SCHEDULE if lab == "Complete")
    assert xgb.loc["treated_unresolved", "n"] == n_treated - n_complete
    # Stage r32 unresolved shrinks further (resolved from R16).
    r32 = tab[
        (tab["grain"] == "cycle")
        & (tab["model_name"] == "xgboost")
        & (tab["stage"] == "entropy_r32")
        & (tab["window"] == "treated_unresolved")
    ].iloc[0]
    n_before_r16 = sum(
        n for lab, n in _LABEL_SCHEDULE if lab not in {"1", "2"} and MATCHDAY_ORDER.index(lab) < MATCHDAY_ORDER.index("R16")
    )
    assert r32["n"] == n_before_r16


def test_table_columns_and_determinism() -> None:
    pairs = mark_states(pair_cycles(_rq2_frame()))
    a = stage_table(pairs, n_boot=300, seed=42)
    b = stage_table(pairs, n_boot=300, seed=42)
    c = stage_table(pairs, n_boot=300, seed=43)
    assert list(a.columns) == ["window", "grain", "model_name", "stage", *SUMMARY_COLUMNS]
    pd.testing.assert_frame_equal(a, b)
    assert not a["ci_lo"].equals(c["ci_lo"])
    assert set(a["window"]) == set(WINDOWS)
    assert set(a["grain"]) == set(GRAINS)
    assert set(a["stage"]) == set(STAGES)
    assert set(a["model_name"]) == set(ROSTER_MODELS)

    md = matchday_table(pairs, n_boot=100, seed=0)
    assert list(md.columns) == ["matchday_label", "grain", "model_name", "stage", *SUMMARY_COLUMNS]
    assert list(md["matchday_label"].unique()) == [lab for lab, _ in _LABEL_SCHEDULE]


def test_plot_entropy_stages_smoke() -> None:
    from src.analysis.plot_entropy_stages import plot_entropy_stages

    df = _rq2_frame()
    df["inference_timestamp"] = pd.to_datetime(df["inference_timestamp"], utc=True)
    df["synthetic"] = df["synthetic"].astype(bool)
    # Restrict to roster models already in the frame; crop like the real plot.
    x_start = df["inference_timestamp"].min()
    fig = plot_entropy_stages(df, x_start=x_start)
    assert len(fig.axes) >= 6
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_pairing_coverage_counts() -> None:
    df = _rq2_frame()
    pairs = mark_states(pair_cycles(df))
    cov = pairing_coverage(df, pairs)
    fr = cov[(cov["grain"] == "cadence") & (cov["key"] == "frozen")].iloc[0]
    pr = cov[(cov["grain"] == "cadence") & (cov["key"] == "per_round")].iloc[0]
    n_live = sum(n for _, n in _LABEL_SCHEDULE)
    assert fr["n_paired"] == n_live
    assert fr["n_unpaired"] == 1  # orphan
    assert fr["n_synthetic"] == 1
    assert pr["n_paired"] == n_live
    assert pr["n_unpaired"] == 0
    assert pr["n_synthetic"] == 1
    xgb = cov[(cov["grain"] == "model") & (cov["key"] == "xgboost")].iloc[0]
    assert xgb["n_live_pairs"] == n_live
    assert xgb["n_states"] == len(_LABEL_SCHEDULE)
