"""RQ3: calibration over time, frozen vs per_round (concept-drift probe).

Computes ECE and a 3-way Brier decomposition (reliability / resolution /
uncertainty) from per-match probabilities in ``rq1_matches.csv``. Differences
are ``d = metric_per_round − metric_frozen`` (negative = per_round better for
ECE, REL, BS; RES is higher-is-better; UNC is identical across arms).

ECE / REL / RES are set statistics: paired match bootstrap for the CI and
per-match arm-swap for the p-value. BS is a per-match mean, so it reuses
``summarise`` from RQ1.

The trend test asks whether frozen calibration degrades over the tournament
(period-level OLS slope of ECE / REL vs period index) and whether that slope
is steeper than per_round's. Power is low (4–5 periods); see findings memo.

Outputs:
    docs/tables/rq3_calibration_rounds.csv
    docs/tables/rq3_calibration_phase.csv
    docs/tables/rq3_trend.csv

Usage:
    python -m src.analysis.rq3_calibration_drift [--n-boot 10000] [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.analysis.round_leaderboards import ROUND_BLOCKS, assign_block
from src.analysis.rq1_paired import (
    CONTRAST_MODELS,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    SUMMARY_COLUMNS,
    TABLES_ROOT,
    summarise,
)
from src.analysis.rq_datasets.paths import (
    METRIC_TOL,
    N_PROBABILITY_BINS,
    RQ1_PATH,
    RQ3_PATH,
)

RQ3_ROUNDS_PATH: Final[Path] = TABLES_ROOT / "rq3_calibration_rounds.csv"
RQ3_PHASE_PATH: Final[Path] = TABLES_ROOT / "rq3_calibration_phase.csv"
RQ3_TREND_PATH: Final[Path] = TABLES_ROOT / "rq3_trend.csv"

CI_LEVEL: Final[float] = 0.95
MIN_INTERPRETABLE_N: Final[int] = 9  # interpretable iff n > 8
DEFAULT_NOISE_FLOOR_DRAWS: Final[int] = 200

MD1_BLOCK: Final[str] = "1"
GROUP_BLOCKS: Final[frozenset[str]] = frozenset({"2", "3"})
TREATED_BLOCKS: Final[tuple[str, ...]] = tuple(b for b in ROUND_BLOCKS if b != MD1_BLOCK)

# Periods with n >= 16 for the trend regression (R16+ pools late KO).
PERIOD_ORDER: Final[tuple[str, ...]] = ("MD1", "MD2", "MD3", "R32", "R16+")
PERIOD_INDEX: Final[dict[str, int]] = {p: i for i, p in enumerate(PERIOD_ORDER)}
BLOCK_TO_PERIOD: Final[dict[str, str]] = {
    "1": "MD1",
    "2": "MD2",
    "3": "MD3",
    "R32": "R32",
    "R16": "R16+",
    "QF": "R16+",
    "SF": "R16+",
    "Final": "R16+",
}

SET_METRICS: Final[tuple[str, ...]] = ("ece", "rel", "res")
PAIR_METRICS: Final[tuple[str, ...]] = ("bs",)  # per-match means
# UNC is outcome-only (identical across arms); reported as levels only.

PROB_COLS: Final[tuple[str, ...]] = ("p_home", "p_draw", "p_away")


def calibration_stats(
    probs: np.ndarray,
    outcomes: np.ndarray,
    *,
    n_bins: int = N_PROBABILITY_BINS,
) -> dict[str, float]:
    """ECE and Murphy Brier decomposition for a 3-way outcome forecast.

    ``probs`` is (n, 3); ``outcomes`` is integer in {0, 1, 2}.
    ECE is mass-weighted |p̄ − ō| over classes and bins, denominator 3n
    (matches ``expected_calibration_error`` in rq3_calibration_overview).
    REL / RES / UNC follow Murphy (1973); ``bs_resid = BS − (REL − RES + UNC)``
    is the within-bin residual (0 when forecasts are constant in each bin).
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    n = len(outcomes)
    if n == 0:
        return {
            "n": 0,
            "ece": float("nan"),
            "bs": float("nan"),
            "rel": float("nan"),
            "res": float("nan"),
            "unc": float("nan"),
            "bs_resid": float("nan"),
        }

    y = np.zeros((n, 3), dtype=float)
    y[np.arange(n), outcomes] = 1.0
    bs = float(((probs - y) ** 2).sum(axis=1).mean())

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    cuts = edges[1:-1]
    abs_gap_mass = 0.0
    rel = 0.0
    res = 0.0
    unc = 0.0
    for k in range(3):
        p_k = probs[:, k]
        y_k = y[:, k]
        o_bar = float(y_k.mean())
        unc += o_bar * (1.0 - o_bar)
        bin_idx = np.digitize(p_k, cuts, right=False)
        # bincount over occupied bins only
        counts = np.bincount(bin_idx, minlength=n_bins)
        sum_p = np.bincount(bin_idx, weights=p_k, minlength=n_bins)
        sum_y = np.bincount(bin_idx, weights=y_k, minlength=n_bins)
        occupied = counts > 0
        n_b = counts[occupied].astype(float)
        p_bar = sum_p[occupied] / n_b
        o_b = sum_y[occupied] / n_b
        abs_gap_mass += float(np.abs(p_bar - o_b).dot(n_b))
        rel += float((n_b * (p_bar - o_b) ** 2).sum())
        res += float((n_b * (o_b - o_bar) ** 2).sum())

    ece = abs_gap_mass / (3.0 * n)
    rel /= n
    res /= n
    return {
        "n": n,
        "ece": ece,
        "bs": bs,
        "rel": rel,
        "res": res,
        "unc": unc,
        "bs_resid": bs - (rel - res + unc),
    }


def ece_from_reliability_bins(bins: pd.DataFrame) -> float:
    """ECE from pre-binned reliability rows (same formula as overview)."""
    sub = bins[bins["n"] > 0]
    if sub.empty:
        return float("nan")
    abs_gap_mass = (sub["observed_freq"] - sub["mean_predicted_p"]).abs() * sub["n"]
    return float(abs_gap_mass.sum() / sub["n"].sum())


def ece_noise_floor(
    probs: np.ndarray,
    *,
    n_bins: int = N_PROBABILITY_BINS,
    n_draws: int = DEFAULT_NOISE_FLOOR_DRAWS,
    rng: np.random.Generator | None = None,
) -> float:
    """Mean ECE under outcomes drawn from the forecast itself (consistency)."""
    probs = np.asarray(probs, dtype=float)
    n = len(probs)
    if n == 0:
        return float("nan")
    if rng is None:
        rng = np.random.default_rng(0)
    # Cumulate class probs for multinomial draws.
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0  # guard float drift
    eces: list[float] = []
    for _ in range(n_draws):
        u = rng.random(n)
        outcomes = (u[:, None] > cdf[:, :-1]).sum(axis=1).astype(int)
        eces.append(calibration_stats(probs, outcomes, n_bins=n_bins)["ece"])
    return float(np.mean(eces))


def _metric_value(
    probs: np.ndarray,
    outcomes: np.ndarray,
    metric: str,
    *,
    n_bins: int = N_PROBABILITY_BINS,
) -> float:
    """Single-metric shortcut used in the bootstrap / permutation inner loop."""
    return float(calibration_stats(probs, outcomes, n_bins=n_bins)[metric])


def summarise_set_stat(
    frozen_probs: np.ndarray,
    per_round_probs: np.ndarray,
    outcomes: np.ndarray,
    *,
    metric: str,
    rng: np.random.Generator,
    n_boot: int = DEFAULT_N_BOOT,
    n_bins: int = N_PROBABILITY_BINS,
) -> dict[str, float]:
    """Paired bootstrap CI and arm-swap p for a set statistic (ECE / REL / RES).

    ``frac_improved`` is NaN (no per-match contribution). ``frac_tied`` is 1
    when the observed difference is within ``METRIC_TOL``, else 0.
    """
    n = len(outcomes)
    if n == 0:
        return {
            "n": 0,
            "mean_frozen": float("nan"),
            "mean_per_round": float("nan"),
            "mean_diff": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_perm": float("nan"),
            "frac_improved": float("nan"),
            "frac_tied": float("nan"),
        }

    frozen_val = _metric_value(frozen_probs, outcomes, metric, n_bins=n_bins)
    per_round_val = _metric_value(per_round_probs, outcomes, metric, n_bins=n_bins)
    obs = per_round_val - frozen_val

    if n < 2:
        return {
            "n": n,
            "mean_frozen": frozen_val,
            "mean_per_round": per_round_val,
            "mean_diff": obs,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_perm": float("nan"),
            "frac_improved": float("nan"),
            "frac_tied": float(abs(obs) <= METRIC_TOL),
        }

    # Paired bootstrap: resample match indices, recompute both arms.
    idx = np.arange(n)
    boot_diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(idx, size=n, replace=True)
        boot_diffs[b] = (
            _metric_value(per_round_probs[sample], outcomes[sample], metric, n_bins=n_bins)
            - _metric_value(frozen_probs[sample], outcomes[sample], metric, n_bins=n_bins)
        )
    alpha = (1.0 - CI_LEVEL) / 2.0
    ci_lo, ci_hi = np.quantile(boot_diffs, [alpha, 1.0 - alpha])

    # Arm swap: under H0 each match's two forecasts are exchangeable.
    swap = rng.random(size=(n_boot, n)) < 0.5
    perm_diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        use_pr_as_frozen = swap[b]
        frozen_slot = np.where(use_pr_as_frozen[:, None], per_round_probs, frozen_probs)
        per_round_slot = np.where(use_pr_as_frozen[:, None], frozen_probs, per_round_probs)
        perm_diffs[b] = (
            _metric_value(per_round_slot, outcomes, metric, n_bins=n_bins)
            - _metric_value(frozen_slot, outcomes, metric, n_bins=n_bins)
        )
    p_perm = (1.0 + np.sum(np.abs(perm_diffs) >= abs(obs) - 1e-12)) / (n_boot + 1.0)

    return {
        "n": n,
        "mean_frozen": frozen_val,
        "mean_per_round": per_round_val,
        "mean_diff": obs,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_perm": float(p_perm),
        "frac_improved": float("nan"),
        "frac_tied": float(abs(obs) <= METRIC_TOL),
    }


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Ordinary least-squares slope of y on x (no intercept constraint)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float(np.dot(x_c, x_c))
    if denom <= 0:
        return float("nan")
    return float(np.dot(x_c, y_c) / denom)


def load_paired(
    df: pd.DataFrame,
    models: tuple[str, ...] = CONTRAST_MODELS,
) -> pd.DataFrame:
    """One row per (match, model) with frozen / per_round probability vectors.

    Asserts MD1 probabilities are identical across arms (placebo convention).
    """
    required = {
        "match_id",
        "kickoff_utc",
        "round_label",
        "cadence_mode",
        "model_name",
        "actual_outcome",
        *PROB_COLS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rq1_matches: missing columns {sorted(missing)}")

    sub = df[df["model_name"].isin(models)].copy()
    dup = sub.duplicated(["match_id", "model_name", "cadence_mode"], keep=False)
    if dup.any():
        raise ValueError(f"duplicate (match, model, cadence) rows: {int(dup.sum())}")
    sub["block"] = sub["round_label"].astype(str).map(assign_block)
    sub["period"] = sub["block"].map(BLOCK_TO_PERIOD)

    wide = sub.pivot_table(
        index=["match_id", "kickoff_utc", "round_label", "block", "period", "model_name", "actual_outcome"],
        columns="cadence_mode",
        values=list(PROB_COLS),
        aggfunc="first",
    )
    if wide.isna().any().any():
        bad = wide[wide.isna().any(axis=1)].index.to_frame(index=False)[["match_id", "model_name"]]
        raise ValueError(
            f"unpaired (match, model) rows: {bad.drop_duplicates().to_dict('records')[:5]}"
        )

    out = wide.index.to_frame(index=False)
    for col in PROB_COLS:
        out[f"{col}_frozen"] = wide[(col, "frozen")].to_numpy()
        out[f"{col}_per_round"] = wide[(col, "per_round")].to_numpy()

    # Design invariant: MD1 arms must tie on probabilities.
    md1 = out[out["block"] == MD1_BLOCK]
    for col in PROB_COLS:
        gap = (md1[f"{col}_per_round"] - md1[f"{col}_frozen"]).abs()
        if (gap > METRIC_TOL).any():
            raise ValueError(
                f"MD1 frozen/per_round differ on {col} for "
                f"{int((gap > METRIC_TOL).sum())} rows "
                f"(premature-refit residue not neutralised in rq1_matches.csv)"
            )

    order = {m: i for i, m in enumerate(models)}
    out["_m"] = out["model_name"].map(order)
    out = out.sort_values(["kickoff_utc", "match_id", "_m"], kind="mergesort").drop(columns="_m")
    return out.reset_index(drop=True)


def _arm_arrays(cell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frozen = cell[[f"{c}_frozen" for c in PROB_COLS]].to_numpy(dtype=float)
    per_round = cell[[f"{c}_per_round" for c in PROB_COLS]].to_numpy(dtype=float)
    outcomes = cell["actual_outcome"].to_numpy(dtype=int)
    return frozen, per_round, outcomes


def _per_match_bs(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    y = np.eye(3, dtype=float)[outcomes]
    return ((probs - y) ** 2).sum(axis=1)


def _summarise_cell(
    cell: pd.DataFrame,
    model: str,
    metric: str,
    rng: np.random.Generator,
    n_boot: int,
    *,
    n_bins: int = N_PROBABILITY_BINS,
    noise_floor_draws: int = DEFAULT_NOISE_FLOOR_DRAWS,
) -> dict[str, float]:
    g = cell[cell["model_name"] == model]
    frozen, per_round, outcomes = _arm_arrays(g)
    n = len(outcomes)

    if metric in SET_METRICS:
        summary = summarise_set_stat(
            frozen, per_round, outcomes, metric=metric, rng=rng, n_boot=n_boot, n_bins=n_bins
        )
    elif metric == "bs":
        d = _per_match_bs(per_round, outcomes) - _per_match_bs(frozen, outcomes)
        summary = summarise(
            d,
            rng,
            n_boot=n_boot,
            frozen=_per_match_bs(frozen, outcomes),
            per_round=_per_match_bs(per_round, outcomes),
        )
    elif metric == "unc":
        # Outcome-only; both arms identical. Report levels, d ≡ 0.
        stats_f = calibration_stats(frozen, outcomes, n_bins=n_bins)
        summary = {
            "n": n,
            "mean_frozen": stats_f["unc"],
            "mean_per_round": stats_f["unc"],
            "mean_diff": 0.0,
            "ci_lo": 0.0 if n >= 2 else float("nan"),
            "ci_hi": 0.0 if n >= 2 else float("nan"),
            "p_perm": 1.0 if n >= 2 else float("nan"),
            "frac_improved": float("nan"),
            "frac_tied": 1.0 if n else float("nan"),
        }
    else:
        raise ValueError(f"unknown metric: {metric}")

    # Noise floor for ECE cells (helps interpret small-n ECE levels).
    if metric == "ece" and n > 0:
        floor_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        summary["ece_noise_floor_frozen"] = ece_noise_floor(
            frozen, n_bins=n_bins, n_draws=noise_floor_draws, rng=floor_rng
        )
        summary["ece_noise_floor_per_round"] = ece_noise_floor(
            per_round, n_bins=n_bins, n_draws=noise_floor_draws, rng=floor_rng
        )
    else:
        summary["ece_noise_floor_frozen"] = float("nan")
        summary["ece_noise_floor_per_round"] = float("nan")

    summary["interpretable"] = bool(n >= MIN_INTERPRETABLE_N)
    return summary


CELL_METRICS: Final[tuple[str, ...]] = ("ece", "rel", "res", "unc", "bs")

EXTRA_COLUMNS: Final[tuple[str, ...]] = (
    "ece_noise_floor_frozen",
    "ece_noise_floor_per_round",
    "interpretable",
)


def round_table(
    paired: pd.DataFrame,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    models: tuple[str, ...] = CONTRAST_MODELS,
    noise_floor_draws: int = DEFAULT_NOISE_FLOOR_DRAWS,
) -> pd.DataFrame:
    """Per-block and cumulative calibration summary."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    seen: list[str] = []
    for block in ROUND_BLOCKS:
        seen.append(block)
        cells = {
            "round": paired[paired["block"] == block],
            "cumulative": paired[paired["block"].isin(seen)],
        }
        for grain, cell in cells.items():
            for model in models:
                for metric in CELL_METRICS:
                    rows.append(
                        {"block": block, "grain": grain, "model_name": model, "metric": metric}
                        | _summarise_cell(
                            cell, model, metric, rng, n_boot, noise_floor_draws=noise_floor_draws
                        )
                    )
    cols = ["block", "grain", "model_name", "metric", *SUMMARY_COLUMNS, *EXTRA_COLUMNS]
    return pd.DataFrame(rows, columns=cols)


def phase_table(
    paired: pd.DataFrame,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    models: tuple[str, ...] = CONTRAST_MODELS,
    noise_floor_draws: int = DEFAULT_NOISE_FLOOR_DRAWS,
) -> pd.DataFrame:
    """Phase / period calibration summary (MD1 excluded from group_ko)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    # group_ko: group = MD2+MD3, KO = R32..Final; plus "all" = treated 80.
    block_map = {
        b: ("group" if b in GROUP_BLOCKS else "KO") for b in TREATED_BLOCKS
    }
    phase_col = paired["block"].map(block_map)
    for phase in ("group", "KO", "all"):
        cell = paired[phase_col.notna()] if phase == "all" else paired[phase_col == phase]
        for model in models:
            for metric in CELL_METRICS:
                rows.append(
                    {
                        "phase_scheme": "group_ko",
                        "phase": phase,
                        "model_name": model,
                        "metric": metric,
                    }
                    | _summarise_cell(
                        cell, model, metric, rng, n_boot, noise_floor_draws=noise_floor_draws
                    )
                )

    # period5: MD1 / MD2 / MD3 / R32 / R16+
    for period in PERIOD_ORDER:
        cell = paired[paired["period"] == period]
        for model in models:
            for metric in CELL_METRICS:
                rows.append(
                    {
                        "phase_scheme": "period5",
                        "phase": period,
                        "model_name": model,
                        "metric": metric,
                    }
                    | _summarise_cell(
                        cell, model, metric, rng, n_boot, noise_floor_draws=noise_floor_draws
                    )
                )

    cols = ["phase_scheme", "phase", "model_name", "metric", *SUMMARY_COLUMNS, *EXTRA_COLUMNS]
    return pd.DataFrame(rows, columns=cols)


def _period_levels(
    paired_model: pd.DataFrame,
    metric: str,
    arm: str,
    periods: tuple[str, ...],
    *,
    n_bins: int = N_PROBABILITY_BINS,
) -> np.ndarray:
    """Period-level metric values for one arm, in ``periods`` order."""
    values = []
    for period in periods:
        cell = paired_model[paired_model["period"] == period]
        frozen, per_round, outcomes = _arm_arrays(cell)
        probs = frozen if arm == "frozen" else per_round
        values.append(calibration_stats(probs, outcomes, n_bins=n_bins)[metric])
    return np.asarray(values, dtype=float)


def trend_table(
    paired: pd.DataFrame,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    models: tuple[str, ...] = CONTRAST_MODELS,
) -> pd.DataFrame:
    """OLS slope of period-level ECE / REL vs period index, with permutation p.

    Variants:
        all5      MD1..R16+ (5 points)
        treated4  MD2..R16+ (4 points; MD1 placebo excluded)

    ``p_frozen`` is one-sided (H1: slope_frozen > 0) under period-label
    permutation that preserves period sizes. ``p_diff`` is one-sided
    (H1: slope_frozen − slope_per_round > 0) under per-match arm swap.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    variants: dict[str, tuple[str, ...]] = {
        "all5": PERIOD_ORDER,
        "treated4": tuple(p for p in PERIOD_ORDER if p != "MD1"),
    }

    for variant, periods in variants.items():
        x = np.asarray([PERIOD_INDEX[p] for p in periods], dtype=float)
        # Restrict to matches in these periods.
        sub = paired[paired["period"].isin(periods)].copy()
        for model in models:
            g = sub[sub["model_name"] == model].reset_index(drop=True)
            for metric in ("ece", "rel"):
                frozen_levels = _period_levels(g, metric, "frozen", periods)
                per_round_levels = _period_levels(g, metric, "per_round", periods)
                slope_f = _ols_slope(x, frozen_levels)
                slope_p = _ols_slope(x, per_round_levels)
                slope_diff = slope_f - slope_p  # positive => frozen degrades more

                # Period-label permutation (preserve period sizes).
                period_labels = g["period"].to_numpy()
                n_matches = len(g)
                frozen_probs = g[[f"{c}_frozen" for c in PROB_COLS]].to_numpy(dtype=float)
                outcomes = g["actual_outcome"].to_numpy(dtype=int)

                n_ge = 0
                for _ in range(n_boot):
                    labels = rng.permutation(period_labels)
                    levels = []
                    for p in periods:
                        mask = labels == p
                        levels.append(
                            calibration_stats(frozen_probs[mask], outcomes[mask])[metric]
                        )
                    s = _ols_slope(x, np.asarray(levels, dtype=float))
                    if s >= slope_f - 1e-12:
                        n_ge += 1
                p_frozen = (1.0 + n_ge) / (n_boot + 1.0)

                # Arm-swap for slope_diff.
                per_round_probs = g[[f"{c}_per_round" for c in PROB_COLS]].to_numpy(dtype=float)
                n_ge_diff = 0
                for _ in range(n_boot):
                    swap = rng.random(n_matches) < 0.5
                    f_slot = np.where(swap[:, None], per_round_probs, frozen_probs)
                    p_slot = np.where(swap[:, None], frozen_probs, per_round_probs)
                    # Attach period from g (unchanged) and recompute slopes.
                    f_levels = []
                    p_levels = []
                    for period in periods:
                        mask = g["period"].to_numpy() == period
                        f_levels.append(
                            calibration_stats(f_slot[mask], outcomes[mask])[metric]
                        )
                        p_levels.append(
                            calibration_stats(p_slot[mask], outcomes[mask])[metric]
                        )
                    diff = _ols_slope(x, np.asarray(f_levels)) - _ols_slope(
                        x, np.asarray(p_levels)
                    )
                    if diff >= slope_diff - 1e-12:
                        n_ge_diff += 1
                p_diff = (1.0 + n_ge_diff) / (n_boot + 1.0)

                rows.append({
                    "variant": variant,
                    "model_name": model,
                    "metric": metric,
                    "n_periods": len(periods),
                    "n_matches": n_matches,
                    "slope_frozen": slope_f,
                    "slope_per_round": slope_p,
                    "slope_diff": slope_diff,
                    "p_frozen": p_frozen,
                    "p_diff": p_diff,
                    "levels_frozen": ";".join(f"{v:.6g}" for v in frozen_levels),
                    "levels_per_round": ";".join(f"{v:.6g}" for v in per_round_levels),
                })

    return pd.DataFrame(rows)


def check_against_bins(
    paired: pd.DataFrame,
    rq3: pd.DataFrame,
    *,
    models: tuple[str, ...] = CONTRAST_MODELS,
    n_bins: int = N_PROBABILITY_BINS,
    tol: float = 1e-9,
) -> None:
    """Assert per-match ECE matches ECE from ``rq3_calibration.csv`` reliability bins.

    Comparison is per (cadence, model, round_label); 3rd_place stays separate
    in the bin table (not merged into Final).
    """
    rel = rq3[rq3["record_type"] == "reliability"].copy()
    rel = rel[rel["model_name"].isin(models)]

    for model in models:
        for cadence, suffix in (("frozen", "_frozen"), ("per_round", "_per_round")):
            m_paired = paired[paired["model_name"] == model]
            for round_label, g in m_paired.groupby("round_label"):
                probs = g[[f"{c}{suffix}" for c in PROB_COLS]].to_numpy(dtype=float)
                outcomes = g["actual_outcome"].to_numpy(dtype=int)
                ece_pm = calibration_stats(probs, outcomes, n_bins=n_bins)["ece"]

                bins = rel[
                    (rel["model_name"] == model)
                    & (rel["cadence_mode"] == cadence)
                    & (rel["round_label"].astype(str) == str(round_label))
                ]
                ece_bin = ece_from_reliability_bins(bins)
                if not np.isfinite(ece_pm) or not np.isfinite(ece_bin):
                    raise AssertionError(
                        f"non-finite ECE for {model}/{cadence}/{round_label}: "
                        f"per-match={ece_pm}, bins={ece_bin}"
                    )
                if abs(ece_pm - ece_bin) > tol:
                    raise AssertionError(
                        f"ECE mismatch {model}/{cadence}/{round_label}: "
                        f"per-match={ece_pm:.10g} vs bins={ece_bin:.10g}"
                    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rq1", type=Path, default=RQ1_PATH)
    parser.add_argument("--rq3", type=Path, default=RQ3_PATH)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--noise-floor-draws", type=int, default=DEFAULT_NOISE_FLOOR_DRAWS
    )
    parser.add_argument("--rounds-out", type=Path, default=RQ3_ROUNDS_PATH)
    parser.add_argument("--phase-out", type=Path, default=RQ3_PHASE_PATH)
    parser.add_argument("--trend-out", type=Path, default=RQ3_TREND_PATH)
    parser.add_argument(
        "--skip-bin-check",
        action="store_true",
        help="Skip consistency check against rq3_calibration.csv reliability bins.",
    )
    args = parser.parse_args(argv)

    rq1 = pd.read_csv(args.rq1)
    paired = load_paired(rq1)

    if not args.skip_bin_check:
        rq3 = pd.read_csv(args.rq3)
        check_against_bins(paired, rq3)
        print("Bin consistency check passed.")

    rounds = round_table(
        paired,
        n_boot=args.n_boot,
        seed=args.seed,
        noise_floor_draws=args.noise_floor_draws,
    )
    phase = phase_table(
        paired,
        n_boot=args.n_boot,
        seed=args.seed,
        noise_floor_draws=args.noise_floor_draws,
    )
    trend = trend_table(paired, n_boot=args.n_boot, seed=args.seed)

    for path, table in (
        (args.rounds_out, rounds),
        (args.phase_out, phase),
        (args.trend_out, trend),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.6g")
        print(f"Wrote {len(table)} rows to {path}")


if __name__ == "__main__":
    main()
