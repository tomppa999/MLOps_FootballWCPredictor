"""RQ2: paired frozen vs per_round Shannon-entropy comparison.

Reads ``data/analysis/rq2_entropy.csv`` and pairs each live per_round cycle to
the frozen twin that immediately precedes it within 30 minutes. Difference is

    d = H_frozen − H_per_round

so **positive d means per_round has resolved more uncertainty** (lower entropy).
This is the opposite orientation from RQ1 (``metric_per_round − metric_frozen``);
``summarise`` is called on ``−d`` and the mean / CI are flipped back so
``frac_improved`` still means "per_round better".

Windows:
    all                every live pair (n = 422)
    placebo            matchday labels ``1`` and ``2`` (premature v16 artefact)
    treated            the rest (n = 342)
    treated_unresolved treated pairs where the stage column is not yet locked

Grains:
    cycle  every live pair
    state  one row per distinct frozen+per_round entropy vector (per model)

Outputs:
    data/analysis/rq2_paired_diffs.csv     one row per live (cycle, model)
    docs/tables/rq2_paired_stage.csv       window × grain × model × stage
    docs/tables/rq2_paired_matchday.csv    matchday × grain × model × stage
    docs/tables/rq2_pairing_coverage.csv   pairing diagnostics

Usage:
    python -m src.analysis.rq2_paired [--n-boot 10000] [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.analysis.rq1_paired import (
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    SUMMARY_COLUMNS,
    TABLES_ROOT,
    summarise,
)
from src.analysis.rq_datasets.paths import ANALYSIS_ROOT, METRIC_TOL, RQ2_PATH

RQ2_PAIRED_DIFFS_PATH: Final[Path] = ANALYSIS_ROOT / "rq2_paired_diffs.csv"
RQ2_PAIRED_STAGE_PATH: Final[Path] = TABLES_ROOT / "rq2_paired_stage.csv"
RQ2_PAIRED_MATCHDAY_PATH: Final[Path] = TABLES_ROOT / "rq2_paired_matchday.csv"
RQ2_PAIRING_COVERAGE_PATH: Final[Path] = TABLES_ROOT / "rq2_pairing_coverage.csv"

STAGES: Final[tuple[str, ...]] = (
    "entropy_r32",
    "entropy_r16",
    "entropy_qf",
    "entropy_sf",
    "entropy_final",
    "entropy_winner",
)

MATCHDAY_ORDER: Final[tuple[str, ...]] = (
    "1", "2", "3", "R32", "R16", "QF", "SF", "Final", "Complete",
)

# Labels at which each stage is fully decided (d ≡ 0 from here onward).
STAGE_RESOLVED_FROM: Final[dict[str, str]] = {
    "entropy_r32": "R16",
    "entropy_r16": "R16",
    "entropy_qf": "QF",
    "entropy_sf": "SF",
    "entropy_final": "Final",
    "entropy_winner": "Complete",
}

PLACEBO_LABELS: Final[frozenset[str]] = frozenset({"1", "2"})
WINDOWS: Final[tuple[str, ...]] = ("all", "placebo", "treated", "treated_unresolved")
GRAINS: Final[tuple[str, ...]] = ("cycle", "state")

PAIR_TOLERANCE: Final[pd.Timedelta] = pd.Timedelta("30min")

ROSTER_MODELS: Final[tuple[str, ...]] = (
    "xgboost",
    "bayesian_poisson",
    "poisson_glm",
    "mean_rate_poisson",
)


def _cycle_frame(df: pd.DataFrame, cadence: str) -> pd.DataFrame:
    """One row per unique cycle for a cadence (models share timestamps)."""
    sub = df[df["cadence_mode"] == cadence][
        ["cycle_index", "inference_timestamp", "matchday_label", "synthetic", "inference_run_id"]
    ].drop_duplicates("cycle_index")
    out = sub.sort_values("inference_timestamp").reset_index(drop=True)
    out = out.rename(
        columns={
            "cycle_index": f"cycle_index_{cadence}",
            "inference_timestamp": f"ts_{cadence}",
            "matchday_label": f"matchday_{cadence}",
            "synthetic": f"synthetic_{cadence}",
            "inference_run_id": f"run_id_{cadence}",
        }
    )
    return out


def pair_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Pair each per_round cycle to its preceding frozen twin within 30 min.

    Returns one row per (live paired cycle, model) with frozen / per_round
    entropy columns and ``d_<stage> = frozen − per_round``. Synthetic pairs
    and unpaired frozen cycles are dropped. Raises on many-to-one twins,
    missing twins for a live per_round cycle, or matchday-label mismatch.
    """
    required = {
        "inference_run_id",
        "inference_timestamp",
        "cadence_mode",
        "model_name",
        "matchday_label",
        "cycle_index",
        "synthetic",
        *STAGES,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rq2_entropy: missing columns {sorted(missing)}")

    work = df.copy()
    work["inference_timestamp"] = pd.to_datetime(
        work["inference_timestamp"], utc=True, format="mixed",
    )
    work["synthetic"] = work["synthetic"].astype(bool)

    frozen_c = _cycle_frame(work, "frozen")
    per_round_c = _cycle_frame(work, "per_round")

    # merge_asof requires the join key sorted; per_round lags frozen by ~9 min.
    paired = pd.merge_asof(
        per_round_c.sort_values("ts_per_round"),
        frozen_c.sort_values("ts_frozen"),
        left_on="ts_per_round",
        right_on="ts_frozen",
        direction="backward",
        tolerance=PAIR_TOLERANCE,
    )
    unpaired_pr = paired["cycle_index_frozen"].isna()
    if unpaired_pr.any():
        bad = paired.loc[unpaired_pr, "cycle_index_per_round"].tolist()[:5]
        raise ValueError(f"per_round cycles with no frozen twin within 30 min: {bad}")

    # 1:1: each frozen twin used at most once.
    dup = paired["cycle_index_frozen"].duplicated(keep=False)
    if dup.any():
        bad = paired.loc[dup, "cycle_index_frozen"].unique().tolist()[:5]
        raise ValueError(f"frozen cycles paired to multiple per_round twins: {bad}")

    label_mismatch = paired["matchday_per_round"] != paired["matchday_frozen"]
    if label_mismatch.any():
        bad = paired.loc[label_mismatch, ["cycle_index_per_round", "matchday_per_round", "matchday_frozen"]]
        raise ValueError(f"matchday_label mismatch on twin pairs: {bad.head().to_dict('records')}")

    paired["offset_min"] = (
        (paired["ts_per_round"] - paired["ts_frozen"]).dt.total_seconds() / 60.0
    )
    paired["synthetic"] = paired["synthetic_per_round"] | paired["synthetic_frozen"]
    paired["matchday_label"] = paired["matchday_per_round"]

    # Join model-level entropy from both arms onto the cycle map.
    cycle_map = paired[
        [
            "cycle_index_per_round",
            "cycle_index_frozen",
            "ts_per_round",
            "ts_frozen",
            "matchday_label",
            "synthetic",
            "offset_min",
            "run_id_per_round",
            "run_id_frozen",
        ]
    ]
    # Drop synthetic snapshot pairs before the model join (identical timestamps).
    cycle_map = cycle_map[~cycle_map["synthetic"]].drop(columns="synthetic")

    pr = work.loc[
        work["cadence_mode"] == "per_round",
        ["cycle_index", "model_name", *STAGES],
    ].rename(columns={"cycle_index": "cycle_index_per_round", **{s: f"{s}_per_round" for s in STAGES}})
    fr = work.loc[
        work["cadence_mode"] == "frozen",
        ["cycle_index", "model_name", *STAGES],
    ].rename(columns={"cycle_index": "cycle_index_frozen", **{s: f"{s}_frozen" for s in STAGES}})

    out = (
        cycle_map
        .merge(pr, on="cycle_index_per_round", how="inner")
        .merge(fr, on=["cycle_index_frozen", "model_name"], how="inner")
    )
    for stage in STAGES:
        out[f"d_{stage}"] = out[f"{stage}_frozen"] - out[f"{stage}_per_round"]

    # Resolved-window invariant: once a stage is locked, d must be ~0.
    _assert_resolved_zeros(out)

    keep = [
        "cycle_index_per_round",
        "cycle_index_frozen",
        "ts_per_round",
        "ts_frozen",
        "matchday_label",
        "offset_min",
        "run_id_per_round",
        "run_id_frozen",
        "model_name",
        *[f"{s}_frozen" for s in STAGES],
        *[f"{s}_per_round" for s in STAGES],
        *[f"d_{s}" for s in STAGES],
    ]
    order = {m: i for i, m in enumerate(ROSTER_MODELS)}
    out["_m"] = out["model_name"].map(order)
    out = out.sort_values(["ts_per_round", "_m"], kind="mergesort").drop(columns="_m")
    return out[keep].reset_index(drop=True)


def _assert_resolved_zeros(pairs: pd.DataFrame, tol: float = METRIC_TOL) -> None:
    """Raise if any stage still differs after its resolution matchday."""
    label_rank = {lab: i for i, lab in enumerate(MATCHDAY_ORDER)}
    for stage, resolved_from in STAGE_RESOLVED_FROM.items():
        thresh = label_rank[resolved_from]
        mask = pairs["matchday_label"].map(label_rank) >= thresh
        bad = pairs.loc[mask, f"d_{stage}"].abs() > tol
        if bad.any():
            n_bad = int(bad.sum())
            raise ValueError(
                f"{stage} differs on {n_bad} pairs at/after {resolved_from} "
                f"(expected d ≡ 0 once the stage is locked)"
            )


def mark_states(pairs: pd.DataFrame, tol: float = METRIC_TOL) -> pd.DataFrame:
    """Add ``is_new_state``: True when the 12-value entropy vector changes.

    Compared within each model against the previously kept row in time order.
    """
    out = pairs.copy()
    out["is_new_state"] = False
    vec_cols = [f"{s}_frozen" for s in STAGES] + [f"{s}_per_round" for s in STAGES]

    for model, g in out.groupby("model_name", sort=False):
        idx = g.index.to_numpy()
        vals = g[vec_cols].to_numpy(dtype=float)
        is_new = np.ones(len(vals), dtype=bool)
        for i in range(1, len(vals)):
            is_new[i] = not np.allclose(vals[i], vals[i - 1], atol=tol, rtol=0.0)
        out.loc[idx, "is_new_state"] = is_new
    return out


def pairing_coverage(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-cadence and per-model pairing diagnostics."""
    work = df.copy()
    work["inference_timestamp"] = pd.to_datetime(
        work["inference_timestamp"], utc=True, format="mixed",
    )
    work["synthetic"] = work["synthetic"].astype(bool)

    rows: list[dict] = []
    paired_pr = set(pairs["cycle_index_per_round"].unique())
    paired_fr = set(pairs["cycle_index_frozen"].unique())
    for cadence, paired_ids, id_col in (
        ("frozen", paired_fr, "cycle_index_frozen"),
        ("per_round", paired_pr, "cycle_index_per_round"),
    ):
        cycles = work[work["cadence_mode"] == cadence].drop_duplicates("cycle_index")
        n_cycles = len(cycles)
        n_synthetic = int(cycles["synthetic"].sum())
        live = cycles[~cycles["synthetic"]]
        n_paired = int(live["cycle_index"].isin(paired_ids).sum())
        n_unpaired = len(live) - n_paired
        rows.append({
            "grain": "cadence",
            "key": cadence,
            "n_cycles": n_cycles,
            "n_paired": n_paired,
            "n_synthetic": n_synthetic,
            "n_unpaired": n_unpaired,
            "offset_min": float(pairs["offset_min"].min()) if len(pairs) else float("nan"),
            "offset_median": float(pairs["offset_min"].median()) if len(pairs) else float("nan"),
            "offset_max": float(pairs["offset_min"].max()) if len(pairs) else float("nan"),
            "n_live_pairs": int(pairs["cycle_index_per_round"].nunique()) if len(pairs) else 0,
            "n_states": float("nan"),
        })

    states = mark_states(pairs) if "is_new_state" not in pairs.columns else pairs
    for model in ROSTER_MODELS:
        g = states[states["model_name"] == model]
        rows.append({
            "grain": "model",
            "key": model,
            "n_cycles": float("nan"),
            "n_paired": float("nan"),
            "n_synthetic": float("nan"),
            "n_unpaired": float("nan"),
            "offset_min": float("nan"),
            "offset_median": float("nan"),
            "offset_max": float("nan"),
            "n_live_pairs": len(g),
            "n_states": int(g["is_new_state"].sum()) if len(g) else 0,
        })
    return pd.DataFrame(rows)


def _window_mask(pairs: pd.DataFrame, window: str, stage: str) -> pd.Series:
    """Boolean mask selecting rows for a (window, stage) cell."""
    is_placebo = pairs["matchday_label"].isin(PLACEBO_LABELS)
    if window == "all":
        return pd.Series(True, index=pairs.index)
    if window == "placebo":
        return is_placebo
    if window == "treated":
        return ~is_placebo
    if window == "treated_unresolved":
        label_rank = {lab: i for i, lab in enumerate(MATCHDAY_ORDER)}
        thresh = label_rank[STAGE_RESOLVED_FROM[stage]]
        unresolved = pairs["matchday_label"].map(label_rank) < thresh
        return (~is_placebo) & unresolved
    raise ValueError(f"unknown window: {window}")


def _summarise_entropy_cell(
    cell: pd.DataFrame,
    model: str,
    stage: str,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, float]:
    """Summarise d = frozen − per_round via summarise(−d) then flip mean/CI."""
    g = cell[cell["model_name"] == model]
    d = g[f"d_{stage}"].to_numpy()  # frozen − per_round
    frozen = g[f"{stage}_frozen"].to_numpy()
    per_round = g[f"{stage}_per_round"].to_numpy()
    # summarise expects per_round − frozen (negative = per_round better).
    s = summarise(-d, rng, n_boot=n_boot, frozen=frozen, per_round=per_round)
    # Flip mean and CI back to frozen − per_round orientation.
    mean = s["mean_diff"]
    lo, hi = s["ci_lo"], s["ci_hi"]
    s["mean_diff"] = -mean if mean == mean else mean  # NaN-safe
    if lo == lo and hi == hi:
        s["ci_lo"] = -hi
        s["ci_hi"] = -lo
    return s


def stage_table(
    pairs: pd.DataFrame,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Paired summary per (window, grain, model, stage)."""
    if "is_new_state" not in pairs.columns:
        pairs = mark_states(pairs)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for grain in GRAINS:
        base = pairs if grain == "cycle" else pairs[pairs["is_new_state"]]
        for window in WINDOWS:
            for stage in STAGES:
                cell = base[_window_mask(base, window, stage)]
                for model in ROSTER_MODELS:
                    rows.append(
                        {
                            "window": window,
                            "grain": grain,
                            "model_name": model,
                            "stage": stage,
                        }
                        | _summarise_entropy_cell(cell, model, stage, rng, n_boot)
                    )
    return pd.DataFrame(
        rows,
        columns=["window", "grain", "model_name", "stage", *SUMMARY_COLUMNS],
    )


def matchday_table(
    pairs: pd.DataFrame,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Paired summary per (matchday_label, grain, model, stage)."""
    if "is_new_state" not in pairs.columns:
        pairs = mark_states(pairs)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    present = [lab for lab in MATCHDAY_ORDER if lab in set(pairs["matchday_label"])]
    for grain in GRAINS:
        base = pairs if grain == "cycle" else pairs[pairs["is_new_state"]]
        for label in present:
            cell = base[base["matchday_label"] == label]
            for stage in STAGES:
                for model in ROSTER_MODELS:
                    rows.append(
                        {
                            "matchday_label": label,
                            "grain": grain,
                            "model_name": model,
                            "stage": stage,
                        }
                        | _summarise_entropy_cell(cell, model, stage, rng, n_boot)
                    )
    return pd.DataFrame(
        rows,
        columns=["matchday_label", "grain", "model_name", "stage", *SUMMARY_COLUMNS],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rq2", type=Path, default=RQ2_PATH)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--diffs-out", type=Path, default=RQ2_PAIRED_DIFFS_PATH)
    parser.add_argument("--stage-out", type=Path, default=RQ2_PAIRED_STAGE_PATH)
    parser.add_argument("--matchday-out", type=Path, default=RQ2_PAIRED_MATCHDAY_PATH)
    parser.add_argument("--coverage-out", type=Path, default=RQ2_PAIRING_COVERAGE_PATH)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.rq2)
    pairs = pair_cycles(df)
    pairs = mark_states(pairs)
    coverage = pairing_coverage(df, pairs)
    stage = stage_table(pairs, n_boot=args.n_boot, seed=args.seed)
    matchday = matchday_table(pairs, n_boot=args.n_boot, seed=args.seed)

    for path, table in (
        (args.diffs_out, pairs),
        (args.stage_out, stage),
        (args.matchday_out, matchday),
        (args.coverage_out, coverage),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.6g")
        print(f"Wrote {len(table)} rows to {path}")


if __name__ == "__main__":
    main()
