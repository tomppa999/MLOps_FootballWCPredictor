"""RQ1: paired per-match comparison of per_round vs frozen cadence.

Reads ``data/analysis/rq1_matches.csv`` and, for every (match, model) pair,
takes the difference ``d = metric_per_round - metric_frozen``. All metrics are
lower-is-better, so **negative d means per_round did better**. Pairing removes
the match-difficulty variance that an unpaired mean comparison would carry.

Per (model, metric, cell) the mean difference is reported with a percentile
bootstrap CI and a paired sign-flip permutation p-value (see ``summarise``).

Only models whose predictions actually differ between cadences are compared
(``CONTRAST_MODELS``); the other five shadows resolved to the same artifact in
both modes during the tournament (shadow-resolution bug, ``wc_live.md`` MD2),
so a paired test on them is degenerate. ``rq1_pairing_coverage.csv`` proves
this from the data rather than from prose (``n_diff`` = 80 for a contrast
model: 104 matches minus the 24 MD1 placebo matches).

Outputs:
    data/analysis/rq1_paired_diffs.csv    one row per (match, model), all metrics
    docs/tables/rq1_paired_phase.csv      group vs KO; group / R32 / R16+ (MD1 excluded)
    docs/tables/rq1_paired_rounds.csv     per round block and cumulative (MD1 = placebo, d ≡ 0)
    docs/tables/rq1_pairing_coverage.csv  n_diff per model (contrast validity)

Usage:
    python -m src.analysis.rq1_paired [--n-boot 10000] [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.analysis.round_leaderboards import ROUND_BLOCKS, assign_block
from src.analysis.rq_datasets.paths import ANALYSIS_ROOT, METRIC_TOL, RQ1_PATH

# Output paths live here, not in rq_datasets/paths.py: that module is a dep of
# every D.1 strand stage, so touching it would invalidate the reconstruction.
# Paper/thesis tables are committed to Git (dvc ``cache: false``), like the figures.
TABLES_ROOT: Final[Path] = Path("docs/tables")
RQ1_PAIRED_DIFFS_PATH: Final[Path] = ANALYSIS_ROOT / "rq1_paired_diffs.csv"
RQ1_PAIRED_PHASE_PATH: Final[Path] = TABLES_ROOT / "rq1_paired_phase.csv"
RQ1_PAIRED_ROUNDS_PATH: Final[Path] = TABLES_ROOT / "rq1_paired_rounds.csv"
RQ1_PAIRING_COVERAGE_PATH: Final[Path] = TABLES_ROOT / "rq1_pairing_coverage.csv"

METRICS: Final[tuple[str, ...]] = ("rps", "nll", "rmse")

# The only models refit at round boundaries whose frozen twin is a distinct
# artifact (live xgboost; Strand 1 reconstruction for the two GLMs).
CONTRAST_MODELS: Final[tuple[str, ...]] = ("xgboost", "poisson_glm", "bayesian_poisson")

# MD1 is excluded from every phase cell: the first correctly gated per_round
# refit ran after MD1, so both arms served the same model there and
# rq_datasets sets per_round MD1 := frozen at source (premature-refit fix).
# MD1 stays in the round table (diff exactly 0) so the cumulative curve keeps
# its 104-match denominator.
MD1_BLOCK: Final[str] = "1"
GROUP_BLOCKS: Final[frozenset[str]] = frozenset({"2", "3"})
TREATED_BLOCKS: Final[tuple[str, ...]] = tuple(b for b in ROUND_BLOCKS if b != MD1_BLOCK)

# phase_scheme -> (ordered phase labels, block -> phase)
PHASE_SCHEMES: Final[dict[str, tuple[tuple[str, ...], dict[str, str]]]] = {
    # RQ1 as worded: group stage vs knockout (48 / 32 treated matches).
    "group_ko": (
        ("group", "KO"),
        {b: ("group" if b in GROUP_BLOCKS else "KO") for b in TREATED_BLOCKS},
    ),
    # Equal-n KO split: at R32 the per_round model has only seen group matches,
    # from R16 onward it has also been refit on knockout matches (16 / 16).
    "group_r32_r16plus": (
        ("group", "R32", "R16+"),
        {b: ("group" if b in GROUP_BLOCKS else ("R32" if b == "R32" else "R16+")) for b in TREATED_BLOCKS},
    ),
}
ALL_PHASE: Final[str] = "all"

DEFAULT_N_BOOT: Final[int] = 10_000
DEFAULT_SEED: Final[int] = 0
CI_LEVEL: Final[float] = 0.95

SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "n",
    "mean_frozen",
    "mean_per_round",
    "mean_diff",
    "ci_lo",
    "ci_hi",
    "p_perm",
    "frac_improved",
    "frac_tied",
)


def paired_differences(df: pd.DataFrame, models: tuple[str, ...] = CONTRAST_MODELS) -> pd.DataFrame:
    """Return one row per (match, model) with frozen / per_round values and diffs.

    Raises if any match is missing one cadence for a requested model, so a
    silent unpaired mean can never enter the tables.
    """
    required = {"match_id", "kickoff_utc", "round_label", "cadence_mode", "model_name", "rps", "nll", "rmse_h", "rmse_a"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rq1_matches: missing columns {sorted(missing)}")

    sub = df[df["model_name"].isin(models)].copy()
    dup = sub.duplicated(["match_id", "model_name", "cadence_mode"], keep=False)
    if dup.any():
        raise ValueError(f"duplicate (match, model, cadence) rows: {int(dup.sum())}")
    sub["rmse"] = (sub["rmse_h"] + sub["rmse_a"]) / 2.0
    sub["block"] = sub["round_label"].astype(str).map(assign_block)

    wide = sub.pivot_table(
        index=["match_id", "kickoff_utc", "round_label", "block", "model_name"],
        columns="cadence_mode",
        values=list(METRICS),
        aggfunc="first",  # keys are unique (checked above); "first" is a no-op
    )
    if wide.isna().any().any():
        bad = wide[wide.isna().any(axis=1)].index.to_frame(index=False)[["match_id", "model_name"]]
        raise ValueError(f"unpaired (match, model) rows: {bad.drop_duplicates().to_dict('records')[:5]}")

    out = wide.index.to_frame(index=False)
    for metric in METRICS:
        out[f"{metric}_frozen"] = wide[(metric, "frozen")].to_numpy()
        out[f"{metric}_per_round"] = wide[(metric, "per_round")].to_numpy()
        out[f"d_{metric}"] = out[f"{metric}_per_round"] - out[f"{metric}_frozen"]

    expected = len(models) * sub["match_id"].nunique()
    if len(out) != expected:
        raise ValueError(f"expected {expected} paired rows, got {len(out)}")

    # Design invariant: no refit before MD1, so MD1 must be exactly tied.
    md1 = out[out["block"] == MD1_BLOCK]
    for metric in METRICS:
        bad = md1[md1[f"d_{metric}"].abs() > METRIC_TOL]
        if not bad.empty:
            raise ValueError(
                f"MD1 frozen/per_round differ on {metric} for {len(bad)} rows "
                f"(premature-refit residue not neutralised in rq1_matches.csv)"
            )

    order = {m: i for i, m in enumerate(models)}
    out["_m"] = out["model_name"].map(order)
    out = out.sort_values(["kickoff_utc", "match_id", "_m"], kind="mergesort").drop(columns="_m")
    return out.reset_index(drop=True)


def pairing_coverage(df: pd.DataFrame, tol: float = METRIC_TOL) -> pd.DataFrame:
    """Per model: how many paired matches have any metric differing across cadences.

    ``is_contrast`` is the explicit ``CONTRAST_MODELS`` membership; the function
    raises if the data disagree with that list in either direction.
    """
    models = tuple(sorted(df["model_name"].unique()))
    diffs = paired_differences(df, models=models)
    any_diff = np.zeros(len(diffs), dtype=bool)
    for metric in METRICS:
        any_diff |= diffs[f"d_{metric}"].abs().to_numpy() > tol
    diffs = diffs.assign(any_diff=any_diff)

    cov = (
        diffs.groupby("model_name", as_index=False)
        .agg(n_pairs=("any_diff", "size"), n_diff=("any_diff", "sum"))
        .assign(frac_diff=lambda t: t["n_diff"] / t["n_pairs"])
    )
    cov["is_contrast"] = cov["model_name"].isin(CONTRAST_MODELS)

    wrong = cov[(cov["n_diff"] > 0) != cov["is_contrast"]]
    if not wrong.empty:
        raise ValueError(
            "CONTRAST_MODELS disagrees with the data: "
            + wrong[["model_name", "n_diff", "is_contrast"]].to_dict("records").__repr__()
        )
    return cov.sort_values("model_name", kind="mergesort").reset_index(drop=True)


def summarise(
    d: np.ndarray,
    rng: np.random.Generator,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    frozen: np.ndarray | None = None,
    per_round: np.ndarray | None = None,
    tol: float = METRIC_TOL,
) -> dict[str, float]:
    """Mean paired difference with bootstrap CI and sign-flip permutation p-value.

    Bootstrap: resample the n differences with replacement ``n_boot`` times and
    take the 2.5 / 97.5 percentiles of the resampled means.

    Permutation: under H0 (cadence irrelevant) each d_i is equally likely to
    carry either sign. Flip signs at random ``n_boot`` times; p is the share of
    flipped means at least as extreme as the observed one, with the observed
    statistic counted once so p is never exactly 0.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    obs = float(d.mean()) if n else float("nan")

    if n >= 2:
        boot = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
        alpha = (1.0 - CI_LEVEL) / 2.0
        ci_lo, ci_hi = np.quantile(boot, [alpha, 1.0 - alpha])
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, n))
        perm = (d * signs).mean(axis=1)
        p_perm = (1.0 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (n_boot + 1.0)
    else:
        ci_lo = ci_hi = p_perm = float("nan")

    return {
        "n": n,
        "mean_frozen": float(np.mean(frozen)) if frozen is not None and n else float("nan"),
        "mean_per_round": float(np.mean(per_round)) if per_round is not None and n else float("nan"),
        "mean_diff": obs,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_perm": float(p_perm),
        "frac_improved": float(np.mean(d < -tol)) if n else float("nan"),
        "frac_tied": float(np.mean(np.abs(d) <= tol)) if n else float("nan"),
    }


def _summarise_cell(
    cell: pd.DataFrame, model: str, metric: str, rng: np.random.Generator, n_boot: int
) -> dict[str, float]:
    g = cell[cell["model_name"] == model]
    return summarise(
        g[f"d_{metric}"].to_numpy(),
        rng,
        n_boot=n_boot,
        frozen=g[f"{metric}_frozen"].to_numpy(),
        per_round=g[f"{metric}_per_round"].to_numpy(),
    )


def phase_table(
    diffs: pd.DataFrame,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    models: tuple[str, ...] = CONTRAST_MODELS,
) -> pd.DataFrame:
    """Paired summary per (phase_scheme, phase, model, metric), plus ``all`` rows.

    Every scheme also gets an ``all`` phase covering the union of its phases,
    i.e. the 80 treated matches (MD2–Final).  MD1 never enters this table.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for scheme, (phases, block_map) in PHASE_SCHEMES.items():
        phase_col = diffs["block"].map(block_map)
        for phase in (*phases, ALL_PHASE):
            cell = diffs[phase_col.notna()] if phase == ALL_PHASE else diffs[phase_col == phase]
            for model in models:
                for metric in METRICS:
                    rows.append(
                        {"phase_scheme": scheme, "phase": phase, "model_name": model, "metric": metric}
                        | _summarise_cell(cell, model, metric, rng, n_boot)
                    )
    return pd.DataFrame(rows, columns=["phase_scheme", "phase", "model_name", "metric", *SUMMARY_COLUMNS])


def round_table(
    diffs: pd.DataFrame,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    models: tuple[str, ...] = CONTRAST_MODELS,
) -> pd.DataFrame:
    """Paired summary per round block (``grain=round``) and cumulative through it."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    seen: list[str] = []
    for block in ROUND_BLOCKS:
        seen.append(block)
        cells = {"round": diffs[diffs["block"] == block], "cumulative": diffs[diffs["block"].isin(seen)]}
        for grain, cell in cells.items():
            for model in models:
                for metric in METRICS:
                    rows.append(
                        {"block": block, "grain": grain, "model_name": model, "metric": metric}
                        | _summarise_cell(cell, model, metric, rng, n_boot)
                    )
    return pd.DataFrame(rows, columns=["block", "grain", "model_name", "metric", *SUMMARY_COLUMNS])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rq1", type=Path, default=RQ1_PATH)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--diffs-out", type=Path, default=RQ1_PAIRED_DIFFS_PATH)
    parser.add_argument("--phase-out", type=Path, default=RQ1_PAIRED_PHASE_PATH)
    parser.add_argument("--rounds-out", type=Path, default=RQ1_PAIRED_ROUNDS_PATH)
    parser.add_argument("--coverage-out", type=Path, default=RQ1_PAIRING_COVERAGE_PATH)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.rq1)
    coverage = pairing_coverage(df)
    diffs = paired_differences(df)
    phase = phase_table(diffs, n_boot=args.n_boot, seed=args.seed)
    rounds = round_table(diffs, n_boot=args.n_boot, seed=args.seed)

    for path, table in (
        (args.diffs_out, diffs),
        (args.phase_out, phase),
        (args.rounds_out, rounds),
        (args.coverage_out, coverage),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.6g")
        print(f"Wrote {len(table)} rows to {path}")


if __name__ == "__main__":
    main()
