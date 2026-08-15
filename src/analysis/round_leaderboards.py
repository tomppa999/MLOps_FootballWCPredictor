"""Recompute per-round and cumulative leaderboards from RQ1 match rows.

Reads ``data/analysis/rq1_matches.csv`` (live + D.1 reconstruction) and emits
``data/analysis/round_leaderboards.csv`` with one row per
(block, cadence_mode, model_name). Leaderboard RMSE is
``(mean(rmse_h) + mean(rmse_a)) / 2``.

Final + third-place are treated as one block (``Final``), matching the
``wc_live.md`` convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import pandas as pd

from src.analysis.rq_datasets.paths import ANALYSIS_ROOT, RQ1_PATH

ROUND_BLOCKS: Final[tuple[str, ...]] = (
    "1",
    "2",
    "3",
    "R32",
    "R16",
    "QF",
    "SF",
    "Final",
)

# Map rq1 round_label -> leaderboard block.
_BLOCK_MAP: Final[dict[str, str]] = {
    "1": "1",
    "2": "2",
    "3": "3",
    "R32": "R32",
    "R16": "R16",
    "QF": "QF",
    "SF": "SF",
    "3rd_place": "Final",
    "Final": "Final",
}

ROUND_LEADERBOARDS_PATH: Final[Path] = ANALYSIS_ROOT / "round_leaderboards.csv"


def _assign_block(round_label: str) -> str:
    try:
        return _BLOCK_MAP[str(round_label)]
    except KeyError as exc:
        raise ValueError(f"unknown round_label: {round_label!r}") from exc


def build_round_leaderboards(rq1_path: Path = RQ1_PATH) -> pd.DataFrame:
    """Return per-block round and cumulative (overall) means."""
    df = pd.read_csv(rq1_path)
    required = {"round_label", "cadence_mode", "model_name", "rps", "nll", "rmse_h", "rmse_a"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{rq1_path}: missing columns {sorted(missing)}")

    df = df.copy()
    df["block"] = df["round_label"].map(_assign_block)
    df["rmse"] = (df["rmse_h"] + df["rmse_a"]) / 2.0

    round_rows = (
        df.groupby(["block", "cadence_mode", "model_name"], as_index=False)
        .agg(
            n_round=("rps", "size"),
            round_rps=("rps", "mean"),
            round_nll=("nll", "mean"),
            round_rmse=("rmse", "mean"),
        )
    )

    overall_frames: list[pd.DataFrame] = []
    seen: list[str] = []
    for block in ROUND_BLOCKS:
        seen.append(block)
        sub = df[df["block"].isin(seen)]
        g = (
            sub.groupby(["cadence_mode", "model_name"], as_index=False)
            .agg(
                n_overall=("rps", "size"),
                overall_rps=("rps", "mean"),
                overall_nll=("nll", "mean"),
                overall_rmse=("rmse", "mean"),
            )
        )
        g["block"] = block
        overall_frames.append(g)
    overall = pd.concat(overall_frames, ignore_index=True)

    out = round_rows.merge(
        overall,
        on=["block", "cadence_mode", "model_name"],
        how="left",
        validate="one_to_one",
    )
    # Stable column order.
    cols = [
        "block",
        "cadence_mode",
        "model_name",
        "n_round",
        "n_overall",
        "round_rps",
        "overall_rps",
        "round_rmse",
        "overall_rmse",
        "round_nll",
        "overall_nll",
    ]
    out = out[cols]
    # Sort for determinism.
    block_cat = pd.Categorical(out["block"], categories=list(ROUND_BLOCKS), ordered=True)
    out = out.assign(block=block_cat).sort_values(
        ["block", "cadence_mode", "model_name"], kind="mergesort"
    ).reset_index(drop=True)
    out["block"] = out["block"].astype(str)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rq1",
        type=Path,
        default=RQ1_PATH,
        help="Path to rq1_matches.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROUND_LEADERBOARDS_PATH,
        help="Output CSV path",
    )
    args = parser.parse_args(argv)
    out = build_round_leaderboards(args.rq1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
