"""Merge live exports with D.1 reconstruction strands into RQ-ready CSVs.

Produces:

- ``data/analysis/rq1_matches.csv``
- ``data/analysis/rq2_entropy.csv``
- ``data/analysis/rq3_calibration.csv``

Run after :mod:`src.analysis.rq_datasets.export_live` has frozen the live tables.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.replay_common import score_prediction_row
from src.analysis.rq_datasets.export_live import (
    fixtures_path,
    inference_cycles_path,
    monitoring_path,
)
from src.analysis.rq_datasets.paths import (
    ANALYSIS_ROOT,
    CADENCE_MODES,
    ENTROPY_H_MAX,
    EXPECTED_ROWS_PER_CADENCE,
    LIVE_ROOT,
    METRIC_TOL,
    MONITORING_COLUMNS,
    N_PROBABILITY_BINS,
    PROVENANCE_BACKFILL_COPIED,
    PROVENANCE_BACKFILL_REPRED,
    PROVENANCE_FROZEN_SHADOW,
    PROVENANCE_LIVE,
    PROVENANCE_VALUES,
    RQ1_KEY,
    RQ1_PATH,
    RQ2_KEY,
    RQ2_PATH,
    RQ3_PATH,
    STRAND1_COMBINED,
    STRAND2_TRAJECTORY,
    STRAND3_BACKFILL,
    STRAND4_SNAPSHOTS,
)

logger = logging.getLogger(__name__)

_ENTROPY_COLS = (
    "entropy_r32",
    "entropy_r16",
    "entropy_qf",
    "entropy_sf",
    "entropy_final",
    "entropy_winner",
)


def _read_monitoring(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "kickoff_utc" in df.columns:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    missing = [c for c in MONITORING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[list(MONITORING_COLUMNS)].copy()


def _key_frame(df: pd.DataFrame) -> pd.Series:
    return (
        df["match_id"].astype(str)
        + "|"
        + df["model_name"].astype(str)
        + "|"
        + df["cadence_mode"].astype(str)
    )


def _backfill_provenance(inference_run_id: str) -> str:
    if inference_run_id.startswith("backfill_copy_from_"):
        return PROVENANCE_BACKFILL_COPIED
    if inference_run_id == "backfill_ridge_md3":
        return PROVENANCE_BACKFILL_REPRED
    raise ValueError(f"Unknown backfill inference_run_id: {inference_run_id!r}")


def assert_unique_keys(df: pd.DataFrame, key_cols: tuple[str, ...], *, label: str) -> None:
    dup = df.duplicated(subset=list(key_cols), keep=False)
    if dup.any():
        n = int(dup.sum())
        raise AssertionError(f"{label}: {n} duplicate key rows on {key_cols}")


def assert_rps_nll_recomputable(df: pd.DataFrame, *, tol: float = METRIC_TOL) -> None:
    """Spot-check that stored rps/nll match recomputation from lambdas."""
    mismatches = 0
    max_abs = 0.0
    sample = df if len(df) <= 200 else df.sample(n=200, random_state=0)
    for _, row in sample.iterrows():
        scored = score_prediction_row(
            row,
            str(row["model_name"]),
            {
                "lambda_h": float(row["lambda_h"]),
                "lambda_a": float(row["lambda_a"]),
                "p_home": float(row["p_home"]),
                "p_draw": float(row["p_draw"]),
                "p_away": float(row["p_away"]),
            },
            inference_run_id=str(row["inference_run_id"]),
            cadence_mode=str(row["cadence_mode"]),
        )
        for col in ("rps", "nll"):
            delta = abs(float(scored[col]) - float(row[col]))
            max_abs = max(max_abs, delta)
            if delta > tol:
                mismatches += 1
    if mismatches:
        raise AssertionError(
            f"rps/nll recomputation failed on {mismatches} checks (max |Δ|={max_abs:.3e})",
        )


def build_rq1(
    *,
    live_root: Path = LIVE_ROOT,
    strand1_path: Path = STRAND1_COMBINED,
    strand3_path: Path = STRAND3_BACKFILL,
    fixtures: Path | None = None,
) -> pd.DataFrame:
    """Merge live monitoring with strand 1 replacements and strand 3 backfills."""
    frames = [_read_monitoring(monitoring_path(c, live_root)) for c in CADENCE_MODES]
    live = pd.concat(frames, ignore_index=True)
    live["provenance"] = PROVENANCE_LIVE
    live["source_run_id"] = live["inference_run_id"]

    strand1 = _read_monitoring(strand1_path)
    if not (strand1["cadence_mode"] == "frozen").all():
        raise ValueError("strand1 must be cadence_mode=frozen only")
    s1_keys = set(_key_frame(strand1))
    live_keys = _key_frame(live)
    replace_mask = live_keys.isin(s1_keys)
    # Preserve original live run ids for replaced rows.
    source_lookup = (
        live.loc[replace_mask, list(RQ1_KEY) + ["inference_run_id"]]
        .drop_duplicates(subset=list(RQ1_KEY))
        .rename(columns={"inference_run_id": "source_run_id"})
    )
    kept = live.loc[~replace_mask].copy()
    strand1 = strand1.copy()
    strand1["provenance"] = PROVENANCE_FROZEN_SHADOW
    strand1 = strand1.merge(source_lookup, on=list(RQ1_KEY), how="left")
    # Keys present only in reconstruction (should not happen for strand1) get NA.
    if strand1["source_run_id"].isna().any():
        logger.warning(
            "strand1: %d rows had no live twin to preserve source_run_id",
            int(strand1["source_run_id"].isna().sum()),
        )

    strand3 = _read_monitoring(strand3_path)
    strand3 = strand3.copy()
    strand3["provenance"] = strand3["inference_run_id"].map(_backfill_provenance)
    strand3["source_run_id"] = pd.NA
    s3_keys = set(_key_frame(strand3))
    already = set(_key_frame(kept)) | set(_key_frame(strand1))
    overlap = s3_keys & already
    if overlap:
        raise ValueError(
            f"strand3 keys already present after live+strand1 merge: {sorted(overlap)[:5]}",
        )

    rq1 = pd.concat([kept, strand1, strand3], ignore_index=True)

    fx_path = fixtures if fixtures is not None else fixtures_path(live_root)
    fixtures_df = pd.read_csv(fx_path)
    if "kickoff_utc" in fixtures_df.columns:
        fixtures_df["kickoff_utc"] = pd.to_datetime(fixtures_df["kickoff_utc"], utc=True)
    if "round_label" not in fixtures_df.columns:
        raise ValueError(f"{fx_path}: missing round_label")
    rq1 = rq1.merge(
        fixtures_df[["match_id", "round_label"]],
        on="match_id",
        how="left",
        validate="many_to_one",
    )
    if rq1["round_label"].isna().any():
        missing_ids = rq1.loc[rq1["round_label"].isna(), "match_id"].unique().tolist()
        raise ValueError(f"round_label missing for match_ids: {missing_ids[:10]}")

    unknown = set(rq1["provenance"]) - PROVENANCE_VALUES
    if unknown:
        raise AssertionError(f"Unexpected provenance values: {unknown}")

    assert_unique_keys(rq1, RQ1_KEY, label="rq1_matches")
    for cadence in CADENCE_MODES:
        n = int((rq1["cadence_mode"] == cadence).sum())
        if n != EXPECTED_ROWS_PER_CADENCE:
            raise AssertionError(
                f"rq1 {cadence}: expected {EXPECTED_ROWS_PER_CADENCE} rows, got {n}",
            )
    assert_rps_nll_recomputable(rq1)

    col_order = list(MONITORING_COLUMNS) + ["provenance", "source_run_id", "round_label"]
    return rq1[col_order].sort_values(
        ["cadence_mode", "kickoff_utc", "model_name", "match_id"],
    ).reset_index(drop=True)


def build_rq2(
    *,
    live_root: Path = LIVE_ROOT,
    trajectory_path: Path = STRAND2_TRAJECTORY,
    snapshots_path: Path = STRAND4_SNAPSHOTS,
) -> pd.DataFrame:
    """Corrected entropy trajectory + synthetic gap-fill snapshots."""
    traj = pd.read_csv(trajectory_path)
    traj["inference_timestamp"] = pd.to_datetime(
        traj["inference_timestamp"], utc=True, format="ISO8601",
    )
    traj["synthetic"] = False
    traj["snapshot_label"] = pd.NA

    snaps = pd.read_csv(snapshots_path)
    snaps["inference_timestamp"] = pd.to_datetime(
        snaps["inference_timestamp"], utc=True, format="ISO8601",
    )
    if "synthetic" not in snaps.columns:
        snaps["synthetic"] = True
    else:
        snaps["synthetic"] = snaps["synthetic"].astype(bool)
    if "snapshot_label" not in snaps.columns:
        raise ValueError(f"{snapshots_path}: missing snapshot_label")

    shared = [
        "inference_run_id",
        "inference_timestamp",
        "cadence_mode",
        "model_name",
        *_ENTROPY_COLS,
        "synthetic",
        "snapshot_label",
    ]
    for col in shared:
        if col not in traj.columns and col in ("synthetic", "snapshot_label"):
            continue
        if col not in traj.columns:
            raise ValueError(f"{trajectory_path}: missing {col}")
        if col not in snaps.columns:
            raise ValueError(f"{snapshots_path}: missing {col}")

    rq2 = pd.concat([traj[shared], snaps[shared]], ignore_index=True)

    cycles_path = inference_cycles_path(live_root)
    if cycles_path.exists():
        cycles = pd.read_csv(cycles_path)
        cycles["inference_timestamp"] = pd.to_datetime(
            cycles["inference_timestamp"], utc=True, format="ISO8601",
        )
        rq2 = rq2.merge(
            cycles[["inference_run_id", "matchday_label"]].drop_duplicates("inference_run_id"),
            on="inference_run_id",
            how="left",
        )
    else:
        rq2["matchday_label"] = pd.Series(pd.NA, index=rq2.index, dtype="object")

    rq2["matchday_label"] = rq2["matchday_label"].astype("object")

    # Synthetic strand4 rows have no live cycle; map from snapshot_label.
    _snap_to_md = {
        "r32_pre_japan_brazil": "R32",
        "r32_japan_brazil_locked": "R32",
        "r32_germany_paraguay_locked": "R32",
        "r16_brazil_norway_locked": "R16",
    }
    need_md = rq2["matchday_label"].isna() & rq2["snapshot_label"].notna()
    rq2.loc[need_md, "matchday_label"] = (
        rq2.loc[need_md, "snapshot_label"].map(_snap_to_md).astype("object")
    )

    # cycle_index: dense rank of unique timestamps within cadence (models share).
    ts_rank = (
        rq2[["cadence_mode", "inference_timestamp"]]
        .drop_duplicates()
        .sort_values(["cadence_mode", "inference_timestamp"])
    )
    ts_rank["cycle_index"] = ts_rank.groupby("cadence_mode").cumcount()
    rq2 = rq2.merge(ts_rank, on=["cadence_mode", "inference_timestamp"], how="left")

    assert_unique_keys(rq2, RQ2_KEY, label="rq2_entropy")

    # Sanity: entropy already column-sum normalised → H ≤ log(48).
    for col in _ENTROPY_COLS:
        finite = rq2[col].dropna()
        if finite.empty:
            continue
        mx = float(finite.max())
        if mx > ENTROPY_H_MAX + 1e-6:
            raise AssertionError(
                f"{col} max={mx:.6f} exceeds log(48)={ENTROPY_H_MAX:.6f}",
            )

    mr = rq2.loc[rq2["model_name"] == "mean_rate_poisson"]
    if not mr.empty:
        for cadence, g in rq2.groupby("cadence_mode"):
            mr_mean = float(
                g.loc[g["model_name"] == "mean_rate_poisson", "entropy_winner"].mean()
            )
            others = g.loc[g["model_name"] != "mean_rate_poisson"]
            if others.empty:
                continue
            other_means = others.groupby("model_name")["entropy_winner"].mean()
            if (other_means >= mr_mean).any():
                raise AssertionError(
                    f"{cadence}: mean_rate_poisson entropy_winner mean={mr_mean:.4f} "
                    f"is not strictly above all other models (floor violated): "
                    f"{other_means.to_dict()}",
                )

    # Known audit note: some rows break monotonic stage ordering (MC noise).
    mono_breaks = 0
    for _, row in rq2.iterrows():
        vals = [row[c] for c in _ENTROPY_COLS]
        if any(pd.isna(v) for v in vals):
            continue
        if any(vals[i] + 1e-9 < vals[i + 1] for i in range(len(vals) - 1)):
            mono_breaks += 1
    if mono_breaks:
        logger.info(
            "rq2 note: %d/%d rows break entropy_r32≥…≥entropy_winner "
            "(Monte Carlo noise; carried forward from strand2 audit)",
            mono_breaks,
            len(rq2),
        )

    col_order = [
        "inference_run_id",
        "inference_timestamp",
        "cadence_mode",
        "model_name",
        "matchday_label",
        "cycle_index",
        "synthetic",
        "snapshot_label",
        *_ENTROPY_COLS,
    ]
    return rq2[col_order].sort_values(
        ["cadence_mode", "inference_timestamp", "model_name"],
    ).reset_index(drop=True)


def _reliability_bins(
    rq1: pd.DataFrame,
    *,
    n_bins: int = N_PROBABILITY_BINS,
) -> pd.DataFrame:
    """Long reliability table: one row per cadence×model×round×outcome×bin."""
    outcome_map = {
        0: ("home", "p_home"),
        1: ("draw", "p_draw"),
        2: ("away", "p_away"),
    }
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []

    group_cols = ["cadence_mode", "model_name", "round_label"]
    phase_stats = (
        rq1.groupby(group_cols, dropna=False)
        .agg(mean_rps=("rps", "mean"), mean_nll=("nll", "mean"), n_matches=("rps", "size"))
        .reset_index()
    )

    for (cadence, model, round_label), g in rq1.groupby(group_cols, dropna=False):
        for outcome_code, (outcome_name, p_col) in outcome_map.items():
            predicted = g[p_col].to_numpy(dtype=float)
            observed = (g["actual_outcome"].to_numpy(dtype=int) == outcome_code).astype(float)
            bin_idx = np.digitize(predicted, edges[1:-1], right=False)
            for b in range(n_bins):
                mask = bin_idx == b
                n = int(mask.sum())
                if n == 0:
                    mean_p = float("nan")
                    obs_freq = float("nan")
                else:
                    mean_p = float(predicted[mask].mean())
                    obs_freq = float(observed[mask].mean())
                rows.append({
                    "record_type": "reliability",
                    "cadence_mode": cadence,
                    "model_name": model,
                    "round_label": round_label,
                    "outcome_class": outcome_name,
                    "bin_index": b,
                    "bin_lo": float(edges[b]),
                    "bin_hi": float(edges[b + 1]),
                    "n": n,
                    "mean_predicted_p": mean_p,
                    "observed_freq": obs_freq,
                    "mean_rps": float(
                        phase_stats.loc[
                            (phase_stats["cadence_mode"] == cadence)
                            & (phase_stats["model_name"] == model)
                            & (phase_stats["round_label"] == round_label),
                            "mean_rps",
                        ].iloc[0]
                    ),
                    "mean_nll": float(
                        phase_stats.loc[
                            (phase_stats["cadence_mode"] == cadence)
                            & (phase_stats["model_name"] == model)
                            & (phase_stats["round_label"] == round_label),
                            "mean_nll",
                        ].iloc[0]
                    ),
                    "n_matches": int(
                        phase_stats.loc[
                            (phase_stats["cadence_mode"] == cadence)
                            & (phase_stats["model_name"] == model)
                            & (phase_stats["round_label"] == round_label),
                            "n_matches",
                        ].iloc[0]
                    ),
                    "match_id": pd.NA,
                    "kickoff_utc": pd.NaT,
                    "rps": float("nan"),
                    "cum_rps": float("nan"),
                    "match_index": pd.NA,
                })
    return pd.DataFrame(rows)


def _cum_rps_rows(rq1: pd.DataFrame) -> pd.DataFrame:
    """Expanding mean RPS ordered by kickoff within cadence×model."""
    pieces: list[pd.DataFrame] = []
    for (cadence, model), g in rq1.groupby(["cadence_mode", "model_name"], dropna=False):
        ordered = g.sort_values(["kickoff_utc", "match_id"]).copy()
        ordered["match_index"] = np.arange(len(ordered), dtype=int)
        ordered["cum_rps"] = ordered["rps"].expanding().mean()
        phase_means = ordered.groupby("round_label")["rps"].transform("mean")
        phase_nll = ordered.groupby("round_label")["nll"].transform("mean")
        phase_n = ordered.groupby("round_label")["rps"].transform("size")
        piece = pd.DataFrame({
            "record_type": "cum_rps",
            "cadence_mode": cadence,
            "model_name": model,
            "round_label": ordered["round_label"].to_numpy(),
            "outcome_class": pd.NA,
            "bin_index": pd.NA,
            "bin_lo": float("nan"),
            "bin_hi": float("nan"),
            "n": 1,
            "mean_predicted_p": float("nan"),
            "observed_freq": float("nan"),
            "mean_rps": phase_means.to_numpy(),
            "mean_nll": phase_nll.to_numpy(),
            "n_matches": phase_n.to_numpy(dtype=int),
            "match_id": ordered["match_id"].to_numpy(),
            "kickoff_utc": ordered["kickoff_utc"].to_numpy(),
            "rps": ordered["rps"].to_numpy(),
            "cum_rps": ordered["cum_rps"].to_numpy(),
            "match_index": ordered["match_index"].to_numpy(),
        })
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def build_rq3(rq1: pd.DataFrame, *, n_bins: int = N_PROBABILITY_BINS) -> pd.DataFrame:
    """Reliability bins + phase means + expanding cum_rps from RQ1."""
    reliability = _reliability_bins(rq1, n_bins=n_bins)
    cum = _cum_rps_rows(rq1)
    rq3 = pd.concat([reliability, cum], ignore_index=True)

    # Bin counts across outcome classes × bins should cover every RQ1 row once
    # per outcome class (3 × n_matches total n-sum for reliability rows).
    rel = rq3.loc[rq3["record_type"] == "reliability"]
    n_sum = int(rel["n"].sum())
    expected = len(rq1) * 3  # three outcome classes
    if n_sum != expected:
        raise AssertionError(
            f"reliability bin n-sum={n_sum} != 3×len(rq1)={expected}",
        )

    col_order = [
        "record_type",
        "cadence_mode",
        "model_name",
        "round_label",
        "outcome_class",
        "bin_index",
        "bin_lo",
        "bin_hi",
        "n",
        "mean_predicted_p",
        "observed_freq",
        "mean_rps",
        "mean_nll",
        "n_matches",
        "match_id",
        "kickoff_utc",
        "rps",
        "cum_rps",
        "match_index",
    ]
    return rq3[col_order].reset_index(drop=True)


def build_all(
    *,
    live_root: Path = LIVE_ROOT,
    analysis_root: Path = ANALYSIS_ROOT,
    skip_export_check: bool = False,
) -> dict[str, Path]:
    """Build and write all three RQ datasets."""
    if not skip_export_check:
        for cadence in CADENCE_MODES:
            path = monitoring_path(cadence, live_root)
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run: python -m src.analysis.rq_datasets.export_live",
                )
        if not fixtures_path(live_root).exists():
            raise FileNotFoundError(f"Missing {fixtures_path(live_root)}")

    analysis_root.mkdir(parents=True, exist_ok=True)

    rq1 = build_rq1(live_root=live_root)
    rq1.to_csv(analysis_root / "rq1_matches.csv", index=False)
    logger.info("Wrote %s (%d rows)", analysis_root / "rq1_matches.csv", len(rq1))

    rq2 = build_rq2(live_root=live_root)
    rq2.to_csv(analysis_root / "rq2_entropy.csv", index=False)
    logger.info("Wrote %s (%d rows)", analysis_root / "rq2_entropy.csv", len(rq2))

    rq3 = build_rq3(rq1)
    rq3.to_csv(analysis_root / "rq3_calibration.csv", index=False)
    logger.info("Wrote %s (%d rows)", analysis_root / "rq3_calibration.csv", len(rq3))

    return {
        "rq1": analysis_root / "rq1_matches.csv",
        "rq2": analysis_root / "rq2_entropy.csv",
        "rq3": analysis_root / "rq3_calibration.csv",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, default=LIVE_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=ANALYSIS_ROOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_all(live_root=args.live_root, analysis_root=args.analysis_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
