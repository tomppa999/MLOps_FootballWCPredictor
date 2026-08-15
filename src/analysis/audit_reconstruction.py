"""Read-only audit of ``data/reconstruction/`` before ``wc_live.md`` backfill.

Default run is offline over committed CSVs.  ``--with-mlflow`` and
``--with-determinism N`` are opt-in and may require network / DagsHub.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Sequence

import numpy as np
import pandas as pd

from src.analysis.replay_common import (
    BACKFILL_FIXTURES,
    ENTROPY_COLUMNS,
    OUTPUT_ROOT,
    PINNED_FROZEN_SHADOW_VERSIONS,
    RECONSTRUCTION_EXPERIMENT,
    SETTLE_DELTA,
    check_entropy_trajectory,
    compute_entropy_columns,
    leaderboard_summary,
    load_finished_wc_fixtures,
    load_monitoring_artifact,
    parse_wc_results_before_kickoff,
    score_prediction_row,
)
from src.models.config import EXPERIMENT_MODELS, LIVE_SHADOW_MODELS

logger = logging.getLogger(__name__)

MONITORING_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "kickoff_utc",
    "home",
    "away",
    "actual_h",
    "actual_a",
    "actual_outcome",
    "model_name",
    "lambda_h",
    "lambda_a",
    "p_home",
    "p_draw",
    "p_away",
    "rps",
    "nll",
    "rmse_h",
    "rmse_a",
    "inference_run_id",
    "cadence_mode",
)

ADVANCEMENT_PROB_COLS: Final[tuple[str, ...]] = (
    "p_group",
    "p_r32",
    "p_r16",
    "p_qf",
    "p_sf",
    "p_final",
    "p_winner",
)

ROUND_SLOT_COUNTS: Final[dict[str, int]] = {
    "p_r32": 32,
    "p_r16": 16,
    "p_qf": 8,
    "p_sf": 4,
    "p_final": 2,
    "p_winner": 1,
}

KO_STAGE_TIE_COUNTS: Final[dict[str, int]] = {
    "R32": 16,
    "R16": 8,
    "QF": 4,
    "SF": 2,
    "Final": 1,
}

ENTROPY_ORDER: Final[tuple[str, ...]] = tuple(
    f"entropy_{c.removeprefix('p_')}" for c in ENTROPY_COLUMNS
)

N_WC_TEAMS: Final[int] = 48
N_MATCHES: Final[int] = 104
EXPECTED_ROWS_PER_CADENCE: Final[int] = len(LIVE_SHADOW_MODELS) * N_MATCHES  # 832

DEFAULT_MONITORING: Final[Path] = Path("src/dashboard/_offline_cache/wc2026_monitoring.csv")
DEFAULT_AUDIT_JSON: Final[Path] = OUTPUT_ROOT / "audit_report.json"

PROB_TOL: Final[float] = 1e-6
SUM_TOL: Final[float] = 1e-3
METRIC_TOL: Final[float] = 1e-9
LAMBDA_DISTINCT_EPS: Final[float] = 1e-9


@dataclass
class CheckResult:
    """One audit assertion result."""

    name: str
    status: str  # pass | warn | fail
    detail: str
    numbers: dict[str, float | int | str] = field(default_factory=dict)
    strand: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pass(name: str, detail: str, **numbers: float | int | str) -> CheckResult:
    return CheckResult(name=name, status="pass", detail=detail, numbers=dict(numbers))


def _warn(name: str, detail: str, **numbers: float | int | str) -> CheckResult:
    return CheckResult(name=name, status="warn", detail=detail, numbers=dict(numbers))


def _fail(name: str, detail: str, **numbers: float | int | str) -> CheckResult:
    return CheckResult(name=name, status="fail", detail=detail, numbers=dict(numbers))


def _tag(results: list[CheckResult], strand: str) -> list[CheckResult]:
    for r in results:
        r.strand = strand
    return results


def _expected_outcome(actual_h: int, actual_a: int) -> int:
    """0=home, 1=draw, 2=away — matches monitoring encoding."""
    if actual_h > actual_a:
        return 0
    if actual_h < actual_a:
        return 2
    return 1


def _recompute_metrics_ok(df: pd.DataFrame, *, name: str) -> CheckResult:
    """Recompute rps/nll/rmse via ``score_prediction_row`` and compare."""
    mismatches = 0
    max_abs = 0.0
    for _, row in df.iterrows():
        pred = {
            "lambda_h": float(row["lambda_h"]),
            "lambda_a": float(row["lambda_a"]),
            "p_home": float(row["p_home"]),
            "p_draw": float(row["p_draw"]),
            "p_away": float(row["p_away"]),
        }
        scored = score_prediction_row(
            row,
            str(row["model_name"]),
            pred,
            inference_run_id=str(row.get("inference_run_id", "audit")),
            cadence_mode=str(row.get("cadence_mode", "frozen")),
        )
        for col in ("rps", "nll", "rmse_h", "rmse_a"):
            delta = abs(float(scored[col]) - float(row[col]))
            max_abs = max(max_abs, delta)
            if delta > METRIC_TOL:
                mismatches += 1
                break
    if mismatches:
        return _fail(
            name,
            f"{mismatches}/{len(df)} rows disagree with recomputed metrics "
            f"(max |Δ|={max_abs:.3e})",
            mismatches=mismatches,
            max_abs_delta=max_abs,
        )
    return _pass(
        name,
        f"All {len(df)} rows match recomputed rps/nll/rmse",
        rows=len(df),
        max_abs_delta=max_abs,
    )


def _prob_and_outcome_ok(df: pd.DataFrame, *, name: str) -> list[CheckResult]:
    out: list[CheckResult] = []
    if df.empty:
        return [_fail(name, "empty frame")]

    prob_sum = df["p_home"] + df["p_draw"] + df["p_away"]
    bad_probs = int((np.abs(prob_sum - 1.0) > PROB_TOL).sum())
    if bad_probs:
        out.append(
            _fail(
                f"{name}.prob_sum",
                f"{bad_probs} rows with |p_home+p_draw+p_away-1| > {PROB_TOL}",
                bad_rows=bad_probs,
            ),
        )
    else:
        out.append(_pass(f"{name}.prob_sum", "Outcome probs sum to 1", rows=len(df)))

    expected = [
        _expected_outcome(int(h), int(a))
        for h, a in zip(df["actual_h"], df["actual_a"], strict=True)
    ]
    bad_out = int((df["actual_outcome"].astype(int).to_numpy() != np.asarray(expected)).sum())
    if bad_out:
        out.append(
            _fail(
                f"{name}.actual_outcome",
                f"{bad_out} rows with actual_outcome inconsistent with scores",
                bad_rows=bad_out,
            ),
        )
    else:
        out.append(_pass(f"{name}.actual_outcome", "actual_outcome matches scores"))
    return out


def _entropy_row_ordering(df: pd.DataFrame, *, name: str) -> CheckResult:
    """Warn (not fail) on non-monotonic entropy across rounds within a row."""
    cols = [c for c in ENTROPY_ORDER if c in df.columns]
    if len(cols) < 2:
        return _fail(name, "missing entropy columns")
    violations = 0
    worst = 0.0
    for _, row in df.iterrows():
        vals = [float(row[c]) for c in cols]
        for i in range(len(vals) - 1):
            if np.isnan(vals[i]) or np.isnan(vals[i + 1]):
                continue
            gap = vals[i + 1] - vals[i]
            if gap > PROB_TOL:
                violations += 1
                worst = max(worst, gap)
                break
    if violations:
        return _warn(
            name,
            f"{violations}/{len(df)} rows violate entropy_r32≥…≥entropy_winner "
            f"(worst upward jump={worst:.4f})",
            violations=violations,
            worst_jump=worst,
        )
    return _pass(name, "Per-row entropy ordering holds", rows=len(df))


# ---------------------------------------------------------------------------
# Strand 1
# ---------------------------------------------------------------------------


def audit_strand1(root: Path = OUTPUT_ROOT) -> list[CheckResult]:
    d = root / "strand1_frozen_shadow"
    results: list[CheckResult] = []
    if not d.is_dir():
        return _tag([_fail("s1.exists", f"missing directory {d}")], "1")

    models = ("poisson_glm", "bayesian_poisson")
    per_model: dict[str, pd.DataFrame] = {}
    for model in models:
        path = d / f"frozen_shadow_{model}.csv"
        if not path.exists():
            results.append(_fail(f"s1.{model}.exists", f"missing {path.name}"))
            continue
        df = pd.read_csv(path)
        if "kickoff_utc" in df.columns:
            df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
        per_model[model] = df

        if list(df.columns) != list(MONITORING_COLUMNS):
            results.append(
                _fail(
                    f"s1.{model}.schema",
                    f"column mismatch: got {list(df.columns)}",
                ),
            )
        else:
            results.append(_pass(f"s1.{model}.schema", "monitoring schema OK"))

        n = len(df)
        n_unique = df["match_id"].nunique() if "match_id" in df.columns else 0
        if n != N_MATCHES or n_unique != N_MATCHES:
            results.append(
                _fail(
                    f"s1.{model}.rows",
                    f"expected {N_MATCHES} unique matches, got rows={n} unique={n_unique}",
                    rows=n,
                    unique=n_unique,
                ),
            )
        else:
            results.append(
                _pass(f"s1.{model}.rows", f"{N_MATCHES} unique match_ids", rows=n),
            )

        nan_cols = [
            c
            for c in ("lambda_h", "lambda_a", "rps", "nll")
            if c in df.columns and int(df[c].isna().sum()) > 0
        ]
        if nan_cols:
            results.append(_fail(f"s1.{model}.nan", f"NaN in {nan_cols}"))
        else:
            results.append(_pass(f"s1.{model}.nan", "no NaN in key metric columns"))

        if "cadence_mode" in df.columns and (df["cadence_mode"] != "frozen").any():
            results.append(_fail(f"s1.{model}.cadence", "non-frozen cadence_mode present"))
        else:
            results.append(_pass(f"s1.{model}.cadence", "all rows cadence_mode=frozen"))

        results.extend(_prob_and_outcome_ok(df, name=f"s1.{model}"))
        results.append(_recompute_metrics_ok(df, name=f"s1.{model}.metrics"))

        summary_path = d / f"frozen_shadow_{model}_summary.csv"
        if summary_path.exists() and not df.empty:
            summary = pd.read_csv(summary_path)
            recomputed = leaderboard_summary(df)
            if len(summary) == 1 and len(recomputed) == 1:
                ok = True
                for col in ("mean_rps", "mean_nll", "mean_rmse_h", "mean_rmse_a", "mean_rmse"):
                    if col not in summary.columns:
                        continue
                    if abs(float(summary.iloc[0][col]) - float(recomputed.iloc[0][col])) > METRIC_TOL:
                        ok = False
                        break
                results.append(
                    _pass(f"s1.{model}.summary", "summary matches recomputed")
                    if ok
                    else _fail(f"s1.{model}.summary", "summary ≠ leaderboard_summary(df)"),
                )

    combined_path = d / "frozen_shadow_combined.csv"
    lb_path = d / "frozen_shadow_leaderboard.csv"
    if combined_path.exists() and len(per_model) == 2:
        combined = pd.read_csv(combined_path)
        concat = pd.concat(per_model.values(), ignore_index=True)
        if len(combined) != len(concat):
            results.append(
                _fail(
                    "s1.combined.rows",
                    f"combined has {len(combined)} rows, concat has {len(concat)}",
                    combined=len(combined),
                    concat=len(concat),
                ),
            )
        else:
            # Compare on match_id + model_name + lambda_h (stable key).
            a = combined.sort_values(["model_name", "match_id"]).reset_index(drop=True)
            b = concat.sort_values(["model_name", "match_id"]).reset_index(drop=True)
            if not np.allclose(
                a["lambda_h"].astype(float),
                b["lambda_h"].astype(float),
                rtol=0,
                atol=METRIC_TOL,
                equal_nan=True,
            ):
                results.append(_fail("s1.combined.content", "combined ≠ concat of per-model files"))
            else:
                results.append(
                    _pass("s1.combined.content", "combined matches per-model concat", rows=len(combined)),
                )

        if lb_path.exists():
            lb = pd.read_csv(lb_path)
            recomputed = leaderboard_summary(combined)
            # Align on model_name
            merged = lb.merge(
                recomputed,
                on=["model_name", "cadence_mode"],
                suffixes=("_file", "_re"),
                how="outer",
            )
            ok = True
            detail_bits: list[str] = []
            for col in ("mean_rps", "mean_nll", "mean_rmse"):
                left, right = f"{col}_file", f"{col}_re"
                if left not in merged.columns or right not in merged.columns:
                    ok = False
                    detail_bits.append(f"missing {col}")
                    continue
                if not np.allclose(
                    merged[left].astype(float),
                    merged[right].astype(float),
                    rtol=0,
                    atol=1e-12,
                    equal_nan=True,
                ):
                    ok = False
                    detail_bits.append(col)
            # mean_rmse identity
            if "mean_rmse_h_file" in merged.columns:
                expected_rmse = (
                    merged["mean_rmse_h_file"].astype(float)
                    + merged["mean_rmse_a_file"].astype(float)
                ) / 2.0
                if not np.allclose(
                    merged["mean_rmse_file"].astype(float),
                    expected_rmse,
                    rtol=0,
                    atol=1e-12,
                ):
                    ok = False
                    detail_bits.append("mean_rmse≠(h+a)/2")
            if ok:
                results.append(_pass("s1.leaderboard", "leaderboard reproduces from combined"))
            else:
                results.append(
                    _fail("s1.leaderboard", "leaderboard mismatch: " + ", ".join(detail_bits)),
                )

    if len(per_model) == 2:
        a = per_model["poisson_glm"].sort_values("match_id")
        b = per_model["bayesian_poisson"].sort_values("match_id")
        if set(a["match_id"]) == set(b["match_id"]):
            merged = a[["match_id", "lambda_h"]].merge(
                b[["match_id", "lambda_h"]],
                on="match_id",
                suffixes=("_glm", "_bayes"),
            )
            delta = (merged["lambda_h_glm"] - merged["lambda_h_bayes"]).abs()
            max_delta = float(delta.max()) if len(delta) else 0.0
            if max_delta <= LAMBDA_DISTINCT_EPS:
                results.append(
                    _fail(
                        "s1.lambda_distinct",
                        "poisson_glm and bayesian_poisson λ_h vectors are identical "
                        f"(max|Δ|={max_delta:.3e}) — possible resolver collapse",
                        max_abs_delta=max_delta,
                    ),
                )
            else:
                results.append(
                    _pass(
                        "s1.lambda_distinct",
                        f"models produce distinct λ_h (max|Δ|={max_delta:.4f})",
                        max_abs_delta=max_delta,
                    ),
                )
        else:
            results.append(_warn("s1.lambda_distinct", "match_id sets differ; skipped Δλ check"))

    # Leakage guard: a replayed match must never see its own result — nor any
    # fixture still in play at its kickoff — in the known-results snapshot its
    # features are built from. Calling parse_wc_results_before_kickoff without
    # SETTLE_DELTA silently reintroduces exactly that.
    try:
        from src.monitoring.monitor import parse_wc_settled_matches

        settled = parse_wc_settled_matches()
        self_leaks = 0
        unsettled_leaks = 0
        for _, match in settled.iterrows():
            kickoff = pd.Timestamp(match["kickoff_utc"])
            wc = parse_wc_results_before_kickoff(kickoff, settle_delta=SETTLE_DELTA)
            for fx in wc["finished_fixtures"]:
                if int(fx["fixture_id"]) == int(match["match_id"]):
                    self_leaks += 1
                if pd.to_datetime(fx["date_utc"], utc=True) + SETTLE_DELTA > kickoff:
                    unsettled_leaks += 1
        if self_leaks or unsettled_leaks:
            results.append(
                _fail(
                    "s1.leakage",
                    f"{self_leaks} matches see their own result and "
                    f"{unsettled_leaks} unsettled fixtures leak into replayed "
                    f"features under SETTLE_DELTA={SETTLE_DELTA}",
                    self_leaks=self_leaks,
                    unsettled_leaks=unsettled_leaks,
                ),
            )
        else:
            results.append(
                _pass(
                    "s1.leakage",
                    f"no match sees its own or an unsettled result "
                    f"(n={len(settled)}, SETTLE_DELTA={SETTLE_DELTA})",
                    matches=int(len(settled)),
                ),
            )
    except Exception as exc:  # noqa: BLE001
        results.append(_warn("s1.leakage", f"leakage check skipped: {exc}"))

    return _tag(results, "1")


# ---------------------------------------------------------------------------
# Strand 3 (run early — most likely to fail)
# ---------------------------------------------------------------------------


def _expected_backfill_keys() -> list[tuple[int, str, str]]:
    keys: list[tuple[int, str, str]] = []
    for spec in BACKFILL_FIXTURES.values():
        for fid in spec["fixture_ids"]:
            keys.append((int(fid), str(spec["model_name"]), str(spec["cadence_mode"])))
    return keys


def audit_strand3(
    root: Path = OUTPUT_ROOT,
    *,
    monitoring_path: Path | None = None,
    per_round_monitoring_path: Path | None = None,
    frozen_monitoring_path: Path | None = None,
) -> list[CheckResult]:
    d = root / "strand3_backfill"
    results: list[CheckResult] = []
    path = d / "backfill_rows.csv"
    if not path.exists():
        return _tag([_fail("s3.exists", f"missing {path}")], "3")

    bf = pd.read_csv(path)
    if "kickoff_utc" in bf.columns:
        bf["kickoff_utc"] = pd.to_datetime(bf["kickoff_utc"], utc=True)

    # --- 832 headline first ---
    mon_frames: list[pd.DataFrame] = []
    resolved_paths: list[str] = []
    for candidate in (
        per_round_monitoring_path,
        frozen_monitoring_path,
        monitoring_path,
        DEFAULT_MONITORING if DEFAULT_MONITORING.exists() else None,
    ):
        if candidate is None:
            continue
        p = Path(candidate)
        if not p.exists():
            results.append(_warn("s3.monitoring.exists", f"monitoring path missing: {p}"))
            continue
        if str(p) in resolved_paths:
            continue
        mon_frames.append(load_monitoring_artifact(p))
        resolved_paths.append(str(p))

    if not mon_frames:
        results.append(
            _warn(
                "s3.coverage_832",
                "no production monitoring CSV found — cannot verify 832/832; "
                "pass --monitoring / --per-round-monitoring / --frozen-monitoring",
            ),
        )
    else:
        mon = pd.concat(mon_frames, ignore_index=True)
        key_cols = ["match_id", "model_name", "cadence_mode"]
        before = mon[key_cols].drop_duplicates()
        union = pd.concat([mon[key_cols], bf[key_cols]], ignore_index=True)
        after = union.drop_duplicates()
        mon_keys = set(map(tuple, before.itertuples(index=False, name=None)))
        bf_keys = set(map(tuple, bf[key_cols].itertuples(index=False, name=None)))
        overlap = mon_keys & bf_keys

        per_cadence = after.groupby("cadence_mode").size().to_dict()
        numbers = {
            "monitoring_rows": len(mon),
            "monitoring_unique_keys": len(mon_keys),
            "backfill_keys": len(bf_keys),
            "overlap": len(overlap),
            **{f"union_{k}": int(v) for k, v in per_cadence.items()},
        }
        if overlap:
            results.append(
                _fail(
                    "s3.coverage_832.no_dup",
                    f"backfill keys already present in monitoring: {sorted(overlap)[:5]}",
                    **numbers,
                ),
            )
        else:
            results.append(
                _pass(
                    "s3.coverage_832.no_dup",
                    "backfill keys do not collide with monitoring",
                    **numbers,
                ),
            )

        # Full end-of-tournament exports should be near 832.  The dashboard
        # offline cache is often a mid-tournament partial — treat that as warn,
        # not fail, so the audit stays actionable without full MLflow exports.
        n_matches_in_mon = int(mon["match_id"].nunique()) if "match_id" in mon.columns else 0
        numbers["monitoring_unique_matches"] = n_matches_in_mon
        export_looks_complete = n_matches_in_mon >= N_MATCHES - 2

        incomplete = False
        for cadence in ("frozen", "per_round"):
            n = int(per_cadence.get(cadence, 0))
            if n == 0:
                incomplete = True
                results.append(
                    _warn(
                        f"s3.coverage_832.{cadence}",
                        f"no {cadence} keys after union — monitoring likely incomplete "
                        f"(offline cache may be partial)",
                        **numbers,
                    ),
                )
            elif n != EXPECTED_ROWS_PER_CADENCE:
                incomplete = True
                if export_looks_complete:
                    results.append(
                        _fail(
                            f"s3.coverage_832.{cadence}",
                            f"union has {n}/{EXPECTED_ROWS_PER_CADENCE} unique keys "
                            f"(monitoring covers {n_matches_in_mon} matches)",
                            count=n,
                            expected=EXPECTED_ROWS_PER_CADENCE,
                        ),
                    )
                else:
                    results.append(
                        _warn(
                            f"s3.coverage_832.{cadence}",
                            f"union has {n}/{EXPECTED_ROWS_PER_CADENCE} {cadence} keys — "
                            f"monitoring covers only {n_matches_in_mon}/{N_MATCHES} matches; "
                            "pass full frozen/per_round exports to verify 832/832",
                            count=n,
                            expected=EXPECTED_ROWS_PER_CADENCE,
                            monitoring_unique_matches=n_matches_in_mon,
                        ),
                    )
            else:
                results.append(
                    _pass(
                        f"s3.coverage_832.{cadence}",
                        f"{EXPECTED_ROWS_PER_CADENCE}/{EXPECTED_ROWS_PER_CADENCE} unique keys",
                        count=n,
                    ),
                )
        if incomplete and not any(
            r.name.startswith("s3.coverage_832.") and r.status == "fail" for r in results
        ):
            results.append(
                _warn(
                    "s3.coverage_832",
                    "full 832/832 claim not verifiable from available monitoring exports",
                    **numbers,
                ),
            )

    # --- structural checks ---
    if list(bf.columns) != list(MONITORING_COLUMNS):
        results.append(
            _fail("s3.schema", f"column mismatch: got {list(bf.columns)}"),
        )
    else:
        results.append(_pass("s3.schema", "schema matches monitoring columns"))

    if len(bf) != 4:
        results.append(_fail("s3.row_count", f"expected 4 rows, got {len(bf)}", rows=len(bf)))
    else:
        results.append(_pass("s3.row_count", "exactly 4 backfill rows", rows=4))

    expected_keys = set(_expected_backfill_keys())
    got_keys = set(
        (int(r.match_id), str(r.model_name), str(r.cadence_mode))
        for r in bf.itertuples(index=False)
    )
    if got_keys != expected_keys:
        results.append(
            _fail(
                "s3.keys",
                f"key mismatch: missing={sorted(expected_keys - got_keys)} "
                f"extra={sorted(got_keys - expected_keys)}",
            ),
        )
    else:
        results.append(_pass("s3.keys", "fixture/model/cadence keys match BACKFILL_FIXTURES"))

    non_copy = bf[~bf["inference_run_id"].str.startswith("backfill_copy_from_")]
    if not non_copy.empty:
        results.append(
            _fail(
                "s3.all_copied",
                f"{len(non_copy)} backfill row(s) are not cadence copies: "
                f"{sorted(set(non_copy['inference_run_id']))}",
            ),
        )
    else:
        results.append(_pass("s3.all_copied", "all backfill rows copied from a cadence twin"))
        results.append(_recompute_metrics_ok(bf, name="s3.backfill.metrics"))
        results.extend(_prob_and_outcome_ok(bf, name="s3.backfill"))

    # Copy-row twin identity against the source cadence monitoring export.
    twin_sources: dict[str, pd.DataFrame] = {}
    for candidate in (
        per_round_monitoring_path,
        frozen_monitoring_path,
        monitoring_path,
        DEFAULT_MONITORING,
    ):
        if candidate is None:
            continue
        p = Path(candidate)
        if not p.exists():
            continue
        src = load_monitoring_artifact(p)
        for cadence in ("frozen", "per_round"):
            if cadence not in twin_sources and (src["cadence_mode"] == cadence).any():
                twin_sources[cadence] = src

    copy_specs = tuple(
        (label, spec["model_name"], fixture_id, spec["copy_from_cadence"], spec["cadence_mode"])
        for label, spec in BACKFILL_FIXTURES.items()
        for fixture_id in spec["fixture_ids"]
    )
    if not twin_sources:
        results.append(_warn("s3.copy_twins", "no monitoring CSV for twin comparison"))
    else:
        for label, model_name, fixture_id, src_cadence, tgt_cadence in copy_specs:
            bf_row = bf[
                (bf["match_id"] == fixture_id)
                & (bf["model_name"] == model_name)
                & (bf["cadence_mode"] == tgt_cadence)
            ]
            twin_source = twin_sources.get(src_cadence)
            twin = (
                twin_source[
                    (twin_source["match_id"] == fixture_id)
                    & (twin_source["model_name"] == model_name)
                    & (twin_source["cadence_mode"] == src_cadence)
                ]
                if twin_source is not None
                else pd.DataFrame()
            )
            if bf_row.empty:
                results.append(_fail(f"s3.copy.{label}", "backfill row missing"))
                continue
            if twin.empty:
                results.append(
                    _warn(
                        f"s3.copy.{label}",
                        f"{src_cadence} twin not in monitoring export — "
                        f"cannot verify byte identity",
                    ),
                )
                continue
            float_cols = [
                c
                for c in (
                    "lambda_h",
                    "lambda_a",
                    "p_home",
                    "p_draw",
                    "p_away",
                    "rps",
                    "nll",
                    "rmse_h",
                    "rmse_a",
                )
                if c in bf_row.columns
            ]
            ok = True
            for c in float_cols:
                if abs(float(bf_row.iloc[0][c]) - float(twin.iloc[0][c])) > METRIC_TOL:
                    ok = False
                    break
            if ok:
                results.append(
                    _pass(
                        f"s3.copy.{label}",
                        f"{tgt_cadence} copy matches {src_cadence} twin floats",
                    ),
                )
            else:
                results.append(
                    _fail(f"s3.copy.{label}", f"{tgt_cadence} copy ≠ {src_cadence} twin"),
                )

    return _tag(results, "3")


# ---------------------------------------------------------------------------
# Strand 2
# ---------------------------------------------------------------------------


def _list_run_ids(model_dir: Path) -> dict[str, set[str]]:
    """Return {advancement, ko_pairings, both, orphan_*} run_id sets."""
    adv: set[str] = set()
    ko: set[str] = set()
    for p in model_dir.glob("*_advancement.csv"):
        adv.add(p.name.removesuffix("_advancement.csv"))
    for p in model_dir.glob("*_ko_pairings.csv"):
        ko.add(p.name.removesuffix("_ko_pairings.csv"))
    return {
        "advancement": adv,
        "ko_pairings": ko,
        "both": adv & ko,
        "orphan_adv": adv - ko,
        "orphan_ko": ko - adv,
    }


def _check_one_advancement(path: Path) -> list[str]:
    """Return list of problem strings (empty if OK)."""
    problems: list[str] = []
    df = pd.read_csv(path)
    if "team" not in df.columns:
        return ["missing team column"]
    n_teams = df["team"].nunique()
    if n_teams != N_WC_TEAMS:
        problems.append(f"teams={n_teams} (expected {N_WC_TEAMS})")
    for col in ADVANCEMENT_PROB_COLS:
        if col not in df.columns:
            problems.append(f"missing {col}")
            continue
        vals = df[col].astype(float)
        if ((vals < -PROB_TOL) | (vals > 1.0 + PROB_TOL)).any():
            problems.append(f"{col} outside [0,1]")
    if "p_group" in df.columns and not np.allclose(df["p_group"].astype(float), 1.0, atol=PROB_TOL):
        problems.append("p_group not all 1.0")
    for col, slots in ROUND_SLOT_COUNTS.items():
        if col not in df.columns:
            continue
        s = float(df[col].astype(float).sum())
        if abs(s - slots) > SUM_TOL:
            problems.append(f"{col} sum={s:.4f} (expected {slots})")
    # Monotonic non-increase across rounds per team
    round_cols = [c for c in ADVANCEMENT_PROB_COLS if c != "p_group" and c in df.columns]
    for _, row in df.iterrows():
        vals = [float(row[c]) for c in round_cols]
        for i in range(len(vals) - 1):
            if vals[i + 1] > vals[i] + PROB_TOL:
                problems.append("per-team round probs not non-increasing")
                break
        else:
            continue
        break
    # Zero stays zero downstream
    for _, row in df.iterrows():
        vals = [float(row[c]) for c in round_cols]
        seen_zero = False
        for v in vals:
            if seen_zero and v > PROB_TOL:
                problems.append("zero probability revived downstream")
                break
            if v <= PROB_TOL:
                seen_zero = True
        else:
            continue
        break
    return problems


def _check_one_ko_pairings(path: Path) -> list[str]:
    problems: list[str] = []
    df = pd.read_csv(path)
    if df.empty:
        return ["empty ko_pairings"]
    for col in ("stage", "count", "frequency"):
        if col not in df.columns:
            return [f"missing {col}"]
    freq = df["frequency"].astype(float)
    count = df["count"].astype(float)
    if ((freq < -PROB_TOL) | (freq > 1.0 + PROB_TOL)).any():
        problems.append("frequency outside [0,1]")
    # Infer n_sims from count/frequency where frequency > 0
    mask = freq > PROB_TOL
    if mask.any():
        n_sims_vals = (count[mask] / freq[mask]).round().astype(int)
        if n_sims_vals.nunique() != 1:
            problems.append(f"inconsistent n_sims values={sorted(n_sims_vals.unique())[:5]}")
        else:
            n_sims = int(n_sims_vals.iloc[0])
            if not np.allclose(freq[mask], count[mask] / n_sims, atol=1e-6):
                problems.append("frequency ≠ count/n_sims")
    for stage, expected in KO_STAGE_TIE_COUNTS.items():
        stage_df = df[df["stage"] == stage]
        if stage_df.empty:
            # Early tournament snapshots may lack later stages — skip.
            continue
        s = float(stage_df["frequency"].astype(float).sum())
        if abs(s - expected) > SUM_TOL:
            problems.append(f"{stage} frequency sum={s:.4f} (expected {expected})")
    return problems


def audit_strand2(
    root: Path = OUTPUT_ROOT,
    *,
    sample_entropy_recompute: int = 20,
    rng: np.random.Generator | None = None,
) -> list[CheckResult]:
    d = root / "strand2_brackets"
    results: list[CheckResult] = []
    if not d.is_dir():
        return _tag([_fail("s2.exists", f"missing {d}")], "2")

    rng = rng or np.random.default_rng(0)
    cadences = ("frozen", "per_round")
    models = tuple(EXPERIMENT_MODELS)
    cycle_counts: dict[tuple[str, str], int] = {}
    all_run_ids: dict[tuple[str, str], set[str]] = {}
    orphan_total = 0

    for cadence in cadences:
        for model in models:
            model_dir = d / cadence / model
            if not model_dir.is_dir():
                results.append(_fail(f"s2.{cadence}.{model}.dir", f"missing {model_dir}"))
                continue
            ids = _list_run_ids(model_dir)
            n_cycles = len(ids["both"])
            cycle_counts[(cadence, model)] = n_cycles
            all_run_ids[(cadence, model)] = set(ids["both"])
            orphan_total += len(ids["orphan_adv"]) + len(ids["orphan_ko"])
            if ids["orphan_adv"] or ids["orphan_ko"]:
                results.append(
                    _fail(
                        f"s2.{cadence}.{model}.pairing",
                        f"orphan advancement={len(ids['orphan_adv'])} "
                        f"ko_pairings={len(ids['orphan_ko'])}",
                        orphan_adv=len(ids["orphan_adv"]),
                        orphan_ko=len(ids["orphan_ko"]),
                    ),
                )
            else:
                results.append(
                    _pass(
                        f"s2.{cadence}.{model}.pairing",
                        f"{n_cycles} paired cycles, no orphans",
                        cycles=n_cycles,
                    ),
                )

    # Per-model frozen − per_round cycle-count delta (silent-skip asymmetry).
    # Run ids are cadence-specific, so only the count delta is meaningful.
    for model in models:
        n_f = cycle_counts.get(("frozen", model), 0)
        n_p = cycle_counts.get(("per_round", model), 0)
        delta = n_f - n_p
        if delta != 0:
            results.append(
                _warn(
                    f"s2.delta.{model}",
                    f"frozen={n_f} per_round={n_p} Δ={delta} "
                    "(cadence-asymmetric cycle counts; expected under silent skips)",
                    frozen=n_f,
                    per_round=n_p,
                    delta=delta,
                ),
            )
        else:
            results.append(
                _pass(
                    f"s2.delta.{model}",
                    f"frozen={n_f} per_round={n_p} (matched)",
                    frozen=n_f,
                    per_round=n_p,
                ),
            )

    # Sample advancement / ko_pairings content checks (full scan is fine but slow)
    adv_problems = 0
    ko_problems = 0
    adv_checked = 0
    ko_checked = 0
    sample_paths: list[Path] = []
    for cadence in cadences:
        for model in models:
            model_dir = d / cadence / model
            if not model_dir.is_dir():
                continue
            for adv_path in model_dir.glob("*_advancement.csv"):
                adv_checked += 1
                probs = _check_one_advancement(adv_path)
                if probs:
                    adv_problems += 1
                    if adv_problems <= 5:
                        results.append(
                            _fail(
                                f"s2.adv.{adv_path.stem}",
                                "; ".join(probs[:3]),
                            ),
                        )
                sample_paths.append(adv_path)
                ko_path = model_dir / f"{adv_path.name.removesuffix('_advancement.csv')}_ko_pairings.csv"
                if ko_path.exists():
                    ko_checked += 1
                    kprobs = _check_one_ko_pairings(ko_path)
                    if kprobs:
                        ko_problems += 1
                        if ko_problems <= 5:
                            results.append(
                                _fail(
                                    f"s2.ko.{ko_path.stem}",
                                    "; ".join(kprobs[:3]),
                                ),
                            )

    if adv_problems == 0:
        results.append(
            _pass("s2.advancement.content", f"all {adv_checked} advancement files OK", checked=adv_checked),
        )
    else:
        results.append(
            _fail(
                "s2.advancement.content",
                f"{adv_problems}/{adv_checked} advancement files have problems",
                problems=adv_problems,
                checked=adv_checked,
            ),
        )
    if ko_problems == 0:
        results.append(
            _pass("s2.ko_pairings.content", f"all {ko_checked} ko_pairings files OK", checked=ko_checked),
        )
    else:
        results.append(
            _fail(
                "s2.ko_pairings.content",
                f"{ko_problems}/{ko_checked} ko_pairings files have problems",
                problems=ko_problems,
                checked=ko_checked,
            ),
        )

    # Entropy trajectory
    et_path = d / "entropy_trajectory.csv"
    if not et_path.exists():
        results.append(_fail("s2.entropy.exists", "missing entropy_trajectory.csv"))
        return _tag(results, "2")

    et = pd.read_csv(et_path)
    et["inference_timestamp"] = pd.to_datetime(et["inference_timestamp"], utc=True)
    nan_entropy = int(et[list(ENTROPY_ORDER)].isna().any(axis=1).sum()) if set(ENTROPY_ORDER) <= set(et.columns) else len(et)
    if nan_entropy:
        results.append(_fail("s2.entropy.nan", f"{nan_entropy} rows with NaN entropy", nan_rows=nan_entropy))
    else:
        results.append(_pass("s2.entropy.nan", "no NaN entropy columns", rows=len(et)))

    results.append(_entropy_row_ordering(et, name="s2.entropy.ordering"))

    # run_id set vs disk
    for cadence in cadences:
        for model in models:
            disk_ids = all_run_ids.get((cadence, model), set())
            et_ids = set(
                et.loc[
                    (et["cadence_mode"] == cadence) & (et["model_name"] == model),
                    "inference_run_id",
                ].astype(str),
            )
            if disk_ids != et_ids:
                results.append(
                    _fail(
                        f"s2.entropy.ids.{cadence}.{model}",
                        f"trajectory≠disk: only_et={len(et_ids - disk_ids)} "
                        f"only_disk={len(disk_ids - et_ids)}",
                        only_et=len(et_ids - disk_ids),
                        only_disk=len(disk_ids - et_ids),
                    ),
                )
            else:
                results.append(
                    _pass(
                        f"s2.entropy.ids.{cadence}.{model}",
                        f"{len(disk_ids)} run_ids match disk",
                        n=len(disk_ids),
                    ),
                )

    # Recompute entropy for a sample of advancement files
    if sample_paths:
        sample_n = min(sample_entropy_recompute, len(sample_paths))
        idxs = rng.choice(len(sample_paths), size=sample_n, replace=False)
        mismatches = 0
        max_abs = 0.0
        for i in idxs:
            adv_path = sample_paths[int(i)]
            # path: .../{cadence}/{model}/{run_id}_advancement.csv
            run_id = adv_path.name.removesuffix("_advancement.csv")
            model = adv_path.parent.name
            cadence = adv_path.parent.parent.name
            adv = pd.read_csv(adv_path)
            recomputed = compute_entropy_columns(adv)
            row = et[
                (et["inference_run_id"] == run_id)
                & (et["model_name"] == model)
                & (et["cadence_mode"] == cadence)
            ]
            if row.empty:
                mismatches += 1
                continue
            for k, v in recomputed.items():
                delta = abs(float(row.iloc[0][k]) - float(v))
                max_abs = max(max_abs, delta)
                if delta > 1e-6:
                    mismatches += 1
                    break
        if mismatches:
            results.append(
                _fail(
                    "s2.entropy.recompute",
                    f"{mismatches}/{sample_n} sampled rows disagree with recomputed entropy",
                    mismatches=mismatches,
                    sampled=sample_n,
                    max_abs_delta=max_abs,
                ),
            )
        else:
            results.append(
                _pass(
                    "s2.entropy.recompute",
                    f"{sample_n} sampled advancement→entropy recomputations match",
                    sampled=sample_n,
                    max_abs_delta=max_abs,
                ),
            )

    # Degeneracy: constant entropy_winner per (cadence, model)
    for cadence in cadences:
        for model in models:
            sub = et[(et["cadence_mode"] == cadence) & (et["model_name"] == model)]
            if sub.empty or "entropy_winner" not in sub.columns:
                continue
            vals = sub["entropy_winner"].dropna()
            if len(vals) and vals.nunique() == 1:
                results.append(
                    _fail(
                        f"s2.entropy.degen.{cadence}.{model}",
                        f"entropy_winner constant at {float(vals.iloc[0]):.6f} "
                        f"across {len(vals)} points — D.1 leak signature",
                        constant=float(vals.iloc[0]),
                        points=len(vals),
                    ),
                )
            else:
                # Also run helper for logging-style metrics (warns on constant)
                check_entropy_trajectory(sub)
                results.append(
                    _pass(
                        f"s2.entropy.degen.{cadence}.{model}",
                        f"entropy_winner varies across {len(vals)} points",
                        nunique=int(vals.nunique()),
                    ),
                )

    return _tag(results, "2")


# ---------------------------------------------------------------------------
# Strand 4
# ---------------------------------------------------------------------------


def audit_strand4(root: Path = OUTPUT_ROOT) -> list[CheckResult]:
    d = root / "strand4_entropy"
    results: list[CheckResult] = []
    path = d / "reconstructed_entropy_snapshots.csv"
    if not path.exists():
        return _tag([_fail("s4.exists", f"missing {path}")], "4")

    s4 = pd.read_csv(path)
    s4["inference_timestamp"] = pd.to_datetime(s4["inference_timestamp"], utc=True)

    expected_labels = {
        "r32_pre_japan_brazil",
        "r32_japan_brazil_locked",
        "r32_germany_paraguay_locked",
        "r16_brazil_norway_locked",
    }
    labels = set(s4["snapshot_label"].astype(str))
    if labels != expected_labels:
        results.append(
            _fail(
                "s4.labels",
                f"expected {sorted(expected_labels)}, got {sorted(labels)}",
            ),
        )
    else:
        results.append(_pass("s4.labels", "4 expected snapshot labels present"))

    if "synthetic" in s4.columns and not s4["synthetic"].astype(bool).all():
        results.append(_fail("s4.synthetic", "not all rows marked synthetic=True"))
    else:
        results.append(_pass("s4.synthetic", "all rows synthetic=True"))

    models = set(EXPERIMENT_MODELS)
    for label in sorted(expected_labels):
        sub = s4[s4["snapshot_label"] == label]
        cadences = set(sub["cadence_mode"])
        if cadences != {"frozen", "per_round"}:
            results.append(
                _fail(f"s4.{label}.cadence", f"cadences={sorted(cadences)}"),
            )
        else:
            results.append(_pass(f"s4.{label}.cadence", "both cadences present"))
        for cadence in ("frozen", "per_round"):
            got = set(sub.loc[sub["cadence_mode"] == cadence, "model_name"])
            if got != models:
                results.append(
                    _fail(
                        f"s4.{label}.{cadence}.models",
                        f"models={sorted(got)} expected={sorted(models)}",
                    ),
                )
            else:
                results.append(
                    _pass(f"s4.{label}.{cadence}.models", "EXPERIMENT_MODELS complete"),
                )

    nan_entropy = int(s4[list(ENTROPY_ORDER)].isna().any(axis=1).sum()) if set(ENTROPY_ORDER) <= set(s4.columns) else len(s4)
    if nan_entropy:
        results.append(_fail("s4.entropy.nan", f"{nan_entropy} NaN entropy rows"))
    else:
        results.append(_pass("s4.entropy.nan", "no NaN entropy", rows=len(s4)))
    results.append(_entropy_row_ordering(s4, name="s4.entropy.ordering"))

    # Timestamp bracketing via strand4 specs + Bronze kickoffs
    try:
        from src.analysis.strand4_entropy import build_entropy_snapshot_specs
        from src.monitoring.monitor import parse_wc_settled_matches

        settled = parse_wc_settled_matches()
        specs = {s.label: s for s in build_entropy_snapshot_specs(settled)}
        for label, spec in specs.items():
            sub = s4[s4["snapshot_label"] == label]
            if sub.empty:
                continue
            ts = pd.to_datetime(sub["inference_timestamp"].iloc[0], utc=True)
            # Synthetic timestamp should match the spec
            if abs((ts - spec.synthetic_timestamp).total_seconds()) > 1:
                results.append(
                    _fail(
                        f"s4.{label}.ts_match",
                        f"timestamp {ts} ≠ spec {spec.synthetic_timestamp}",
                    ),
                )
            else:
                results.append(
                    _pass(f"s4.{label}.ts_match", "synthetic timestamp matches spec"),
                )
    except Exception as exc:  # noqa: BLE001
        results.append(_warn("s4.specs", f"could not build snapshot specs: {exc}"))

    # No collision with real entropy trajectory timestamps
    et_path = root / "strand2_brackets" / "entropy_trajectory.csv"
    if et_path.exists():
        et = pd.read_csv(et_path)
        et["inference_timestamp"] = pd.to_datetime(et["inference_timestamp"], utc=True)
        real_ts = set(et["inference_timestamp"])
        synth_ts = set(s4["inference_timestamp"])
        collide = real_ts & synth_ts
        if collide:
            results.append(
                _fail(
                    "s4.ts_collision",
                    f"{len(collide)} synthetic timestamps collide with real trajectory",
                    collisions=len(collide),
                ),
            )
        else:
            results.append(_pass("s4.ts_collision", "no timestamp collisions with real trajectory"))

        # Continuity: entropy_winner jump vs neighbours
        jumps_real: list[float] = []
        for (cadence, model), g in et.groupby(["cadence_mode", "model_name"]):
            g = g.sort_values("inference_timestamp")
            vals = g["entropy_winner"].astype(float).to_numpy()
            if len(vals) >= 2:
                jumps_real.extend(np.abs(np.diff(vals)).tolist())
        p95 = float(np.percentile(jumps_real, 95)) if jumps_real else float("inf")

        big_jumps = 0
        worst = 0.0
        for _, row in s4.iterrows():
            cadence, model = row["cadence_mode"], row["model_name"]
            ts = row["inference_timestamp"]
            series = et[
                (et["cadence_mode"] == cadence) & (et["model_name"] == model)
            ].sort_values("inference_timestamp")
            if series.empty:
                continue
            before = series[series["inference_timestamp"] < ts]
            after = series[series["inference_timestamp"] > ts]
            neighbours: list[float] = []
            if not before.empty:
                neighbours.append(float(before.iloc[-1]["entropy_winner"]))
            if not after.empty:
                neighbours.append(float(after.iloc[0]["entropy_winner"]))
            for nval in neighbours:
                jump = abs(float(row["entropy_winner"]) - nval)
                worst = max(worst, jump)
                if jump > p95 and p95 < float("inf"):
                    big_jumps += 1
        if big_jumps:
            results.append(
                _warn(
                    "s4.continuity",
                    f"{big_jumps} neighbour jumps exceed real p95={p95:.4f} "
                    f"(worst={worst:.4f})",
                    big_jumps=big_jumps,
                    p95=p95,
                    worst=worst,
                ),
            )
        else:
            results.append(
                _pass(
                    "s4.continuity",
                    f"synthetic neighbour jumps within real p95={p95:.4f}",
                    p95=p95,
                    worst=worst,
                ),
            )
    else:
        results.append(_warn("s4.trajectory", "strand2 entropy_trajectory.csv missing; skipped continuity"))

    # Leakage guard: parse_wc_results_before_kickoff with SETTLE_DELTA
    try:
        from src.analysis.strand4_entropy import build_entropy_snapshot_specs
        from src.monitoring.monitor import parse_wc_settled_matches

        settled = parse_wc_settled_matches()
        specs = build_entropy_snapshot_specs(settled)
        finished = load_finished_wc_fixtures()
        leak_count = 0
        for spec in specs:
            wc = parse_wc_results_before_kickoff(spec.max_kickoff, settle_delta=SETTLE_DELTA)
            locked_keys = set(wc["ko_results"].keys())
            for fx in finished:
                if not fx.round_str.lower().startswith("group"):
                    key = frozenset({fx.home_team, fx.away_team})
                    if key in locked_keys and fx.kickoff + SETTLE_DELTA > spec.max_kickoff:
                        leak_count += 1
        if leak_count:
            results.append(
                _fail(
                    "s4.leakage",
                    f"{leak_count} unsettled KO fixtures appear in locked ko_results "
                    f"under SETTLE_DELTA={SETTLE_DELTA}",
                    leaks=leak_count,
                ),
            )
        else:
            results.append(
                _pass(
                    "s4.leakage",
                    f"no unsettled KO fixtures in locked results (SETTLE_DELTA={SETTLE_DELTA})",
                ),
            )
    except Exception as exc:  # noqa: BLE001
        results.append(_warn("s4.leakage", f"leakage check skipped: {exc}"))

    return _tag(results, "4")


# ---------------------------------------------------------------------------
# Optional MLflow / determinism
# ---------------------------------------------------------------------------


def audit_mlflow() -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        from src.models.mlflow_utils import SHADOW_MODEL_NAME, setup_mlflow

        setup_mlflow()
        client = MlflowClient()
        exp = client.get_experiment_by_name(RECONSTRUCTION_EXPERIMENT)
        if exp is None:
            return [_fail("mlflow.experiment", f"experiment {RECONSTRUCTION_EXPERIMENT!r} not found")]
        results.append(_pass("mlflow.experiment", f"found {RECONSTRUCTION_EXPERIMENT}"))

        runs = client.search_runs(
            [exp.experiment_id],
            max_results=500,
            order_by=["attributes.start_time DESC"],
        )
        if not runs:
            results.append(_warn("mlflow.runs", "no runs in reconstruction experiment"))
            return _tag(results, "mlflow")

        bad_stage = 0
        missing_strand = 0
        for run in runs:
            tags = run.data.tags
            if tags.get("stage") != "reconstruction":
                bad_stage += 1
            if "strand" not in tags:
                missing_strand += 1
        if bad_stage:
            results.append(
                _fail("mlflow.stage_tag", f"{bad_stage} runs without stage=reconstruction"),
            )
        else:
            results.append(_pass("mlflow.stage_tag", "all runs tagged stage=reconstruction"))
        if missing_strand:
            results.append(_fail("mlflow.strand_tag", f"{missing_strand} runs missing strand tag"))
        else:
            results.append(_pass("mlflow.strand_tag", "all runs tagged with strand"))

        # Pinned frozen shadow versions must resolve to distinct run ids
        run_ids: dict[str, str] = {}
        for model, version in PINNED_FROZEN_SHADOW_VERSIONS.items():
            mv = client.get_model_version(SHADOW_MODEL_NAME, str(version))
            run_ids[model] = mv.run_id
        if len(set(run_ids.values())) != len(run_ids):
            results.append(
                _fail(
                    "mlflow.pinned_distinct",
                    f"pinned versions share run ids: {run_ids}",
                ),
            )
        else:
            results.append(
                _pass(
                    "mlflow.pinned_distinct",
                    f"pinned versions distinct: {run_ids}",
                ),
            )

        # Soft check: no newer shadow versions beyond known pins for those models
        # (informational — registry may have grown with per_round versions)
        for model, version in PINNED_FROZEN_SHADOW_VERSIONS.items():
            results.append(
                _pass(
                    f"mlflow.pinned.{model}",
                    f"{SHADOW_MODEL_NAME} v{version} resolves",
                    version=version,
                ),
            )

    except Exception as exc:  # noqa: BLE001
        results.append(_fail("mlflow.error", f"MLflow audit failed: {exc}"))
    return _tag(results, "mlflow")


def audit_determinism(n: int, root: Path = OUTPUT_ROOT) -> list[CheckResult]:
    """Re-run N sampled strand-2 cycles and diff advancement CSVs."""
    results: list[CheckResult] = []
    et_path = root / "strand2_brackets" / "entropy_trajectory.csv"
    if not et_path.exists() or n <= 0:
        return _tag([_warn("det.skip", "no entropy trajectory or N<=0")], "determinism")

    try:
        from src.analysis.strand2_brackets import _replay_simulation_for_run
    except Exception as exc:  # noqa: BLE001
        return _tag([_fail("det.import", str(exc))], "determinism")

    et = pd.read_csv(et_path)
    et["inference_timestamp"] = pd.to_datetime(et["inference_timestamp"], utc=True)
    # Prefer frozen/xgboost for sampling (largest set)
    pool = et[(et["cadence_mode"] == "frozen") & (et["model_name"] == "xgboost")]
    if pool.empty:
        pool = et
    sample = pool.drop_duplicates("inference_run_id").sample(
        n=min(n, pool["inference_run_id"].nunique()),
        random_state=0,
    )

    mismatches = 0
    checked = 0
    for _, row in sample.iterrows():
        run_id = str(row["inference_run_id"])
        cadence = str(row["cadence_mode"])
        model = str(row["model_name"])
        ts = row["inference_timestamp"]
        disk = root / "strand2_brackets" / cadence / model / f"{run_id}_advancement.csv"
        if not disk.exists():
            continue
        try:
            per_model = _replay_simulation_for_run(run_id, cadence_mode=cadence, as_of=ts)
            adv = per_model.get(model, {}).get("advancement")
            if adv is None or adv.empty:
                results.append(_warn(f"det.{run_id}", "replay returned empty advancement"))
                continue
            disk_df = pd.read_csv(disk)
            checked += 1
            # Compare on team-aligned p_winner
            merged = disk_df[["team", "p_winner"]].merge(
                adv[["team", "p_winner"]],
                on="team",
                suffixes=("_disk", "_re"),
            )
            if not np.allclose(
                merged["p_winner_disk"].astype(float),
                merged["p_winner_re"].astype(float),
                atol=1e-6,
                equal_nan=True,
            ):
                mismatches += 1
                results.append(_fail(f"det.{run_id}", "p_winner differs on re-run"))
        except Exception as exc:  # noqa: BLE001
            results.append(_warn(f"det.{run_id}", f"replay failed: {exc}"))

    if checked == 0:
        results.append(_warn("det.summary", "no cycles successfully re-run"))
    elif mismatches == 0:
        results.append(_pass("det.summary", f"{checked} cycles deterministic", checked=checked))
    else:
        results.append(
            _fail(
                "det.summary",
                f"{mismatches}/{checked} cycles non-deterministic",
                mismatches=mismatches,
                checked=checked,
            ),
        )
    return _tag(results, "determinism")


# ---------------------------------------------------------------------------
# Report / CLI
# ---------------------------------------------------------------------------


def run_audit(
    *,
    strands: Sequence[str] = ("1", "2", "3", "4"),
    root: Path = OUTPUT_ROOT,
    with_mlflow: bool = False,
    with_determinism: int = 0,
    monitoring_path: Path | None = None,
    per_round_monitoring_path: Path | None = None,
    frozen_monitoring_path: Path | None = None,
) -> list[CheckResult]:
    """Run selected strand audits. Strand 3 runs before 2 when both selected."""
    results: list[CheckResult] = []
    selected = {s.strip() for s in strands}

    # Prefer strand 3 early among the CSV strands
    order = [s for s in ("1", "3", "2", "4") if s in selected]
    dispatch: dict[str, Callable[[], list[CheckResult]]] = {
        "1": lambda: audit_strand1(root),
        "3": lambda: audit_strand3(
            root,
            monitoring_path=monitoring_path,
            per_round_monitoring_path=per_round_monitoring_path,
            frozen_monitoring_path=frozen_monitoring_path,
        ),
        "2": lambda: audit_strand2(root),
        "4": lambda: audit_strand4(root),
    }
    for key in order:
        logger.info("Auditing strand %s …", key)
        results.extend(dispatch[key]())

    if with_mlflow:
        logger.info("Auditing MLflow …")
        results.extend(audit_mlflow())
    if with_determinism > 0:
        logger.info("Auditing determinism (N=%d) …", with_determinism)
        results.extend(audit_determinism(with_determinism, root))

    return results


def print_report(results: Iterable[CheckResult]) -> None:
    rows = list(results)
    width = max((len(r.name) for r in rows), default=10)
    print()
    print(f"{'STATUS':<6}  {'STRAND':<10}  {'CHECK':<{width}}  DETAIL")
    print("-" * (width + 60))
    for r in rows:
        print(f"{r.status.upper():<6}  {r.strand:<10}  {r.name:<{width}}  {r.detail}")
    counts = {s: sum(1 for r in rows if r.status == s) for s in ("pass", "warn", "fail")}
    print("-" * (width + 60))
    print(
        f"Summary: {counts['pass']} pass, {counts['warn']} warn, {counts['fail']} fail "
        f"(total {len(rows)})",
    )


def write_report_json(results: Sequence[CheckResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_pass": sum(1 for r in results if r.status == "pass"),
        "n_warn": sum(1 for r in results if r.status == "warn"),
        "n_fail": sum(1 for r in results if r.status == "fail"),
        "checks": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote audit report to %s", path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit data/reconstruction/ completeness and consistency.",
    )
    parser.add_argument(
        "--strand",
        default="1,2,3,4",
        help="Comma-separated strand numbers (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Reconstruction root (default: data/reconstruction)",
    )
    parser.add_argument(
        "--with-mlflow",
        action="store_true",
        help="Also audit MLflow reconstruction experiment / pinned versions",
    )
    parser.add_argument(
        "--with-determinism",
        type=int,
        default=0,
        metavar="N",
        help="Re-run N sampled strand-2 cycles and diff (needs MLflow artifacts)",
    )
    parser.add_argument(
        "--monitoring",
        type=Path,
        default=None,
        help="Production monitoring CSV (default: dashboard offline cache if present)",
    )
    parser.add_argument(
        "--per-round-monitoring",
        type=Path,
        default=None,
        help="Per-round monitoring CSV for strand-3 twin / 832 checks",
    )
    parser.add_argument(
        "--frozen-monitoring",
        type=Path,
        default=None,
        help="Frozen monitoring CSV for strand-3 832 checks",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_AUDIT_JSON,
        help=f"Write machine-readable report (default: {DEFAULT_AUDIT_JSON})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    strands = [s.strip() for s in args.strand.split(",") if s.strip()]
    results = run_audit(
        strands=strands,
        root=args.root,
        with_mlflow=args.with_mlflow,
        with_determinism=args.with_determinism,
        monitoring_path=args.monitoring,
        per_round_monitoring_path=args.per_round_monitoring,
        frozen_monitoring_path=args.frozen_monitoring,
    )
    print_report(results)
    if args.json:
        write_report_json(results, args.json)

    n_fail = sum(1 for r in results if r.status == "fail")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
