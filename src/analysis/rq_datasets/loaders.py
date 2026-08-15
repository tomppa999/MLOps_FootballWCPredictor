"""Validated loaders for the frozen RQ analysis CSVs.

Thesis analysis code should import only from here::

    from src.analysis.rq_datasets import load_rq1, load_rq2, load_rq3
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.rq_datasets.build_rq_datasets import assert_unique_keys
from src.analysis.rq_datasets.paths import (
    ANALYSIS_ROOT,
    CADENCE_MODES,
    EXPECTED_ROWS_PER_CADENCE,
    PROVENANCE_VALUES,
    RQ1_KEY,
    RQ1_PATH,
    RQ2_KEY,
    RQ2_PATH,
    RQ3_PATH,
)


def _resolve(path: Path | None, default: Path, analysis_root: Path, filename: str) -> Path:
    if path is not None:
        resolved = path
    elif analysis_root == ANALYSIS_ROOT:
        resolved = default
    else:
        resolved = analysis_root / filename
    if not resolved.exists():
        raise FileNotFoundError(
            f"Missing {resolved}. Build with: "
            "python -m src.analysis.rq_datasets.export_live && "
            "python -m src.analysis.rq_datasets.build_rq_datasets",
        )
    return resolved


def validate_rq1(df: pd.DataFrame) -> None:
    assert_unique_keys(df, RQ1_KEY, label="rq1_matches")
    for cadence in CADENCE_MODES:
        n = int((df["cadence_mode"] == cadence).sum())
        if n != EXPECTED_ROWS_PER_CADENCE:
            raise AssertionError(
                f"rq1 {cadence}: expected {EXPECTED_ROWS_PER_CADENCE} rows, got {n}",
            )
    unknown = set(df["provenance"]) - PROVENANCE_VALUES
    if unknown:
        raise AssertionError(f"Unexpected provenance values: {unknown}")
    if df["round_label"].isna().any():
        raise AssertionError("rq1: round_label has nulls")


def validate_rq2(df: pd.DataFrame) -> None:
    assert_unique_keys(df, RQ2_KEY, label="rq2_entropy")
    if "synthetic" not in df.columns:
        raise AssertionError("rq2: missing synthetic column")
    if df["cycle_index"].isna().any():
        raise AssertionError("rq2: cycle_index has nulls")


def validate_rq3(df: pd.DataFrame) -> None:
    if "record_type" not in df.columns:
        raise AssertionError("rq3: missing record_type")
    kinds = set(df["record_type"].unique())
    if not {"reliability", "cum_rps"}.issubset(kinds):
        raise AssertionError(f"rq3: unexpected record_type set {kinds}")
    if df.loc[df["record_type"] == "reliability"].empty:
        raise AssertionError("rq3: no reliability rows")


def load_rq1(
    path: Path | None = None,
    *,
    validate: bool = True,
    analysis_root: Path = ANALYSIS_ROOT,
) -> pd.DataFrame:
    """Load ``rq1_matches.csv`` with typed kickoff timestamps."""
    p = _resolve(path, RQ1_PATH, analysis_root, "rq1_matches.csv")
    df = pd.read_csv(p)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True, format="ISO8601")
    if validate:
        validate_rq1(df)
    return df


def load_rq2(
    path: Path | None = None,
    *,
    validate: bool = True,
    analysis_root: Path = ANALYSIS_ROOT,
) -> pd.DataFrame:
    """Load ``rq2_entropy.csv`` with typed timestamps."""
    p = _resolve(path, RQ2_PATH, analysis_root, "rq2_entropy.csv")
    df = pd.read_csv(p)
    df["inference_timestamp"] = pd.to_datetime(
        df["inference_timestamp"], utc=True, format="ISO8601",
    )
    if "synthetic" in df.columns:
        df["synthetic"] = df["synthetic"].astype(bool)
    if validate:
        validate_rq2(df)
    return df


def load_rq3(
    path: Path | None = None,
    *,
    validate: bool = True,
    analysis_root: Path = ANALYSIS_ROOT,
) -> pd.DataFrame:
    """Load ``rq3_calibration.csv``."""
    p = _resolve(path, RQ3_PATH, analysis_root, "rq3_calibration.csv")
    df = pd.read_csv(p)
    if "kickoff_utc" in df.columns:
        df["kickoff_utc"] = pd.to_datetime(
            df["kickoff_utc"], utc=True, format="ISO8601",
        )
    if validate:
        validate_rq3(df)
    return df
