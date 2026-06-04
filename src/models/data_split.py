"""Gold data loading, time-based splitting, and walk-forward CV."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.gold.schema import TARGET_COLUMNS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GOLD_PATH: Path = Path("data/gold/matches.parquet")

WC_2022_START: str = "2022-11-20"
WC_2022_END: str = "2022-12-18"

# A.3 — Expanded holdout: WC 2022 + continental final tournaments through 2026.
# Each entry is (name, league_id, start, end).  Both the date window and the
# league_id are required so that friendlies/qualifiers inside the window are
# excluded and the two editions of AFCON/Gold Cup (same league_id) are kept
# apart.  All windows start after WC_2022_START, so there is no overlap with
# the training period.
HOLDOUT_TOURNAMENTS: list[tuple[str, int, str, str]] = [
    ("WC 2022", 1, "2022-11-20", "2022-12-18"),
    ("Gold Cup 2023", 22, "2023-06-01", "2023-07-31"),
    ("Asian Cup 2024", 7, "2024-01-01", "2024-02-29"),
    ("AFCON 2024", 6, "2024-01-01", "2024-02-29"),
    ("EURO 2024", 4, "2024-06-01", "2024-07-31"),
    ("Copa 2024", 9, "2024-06-01", "2024-07-31"),
    ("Gold Cup 2025", 22, "2025-06-01", "2025-07-31"),
    ("AFCON 2025", 6, "2025-12-01", "2026-02-28"),
]

# A.4 — Match-importance sample weights (Ley et al. 2019 / pre-2018 FIFA).
# Mapped from Gold ``competition_tier``.
IMPORTANCE_WEIGHTS: dict[int, float] = {1: 4.0, 2: 3.0, 3: 2.5, 4: 1.0}

# Default exponential time-decay half-life (Ley et al.'s national-team optimum).
DEFAULT_HALF_PERIOD_YEARS: float = 3.0
_DAYS_PER_YEAR: float = 365.25

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_gold(path: Path = GOLD_PATH) -> pd.DataFrame:
    """Load the Gold parquet and sort by date."""
    df = pd.read_parquet(path)
    return df.sort_values("date_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


class DataSplits(NamedTuple):
    """Container for the three data partitions (train / holdout / full).

    ``w_train`` and ``w_full`` are per-row sample weights (time-decay x match
    importance) aligned with ``X_train`` / ``X_full`` (A.4).
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray
    X_full: np.ndarray
    y_full: np.ndarray
    df_train: pd.DataFrame
    df_holdout: pd.DataFrame
    df_full: pd.DataFrame
    # Sample weights default to None so callers that construct DataSplits
    # directly (e.g. unit-test fixtures) need not supply them; make_splits
    # always populates them.  None → unweighted fit.
    w_train: np.ndarray | None = None
    w_full: np.ndarray | None = None


def _to_float_array(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Convert a subset of a DataFrame to a float64 numpy array."""
    return df[cols].to_numpy(dtype="float64", na_value=np.nan)


def compute_weight_components(
    df: pd.DataFrame,
    *,
    reference_date: pd.Timestamp | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (days_ago, w_importance) arrays aligned with *df* rows.

    These are the two stable components of the sample weight that do not
    depend on ``half_period_years``.  Call this once before Optuna tuning,
    then recompute ``w_time = 0.5 ** (days_ago / (hp * _DAYS_PER_YEAR))``
    per trial — one numpy operation — and multiply by ``w_importance``.

    Args:
        df: DataFrame with ``date_utc`` and ``competition_tier`` columns.
        reference_date: "Present" reference for recency; defaults to df max date.

    Returns:
        (days_ago, w_importance) — both float64 arrays of shape (len(df),).
        Empty df → (empty, empty).
    """
    if len(df) == 0:
        empty = np.empty(0, dtype="float64")
        return empty, empty

    dates = pd.to_datetime(df["date_utc"])
    ref = reference_date if reference_date is not None else dates.max()
    days_ago = (ref - dates).dt.days.clip(lower=0).to_numpy(dtype="float64")

    tier = df["competition_tier"].astype("int64")
    w_importance = (
        tier.map(IMPORTANCE_WEIGHTS).fillna(IMPORTANCE_WEIGHTS[4]).to_numpy(dtype="float64")
    )
    return days_ago, w_importance


def compute_sample_weights(
    df: pd.DataFrame,
    *,
    half_period_years: float = DEFAULT_HALF_PERIOD_YEARS,
    reference_date: pd.Timestamp | None = None,
) -> np.ndarray:
    """Compute time-decay x match-importance training weights (A.4).

    ``w_time = 0.5 ** (days_ago / (half_period_years * 365.25))`` where
    ``days_ago`` is measured relative to *reference_date* (default: the most
    recent ``date_utc`` in *df*, so the newest match has weight ~= importance
    and weights never exceed the importance ceiling).  ``w_importance`` maps
    ``competition_tier`` via ``IMPORTANCE_WEIGHTS``.

    Args:
        df: DataFrame with ``date_utc`` and ``competition_tier`` columns.
        half_period_years: Time-decay half-life in years.
        reference_date: "Present" reference for recency; defaults to df max date.

    Returns:
        Float64 array of weights aligned with *df* rows.  Empty df → empty array.
    """
    days_ago, w_importance = compute_weight_components(df, reference_date=reference_date)
    if len(days_ago) == 0:
        return np.empty(0, dtype="float64")
    w_time = 0.5 ** (days_ago / (half_period_years * _DAYS_PER_YEAR))
    return w_time * w_importance


def make_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str] | None = None,
    *,
    dropna: bool = True,
    half_period_years: float = DEFAULT_HALF_PERIOD_YEARS,
) -> DataSplits:
    """Split Gold data into train (pre-WC 2022), holdout (WC 2022), and full.

    Args:
        df: Gold DataFrame (sorted by date_utc).
        feature_cols: Columns to use as features.
        target_cols: Target columns (default: schema TARGET_COLUMNS).
        dropna: If True, drop rows with NaN in feature_cols + target_cols.
            Set False for models that handle NaN natively (e.g. XGBoost).
        half_period_years: Time-decay half-life passed to
            ``compute_sample_weights`` for ``w_train`` and ``w_full``.
    """
    if target_cols is None:
        target_cols = list(TARGET_COLUMNS)

    df = df.sort_values("date_utc").reset_index(drop=True)

    if dropna:
        df = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

    dates = pd.to_datetime(df["date_utc"])
    train_mask = dates < pd.Timestamp(WC_2022_START)
    holdout_mask = _holdout_mask(df, dates)

    df_train = df.loc[train_mask].reset_index(drop=True)
    df_holdout = df.loc[holdout_mask].reset_index(drop=True)
    df_full = df.reset_index(drop=True)

    return DataSplits(
        X_train=_to_float_array(df_train, feature_cols),
        y_train=_to_float_array(df_train, target_cols),
        X_holdout=_to_float_array(df_holdout, feature_cols),
        y_holdout=_to_float_array(df_holdout, target_cols),
        X_full=_to_float_array(df_full, feature_cols),
        y_full=_to_float_array(df_full, target_cols),
        df_train=df_train,
        df_holdout=df_holdout,
        df_full=df_full,
        w_train=compute_sample_weights(df_train, half_period_years=half_period_years),
        w_full=compute_sample_weights(df_full, half_period_years=half_period_years),
    )


def _holdout_mask(df: pd.DataFrame, dates: pd.Series) -> pd.Series:
    """Boolean mask selecting rows belonging to any holdout tournament (A.3).

    A row qualifies if it falls inside a tournament's date window and matches
    that tournament's ``league_id``.  When ``league_id`` is absent (synthetic
    test frames), the date window alone is used so existing fixtures still work.
    """
    has_league = "league_id" in df.columns
    mask = pd.Series(False, index=df.index)
    for _name, league_id, start, end in HOLDOUT_TOURNAMENTS:
        window = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        if has_league:
            window &= df["league_id"] == league_id
        mask |= window
    return mask


# ---------------------------------------------------------------------------
# Walk-forward expanding-window cross-validation
# ---------------------------------------------------------------------------


def walk_forward_cv(
    n_samples: int,
    n_splits: int = 5,
    min_train_frac: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward expanding-window CV folds.

    The first ``min_train_frac`` of data forms the initial training window.
    The remainder is divided into ``n_splits`` equal validation chunks.
    Each successive fold expands training to include prior validation data.

    Returns:
        List of (train_indices, val_indices) pairs.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if not 0 < min_train_frac < 1:
        raise ValueError("min_train_frac must be in (0, 1)")

    min_train = int(n_samples * min_train_frac)
    remaining = n_samples - min_train
    val_size = remaining // n_splits

    if val_size < 1:
        raise ValueError(
            f"Not enough samples ({n_samples}) for {n_splits} splits "
            f"with min_train_frac={min_train_frac}"
        )

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        train_end = min_train + i * val_size
        val_start = train_end
        val_end = val_start + val_size if i < n_splits - 1 else n_samples
        folds.append(
            (np.arange(train_end), np.arange(val_start, val_end))
        )
    return folds
