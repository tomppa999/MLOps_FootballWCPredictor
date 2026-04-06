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
    """Container for the three data partitions (train / holdout / full)."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray
    X_full: np.ndarray
    y_full: np.ndarray
    df_train: pd.DataFrame
    df_holdout: pd.DataFrame
    df_full: pd.DataFrame


def _to_float_array(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Convert a subset of a DataFrame to a float64 numpy array."""
    return df[cols].to_numpy(dtype="float64", na_value=np.nan)


def make_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str] | None = None,
    *,
    dropna: bool = True,
) -> DataSplits:
    """Split Gold data into train (pre-WC 2022), holdout (WC 2022), and full.

    Args:
        df: Gold DataFrame (sorted by date_utc).
        feature_cols: Columns to use as features.
        target_cols: Target columns (default: schema TARGET_COLUMNS).
        dropna: If True, drop rows with NaN in feature_cols + target_cols.
            Set False for models that handle NaN natively (e.g. XGBoost).
    """
    if target_cols is None:
        target_cols = list(TARGET_COLUMNS)

    df = df.sort_values("date_utc").reset_index(drop=True)

    if dropna:
        df = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

    train_mask = df["date_utc"] < WC_2022_START
    holdout_mask = (df["date_utc"] >= WC_2022_START) & (df["date_utc"] <= WC_2022_END)

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
    )


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
