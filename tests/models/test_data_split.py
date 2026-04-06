"""Tests for src.models.data_split — splits, date boundaries, leakage, CV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.data_split import (
    WC_2022_END,
    WC_2022_START,
    DataSplits,
    make_splits,
    walk_forward_cv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FEATURE_COLS = ["f1", "f2"]
_TARGET_COLS = ["home_goals", "away_goals"]


def _make_gold_df(n: int = 200) -> pd.DataFrame:
    """Synthetic Gold-like DataFrame spanning 2020–2024."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=n, freq="5D")
    return pd.DataFrame(
        {
            "date_utc": dates,
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "home_goals": rng.integers(0, 5, size=n),
            "away_goals": rng.integers(0, 5, size=n),
        }
    )


# ---------------------------------------------------------------------------
# make_splits
# ---------------------------------------------------------------------------


class TestMakeSplits:
    def test_returns_named_tuple(self):
        df = _make_gold_df()
        splits = make_splits(df, _FEATURE_COLS, _TARGET_COLS)
        assert isinstance(splits, DataSplits)

    def test_train_dates_before_wc2022(self):
        df = _make_gold_df()
        splits = make_splits(df, _FEATURE_COLS, _TARGET_COLS)
        assert (splits.df_train["date_utc"] < WC_2022_START).all()

    def test_holdout_dates_within_wc2022(self):
        df = _make_gold_df()
        splits = make_splits(df, _FEATURE_COLS, _TARGET_COLS)
        if len(splits.df_holdout) > 0:
            assert (splits.df_holdout["date_utc"] >= WC_2022_START).all()
            assert (splits.df_holdout["date_utc"] <= WC_2022_END).all()

    def test_no_overlap_between_train_and_holdout(self):
        df = _make_gold_df()
        splits = make_splits(df, _FEATURE_COLS, _TARGET_COLS)
        if len(splits.df_holdout) > 0:
            assert splits.df_train["date_utc"].max() < splits.df_holdout["date_utc"].min()

    def test_full_includes_all_rows(self):
        df = _make_gold_df()
        splits = make_splits(df, _FEATURE_COLS, _TARGET_COLS)
        assert len(splits.df_full) == len(df)

    def test_array_shapes_consistent(self):
        df = _make_gold_df()
        splits = make_splits(df, _FEATURE_COLS, _TARGET_COLS)
        assert splits.X_train.shape == (len(splits.df_train), len(_FEATURE_COLS))
        assert splits.y_train.shape == (len(splits.df_train), len(_TARGET_COLS))

    def test_dropna_removes_nan_rows(self):
        df = _make_gold_df()
        df.loc[0, "f1"] = np.nan
        df.loc[1, "f2"] = np.nan
        splits = make_splits(df, _FEATURE_COLS, _TARGET_COLS, dropna=True)
        assert len(splits.df_full) == len(df) - 2

    def test_dropna_false_keeps_nan(self):
        df = _make_gold_df()
        df.loc[0, "f1"] = np.nan
        splits = make_splits(df, _FEATURE_COLS, _TARGET_COLS, dropna=False)
        assert len(splits.df_full) == len(df)

    def test_no_future_leakage_train_before_holdout(self):
        """Training set must never contain dates >= WC 2022 start."""
        df = _make_gold_df(400)
        splits = make_splits(df, _FEATURE_COLS, _TARGET_COLS)
        if len(splits.df_train) > 0:
            assert splits.df_train["date_utc"].max() < pd.Timestamp(WC_2022_START)


# ---------------------------------------------------------------------------
# walk_forward_cv
# ---------------------------------------------------------------------------


class TestWalkForwardCV:
    def test_correct_number_of_folds(self):
        folds = walk_forward_cv(100, n_splits=5)
        assert len(folds) == 5

    def test_training_window_expands(self):
        folds = walk_forward_cv(100, n_splits=5, min_train_frac=0.5)
        train_sizes = [len(tr) for tr, _ in folds]
        assert train_sizes == sorted(train_sizes)
        assert len(set(train_sizes)) == 5  # all different

    def test_no_overlap_between_train_and_val(self):
        folds = walk_forward_cv(100, n_splits=3)
        for train_idx, val_idx in folds:
            assert len(np.intersect1d(train_idx, val_idx)) == 0

    def test_val_always_after_train(self):
        folds = walk_forward_cv(100, n_splits=4)
        for train_idx, val_idx in folds:
            assert train_idx.max() < val_idx.min()

    def test_last_fold_covers_tail(self):
        folds = walk_forward_cv(100, n_splits=5)
        _, last_val = folds[-1]
        assert last_val[-1] == 99  # last sample index

    def test_all_val_indices_cover_tail_partition(self):
        """Union of all validation indices should cover the non-initial portion."""
        folds = walk_forward_cv(100, n_splits=5, min_train_frac=0.5)
        all_val = np.concatenate([v for _, v in folds])
        assert len(all_val) == len(np.unique(all_val)), "Validation indices must not overlap"
        assert all_val.min() == 50
        assert all_val.max() == 99

    def test_raises_on_bad_params(self):
        with pytest.raises(ValueError):
            walk_forward_cv(10, n_splits=0)
        with pytest.raises(ValueError):
            walk_forward_cv(10, n_splits=5, min_train_frac=1.0)
        with pytest.raises(ValueError):
            walk_forward_cv(5, n_splits=10, min_train_frac=0.5)
