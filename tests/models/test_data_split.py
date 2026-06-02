"""Tests for src.models.data_split — splits, date boundaries, leakage, CV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.data_split import (
    DEFAULT_HALF_PERIOD_YEARS,
    HOLDOUT_TOURNAMENTS,
    IMPORTANCE_WEIGHTS,
    WC_2022_END,
    WC_2022_START,
    DataSplits,
    compute_sample_weights,
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
            "competition_tier": 1,
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

    def test_holdout_after_training_cutoff(self):
        df = _make_gold_df()
        splits = make_splits(df, _FEATURE_COLS, _TARGET_COLS)
        if len(splits.df_holdout) > 0:
            assert (splits.df_holdout["date_utc"] >= WC_2022_START).all()

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
# Expanded holdout (A.3)
# ---------------------------------------------------------------------------


def _tournament_rows(league_id: int, tier: int, start: str, n: int) -> pd.DataFrame:
    """n consecutive daily rows for a tournament window."""
    rng = np.random.default_rng(league_id)
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({
        "date_utc": dates,
        "league_id": league_id,
        "competition_tier": tier,
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
        "home_goals": rng.integers(0, 5, size=n),
        "away_goals": rng.integers(0, 5, size=n),
    })


class TestExpandedHoldout:
    def _df(self) -> pd.DataFrame:
        pre = _make_gold_df(100)
        pre["league_id"] = 10  # friendlies before the cutoff (training)
        wc = _tournament_rows(1, 1, "2022-11-21", 8)              # WC 2022
        euro = _tournament_rows(4, 2, "2024-06-15", 6)            # EURO 2024
        afcon24 = _tournament_rows(6, 2, "2024-01-15", 5)         # AFCON 2024
        afcon25 = _tournament_rows(6, 2, "2025-12-22", 4)         # AFCON 2025
        # Friendlies inside the EURO window must be excluded (league_id 10).
        friendly = _tournament_rows(10, 4, "2024-06-15", 3)
        return pd.concat(
            [pre, wc, euro, afcon24, afcon25, friendly], ignore_index=True
        )

    def test_holdout_unions_multiple_tournaments(self):
        splits = make_splits(self._df(), _FEATURE_COLS, _TARGET_COLS)
        assert len(splits.df_holdout) == 8 + 6 + 5 + 4  # 23

    def test_holdout_excludes_friendlies_inside_window(self):
        splits = make_splits(self._df(), _FEATURE_COLS, _TARGET_COLS)
        assert (splits.df_holdout["league_id"] != 10).all()

    def test_two_editions_same_league_kept_apart(self):
        """AFCON 2024 and 2025 share league_id 6 but distinct date windows."""
        splits = make_splits(self._df(), _FEATURE_COLS, _TARGET_COLS)
        afcon = splits.df_holdout[splits.df_holdout["league_id"] == 6]
        assert len(afcon) == 5 + 4

    def test_training_excludes_all_holdout(self):
        splits = make_splits(self._df(), _FEATURE_COLS, _TARGET_COLS)
        assert (splits.df_train["date_utc"] < pd.Timestamp(WC_2022_START)).all()

    def test_all_tournaments_have_distinct_windows(self):
        """Sanity: no two configured windows overlap for the same league_id."""
        by_league: dict[int, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for _name, lid, start, end in HOLDOUT_TOURNAMENTS:
            by_league.setdefault(lid, []).append(
                (pd.Timestamp(start), pd.Timestamp(end))
            )
        for windows in by_league.values():
            windows.sort()
            for (s1, e1), (s2, e2) in zip(windows, windows[1:]):
                assert e1 < s2


# ---------------------------------------------------------------------------
# Sample weights (A.4)
# ---------------------------------------------------------------------------


class TestSampleWeights:
    def test_weight_at_half_period_is_half_importance(self):
        ref = pd.Timestamp("2024-01-01")
        # 4 years = exactly 1461 days (4 * 365.25), so days_ago == half-life and
        # w_time == 0.5 exactly (avoids integer-day rounding noise).
        half_period = 4.0
        half_days = 1461
        df = pd.DataFrame({
            "date_utc": [ref, ref - pd.Timedelta(days=half_days)],
            "competition_tier": [1, 1],  # importance 4
        })
        w = compute_sample_weights(df, half_period_years=half_period, reference_date=ref)
        assert w[0] == pytest.approx(IMPORTANCE_WEIGHTS[1])          # today: full importance
        assert w[1] == pytest.approx(IMPORTANCE_WEIGHTS[1] * 0.5)    # half-life ago: halved

    def test_recent_wc_match_has_highest_weight(self):
        ref = pd.Timestamp("2024-01-01")
        df = pd.DataFrame({
            "date_utc": [ref, ref, ref - pd.Timedelta(days=30)],
            "competition_tier": [1, 4, 1],  # WC today, friendly today, WC last month
        })
        w = compute_sample_weights(df, reference_date=ref)
        assert w[0] == max(w)

    def test_old_friendly_near_zero(self):
        ref = pd.Timestamp("2024-01-01")
        df = pd.DataFrame({
            "date_utc": [ref - pd.Timedelta(days=365 * 30)],  # 30 years (~10 half-lives)
            "competition_tier": [4],
        })
        w = compute_sample_weights(df, reference_date=ref)
        assert w[0] < 0.01

    def test_importance_ordering(self):
        ref = pd.Timestamp("2024-01-01")
        df = pd.DataFrame({
            "date_utc": [ref, ref, ref, ref],
            "competition_tier": [1, 2, 3, 4],
        })
        w = compute_sample_weights(df, reference_date=ref)
        assert list(w) == sorted(w, reverse=True)  # tier 1 > 2 > 3 > 4

    def test_make_splits_populates_weight_arrays(self):
        df = _make_gold_df(120)
        splits = make_splits(df, _FEATURE_COLS, _TARGET_COLS)
        assert splits.w_train.shape == (len(splits.df_train),)
        assert splits.w_full.shape == (len(splits.df_full),)
        assert (splits.w_train > 0).all()

    def test_empty_df_returns_empty_weights(self):
        empty = pd.DataFrame({"date_utc": pd.to_datetime([]), "competition_tier": []})
        w = compute_sample_weights(empty)
        assert w.shape == (0,)


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
