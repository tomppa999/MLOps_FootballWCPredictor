"""Tests for src.gold.tactical_clustering."""

import numpy as np
import pandas as pd
import pytest

from src.gold.tactical_clustering import (
    TACTICAL_PROFILE_COLUMNS_HOME,
    TACTICAL_PROFILE_COLUMNS_AWAY,
    assign_tactical_clusters,
    fit_tactical_clusters,
)


class TestFitClusters:
    def _make_profiles(self, n: int = 100, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, 5))

    def test_returns_pipeline(self):
        pipe = fit_tactical_clusters(self._make_profiles(), n_clusters=3)
        assert hasattr(pipe, "named_steps")
        assert "scaler" in pipe.named_steps
        assert "kmeans" in pipe.named_steps

    def test_correct_n_clusters(self):
        pipe = fit_tactical_clusters(self._make_profiles(), n_clusters=5)
        assert pipe.named_steps["kmeans"].n_clusters == 5

    def test_deterministic(self):
        profiles = self._make_profiles()
        pipe1 = fit_tactical_clusters(profiles, n_clusters=3, random_state=0)
        pipe2 = fit_tactical_clusters(profiles, n_clusters=3, random_state=0)
        np.testing.assert_array_equal(
            pipe1.named_steps["kmeans"].cluster_centers_,
            pipe2.named_steps["kmeans"].cluster_centers_,
        )


class TestAssignClusters:
    def _make_df_and_pipeline(self, n_rows: int = 50, n_clusters: int = 3):
        rng = np.random.default_rng(42)
        data = {}
        for col in TACTICAL_PROFILE_COLUMNS_HOME + TACTICAL_PROFILE_COLUMNS_AWAY:
            data[col] = rng.standard_normal(n_rows)
        df = pd.DataFrame(data)

        profiles = rng.standard_normal((200, 5))
        pipeline = fit_tactical_clusters(profiles, n_clusters=n_clusters)
        return df, pipeline

    def test_cluster_labels_in_range(self):
        df, pipe = self._make_df_and_pipeline(n_clusters=3)
        result = assign_tactical_clusters(df, pipe)
        for side in ("home", "away"):
            labels = result[f"{side}_tactical_cluster"].dropna()
            assert labels.min() >= 0
            assert labels.max() < 3

    def test_distances_non_negative(self):
        df, pipe = self._make_df_and_pipeline()
        result = assign_tactical_clusters(df, pipe)
        for side in ("home", "away"):
            dists = result[f"{side}_tactical_cluster_dist"].dropna()
            assert (dists >= 0).all()

    def test_nan_profile_gets_nan_cluster(self):
        df, pipe = self._make_df_and_pipeline(n_rows=5)
        df.iloc[0, :5] = np.nan  # set home profile to NaN
        result = assign_tactical_clusters(df, pipe)
        assert pd.isna(result["home_tactical_cluster"].iloc[0])
        assert pd.isna(result["home_tactical_cluster_dist"].iloc[0])

    def test_does_not_mutate_input(self):
        df, pipe = self._make_df_and_pipeline(n_rows=5)
        original_cols = set(df.columns)
        _ = assign_tactical_clusters(df, pipe)
        assert set(df.columns) == original_cols
