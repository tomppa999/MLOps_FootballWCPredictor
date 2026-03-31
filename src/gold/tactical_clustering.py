"""Tactical clustering: fit KMeans on rolling tactical profiles, assign labels + distances.

The clustering input is the 5-column rolling tactical profile per team entering
each match. Epoch-based fitting: the model is fit once on all available profiles
and then applied to every row. Rows with NaN profiles receive NaN cluster/distance.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.gold.rolling_features import TACTICAL_STAT_SUFFIXES

logger = logging.getLogger(__name__)

DEFAULT_N_CLUSTERS: Final[int] = 4
DEFAULT_RANDOM_STATE: Final[int] = 42

TACTICAL_PROFILE_COLUMNS_HOME: Final[list[str]] = [
    f"home_team_rolling_tac_{s}" for s in TACTICAL_STAT_SUFFIXES
]
TACTICAL_PROFILE_COLUMNS_AWAY: Final[list[str]] = [
    f"away_team_rolling_tac_{s}" for s in TACTICAL_STAT_SUFFIXES
]


def fit_tactical_clusters(
    profiles: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Fit StandardScaler + KMeans on a matrix of tactical profiles.

    Parameters
    ----------
    profiles : np.ndarray
        Shape (n_samples, 5). Must not contain NaN.
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline with steps ``scaler`` and ``kmeans``.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)),
    ])
    pipe.fit(profiles)
    logger.info(
        "Fitted tactical clusters: k=%d on %d profiles, inertia=%.1f",
        n_clusters, len(profiles), pipe.named_steps["kmeans"].inertia_,
    )
    return pipe


def _predict_and_distance(
    row_profile: np.ndarray, pipeline: Pipeline
) -> tuple[int, float]:
    """Predict cluster label and compute distance to assigned centroid."""
    scaled = pipeline.named_steps["scaler"].transform(row_profile.reshape(1, -1))
    label = pipeline.named_steps["kmeans"].predict(scaled)[0]
    centroid = pipeline.named_steps["kmeans"].cluster_centers_[label]
    dist = float(np.linalg.norm(scaled[0] - centroid))
    return int(label), dist


def assign_tactical_clusters(
    df: pd.DataFrame, pipeline: Pipeline
) -> pd.DataFrame:
    """Assign tactical cluster labels and centroid distances to Gold rows.

    For each side (home/away), extracts the rolling tactical profile, predicts
    the cluster, and computes the Euclidean distance to the assigned centroid
    in scaled space. Rows with any NaN in the profile get NaN for both columns.
    """
    df = df.copy()

    for side, profile_cols in [
        ("home", TACTICAL_PROFILE_COLUMNS_HOME),
        ("away", TACTICAL_PROFILE_COLUMNS_AWAY),
    ]:
        labels = []
        dists = []
        for _, row in df.iterrows():
            raw = row[profile_cols]
            has_na = raw.isna().any()
            if has_na:
                labels.append(pd.NA)
                dists.append(np.nan)
            else:
                profile = np.array(raw.values, dtype=np.float64)
                lbl, d = _predict_and_distance(profile, pipeline)
                labels.append(lbl)
                dists.append(d)

        df[f"{side}_tactical_cluster"] = pd.array(labels, dtype="Int8")
        df[f"{side}_tactical_cluster_dist"] = dists

    return df
