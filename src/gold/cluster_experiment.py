"""Experiment script: silhouette + elbow analysis for tactical clustering k selection.

Usage:
    python -m src.gold.cluster_experiment [--data-root DATA_ROOT] [--save-plot]

Loads the Gold parquet (after rolling features are computed), extracts non-NaN
tactical profiles, and evaluates KMeans for k = 2..10. Does NOT modify Gold.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.gold.rolling_features import TACTICAL_STAT_SUFFIXES
from src.gold.tactical_clustering import (
    TACTICAL_PROFILE_COLUMNS_HOME,
    TACTICAL_PROFILE_COLUMNS_AWAY,
)

logger = logging.getLogger(__name__)

K_RANGE = range(2, 11)


def extract_profiles(df: pd.DataFrame) -> np.ndarray:
    """Extract all non-NaN tactical profiles (both home and away) from Gold rows."""
    home = df[TACTICAL_PROFILE_COLUMNS_HOME].rename(
        columns=dict(zip(TACTICAL_PROFILE_COLUMNS_HOME, TACTICAL_STAT_SUFFIXES))
    )
    away = df[TACTICAL_PROFILE_COLUMNS_AWAY].rename(
        columns=dict(zip(TACTICAL_PROFILE_COLUMNS_AWAY, TACTICAL_STAT_SUFFIXES))
    )
    combined = pd.concat([home, away], ignore_index=True)
    clean = combined.dropna()
    logger.info(
        "Extracted %d non-NaN profiles out of %d total", len(clean), len(combined)
    )
    return clean.values


def run_experiment(profiles: np.ndarray, random_state: int = 42) -> pd.DataFrame:
    """Run KMeans for k = 2..10, return DataFrame with inertia and silhouette."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(profiles)

    results = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(scaled)
        sil = silhouette_score(scaled, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        logger.info("k=%d  inertia=%.1f  silhouette=%.4f", k, km.inertia_, sil)

    return pd.DataFrame(results)


def save_plot(results: pd.DataFrame, output_path: Path) -> None:
    """Save elbow + silhouette plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(results["k"], results["inertia"], "bo-")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax1.set_xticks(list(K_RANGE))

    ax2.plot(results["k"], results["silhouette"], "rs-")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    ax2.set_xticks(list(K_RANGE))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Tactical clustering k-selection experiment")
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--save-plot", action="store_true",
        help="Save elbow + silhouette plot to docs/figures/",
    )
    args = parser.parse_args()

    gold_path = args.data_root / "gold" / "matches.parquet"
    if not gold_path.exists():
        logger.error("Gold parquet not found at %s. Run build_gold first.", gold_path)
        return

    df = pd.read_parquet(gold_path)
    profiles = extract_profiles(df)

    if len(profiles) < 20:
        logger.error("Too few profiles (%d) for meaningful clustering", len(profiles))
        return

    results = run_experiment(profiles)

    print("\n=== Cluster k-selection results ===")
    print(results.to_string(index=False))

    best_k = int(results.loc[results["silhouette"].idxmax(), "k"])
    print(f"\nBest k by silhouette: {best_k}")

    if args.save_plot:
        save_plot(results, Path("docs/figures/cluster_k_selection.png"))


if __name__ == "__main__":
    main()
