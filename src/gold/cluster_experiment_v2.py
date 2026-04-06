"""Experiment v2: compare column subsets and PCA for tactical clustering k selection.

Runs five variants side-by-side without touching any production code or Gold data:

    all5    — original 5 columns (baseline)
    core3   — possession_pct, total_shots, fouls
    style2  — possession_pct, shot_precision
    pca2    — PCA(2 components) from all 5 columns
    pca3    — PCA(3 components) from all 5 columns

Usage:
    python -m src.gold.cluster_experiment_v2 [--data-root DATA_ROOT] [--save-plot]

Does NOT modify Gold or any existing figures.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.gold.rolling_features import TACTICAL_STAT_SUFFIXES
from src.gold.tactical_clustering import (
    TACTICAL_PROFILE_COLUMNS_AWAY,
    TACTICAL_PROFILE_COLUMNS_HOME,
)

logger = logging.getLogger(__name__)

K_RANGE = range(2, 11)

# Each entry: (label, column subset or None for PCA, pca_components or None)
VARIANTS: list[tuple[str, list[str] | None, int | None]] = [
    ("all5",   ["total_shots", "shot_precision", "fouls", "corner_kicks", "possession_pct"], None),
    ("core3",  ["possession_pct", "total_shots", "fouls"],                                   None),
    ("style2", ["possession_pct", "shot_precision"],                                         None),
    ("pca2",   None,                                                                          2),
    ("pca3",   None,                                                                          3),
]


def extract_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean DataFrame of tactical profiles (home + away, no NaN rows)."""
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
    return clean


def scale_and_optionally_pca(
    profiles_df: pd.DataFrame,
    columns: list[str] | None,
    pca_components: int | None,
    random_state: int = 42,
) -> tuple[np.ndarray, list[float] | None]:
    """Scale profiles, optionally select columns, optionally apply PCA.

    Returns
    -------
    scaled : np.ndarray
        Input ready for KMeans.
    explained_variance_ratio : list[float] or None
        Per-component explained variance when PCA is used, else None.
    """
    scaler = StandardScaler()

    if pca_components is not None:
        # Use all 5 columns, then reduce with PCA
        scaled_full = scaler.fit_transform(profiles_df[TACTICAL_STAT_SUFFIXES].values)
        pca = PCA(n_components=pca_components, random_state=random_state)
        scaled = pca.fit_transform(scaled_full)
        return scaled, list(pca.explained_variance_ratio_)
    else:
        assert columns is not None
        scaled = scaler.fit_transform(profiles_df[columns].values)
        return scaled, None


def run_variant(
    profiles_df: pd.DataFrame,
    columns: list[str] | None,
    pca_components: int | None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[float] | None]:
    """Run KMeans for k=2..10 for one variant.

    Returns a results DataFrame and the PCA explained variance (or None).
    """
    scaled, evr = scale_and_optionally_pca(
        profiles_df, columns, pca_components, random_state
    )

    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(scaled)
        sil = silhouette_score(scaled, labels)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})

    return pd.DataFrame(rows), evr


def save_comparison_plot(
    all_results: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Save a 2-row grid: elbow (top) and silhouette (bottom), one column per variant."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    variants = list(all_results.keys())
    n = len(variants)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), sharey="row")

    for col, name in enumerate(variants):
        res = all_results[name]

        ax_elbow = axes[0, col]
        ax_sil = axes[1, col]

        ax_elbow.plot(res["k"], res["inertia"], "bo-", markersize=4)
        ax_elbow.set_title(name, fontsize=11, fontweight="bold")
        ax_elbow.set_xlabel("k")
        ax_elbow.set_xticks(list(K_RANGE))
        if col == 0:
            ax_elbow.set_ylabel("Inertia")

        ax_sil.plot(res["k"], res["silhouette"], "rs-", markersize=4)
        ax_sil.set_xlabel("k")
        ax_sil.set_xticks(list(K_RANGE))
        if col == 0:
            ax_sil.set_ylabel("Silhouette Score")

        best_k = int(res.loc[res["silhouette"].idxmax(), "k"])
        best_sil = res["silhouette"].max()
        ax_sil.axvline(best_k, color="gray", linestyle="--", linewidth=0.8)
        ax_sil.annotate(
            f"k={best_k}\n({best_sil:.3f})",
            xy=(best_k, best_sil),
            xytext=(best_k + 0.3, best_sil - 0.01),
            fontsize=7,
            color="darkred",
        )

    fig.suptitle("Tactical clustering: column subset / PCA comparison", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Comparison plot saved to %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Tactical clustering column-subset / PCA comparison experiment"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--save-plot", action="store_true",
        help="Save comparison plot to docs/figures/cluster_experiment_v2.png",
    )
    args = parser.parse_args()

    gold_path = args.data_root / "gold" / "matches.parquet"
    if not gold_path.exists():
        logger.error("Gold parquet not found at %s. Run build_gold first.", gold_path)
        return

    df = pd.read_parquet(gold_path)
    profiles_df = extract_profiles(df)

    if len(profiles_df) < 20:
        logger.error("Too few profiles (%d) for meaningful clustering", len(profiles_df))
        return

    all_results: dict[str, pd.DataFrame] = {}
    summary_rows = []

    print("\n" + "=" * 72)
    print(f"{'variant':<10} {'k_best_sil':>10} {'best_sil':>10} {'k_best_inertia_drop':>20}")
    print("=" * 72)

    for name, columns, pca_components in VARIANTS:
        logger.info("--- Running variant: %s ---", name)
        results, evr = run_variant(profiles_df, columns, pca_components)
        all_results[name] = results

        best_sil_k = int(results.loc[results["silhouette"].idxmax(), "k"])
        best_sil = results["silhouette"].max()

        # Elbow heuristic: largest second-derivative of inertia
        inertia = results["inertia"].values
        second_diff = np.diff(np.diff(inertia))
        elbow_k = int(results["k"].values[1:-1][np.argmax(second_diff)])

        print(f"{name:<10} {best_sil_k:>10} {best_sil:>10.4f} {elbow_k:>20}")

        evr_str = (
            "  PCA explained variance: " + ", ".join(f"{v:.3f}" for v in evr)
            + f"  (total {sum(evr):.3f})"
            if evr is not None
            else ""
        )
        if evr_str:
            print(f"{'':10}{evr_str}")

        summary_rows.append({
            "variant": name,
            "best_sil_k": best_sil_k,
            "best_silhouette": round(best_sil, 4),
            "elbow_k": elbow_k,
        })

    print("=" * 72)

    summary = pd.DataFrame(summary_rows)
    print("\n=== Full silhouette tables ===")
    for name in all_results:
        print(f"\n-- {name} --")
        print(all_results[name].to_string(index=False))

    if args.save_plot:
        save_comparison_plot(
            all_results,
            Path("docs/figures/cluster_experiment_v2.png"),
        )


if __name__ == "__main__":
    main()
