"""Plot the RQ2 Shannon-entropy trajectory: frozen vs per-round retraining.

Usage:
    python -m src.analysis.plot_entropy_trajectory [--output docs/figures/entropy-trajectory.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.rq_datasets.paths import ENTROPY_H_MAX, RQ1_PATH, RQ2_PATH

DEFAULT_OUTPUT: Final[Path] = Path("docs/figures/entropy-trajectory.png")

# Darker shade = per-round, lighter shade = frozen.
MODEL_STYLES: Final[dict[str, dict[str, str]]] = {
    "xgboost": {
        "per_round": "#1b5e20",
        "frozen": "#81c784",
        "label": "XGBoost",
    },
    "bayesian_poisson": {
        "per_round": "#0d47a1",
        "frozen": "#90caf9",
        "label": "Bayesian Poisson",
    },
    "poisson_glm": {
        "per_round": "#b71c1c",
        "frozen": "#ef9a9a",
        "label": "Bivariate Poisson GLM",
    },
    "mean_rate_poisson": {
        "per_round": "#f9a825",
        "frozen": "#f9a825",
        "label": "Mean-rate Poisson (baseline)",
    },
}

MATCHDAY_ORDER: Final[tuple[str, ...]] = (
    "1", "2", "3", "R32", "R16", "QF", "SF", "Final", "Complete",
)


def load_entropy(path: Path = RQ2_PATH) -> pd.DataFrame:
    """Read the RQ2 dataset, restricted to the simulated roster."""
    df = pd.read_csv(path)
    # The logged offsets carry mixed sub-second precision, which `parse_dates`
    # leaves as object dtype — matplotlib would then treat every cycle as a
    # categorical tick instead of a point in time.
    df["inference_timestamp"] = pd.to_datetime(
        df["inference_timestamp"], utc=True, format="mixed",
    )
    df = df[df["model_name"].isin(MODEL_STYLES)]
    return df.sort_values("inference_timestamp")


def first_kickoff(path: Path = RQ1_PATH) -> pd.Timestamp:
    """Kickoff of the tournament opener, used as the left edge of the plot."""
    kickoffs = pd.read_csv(path, usecols=["kickoff_utc"])["kickoff_utc"]
    return pd.to_datetime(kickoffs, utc=True, format="mixed").min()


def matchday_boundaries(df: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """First cycle timestamp of each matchday after the first, in order."""
    firsts = df.groupby("matchday_label")["inference_timestamp"].min()
    present = [label for label in MATCHDAY_ORDER if label in firsts.index]
    return {label: firsts[label] for label in present[1:]}


def plot_entropy(df: pd.DataFrame, *, x_start: pd.Timestamp) -> plt.Figure:
    """Render winner-entropy trajectories for both cadence modes.

    Pre-tournament cycles stay in the data but are cropped from view, so the
    curves begin at the opener rather than at the dress-rehearsal runs.
    """
    fig, ax = plt.subplots(figsize=(13, 6.5))

    for label, boundary in matchday_boundaries(df).items():
        ax.axvline(boundary, color="0.85", linewidth=0.8, zorder=1)
        ax.annotate(
            label,
            xy=(boundary, ENTROPY_H_MAX * 1.02),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=7,
            color="0.45",
            va="center",
        )

    for model, style in MODEL_STYLES.items():
        for mode in ("frozen", "per_round"):
            # The baseline is never refitted, so both cadences are identical.
            if model == "mean_rate_poisson" and mode == "frozen":
                continue

            series = df[(df["model_name"] == model) & (df["cadence_mode"] == mode)]
            if series.empty:
                continue

            legend_label = style["label"]
            if model != "mean_rate_poisson":
                legend_label = f"{legend_label} ({mode.replace('_', '-')})"

            ax.plot(
                series["inference_timestamp"],
                series["entropy_winner"],
                color=style[mode],
                linewidth=1.4 if mode == "per_round" else 1.1,
                linestyle="-" if mode == "per_round" else "--",
                label=legend_label,
                # Frozen sits on top: dashed and lighter, so the per-round line
                # underneath stays visible where the two nearly coincide.
                zorder=3 if mode == "frozen" else 2,
            )

            reconstructed = series[series["synthetic"]]
            if not reconstructed.empty:
                ax.scatter(
                    reconstructed["inference_timestamp"],
                    reconstructed["entropy_winner"],
                    color=style[mode],
                    marker="x",
                    s=45,
                    linewidths=1.4,
                    zorder=5,
                )

    ax.axhline(
        ENTROPY_H_MAX,
        color="0.6",
        linewidth=0.9,
        linestyle=":",
        zorder=1,
        label=f"Uniform prior over 48 teams (ln 48 = {ENTROPY_H_MAX:.2f})",
    )

    ax.set_xlabel("Inference cycle timestamp (UTC)")
    ax.set_ylabel("Shannon entropy of winner probability (nats)")
    ax.set_title(
        "Uncertainty resolution over WC 2026: frozen vs per-round retraining\n"
        "(× marks reconstructed snapshots)",
    )
    ax.set_ylim(0, ENTROPY_H_MAX * 1.06)
    x_end = df["inference_timestamp"].max()
    ax.set_xlim(x_start, x_end + (x_end - x_start) * 0.01)
    ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RQ2_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = load_entropy(args.input)
    print(
        f"Loaded {len(df)} rows "
        f"({int(df['synthetic'].sum())} reconstructed) "
        f"for {df['model_name'].nunique()} models",
    )

    x_start = first_kickoff()
    print(f"Cropping x-axis to the tournament window, from {x_start}")

    figure = plot_entropy(df, x_start=x_start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
