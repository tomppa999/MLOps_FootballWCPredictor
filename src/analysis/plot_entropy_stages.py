"""Plot RQ2 Shannon-entropy trajectories by tournament stage.

Sibling of ``plot_entropy_trajectory``: one panel per stage column
(``entropy_r32`` … ``entropy_winner``), frozen vs per-round for each roster
model, with mean-rate as the flat floor.

Usage:
    python -m src.analysis.plot_entropy_stages [--output docs/figures/entropy-stages.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt

from src.analysis.plot_entropy_trajectory import (
    MODEL_STYLES,
    first_kickoff,
    load_entropy,
    matchday_boundaries,
)
from src.analysis.rq2_paired import STAGES
from src.analysis.rq_datasets.paths import ENTROPY_H_MAX, RQ2_PATH

DEFAULT_OUTPUT: Final[Path] = Path("docs/figures/entropy-stages.png")

STAGE_TITLES: Final[dict[str, str]] = {
    "entropy_r32": "R32 advancement",
    "entropy_r16": "R16 advancement",
    "entropy_qf": "QF advancement",
    "entropy_sf": "SF advancement",
    "entropy_final": "Final advancement",
    "entropy_winner": "Winner",
}


def plot_entropy_stages(df, *, x_start) -> plt.Figure:
    """Render a 2×3 grid of stage-entropy trajectories."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    axes_flat = axes.ravel()

    boundaries = matchday_boundaries(df)
    x_end = df["inference_timestamp"].max()
    x_pad = (x_end - x_start) * 0.01

    legend_handles = None
    legend_labels = None

    for ax, stage in zip(axes_flat, STAGES):
        for label, boundary in boundaries.items():
            ax.axvline(boundary, color="0.85", linewidth=0.7, zorder=1)

        for model, style in MODEL_STYLES.items():
            for mode in ("frozen", "per_round"):
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
                    series[stage],
                    color=style[mode],
                    linewidth=1.3 if mode == "per_round" else 1.0,
                    linestyle="-" if mode == "per_round" else "--",
                    label=legend_label,
                    zorder=3 if mode == "frozen" else 2,
                )
                reconstructed = series[series["synthetic"]]
                if not reconstructed.empty:
                    ax.scatter(
                        reconstructed["inference_timestamp"],
                        reconstructed[stage],
                        color=style[mode],
                        marker="x",
                        s=30,
                        linewidths=1.2,
                        zorder=5,
                    )

        ax.axhline(
            ENTROPY_H_MAX,
            color="0.6",
            linewidth=0.8,
            linestyle=":",
            zorder=1,
        )
        ax.set_title(STAGE_TITLES[stage], fontsize=10)
        ax.set_ylim(0, ENTROPY_H_MAX * 1.06)
        ax.set_xlim(x_start, x_end + x_pad)
        ax.grid(axis="y", alpha=0.25)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # Matchday labels on the top row only.
    for ax in axes[0]:
        for label, boundary in boundaries.items():
            ax.annotate(
                label,
                xy=(boundary, ENTROPY_H_MAX * 1.02),
                xytext=(2, 0),
                textcoords="offset points",
                fontsize=6,
                color="0.45",
                va="center",
            )

    for ax in axes[1]:
        ax.set_xlabel("Inference cycle (UTC)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Shannon entropy (nats)", fontsize=8)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        fontsize=7,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Uncertainty resolution by tournament stage: frozen vs per-round\n"
        "(× marks reconstructed snapshots; dotted line = ln 48)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
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
    figure = plot_entropy_stages(df, x_start=x_start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
