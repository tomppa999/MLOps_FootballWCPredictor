"""Plot RQ2 Shannon-entropy trajectories by tournament stage.

Generate the six-panel overview and a full-size plot for each target
(``entropy_r32`` … ``entropy_winner``). Individual plots use a clearly labelled
zoomed y-axis where the target's entropy range does not reach zero.

Usage:
    python -m src.analysis.plot_entropy_stages [--output docs/figures/entropy-stages.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.plot_entropy_trajectory import (
    MODEL_STYLES,
    first_kickoff,
    load_entropy,
    load_round_kickoffs,
    matchday_boundaries,
)
from src.analysis.rq2_paired import STAGES
from src.analysis.rq_datasets.paths import ENTROPY_H_MAX, RQ2_PATH

DEFAULT_OUTPUT: Final[Path] = Path("docs/figures/entropy-stages.png")
DEFAULT_TARGETS_DIR: Final[Path] = Path("docs/figures/entropy-targets")

STAGE_TITLES: Final[dict[str, str]] = {
    "entropy_r32": "R32 advancement",
    "entropy_r16": "R16 advancement",
    "entropy_qf": "QF advancement",
    "entropy_sf": "SF advancement",
    "entropy_final": "Final advancement",
    "entropy_winner": "Winner",
}


def _plot_stage_lines(ax, df: pd.DataFrame, stage: str, *, standalone: bool = False) -> None:
    """Draw the same saved curves in the overview and individual target views."""
    for model, style in MODEL_STYLES.items():
        for mode in ("frozen", "per_round"):
            if model == "mean_rate_poisson" and mode == "frozen":
                continue
            series = df[(df["model_name"] == model) & (df["cadence_mode"] == mode)].sort_values(
                "inference_timestamp"
            )
            if series.empty:
                continue
            label = style["label"]
            if model != "mean_rate_poisson":
                label = f"{label} ({mode.replace('_', '-')})"
            width = (
                (1.8 if mode == "per_round" else 1.4)
                if standalone
                else (1.3 if mode == "per_round" else 1.0)
            )
            ax.plot(
                series["inference_timestamp"],
                series[stage],
                color=style[mode],
                linewidth=width,
                linestyle="-" if mode == "per_round" else "--",
                label=label,
                zorder=3 if mode == "frozen" else 2,
            )
            reconstructed = series[series["synthetic"]]
            if not reconstructed.empty:
                ax.scatter(
                    reconstructed["inference_timestamp"],
                    reconstructed[stage],
                    color=style[mode],
                    marker="x",
                    s=40 if standalone else 30,
                    linewidths=1.3 if standalone else 1.2,
                    zorder=5,
                )


def plot_entropy_target(
    df: pd.DataFrame,
    *,
    stage: str,
    x_start: pd.Timestamp,
    round_kickoffs: dict[str, pd.Timestamp],
) -> plt.Figure:
    """Full-size target plot with date labels, its own legend and an honest zoom."""
    if stage not in STAGES:
        raise ValueError(f"Unknown entropy target: {stage}")
    visible = df[(df["inference_timestamp"] >= x_start) & df["model_name"].isin(MODEL_STYLES)]
    values = visible[stage].to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"Missing or non-finite visible entropy for {stage}")
    slots = {
        "entropy_r32": 32,
        "entropy_r16": 16,
        "entropy_qf": 8,
        "entropy_sf": 4,
        "entropy_final": 2,
        "entropy_winner": 1,
    }
    settled = float(np.log(slots[stage]))
    low, high = min(float(values.min()), settled), max(float(values.max()), ENTROPY_H_MAX)
    padding = max((high - low) * 0.04, 0.01)
    y_min, y_max = max(0.0, low - padding), high + padding

    fig, ax = plt.subplots(figsize=(11, 7))
    _plot_stage_lines(ax, df, stage, standalone=True)
    for label, cycle_start in matchday_boundaries(df).items():
        boundary = round_kickoffs.get(label, cycle_start)
        ax.axvline(boundary, color="0.85", linewidth=0.8, zorder=1)
        ax.annotate(
            label,
            xy=(boundary, 0.90 if label == "Complete" else 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(-3 if label in {"Final", "Complete"} else 3, 0),
            textcoords="offset points",
            fontsize=10,
            color="0.4",
            ha="right" if label in {"Final", "Complete"} else "left",
            va="top",
        )
    ax.axhline(ENTROPY_H_MAX, color="0.6", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_ylim(y_min, y_max)
    x_end = visible["inference_timestamp"].max()
    ax.set_xlim(x_start, x_end + (x_end - x_start) * 0.01)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(labelsize=11)
    ax.set_xlabel("Forecast timestamp (UTC)", fontsize=12)
    ax.set_ylabel("Shannon entropy (nats)", fontsize=12)
    ax.set_title(
        "Frozen vs per-round refitting" + ("; zoomed y-axis" if y_min > 0 else ""),
        fontsize=11,
        pad=20,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{STAGE_TITLES[stage]} forecast entropy over WC 2026", fontsize=15, y=0.97)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=10.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.04),
    )
    fig.text(
        0.5,
        0.01,
        "Both arms use frozen forecasts until the first valid refit. × marks reconstructed snapshots.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.29, top=0.84)
    return fig


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

        _plot_stage_lines(ax, df, stage)

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
        "Forecast entropy by tournament target: frozen vs per-round\n"
        "Both arms use frozen forecasts before the first valid refit. × marks reconstructed snapshots.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RQ2_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets-dir", type=Path, default=DEFAULT_TARGETS_DIR)
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
    round_kickoffs = load_round_kickoffs()
    args.targets_dir.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        target = plot_entropy_target(
            df, stage=stage, x_start=x_start, round_kickoffs=round_kickoffs
        )
        target_path = args.targets_dir / f"{stage.replace('_', '-')}.png"
        target.savefig(target_path, dpi=240)
        plt.close(target)
        print(f"Saved individual target figure to {target_path}")


if __name__ == "__main__":
    main()
